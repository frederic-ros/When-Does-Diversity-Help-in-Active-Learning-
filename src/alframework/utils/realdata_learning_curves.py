# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 06:37:26 2026

@author: frederic.ros
"""

# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import train_test_split


@dataclass
class SimpleALState:
    """
    État minimal compatible avec beaucoup d'implémentations AL:
      - pools labeled / unlabeled
      - modèle courant (cloné à chaque run)
      - test set pour évaluer dans la loop (si besoin)
    """
    X_labeled: np.ndarray
    y_labeled: np.ndarray
    X_unlabeled: np.ndarray

    # Optionnels mais souvent utiles dans ton loop/stratégies
    labels: List[int]
    model: Any

    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None


def _stratified_init_indices(
    y_train: np.ndarray,
    *,
    n_init: int,
    labels: List[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sélection initiale stratifiée dans le train:
      - assure au moins 1 par classe si possible
      - complète proportionnellement ensuite
    """
    y_train = np.asarray(y_train)
    idx_by_class = {c: np.flatnonzero(y_train == c) for c in labels}

    # classes présentes
    present = [c for c in labels if len(idx_by_class[c]) > 0]
    if len(present) == 0:
        raise ValueError("No labels present in y_train.")

    # si n_init trop petit pour couvrir toutes les classes présentes: on fait au mieux
    # (on garde la stratification en tirant aléatoirement parmi classes)
    if n_init < len(present):
        chosen_classes = rng.choice(present, size=n_init, replace=False)
        init_idx = []
        for c in chosen_classes:
            pool = idx_by_class[c]
            init_idx.append(int(rng.choice(pool)))
        return np.array(sorted(set(init_idx)), dtype=int)

    # 1 par classe présente
    init_idx = []
    for c in present:
        pool = idx_by_class[c]
        init_idx.append(int(rng.choice(pool)))

    remaining = n_init - len(init_idx)
    if remaining <= 0:
        return np.array(sorted(set(init_idx)), dtype=int)

    # Poids proportionnels à la taille de classe (sur le train)
    sizes = np.array([len(idx_by_class[c]) for c in present], dtype=float)
    probs = sizes / sizes.sum()

    # On complète en tirant des classes selon probs puis un index dans la classe
    for _ in range(remaining):
        c = int(rng.choice(present, p=probs))
        pool = idx_by_class[c]
        init_idx.append(int(rng.choice(pool)))

    # dédup: si collision, on complète avec du sampling global restant
    init_idx = list(sorted(set(init_idx)))
    if len(init_idx) < n_init:
        all_train = np.arange(len(y_train), dtype=int)
        mask = np.ones(len(all_train), dtype=bool)
        mask[init_idx] = False
        remaining_pool = all_train[mask]
        if len(remaining_pool) < (n_init - len(init_idx)):
            raise ValueError("Not enough points in train to pick n_init without duplicates.")
        extra = rng.choice(remaining_pool, size=(n_init - len(init_idx)), replace=False).tolist()
        init_idx.extend(extra)

    return np.array(sorted(init_idx), dtype=int)


def compute_learning_curves_one_dataset_iterative_from_arrays(
    *,
    X: np.ndarray,
    y: np.ndarray,
    strategy_names: Optional[List[str]] = None,
    parameterconfig: bool = False,
    strategy_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    max_budget: int = 500,
    budget_step: int = 10,
    tracked_metrics: Optional[List[str]] = None,
    n_init: int = 20,
    labels: Optional[List[int]] = None,
    test_size: float = 0.25,
    split_random_state: int = 42,
    init_random_state: int = 123,
    rng_seed: int = 42,
    print_progress: bool = True,
) -> Dict[str, Any]:
    """
    OPTION B (REAL DATA):
    Reproduit la structure et la sémantique de compute_learning_curves_one_dataset_iterative,
    mais à partir de arrays (X, y) au lieu d'un générateur synthétique.

    -> Ne nécessite AUCUNE modification de learning_curves_one_dataset.py
    -> On réutilise les mêmes STRATEGIES, _instantiate_strategy, active_learning_loop, evaluate, _compute_auc, etc.
    """
    # Import tardif pour ne pas casser si le module n'est pas chargé ailleurs
    import learning_curves_one_dataset as lc  # ton module existant

    X = np.asarray(X)
    y = np.asarray(y)

    if labels is None:
        # labels = classes uniques triées
        labels = sorted([int(c) for c in np.unique(y).tolist()])
    if tracked_metrics is None:
        tracked_metrics = list(lc.DEFAULT_TRACKED_METRICS)
    if strategy_names is None:
        strategy_names = sorted(lc.STRATEGIES.keys())
    else:
        unknown = [s for s in strategy_names if s not in lc.STRATEGIES]
        if unknown:
            raise ValueError(f"Stratégies inconnues: {unknown}")

    # Split train/test stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(test_size),
        random_state=int(split_random_state),
        stratify=y,
    )

    # Init pool labeled/unlabeled stratifié dans train
    rng_init = np.random.default_rng(int(init_random_state))
    init_idx = _stratified_init_indices(y_train, n_init=int(n_init), labels=list(labels), rng=rng_init)

    all_train_idx = np.arange(len(y_train), dtype=int)
    mask = np.ones(len(all_train_idx), dtype=bool)
    mask[init_idx] = False
    unlabeled_idx = all_train_idx[mask]

    X_labeled0 = X_train[init_idx]
    y_labeled0 = y_train[init_idx]
    X_unlabeled0 = X_train[unlabeled_idx]
    y_unlabeled_true0 = y_train[unlabeled_idx]

    base_model = RandomForestClassifier(n_estimators=200, random_state=42)

    # state0 (référence): sans stratégie
    state0 = SimpleALState(
        X_labeled=X_labeled0,
        y_labeled=y_labeled0,
        X_unlabeled=X_unlabeled0,
        labels=list(labels),
        model=clone(base_model),
        X_test=X_test,
        y_test=y_test,
    )

    max_budget = min(int(max_budget), len(state0.X_unlabeled))
    n_rounds = int(math.ceil(max_budget / int(budget_step)))

    results: Dict[str, Any] = {}

    for name in strategy_names:
        overrides = dict((strategy_overrides or {}).get(name, {}))
        strat = lc._instantiate_strategy(name, parameterconfig=parameterconfig, overrides=overrides)

        # reset state par stratégie
        state = copy.deepcopy(state0)

        # labeler sur la vérité du pool unlabeled
        labeler = lc.ArrayLabeler(y_unlabeled_true0.copy())

        history = lc.active_learning_loop(
            state=state,
            strategy=strat,
            labeler=labeler,
            n_rounds=n_rounds,
            budget=int(budget_step),
        )

        # Point initial (avant round 0)
        base_model_0 = clone(base_model).fit(state0.X_labeled, state0.y_labeled)
        m0 = lc.evaluate(base_model_0, X_test, y_test)
        curve: List[Any] = [lc.CurvePoint(
            n_selected=0,
            metrics={k: float(m0.get(k, np.nan)) for k in tracked_metrics},
        )]

        # Points de chaque round (mêmes conventions que ta version)
        for h in history:
            n_selected = int(h["n_labeled"]) - int(n_init)
            if n_selected > max_budget:
                continue
            curve.append(lc.CurvePoint(
                n_selected=n_selected,
                metrics={k: float(h.get(k, np.nan)) for k in tracked_metrics},
            ))

        auc, auc_norm = lc._compute_auc(curve, tracked_metrics)

        results[name] = lc.LearningCurveResult(
            selected_indices=np.array([], dtype=int),
            curve=curve,
            auc=auc,
            auc_norm=auc_norm,
            params=dict(getattr(strat, "__dict__", {})),
        )

        if print_progress:
            last_acc = results[name].curve[-1].metrics.get("accuracy", np.nan)
            print(f"{name:>24} | max_budget={max_budget:>4} | acc@max={last_acc:.4f} "
                  f"| AULC(acc)={auc_norm['accuracy']:.4f}")

    return {
        "setup": dict(
            mode="iterative_realdata_arrays",
            parameterconfig=bool(parameterconfig),
            strategy_overrides={k: dict(v) for k, v in (strategy_overrides or {}).items()},
            max_budget=int(max_budget),
            budget_step=int(budget_step),
            n_rounds=int(n_rounds),
            tracked_metrics=list(tracked_metrics),
            n_init=int(n_init),
            labels=list(labels),
            test_size=float(test_size),
            split_random_state=int(split_random_state),
            init_random_state=int(init_random_state),
            rng_seed=int(rng_seed),
            dataset_params=dict(source="arrays"),
        ),
        "strategies": results,
    }