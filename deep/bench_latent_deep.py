# -*- coding: utf-8 -*-
"""
tests/benchtestrealdata.py
==========================

Benchmark sur données réelles.

Pipeline reproductible :
1) chargement dataset complet
2) création de n_strates stratifiées de taille fixe
3) pour chaque strate : n_splits train/test stratifiés
4) pour chaque split : boucle active learning
5) agrégation par fichier, par série, et all-splits

Paramètres importants :
- n_strates : nombre de strates stratifiées
- strate_size : taille totale de chaque strate
- test_size : pourcentage test, ex. 0.30 = 30%
- init_mode : "random" ou "stratified"

Important :
Le StandardScaler est fit uniquement sur X_train puis appliqué à X_test.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Classifieur utilisé par ce bench (RandomForest only). Sert au nommage des
# historiques pour rester cohérent avec bench_real (history_rf_...).
MODEL_TAG = "rf"


# ─── Path setup ───────────────────────────────────────────────────────────────
this_file = Path(__file__).resolve()
project_root = this_file.parent

while project_root != project_root.parent:
    src = project_root / "src"
    if src.exists():
        break
    project_root = project_root.parent

if str(src) not in sys.path:
    sys.path.insert(0, str(src))

utils_dir = src / "alframework" / "utils"
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))


# ─── Utils existants ──────────────────────────────────────────────────────────
from alframework.utils.realdata_bench_utils import (
    find_dataset_files,
    load_tabular_txt,
    make_stratified_splits,
    aggregate_scores,
)

from alframework.utils.realdata_plot_utils import (
    plot_curves_and_save,
    build_constant_baseline_from_out,
)

# Force registry population for newer strategies
# Import explicite afin que les décorateurs @register(...) soient exécutés.
import alframework.strategies.typiclust
import alframework.strategies.tri_committee
import alframework.strategies.active_pseudolabelv44
import alframework.strategies.active_pseudolabelv51
import alframework.strategies.active_pseudolabelv52
import alframework.strategies.active_pseudolabelv53
import alframework.strategies.active_pseudolabelv54
import alframework.strategies.strategy_v58
import alframework.strategies.coreset
import alframework.strategies.probcover
import alframework.strategies.badge
import alframework.strategies.bait_simple
import alframework.strategies.adaptive_disagreement
import alframework.strategies.diversity_optimized_batch
import alframework.strategies.selftrain_acq
import alframework.strategies.robust_qbc

def _force_register_v4_derivatives() -> None:
    """
    Force l'enregistrement des stratégies récentes utilisées dans ce benchmark.

    Important :
    - Les stratégies appelées ici reprennent le panel du fichier ablation.
    - On importe les modules pour déclencher les décorateurs @register(...).
    - On ajoute un fallback manuel pour éviter les problèmes de cache/reload sous Spyder.
    """
    import importlib
    from alframework.core.registry import STRATEGIES

    modules = [
        "alframework.strategies.random",
        "alframework.strategies.typiclust",
        "alframework.strategies.uncertainty",
        "alframework.strategies.dbal",
        "alframework.strategies.rank2022",
        "alframework.strategies.qbc",
        "alframework.strategies.tri_committee",
        "alframework.strategies.active_pseudolabelv53",
        "alframework.strategies.active_pseudolabelv54",
        "alframework.strategies.active_pseudolabelv55",
        "alframework.strategies.strategy_v58",
        "alframework.strategies.coreset",
        "alframework.strategies.probcover",
        "alframework.strategies.badge",
        "alframework.strategies.bait_simple",
        "alframework.strategies.adaptive_disagreement",
        "alframework.strategies.diversity_optimized_batch",
        "alframework.strategies.selftrain_acq",
        "alframework.strategies.robust_qbc",
    ]

    for mod_path in modules:
        try:
            mod = importlib.import_module(mod_path)
            importlib.reload(mod)
        except Exception as e:
            print(f"[WARN] could not import/reload {mod_path}: {e}")

    # Fallback manuel si reload/decorator ne suffit pas
    manual = [
        ("tri_committee", "alframework.strategies.tri_committee", "TriCommitteeDisagreement"),
        ("typiclust", "alframework.strategies.typiclust", "TypiClustSampling"),
        ("ActivePseudoLabelV53", "alframework.strategies.active_pseudolabelv53", "ActivePseudoLabelV53"),
        ("ActivePseudoLabelV54", "alframework.strategies.active_pseudolabelv54", "ActivePseudoLabelV54"),
        ("ActivePseudoLabelV55", "alframework.strategies.active_pseudolabelv55", "ActivePseudoLabelV55"),
        ("ActivePseudoLabelV58", "alframework.strategies.strategy_v58", "ActivePseudoLabelV58"),

    ]

    for reg_name, mod_path, cls_name in manual:
        try:
            mod = importlib.import_module(mod_path)
            STRATEGIES[reg_name] = getattr(mod, cls_name)
        except Exception as e:
            print(f"[WARN] {reg_name} manual register failed: {e}")
# =============================================================================
# Helpers
# =============================================================================
def _normalize_proba_rows(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, eps, 1.0)

    row_sum = P.sum(axis=1, keepdims=True)
    row_sum[row_sum <= eps] = 1.0

    return P / row_sum
# =============================================================================

def load_tabular_txt_robust(
    file_path: Path,
    *,
    delimiter: str = "\t",
    label_column: int = -1,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Charge un fichier tabulaire de façon robuste.

    Hypothèses :
    - la dernière colonne est la classe par défaut ;
    - les labels peuvent être textuels ;
    - les features peuvent être numériques ou catégorielles ;
    - les valeurs manquantes sont imputées.

    Retour :
    - X : float64, sans NaN
    - y : int64 encodé 0..C-1
    - meta : informations de mapping labels/features
    """
    import pandas as pd

    file_path = Path(file_path)

    # Lecture permissive en chaînes pour ne jamais perdre les labels texte.
    # On garde header=None car les fichiers benchmark sont généralement sans en-tête.
    try:
        df = pd.read_csv(
            file_path,
            sep=delimiter,
            header=None,
            dtype=str,
            engine="python",
        )
    except Exception:
        df = pd.read_csv(
            file_path,
            sep=None,
            header=None,
            dtype=str,
            engine="python",
        )

    # Si le séparateur fourni n'a rien séparé, on retente avec un séparateur large.
    if df.shape[1] <= 1:
        df = pd.read_csv(
            file_path,
            sep=r"[\t;, ]+",
            header=None,
            dtype=str,
            engine="python",
        )

    df = df.dropna(how="all").reset_index(drop=True)

    if df.shape[1] < 2:
        raise ValueError(
            f"{file_path.name}: expected at least 2 columns, got {df.shape[1]}"
        )

    # Nettoyage texte simple.
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "NULL": np.nan, "?": np.nan})

    n_cols = df.shape[1]
    if label_column < 0:
        label_column = n_cols + label_column

    if label_column < 0 or label_column >= n_cols:
        raise ValueError(
            f"{file_path.name}: invalid label_column={label_column} for {n_cols} columns"
        )

    feature_cols = [c for c in range(n_cols) if c != label_column]

    # Détection d'un éventuel header : première ligne très peu numérique côté features,
    # deuxième ligne majoritairement numérique.
    if len(df) >= 2:
        first_num = pd.to_numeric(
            df.iloc[0, feature_cols].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        ).notna().mean()
        second_num = pd.to_numeric(
            df.iloc[1, feature_cols].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        ).notna().mean()

        if first_num < 0.30 and second_num > 0.70:
            if verbose:
                print(f"[INFO] {file_path.name}: header row detected and dropped")
            df = df.iloc[1:].reset_index(drop=True)

    y_raw = df.iloc[:, label_column].astype(str).str.strip()
    X_raw = df.iloc[:, feature_cols].copy()

    # Labels invalides.
    invalid_y = (
        y_raw.isna()
        | (y_raw == "")
        | (y_raw.str.lower().isin(["nan", "none", "null", "?"]))
    )

    if invalid_y.any():
        if verbose:
            print(f"[WARN] {file_path.name}: dropping {int(invalid_y.sum())} rows with invalid labels")
        keep = ~invalid_y
        y_raw = y_raw.loc[keep].reset_index(drop=True)
        X_raw = X_raw.loc[keep].reset_index(drop=True)

    if len(y_raw) == 0:
        raise ValueError(f"{file_path.name}: no valid rows after label cleaning")

    # Encodage labels texte -> 0..C-1.
    le = LabelEncoder()
    y = le.fit_transform(y_raw.to_numpy()).astype(np.int64)

    if len(le.classes_) < 2:
        raise ValueError(
            f"{file_path.name}: need at least 2 classes after label encoding, "
            f"got {len(le.classes_)}: {list(le.classes_)}"
        )

    # Conversion features.
    X_parts = []
    numeric_cols = []
    categorical_cols = []

    for c in X_raw.columns:
        s = X_raw[c].astype(str).str.strip().str.replace(",", ".", regex=False)
        num = pd.to_numeric(s, errors="coerce")

        # Colonne numérique si au moins 80% des valeurs non manquantes sont numériques.
        non_missing = s.replace({"nan": np.nan, "None": np.nan, "NULL": np.nan, "?": np.nan}).notna()
        denom = max(1, int(non_missing.sum()))
        ratio_num = float(num.notna().sum()) / denom

        if ratio_num >= 0.80:
            numeric_cols.append(c)
            X_parts.append(num.to_numpy(dtype=float).reshape(-1, 1))
        else:
            categorical_cols.append(c)
            vals = s.fillna("__MISSING__").replace({"nan": "__MISSING__", "None": "__MISSING__", "NULL": "__MISSING__", "?": "__MISSING__"})
            codes, _ = pd.factorize(vals, sort=True)
            X_parts.append(codes.astype(float).reshape(-1, 1))

    X = np.hstack(X_parts).astype(float)

    # Imputation médiane pour les colonnes numériques/catégorielles codées.
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X).astype(np.float64)

    meta = {
        "label_classes": [str(c) for c in le.classes_],
        "n_classes": int(len(le.classes_)),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "numeric_feature_cols": [int(c) for c in numeric_cols],
        "categorical_feature_cols": [int(c) for c in categorical_cols],
    }

    if verbose:
        cls, cnt = np.unique(y, return_counts=True)
        dist = {
            str(le.classes_[int(c)]): int(n)
            for c, n in zip(cls, cnt)
        }
        print(
            f"[LOAD] {file_path.name}: X={X.shape}, "
            f"classes={meta['n_classes']}, class_counts={dist}"
        )
        if categorical_cols:
            print(
                f"[LOAD] {file_path.name}: encoded {len(categorical_cols)} "
                f"categorical feature columns"
            )

    return X, y, meta


def print_stats_table(
    stats_block: Dict[str, Any],
    *,
    sort_by: str = "accuracy",
    title: str | None = None,
) -> None:
    if not stats_block:
        print("No stats to display.")
        return

    metrics = list(next(iter(stats_block.values())).keys())

    if title:
        print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")

    order = sorted(
        stats_block.keys(),
        key=lambda s: stats_block[s].get(sort_by, {}).get("mean", -999),
        reverse=True,
    )

    sep = "-" * (28 + 26 * len(metrics))

    print(f"\n{'':>28}", end="")
    for m in metrics:
        print(f"| {m:^23}", end="")
    print("")
    print(sep)

    for name in order:
        row = f"{name:>28}"

        for m in metrics:
            mean = stats_block[name][m].get("mean", np.nan)
            std = stats_block[name][m].get("std", np.nan)
            row += f"| {mean:>8.4f} ± {std:<8.4f}  "

        print(row)

    print(sep)


def _safe_json(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    return str(obj)


def make_stratified_strates(
    y: np.ndarray,
    *,
    n_strates: int,
    strate_size: int,
    seed: int,
) -> List[np.ndarray]:
    """
    Crée n_strates sous-échantillons stratifiés reproductibles.

    Chaque strate respecte approximativement la répartition globale des classes.
    Les strates peuvent se recouvrir volontairement.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(int(seed))

    if n_strates <= 0:
        raise ValueError(f"n_strates must be positive, got {n_strates}")

    if strate_size <= 0:
        raise ValueError(f"strate_size must be positive, got {strate_size}")

    if strate_size > len(y):
        print(
            f"[INFO] strate_size={strate_size} > dataset size={len(y)}: "            f"using full dataset (strate_size clamped to {len(y)})"
        )
        strate_size = len(y)

    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()

    raw_sizes = proportions * int(strate_size)
    class_sizes = np.floor(raw_sizes).astype(int)

    remainder = int(strate_size) - int(class_sizes.sum())
    if remainder > 0:
        frac_order = np.argsort(-(raw_sizes - class_sizes))
        for j in frac_order[:remainder]:
            class_sizes[j] += 1

    # Garantit au moins 1 exemple par classe si possible.
    if strate_size >= len(classes):
        for j, n_c in enumerate(class_sizes):
            if n_c == 0:
                class_sizes[j] = 1

        while class_sizes.sum() > strate_size:
            j = int(np.argmax(class_sizes))
            if class_sizes[j] > 1:
                class_sizes[j] -= 1
            else:
                break

    strates: List[np.ndarray] = []

    for _ in range(n_strates):
        idx_parts = []

        for c, n_c in zip(classes, class_sizes):
            candidates = np.flatnonzero(y == c)

            if int(n_c) <= 0:
                continue

            replace = int(n_c) > len(candidates)

            chosen = rng.choice(
                candidates,
                size=int(n_c),
                replace=replace,
            )

            idx_parts.append(chosen)

        idx = np.concatenate(idx_parts).astype(int)
        rng.shuffle(idx)
        strates.append(idx)

    return strates


def print_class_distribution(y: np.ndarray, prefix: str = "") -> None:
    cls, cnt = np.unique(y, return_counts=True)
    total = cnt.sum()
    dist = {
        int(c): f"{int(n)} ({100.0 * n / total:.1f}%)"
        for c, n in zip(cls, cnt)
    }
    print(f"{prefix}{dist}")


# =============================================================================
# Core train/test direct
# =============================================================================

def _build_initial_indices(
    y_train: np.ndarray,
    *,
    labels: Sequence[int],
    n_init: int,
    init_mode: str,
    init_random_state: int,
) -> np.ndarray:
    """
    Construit le set initial AL.

    init_mode="random":
        tirage uniforme sans garantie de classe.

    init_mode="stratified":
        garantit autant que possible une représentation des classes présentes.
    """
    y_train = np.asarray(y_train)
    rng = np.random.default_rng(int(init_random_state))

    if n_init <= 0:
        raise ValueError(f"n_init must be positive, got {n_init}")

    if n_init > len(y_train):
        raise ValueError(
            f"n_init={n_init} cannot exceed train size={len(y_train)}"
        )

    idx_by_class = {int(c): np.flatnonzero(y_train == c) for c in labels}
    present = [int(c) for c in labels if len(idx_by_class[int(c)]) > 0]

    if init_mode == "random":
        return rng.choice(
            np.arange(len(y_train), dtype=int),
            size=int(n_init),
            replace=False,
        ).astype(int)

    if init_mode != "stratified":
        raise ValueError(
            f"unknown init_mode={init_mode!r}; expected 'random' or 'stratified'"
        )

    init_idx: List[int] = []

    if n_init < len(present):
        chosen_classes = rng.choice(present, size=int(n_init), replace=False)

        for c in chosen_classes:
            init_idx.append(int(rng.choice(idx_by_class[int(c)])))

    else:
        for c in present:
            init_idx.append(int(rng.choice(idx_by_class[int(c)])))

        remaining = int(n_init) - len(init_idx)

        if remaining > 0:
            all_idx = np.arange(len(y_train), dtype=int)
            used = np.asarray(init_idx, dtype=int)
            pool = np.setdiff1d(all_idx, used, assume_unique=False)

            extra = rng.choice(
                pool,
                size=remaining,
                replace=False,
            )

            init_idx.extend([int(i) for i in extra])

    return np.asarray(sorted(set(init_idx)), dtype=int)


def _compute_from_train_test(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    strategy_names: List[str],
    parameterconfig: bool,
    strategy_overrides: Optional[Mapping[str, Mapping[str, Any]]],
    max_budget: int,
    budget_step: int,
    tracked_metrics: Optional[List[str]],
    n_init: int,
    labels: List[int],
    init_mode: str,
    init_random_state: int,
    rng_seed: int,
    print_progress: bool,
) -> Dict[str, Any]:
    import learning_curves_one_dataset as lc
    from alframework.core.registry import STRATEGIES

    _force_register_v4_derivatives()

    if tracked_metrics is None:
        tracked_metrics = list(lc.DEFAULT_TRACKED_METRICS)

    # ------------------------------------------------------------------
    # Safety for multiclass datasets.
    # sklearn.log_loss fails if the fitted model has seen fewer classes
    # than the label list used for evaluation. With 26 classes and n_init=10,
    # this happens immediately on letter_recognition.
    # In stratified mode we therefore promote n_init to at least one sample
    # per class.
    # ------------------------------------------------------------------
    n_init_eff = int(n_init)
    n_classes_total = int(len(labels))

    if init_mode == "stratified" and n_init_eff < n_classes_total:
        if print_progress:
            print(
                f"[WARN] n_init={n_init_eff} < n_classes={n_classes_total}; "
                f"using n_init_eff={n_classes_total} for stratified init."
            )
        n_init_eff = n_classes_total

    if n_init_eff > len(y_train):
        raise ValueError(
            f"n_init_eff={n_init_eff} cannot exceed train size={len(y_train)}"
        )

    init_idx = _build_initial_indices(
        y_train,
        labels=labels,
        n_init=n_init_eff,
        init_mode=init_mode,
        init_random_state=init_random_state,
    )

    all_train_idx = np.arange(len(y_train), dtype=int)
    mask = np.ones(len(all_train_idx), dtype=bool)
    mask[init_idx] = False
    unlabeled_idx = all_train_idx[mask]

    X_labeled0 = X_train[init_idx]
    y_labeled0 = y_train[init_idx]
    X_unlabeled0 = X_train[unlabeled_idx]
    y_unlabeled_true0 = y_train[unlabeled_idx]

    base_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=1,
    )

    class _State:
        def __init__(self):
            self.X_labeled = X_labeled0
            self.y_labeled = y_labeled0
            self.X_unlabeled = X_unlabeled0
            self.labels = list(labels)
            self.n_classes = len(labels)
            self.y_pool = y_train
            self.rng = np.random.default_rng(int(rng_seed))
            self.model = clone(base_model)
            self.X_test = X_test
            self.y_test = y_test

    state0 = _State()

    max_budget_eff = min(int(max_budget), len(state0.X_unlabeled))
    n_rounds = int(math.ceil(max_budget_eff / int(budget_step)))

    results: Dict[str, Any] = {}

    # print_progress=True affiche uniquement le résumé final par stratégie.
    # Les informations de split sont déjà affichées dans run_one_file().

    for i_strategy, name in enumerate(strategy_names, start=1):
        overrides = dict((strategy_overrides or {}).get(name, {}))

        if name not in STRATEGIES:
            raise KeyError(
                f"Strategy '{name}' not found in registry. "
                f"Available: {sorted(STRATEGIES.keys())}"
            )

        strat = STRATEGIES[name](**overrides)

        state = copy.deepcopy(state0)
        labeler = lc.ArrayLabeler(y_unlabeled_true0.copy())

        history = lc.active_learning_loop(
            state=state,
            strategy=strat,
            labeler=labeler,
            n_rounds=n_rounds,
            budget=int(budget_step),
            log_indicators=True,
            indicator_rounds=[0, 1, 2],
        )

        base_model_0 = clone(base_model).fit(
            state0.X_labeled,
            state0.y_labeled,
        )
        m0 = lc.evaluate(base_model_0, X_test, y_test)

        curve: List[Any] = [
            lc.CurvePoint(
                n_selected=0,
                metrics={
                    k: float(m0.get(k, np.nan))
                    for k in tracked_metrics
                },
            )
        ]

        for ir, h in enumerate(history, start=1):
            n_selected = int(h["n_labeled"]) - int(n_init_eff)

            if n_selected > max_budget_eff:
                continue

            curve.append(
                lc.CurvePoint(
                    n_selected=n_selected,
                    metrics={
                        k: float(h.get(k, np.nan))
                        for k in tracked_metrics
                    },
                )
            )
            '''
            if print_progress:
                acc = float(h.get("accuracy", np.nan))
                f1 = float(h.get("f1_macro", np.nan))

                print(
                    f"    round={ir:02d} "
                    f"| selected={n_selected:4d}/{max_budget_eff:<4d} "
                    f"| labeled={int(h['n_labeled']):4d} "
                    f"| acc={acc:.4f} "
                    f"| f1={f1:.4f}"
                )
            ''' 
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
            last_f1 = results[name].curve[-1].metrics.get("f1_macro", np.nan)
            auc_acc = auc_norm.get("accuracy", np.nan)
            auc_f1 = auc_norm.get("f1_macro", np.nan)

            print(
                f"[DONE] {name:<24} "
                f"| acc_final={last_acc:.4f} "
                f"| f1_final={last_f1:.4f} "
                f"| AULC(acc)={auc_acc:.4f} "
                f"| AULC(f1)={auc_f1:.4f}"
            )

    return {
        "setup": dict(
            mode="iterative_realdata_train_test",
            parameterconfig=bool(parameterconfig),
            strategy_overrides={
                k: dict(v)
                for k, v in (strategy_overrides or {}).items()
            },
            max_budget=int(max_budget_eff),
            budget_step=int(budget_step),
            n_rounds=int(n_rounds),
            tracked_metrics=list(tracked_metrics),
            n_init=int(n_init_eff),
            init_mode=str(init_mode),
            labels=list(labels),
            requested_n_init=int(n_init),
            init_random_state=int(init_random_state),
            rng_seed=int(rng_seed),
        ),
        "strategies": results,
    }
# =============================================================================
# Run 1 split depuis indices + scaling train-only
# =============================================================================

def run_one_split_from_indices(
    *,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    strategy_names: List[str],
    strategy_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    parameterconfig: bool = True,
    split_seed: int,
    init_seed: int,
    rng_seed: int,
    max_budget: int = 400,
    budget_step: int = 20,
    n_init: int = 10,
    init_mode: str = "stratified",
    tracked_metrics: Optional[List[str]] = None,
    print_progress: bool = False,
) -> Dict[str, Any]:
    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    y_train = y[train_idx]
    y_test = y[test_idx]

    labels = sorted([int(c) for c in np.unique(y).tolist()])

    out = _compute_from_train_test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        strategy_names=strategy_names,
        parameterconfig=parameterconfig,
        strategy_overrides=strategy_overrides,
        max_budget=max_budget,
        budget_step=budget_step,
        tracked_metrics=tracked_metrics,
        n_init=n_init,
        labels=labels,
        init_mode=init_mode,
        init_random_state=init_seed,
        rng_seed=rng_seed,
        print_progress=print_progress,
    )

    out["setup"]["split_seed"] = int(split_seed)
    out["setup"]["test_size_actual"] = float(len(test_idx) / (len(train_idx) + len(test_idx)))
    out["setup"]["n_train"] = int(len(train_idx))
    out["setup"]["n_test"] = int(len(test_idx))

    return out


# =============================================================================
# Benchmark : 1 fichier, multi-strates, multi-splits
# =============================================================================

def run_one_file(
    *,
    file_path: Path,
    series_name: str,
    strategy_names: List[str],
    n_strates: int,
    strate_size: Optional[int],
    n_splits: int,
    test_size: float,
    base_seed: int,
    min_train_per_class: int = 2,
    strategy_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    parameterconfig: bool = True,
    max_budget: int = 400,
    budget_step: int = 20,
    n_init: int = 10,
    init_mode: str = "stratified",
    tracked_metrics: Optional[List[str]] = None,
    metrics_for_auc: Optional[List[str]] = None,
    print_per_split: bool = False,
    plot_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    X, y, data_meta = load_tabular_txt_robust(
        file_path,
        delimiter="\t",
        label_column=-1,
        verbose=print_per_split,
    )

    if tracked_metrics is None:
        import learning_curves_one_dataset as lc
        tracked_metrics = list(lc.DEFAULT_TRACKED_METRICS)

    if metrics_for_auc is None:
        metrics_for_auc = ["accuracy", "f1_macro"]

    ds_name = file_path.stem

    if strate_size is None:
        strates = [np.arange(len(y), dtype=int)]
    else:
        _strate_size_eff = strate_size
        if _strate_size_eff > len(y):
            print(
                f"[INFO] {file_path.name}: strate_size={_strate_size_eff} > "                f"n_samples={len(y)}, using full dataset and keeping "                f"n_strates={n_strates} for statistics."
            )
            _strate_size_eff = len(y)
        strates = make_stratified_strates(
            y,
            n_strates=n_strates,
            strate_size=_strate_size_eff,
            seed=base_seed,
        )

    # Optional history export for analyze_histories.py
    history_dir = None
    if plot_cfg is not None and plot_cfg.get("history_dir", None) is not None:
        history_dir = Path(plot_cfg["history_dir"])
        history_dir.mkdir(parents=True, exist_ok=True)

    all_scores: Dict[str, Dict[str, List[float]]] = {}
    per_split_results: List[Dict[str, Any]] = []

    for strate_id, strate_idx in enumerate(strates):
        X_s = X[strate_idx]
        y_s = y[strate_idx]

        split_seed_for_strate = int(base_seed + 10_000 * strate_id)

        splits = make_stratified_splits(
            y_s,
            n_splits=n_splits,
            test_size=test_size,
            seed=split_seed_for_strate,
            min_train_per_class=min_train_per_class,
        )

        if print_per_split:
            print(f"\n{'#' * 70}")
            print(
                f"{file_path.name} | strate {strate_id + 1}/{len(strates)} "
                f"| n={len(y_s)} | split_seed={split_seed_for_strate}"
            )
            print_class_distribution(y_s, prefix="strate class counts: ")
            print(f"{'#' * 70}")

        for split_id, (train_idx, test_idx) in enumerate(splits):
            split_seed = int(split_seed_for_strate + 1_000 * split_id)
            init_seed = int(split_seed_for_strate + 2_000 * split_id + 123)
            rng_seed = int(split_seed_for_strate + 3_000 * split_id + 999)

            split_global_id = int(strate_id * n_splits + split_id)
            split_tag = f"strate{strate_id + 1:02d}__split{split_id + 1:02d}"

            if print_per_split:
                cls, cnt = np.unique(y_s[train_idx], return_counts=True)
                print(f"\n{'=' * 70}")
                print(
                    f"{file_path.name} | strate {strate_id + 1}/{len(strates)} "
                    f"| split {split_id + 1}/{n_splits} "
                    f"| test_size={test_size:.2f} "
                    f"| split_seed={split_seed} "
                    f"| init_seed={init_seed}"
                )
                print(f"train class counts: {dict(zip(cls.tolist(), cnt.tolist()))}")
                print(f"{'=' * 70}")

            out = run_one_split_from_indices(
                X=X_s,
                y=y_s,
                train_idx=train_idx,
                test_idx=test_idx,
                strategy_names=strategy_names,
                strategy_overrides=strategy_overrides,
                parameterconfig=parameterconfig,
                split_seed=split_seed,
                init_seed=init_seed,
                rng_seed=rng_seed,
                max_budget=max_budget,
                budget_step=budget_step,
                n_init=n_init,
                init_mode=init_mode,
                tracked_metrics=tracked_metrics,
                print_progress=print_per_split,
            )

            out["setup"]["strate_id"] = int(strate_id)
            out["setup"]["split_id"] = int(split_id)
            out["setup"]["split_global_id"] = int(split_global_id)
            out["setup"]["split_tag"] = split_tag
            out["setup"]["strate_size"] = int(len(y_s))
            out["setup"]["test_size_requested"] = float(test_size)

            # --------------------------------------------------------------
            # Save per-strategy histories compatible with analyze_histories.py
            # Expected pattern:
            # history_<dataset>_split<split_id>_<strategy>.json
            # --------------------------------------------------------------
            if history_dir is not None:
                safe_dataset = (
                    str(ds_name)
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace(" ", "_")
                )

                for strategy_name, strat_out in out["strategies"].items():
                    safe_strategy = (
                        str(strategy_name)
                        .replace("/", "_")
                        .replace("\\", "_")
                        .replace(" ", "_")
                    )

                    records = []

                    for point in strat_out.curve:
                        # analyze_histories.py expects n_labeled + accuracy
                        rec = {
                            "n_labeled": int(point.n_selected),
                        }

                        for k, v in point.metrics.items():
                            try:
                                rec[k] = float(v)
                            except Exception:
                                rec[k] = None

                        records.append(rec)

                    history_path = history_dir / (
                        f"history_{MODEL_TAG}_{safe_dataset}_split{split_global_id}_{safe_strategy}.json"
                    )

                    with open(history_path, "w", encoding="utf-8") as f:
                        json.dump(records, f, indent=2, ensure_ascii=False)

            per_split_results.append(out)

            # --------------------------------------------------------------
            # Optional plots per split
            # --------------------------------------------------------------
            if plot_cfg and plot_cfg.get("enabled", False):
                plots_root = Path(plot_cfg["plots_root"])
                do_show = bool(plot_cfg.get("do_show", False))
                show_strats = plot_cfg.get("show_strats", None)

                baseline_acc = build_constant_baseline_from_out(
                    out,
                    metric="accuracy",
                    label="init@supervised",
                )

                plot_curves_and_save(
                    out,
                    metric="accuracy",
                    show=show_strats,
                    title=f"{series_name} — {ds_name} — {split_tag} — accuracy",
                    baseline=baseline_acc,
                    save_path=plots_root / series_name / ds_name / f"{split_tag}__accuracy.png",
                    do_show=do_show,
                )

                baseline_f1 = build_constant_baseline_from_out(
                    out,
                    metric="f1_macro",
                    label="init@supervised",
                )

                plot_curves_and_save(
                    out,
                    metric="f1_macro",
                    show=show_strats,
                    title=f"{series_name} — {ds_name} — {split_tag} — f1_macro",
                    baseline=baseline_f1,
                    save_path=plots_root / series_name / ds_name / f"{split_tag}__f1_macro.png",
                    do_show=do_show,
                )

            for name, strat_out in out["strategies"].items():
                if name not in all_scores:
                    all_scores[name] = {m: [] for m in metrics_for_auc}

                for m in metrics_for_auc:
                    val = strat_out.auc_norm.get(m, np.nan)
                    all_scores[name][m].append(float(val))

    stats = aggregate_scores(all_scores)

    return {
        "setup": {
            "file": str(file_path),
            "n_original": int(len(y)),
            "n_strates": int(len(strates)),
            "strate_size": None if strate_size is None else int(strate_size),
            "n_splits": int(n_splits),
            "test_size": float(test_size),
            "base_seed": int(base_seed),
            "min_train_per_class": int(min_train_per_class),
            "max_budget": int(max_budget),
            "budget_step": int(budget_step),
            "n_init": int(n_init),
            "init_mode": str(init_mode),
            "tracked_metrics": list(tracked_metrics),
            "metrics_for_auc": list(metrics_for_auc),
            "strategy_names": list(strategy_names),
            "history_dir": None if history_dir is None else str(history_dir),
            "data_meta": _safe_json(data_meta),
        },
        "per_split": per_split_results,
        "stats": stats,
    }

# =============================================================================
# Benchmark : une série
# =============================================================================

def run_one_series(
    *,
    series_dir: Path,
    strategy_names: List[str],
    n_strates: int,
    strate_size: Optional[int],
    n_splits: int,
    test_size: float,
    base_seed: int,
    min_train_per_class: int = 2,
    strategy_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    parameterconfig: bool = True,
    max_budget: int = 400,
    budget_step: int = 20,
    n_init: int = 10,
    init_mode: str = "stratified",
    tracked_metrics: Optional[List[str]] = None,
    metrics_for_auc: Optional[List[str]] = None,
    print_per_file: bool = True,
    print_per_split: bool = True,
    also_global_all_splits: bool = True,
    plot_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    files = find_dataset_files(series_dir, pattern="*.txt", recursive=False)

    if len(files) == 0:
        raise FileNotFoundError(f"No .txt files found in {series_dir}")

    per_file: List[Dict[str, Any]] = []

    global_scores_by_filemean: Dict[str, Dict[str, List[float]]] = {}
    global_scores_all_splits: Dict[str, Dict[str, List[float]]] = {}

    series_name = series_dir.name

    for i, fp in enumerate(files):
        file_seed = int(base_seed + 50_000 * i)

        if print_per_file:
            print(f"\n{'#' * 90}")
            print(
                f"Series: {series_name} | file {i + 1}/{len(files)}: "
                f"{fp.name} | seed={file_seed}"
            )
            print(
                f"n_strates={n_strates} | strate_size={strate_size} "
                f"| n_splits={n_splits} | test_size={test_size:.2f} "
                f"| init_mode={init_mode}"
            )
            print(f"{'#' * 90}")

        res_file = run_one_file(
            file_path=fp,
            series_name=series_name,
            strategy_names=strategy_names,
            n_strates=n_strates,
            strate_size=strate_size,
            n_splits=n_splits,
            test_size=test_size,
            base_seed=file_seed,
            min_train_per_class=min_train_per_class,
            strategy_overrides=strategy_overrides,
            parameterconfig=parameterconfig,
            max_budget=max_budget,
            budget_step=budget_step,
            n_init=n_init,
            init_mode=init_mode,
            tracked_metrics=tracked_metrics,
            metrics_for_auc=metrics_for_auc,
            print_per_split=print_per_split,
            plot_cfg=plot_cfg,
        )

        per_file.append(res_file)

        metrics_use = res_file["setup"]["metrics_for_auc"]

        # Global A : moyenne par fichier
        for strat in res_file["stats"].keys():
            if strat not in global_scores_by_filemean:
                global_scores_by_filemean[strat] = {m: [] for m in metrics_use}

            for m in metrics_use:
                global_scores_by_filemean[strat][m].append(
                    res_file["stats"][strat][m]["mean"]
                )

        # Global B : toutes les strates/splits
        if also_global_all_splits:
            for split_out in res_file["per_split"]:
                for strat, strat_out in split_out["strategies"].items():
                    if strat not in global_scores_all_splits:
                        global_scores_all_splits[strat] = {m: [] for m in metrics_use}

                    for m in metrics_use:
                        global_scores_all_splits[strat][m].append(
                            float(strat_out.auc_norm.get(m, np.nan))
                        )

    global_stats_by_filemean = aggregate_scores(global_scores_by_filemean)
    global_stats_all_splits = (
        aggregate_scores(global_scores_all_splits)
        if also_global_all_splits
        else {}
    )

    return {
        "setup": {
            "series_dir": str(series_dir),
            "n_files": int(len(files)),
            "n_strates": int(n_strates),
            "strate_size": None if strate_size is None else int(strate_size),
            "n_splits": int(n_splits),
            "test_size": float(test_size),
            "base_seed": int(base_seed),
            "metrics_for_auc": list(metrics_for_auc or ["accuracy", "f1_macro"]),
            "strategy_names": list(strategy_names),
            "init_mode": str(init_mode),
        },
        "per_file": per_file,
        "global_stats_by_filemean": global_stats_by_filemean,
        "global_stats_all_splits": global_stats_all_splits,
    }

# =============================================================================
def save_histories_for_analyzer(
    *,
    out: Dict[str, Any],
    history_dir: Path,
    dataset_name: str,
    split_global_id: int,
) -> None:
    """
    Sauve un fichier history_*.json par stratégie,
    compatible avec analyze_histories.py.
    """
    history_dir.mkdir(parents=True, exist_ok=True)

    for strategy_name, strat_out in out["strategies"].items():
        records = []

        for point in strat_out.curve:
            rec = {
                "n_labeled": int(point.n_selected),
            }

            for k, v in point.metrics.items():
                rec[k] = float(v)

            records.append(rec)

        safe_strategy = str(strategy_name).replace("/", "_").replace("\\", "_")
        safe_dataset = str(dataset_name).replace("/", "_").replace("\\", "_")

        path = history_dir / (
            f"history_{MODEL_TAG}_{safe_dataset}_split{split_global_id}_{safe_strategy}.json"
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)


def discover_series_dirs(root: Path) -> List[Path]:
    """
    Découvre les séries de benchmark à partir d'un root.

    Comportement :
    - si root contient des .txt directement, root est testé comme une série ;
    - si root contient des sous-dossiers qui contiennent des .txt, chaque sous-dossier
      est testé comme une série séparée ;
    - si les deux existent, on teste root puis chaque sous-dossier ;
    - recherche volontairement non récursive au-delà d'un niveau pour éviter de mélanger
      des séries imbriquées.

    Exemple : Images/ contenant PCA10/, PCA20/ => résultats séparés pour PCA10 et PCA20.
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Benchmark root does not exist: {root}")

    series_dirs: List[Path] = []

    # Cas historique : le dossier passé contient directement les datasets .txt.
    if any(root.glob("*.txt")):
        series_dirs.append(root)

    # Nouveau cas : le root contient des folders de datasets.
    for child in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda x: x.name.lower()):
        if any(child.glob("*.txt")):
            series_dirs.append(child)

    # Déduplication robuste tout en gardant l'ordre.
    seen = set()
    unique: List[Path] = []
    for d in series_dirs:
        key = str(d.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(d)

    if not unique:
        raise FileNotFoundError(
            f"No dataset series found under {root}. Expected .txt files either directly "
            f"in root or in first-level subfolders."
        )

    return unique

# =============================================================================
# Main
# =============================================================================
def _ensure_strategies_registered() -> None:

    must_have = [
        "alframework.strategies.random",
        "alframework.strategies.typiclust",
        "alframework.strategies.uncertainty",
        "alframework.strategies.dbal",
        "alframework.strategies.rank2022",
        "alframework.strategies.qbc",
        "alframework.strategies.tri_committee",
        "alframework.strategies.active_pseudolabelv53",
        "alframework.strategies.active_pseudolabelv54",
        "alframework.strategies.active_pseudolabelv55",
        "alframework.strategies.strategy_v58",
    ]

    for mod_path in must_have:
        try:
            __import__(mod_path)
        except ImportError as e:
            print(f"[FATAL] could not import {mod_path}: {e}")
            sys.exit(1)

    _force_register_v4_derivatives()

            
def main():
    # -------------------------------------------------------------------------
    # Affichage
    # -------------------------------------------------------------------------
   
    # -------------------------------------------------------------------------
    # Stratégies
    # -------------------------------------------------------------------------
    STRATEGIES_TO_TEST = [
        "random",
        "margin",
        "dbal",
        "ActivePseudoLabelV58",
        # "ActivePseudoLabelV54",
        # "ActivePseudoLabelV55",
    ]

    STRATEGY_OVERRIDES = {
        "dbal": {
            "method": "margin",
            "dbal_factor": 5,
        },

        "ActivePseudoLabelV53": {
            "lambda_prop": 0.0,
            "source_policy": "auto_fast",
            "max_u_contrast_for_source": 0.28,
            "min_u_flatness_for_source": 0.88,
            "max_selection_pool": 250,
            "adaptive_representative": True,
            "random_state": 0,
        },

        "ActivePseudoLabelV54": {
            "lambda_prop": 0.0,
            "source_policy": "auto_fast",
            "max_u_contrast_for_source": 0.28,
            "min_u_flatness_for_source": 0.88,
            "max_selection_pool": 250,
            "adaptive_representative": True,
            "random_state": 0,
        },

        "ActivePseudoLabelV55": {
            "lambda_prop": 0.0,
            "source_policy": "auto_fast",
            "max_u_contrast_for_source": 0.28,
            "min_u_flatness_for_source": 0.88,
            "max_selection_pool": 250,
            "adaptive_representative": True,
            "random_state": 0,
        },
        
        "ActivePseudoLabelV58": {
            "variant": "V58b",
            "multiclass_thr": 5,
            "u_flat_trigger": 0.50,
            "eff_dim_thr": 12.0,
            "peak_lo": 0.58,
            "route_round": 1,
            "correction_round": 3,
            "hysteresis": 0.05,
            "kmeans_n_init": 3,
            "random_state": 0,
            "debug_route": False,
        },
    }

    # -------------------------------------------------------------------------
    # Paramètres statistiques
    # -------------------------------------------------------------------------
    n_strates = 3       # 3 strates × n_splits = 9 répétitions par dataset (comme DualSelect)
    strate_size = 5000 # ~10k par strate (datasets images ~10k échantillons)
    n_splits = 3
    test_size = 0.30
    init_mode = "stratified"

    base_seed = 42
    min_train_per_class = 2

    max_budget = 1200   # budget max adapté espace latent
    budget_step = 50    # batch=50
    n_init = 50         # init stratifiée : 5 par classe (10 classes)

    # -------------------------------------------------------------------------
    # Dossiers
    # -------------------------------------------------------------------------
    # Donne ici soit un dossier-série contenant directement des .txt,
    # soit un root parent contenant des sous-dossiers de séries.
    # Exemples :
    #   Images/latent_pca100H3        -> une seule série
    #   Images/                       -> séries Images/PCA10, Images/PCA20, ...
    bench_root = project_root / "tests" / "benchmark_data" / "Images"

    series_to_test = discover_series_dirs(bench_root)

    out_dir = project_root / "tests" / f"bench_results_real_strates_{init_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plots_root = out_dir / "plots"

    plot_cfg = dict(
        enabled=False,
        plots_root=plots_root,
        history_dir=out_dir / "histories",
        do_show=False,
        show_strats=None,
    )

    save_json = True

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------
    for sdir in series_to_test:
        print("\n" + "=" * 100)
        print(f"RUN SERIES: {sdir.name}")
        print("=" * 100)
        print(f"strategies   = {STRATEGIES_TO_TEST}")
        print(f"n_strates    = {n_strates}")
        print(f"strate_size  = {strate_size}")
        print(f"n_splits     = {n_splits}")
        print(f"test_size    = {test_size}")
        print(f"init_mode    = {init_mode}")
        print(f"max_budget   = {max_budget}")
        print(f"budget_step  = {budget_step}")
        print(f"n_init       = {n_init}")
        print("=" * 100)

        # Histories/plots séparés par folder pour éviter les collisions de noms
        # si deux séries contiennent des fichiers .txt au même stem.
        plot_cfg_series = dict(plot_cfg)
        plot_cfg_series["history_dir"] = out_dir / "histories" / sdir.name
        plot_cfg_series["plots_root"] = plots_root

        out = run_one_series(
            series_dir=sdir,
            strategy_names=STRATEGIES_TO_TEST,
            n_strates=n_strates,
            strate_size=strate_size,
            n_splits=n_splits,
            test_size=test_size,
            base_seed=base_seed,
            min_train_per_class=min_train_per_class,
            strategy_overrides=STRATEGY_OVERRIDES,
            parameterconfig=True,
            max_budget=max_budget,
            budget_step=budget_step,
            n_init=n_init,
            init_mode=init_mode,
            tracked_metrics=None,
            metrics_for_auc=["accuracy", "f1_macro"],
            print_per_file=True,
            print_per_split=True,
            also_global_all_splits=True,
            plot_cfg=plot_cfg_series,
        )

        series_name = Path(out["setup"]["series_dir"]).name

        print_stats_table(
            out["global_stats_by_filemean"],
            sort_by="accuracy",
            title=f"GLOBAL (by file mean) — {series_name}",
        )

        if out["global_stats_all_splits"]:
            print_stats_table(
                out["global_stats_all_splits"],
                sort_by="accuracy",
                title=f"GLOBAL (all strates/splits) — {series_name}",
            )

        for fres in out["per_file"]:
            fname = Path(fres["setup"]["file"]).name

            print_stats_table(
                fres["stats"],
                sort_by="accuracy",
                title=f"FILE — {fname}",
            )

        if save_json:
            out_path = out_dir / f"bench_real_{series_name}.json"

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    _safe_json(out),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"\nSaved results to: {out_path}")

        print(f"\nHistories saved under: {plot_cfg_series['history_dir']}")

        if plot_cfg.get("enabled", False):
            print(f"Plots saved under: {plots_root / series_name}")
        
if __name__ == "__main__":
    
    _ensure_strategies_registered()
    main()