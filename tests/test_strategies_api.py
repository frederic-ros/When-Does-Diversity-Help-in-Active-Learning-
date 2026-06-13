# -*- coding: utf-8 -*-
"""
tests/test_strategies_api.py
============================
Tests des stratégies d'active learning.

Deux modes disponibles :
  - Single-shot  : la stratégie sélectionne budget points d'un coup,
                   le train est évalué UNE fois.
  - Itératif     : la boucle AL standard (active_learning_loop) est utilisée,
                   l'état (labeled/unlabeled) évolue à chaque round.

Point d'entrée : bloc ``if __name__ == "__main__":`` en bas du fichier.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from itertools import product
import pandas as pd


# ─── Shared infrastructure ────────────────────────────────────────────────────
from common import (
    STRATEGIES, ALState, active_learning_loop, ArrayLabeler,
    build_synth_state_and_testset, make_strategy_from_config,
    validate_strategy_config, make_strategy, enabled_strategy_names,
    _fit_eval, _validate_indices, _instantiate_strategy, _copy_state,
    DEFAULT_METRIC_KEYS, DEFAULT_GAP_KEYS,
)


# =============================================================================
# Utilitaires d'affichage
# =============================================================================

def _print_metrics(title: str, m: Dict[str, Any], *, verbose: bool = False) -> None:
    if verbose:
        print(f"\n{'-'*70}\n{title}\n{'-'*70}")
        for key in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
                    "precision_macro", "recall_macro", "log_loss", "brier_score"]:
            if key in m:
                print(f"  {key}: {m[key]:.4f}")
    else:
        print(f"  {title} → accuracy={m.get('accuracy', float('nan')):.4f}")


# =============================================================================
# Tests infrastructure
# =============================================================================

def test_registry_is_populated() -> None:
    assert len(STRATEGIES) > 0, "Registry STRATEGIES vide."
    print(f"✅ Registry OK : {len(STRATEGIES)} stratégies enregistrées")
    print("   Stratégies :", sorted(STRATEGIES.keys()))


def test_instantiation_from_config() -> None:
    names = enabled_strategy_names()
    assert names, "Aucune stratégie activée dans strategies_config.py."
    for name in names:
        s = make_strategy(name)
        print(f"  {name} → {type(s).__name__}")
    print(f"✅ Config OK : {len(names)} stratégies activées")


def test_strategy_config_injection() -> None:
    from alframework.config.strategies_parameter import STRATEGY_SPECS
    name = "dbal"
    original = STRATEGY_SPECS[name]
    STRATEGY_SPECS[name] = type(original)(
        enabled=True,
        init_kwargs={"method": "entropy", "dbal_factor": 7, "random_state": 123},
    )
    try:
        strat = make_strategy_from_config(name)
        assert getattr(strat, "method") == "entropy"
        assert getattr(strat, "dbal_factor") == 7
        print(f"✅ Injection config OK pour '{name}'")
    finally:
        STRATEGY_SPECS[name] = original


# =============================================================================
# Runner single-shot
# =============================================================================

def run_singleshot_one_strategy(
    strategy_name: str,
    *,
    parameterconfig: bool = False,
    budget: int = 50,
    n_init: int = 20,
    labels: Optional[List[int]] = None,
    dataset_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.25,
    split_random_state: int = 42,
    init_random_state: int = 123,
    rng_seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Test single-shot d'UNE stratégie : une sélection, une évaluation."""
    if dataset_params is None:
        dataset_params = dict(n_samples=2000, n_features=20, n_informative=10,
                              n_classes=3, class_sep=1.5, flip_y=0.02, random_state=42)
    if labels is None:
        labels = list(range(dataset_params["n_classes"]))

    base_model = RandomForestClassifier(n_estimators=200, random_state=42)
    state, y_unlabeled_true, X_test, y_test = build_synth_state_and_testset(
        dataset_params=dataset_params, test_size=test_size,
        split_random_state=split_random_state, n_init=n_init,
        init_random_state=init_random_state, rng_seed=rng_seed, model=clone(base_model),
    )

    strat = _instantiate_strategy(strategy_name, parameterconfig=parameterconfig)
    print(f"[{strategy_name}] params={strat.__dict__}")

    idx = np.asarray(strat.select(state, budget=budget), dtype=int)
    _validate_indices(idx, len(state.X_unlabeled), budget, strategy_name)

    X_sel = state.X_unlabeled[idx]
    y_sel = y_unlabeled_true[idx]
    X_al = np.vstack([state.X_labeled, X_sel]) if state.X_labeled.shape[0] > 0 else X_sel
    y_al = np.concatenate([state.y_labeled, y_sel]) if state.y_labeled.shape[0] > 0 else y_sel
    X_full = np.vstack([state.X_labeled, state.X_unlabeled]) if state.X_labeled.shape[0] > 0 else state.X_unlabeled
    y_full = np.concatenate([state.y_labeled, y_unlabeled_true]) if state.y_labeled.shape[0] > 0 else y_unlabeled_true

    metrics_al = _fit_eval(base_model, X_al, y_al, X_test, y_test, labels)
    metrics_full = _fit_eval(base_model, X_full, y_full, X_test, y_test, labels)

    _print_metrics(f"{strategy_name} (AL train)", metrics_al, verbose=verbose)
    _print_metrics("Full train", metrics_full, verbose=verbose)
    print(f"✅ {strategy_name} | selected={len(idx)} | acc={metrics_al['accuracy']:.4f}")

    return {
        "strategy": strategy_name, "budget": budget, "n_init": n_init,
        "selected_indices": idx, "metrics_al": metrics_al, "metrics_full": metrics_full,
        "used_config_params": bool(parameterconfig),
    }


def run_singleshot_all_strategies(
    strategy_names: Optional[List[str]] = None,
    *,
    parameterconfig: bool = False,
    printall: bool = False,
    budget: int = 200,
    n_init: int = 20,
    labels: Optional[List[int]] = None,
    dataset_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.25,
    split_random_state: int = 42,
    init_random_state: int = 123,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """
    Test single-shot de TOUTES les stratégies sur le même dataset.
    Chaque stratégie fait UNE sélection de ``budget`` points d'un coup.
    Utile pour un test rapide d'API.
    """
    if labels is None:
        labels = [0, 1, 2]
    if dataset_params is None:
        dataset_params = dict(n_samples=5000, n_features=20, n_informative=20,
                              n_classes=3, class_sep=1.2, flip_y=0.02, random_state=42)

    if strategy_names is None:
        strategy_names = sorted(STRATEGIES.keys())
    else:
        unknown = [s for s in strategy_names if s not in STRATEGIES]
        assert not unknown, f"Stratégies inconnues: {unknown}"

    base_model = RandomForestClassifier(n_estimators=200, random_state=42)
    state, y_unlabeled_true, X_test, y_test = build_synth_state_and_testset(
        dataset_params=dataset_params, test_size=test_size,
        split_random_state=split_random_state, n_init=n_init,
        init_random_state=init_random_state, rng_seed=rng_seed, model=clone(base_model),
    )

    X_full = np.vstack([state.X_labeled, state.X_unlabeled]) if state.X_labeled.shape[0] > 0 else state.X_unlabeled
    y_full = np.concatenate([state.y_labeled, y_unlabeled_true]) if state.y_labeled.shape[0] > 0 else y_unlabeled_true
    metrics_full = _fit_eval(base_model, X_full, y_full, X_test, y_test, labels)

    if printall:
        print(f"{'='*90}")
        print(f"SINGLE-SHOT | budget={budget} | n_init={n_init} | parameterconfig={parameterconfig}")
        print(f"Full train accuracy: {metrics_full['accuracy']:.4f}")
        print(f"{'='*90}")

    results: Dict[str, Any] = {}
    for name in strategy_names:
        strat = _instantiate_strategy(name, parameterconfig=parameterconfig)
        idx = np.asarray(strat.select(state, budget=budget), dtype=int)
        _validate_indices(idx, len(state.X_unlabeled), budget, name)

        X_sel = state.X_unlabeled[idx]
        y_sel = y_unlabeled_true[idx]
        X_al = np.vstack([state.X_labeled, X_sel]) if state.X_labeled.shape[0] > 0 else X_sel
        y_al = np.concatenate([state.y_labeled, y_sel]) if state.y_labeled.shape[0] > 0 else y_sel
        metrics_al = _fit_eval(base_model, X_al, y_al, X_test, y_test, labels)
        delta = {f"{k}_gap": metrics_full.get(k, float("nan")) - metrics_al.get(k, float("nan"))
                 for k in DEFAULT_METRIC_KEYS if k in metrics_full and k in metrics_al}

        print(f"{name:>24} | selected={len(idx):>3} | acc={metrics_al['accuracy']:.4f} "
              f"| gap={delta.get('accuracy_gap', float('nan')):.4f} | params={strat.__dict__}")

        results[name] = {
            "params": dict(getattr(strat, "__dict__", {})),
            "selected_indices": idx,
            "n_selected": int(len(idx)),
            "metrics_al": metrics_al,
            "delta_vs_full": delta,
        }

    return {
        "setup": dict(budget=budget, n_init=n_init, parameterconfig=bool(parameterconfig),
                      dataset_params=dict(dataset_params), test_size=test_size,
                      split_random_state=split_random_state,
                      init_random_state=init_random_state, rng_seed=rng_seed),
        "metrics_full": metrics_full,
        "strategies": results,
    }


# =============================================================================
# Runner itératif (active_learning_loop — état évolue entre les rounds)
# =============================================================================

def run_iterative_all_strategies(
    strategy_names: Optional[List[str]] = None,
    *,
    parameterconfig: bool = False,
    strategy_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    budget_step: int = 20,
    max_budget: int = 200,
    n_init: int = 20,
    labels: Optional[List[int]] = None,
    dataset_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.25,
    split_random_state: int = 42,
    init_random_state: int = 123,
    rng_seed: int = 42,
    print_progress: bool = True,
) -> Dict[str, Any]:
    """
    Test itératif de TOUTES les stratégies sur le même dataset.

    Protocole :
      - ``n_rounds = ceil(max_budget / budget_step)`` rounds AL
      - À chaque round : fit modèle → select → label → update état (labeled/unlabeled)
      - Résultat : métriques du dernier round + historique complet

    Avantage vs single-shot : la stratégie voit un état mis à jour à chaque itération.
    C'est le protocole AL "réaliste".
    """
    if labels is None:
        labels = [0, 1, 2]
    if dataset_params is None:
        dataset_params = dict(n_samples=5000, n_features=20, n_informative=20,
                              n_classes=3, class_sep=1.2, flip_y=0.02, random_state=42)

    if strategy_names is None:
        strategy_names = sorted(STRATEGIES.keys())
    else:
        unknown = [s for s in strategy_names if s not in STRATEGIES]
        assert not unknown, f"Stratégies inconnues: {unknown}"

    base_model = RandomForestClassifier(n_estimators=200, random_state=42)
    state0, y_unlabeled_true0, X_test, y_test = build_synth_state_and_testset(
        dataset_params=dataset_params, test_size=test_size,
        split_random_state=split_random_state, n_init=n_init,
        init_random_state=init_random_state, rng_seed=rng_seed, model=clone(base_model),
    )

    max_budget = min(int(max_budget), len(state0.X_unlabeled))
    n_rounds = int(math.ceil(max_budget / budget_step))

    X_full = np.vstack([state0.X_labeled, state0.X_unlabeled]) if state0.X_labeled.shape[0] > 0 else state0.X_unlabeled
    y_full = np.concatenate([state0.y_labeled, y_unlabeled_true0]) if state0.y_labeled.shape[0] > 0 else y_unlabeled_true0
    metrics_full = _fit_eval(base_model, X_full, y_full, X_test, y_test, labels)

    if print_progress:
        print(f"{'='*90}")
        print(f"ITÉRATIF | max_budget={max_budget} | budget_step={budget_step} | "
              f"n_rounds={n_rounds} | n_init={n_init} | parameterconfig={parameterconfig}")
        print(f"Labeled={state0.X_labeled.shape[0]} | Unlabeled={state0.X_unlabeled.shape[0]} | "
              f"Test={X_test.shape[0]}")
        print(f"Full train accuracy: {metrics_full['accuracy']:.4f}")
        print(f"{'='*90}")

    results: Dict[str, Any] = {}

    for name in strategy_names:
        overrides = dict((strategy_overrides or {}).get(name, {}))
        strat = _instantiate_strategy(name, parameterconfig=parameterconfig, overrides=overrides)

        state = _copy_state(state0, X_test, y_test)
        labeler = ArrayLabeler(y_unlabeled_true0.copy())

        history = active_learning_loop(
            state=state, strategy=strat, labeler=labeler,
            n_rounds=n_rounds, budget=budget_step,
        )

        final_h = history[-1] if history else {}
        metrics_al = {k: float(final_h.get(k, float("nan"))) for k in DEFAULT_METRIC_KEYS}
        n_labeled_final = int(final_h.get("n_labeled", 0))
        n_selected = n_labeled_final - int(n_init)
        delta = {f"{k}_gap": metrics_full.get(k, float("nan")) - metrics_al.get(k, float("nan"))
                 for k in DEFAULT_METRIC_KEYS}

        if print_progress:
            print(f"{name:>24} | labeled={n_labeled_final} | selected={n_selected} "
                  f"| acc={metrics_al.get('accuracy', float('nan')):.4f} "
                  f"| gap={delta.get('accuracy_gap', float('nan')):.4f}")

        results[name] = {
            "params": dict(getattr(strat, "__dict__", {})),
            "n_selected": n_selected,
            "n_labeled_final": n_labeled_final,
            "metrics_al": metrics_al,
            "delta_vs_full": delta,
            "history": history,
        }

    return {
        "setup": dict(max_budget=max_budget, budget_step=budget_step, n_rounds=n_rounds,
                      n_init=n_init, parameterconfig=bool(parameterconfig),
                      dataset_params=dict(dataset_params), test_size=test_size,
                      split_random_state=split_random_state,
                      init_random_state=init_random_state, rng_seed=rng_seed),
        "metrics_full": metrics_full,
        "strategies": results,
    }


# Alias de compatibilité (audit.py et code existant)
main_test_synthetique_allstrategies = run_singleshot_all_strategies

def print_auc_norm_table(out, metrics=("accuracy", "f1_macro"), sort_by="accuracy"):
    strat_out = out["strategies"]
    names = list(strat_out.keys())

    def key_fn(name):
        return strat_out[name].auc_norm.get(sort_by, float("-inf"))

    names = sorted(names, key=key_fn, reverse=True)

    print("\n" + "-" * (26 + 18 * len(metrics)))
    header = f"{'strategy':>24} "
    for m in metrics:
        header += f"| {m:^15} "
    print(header)
    print("-" * (26 + 18 * len(metrics)))

    for name in names:
        row = f"{name:>24} "
        for m in metrics:
            v = strat_out[name].auc_norm.get(m, float("nan"))
            row += f"| {v:>15.4f} "
        print(row)

    print("-" * (26 + 18 * len(metrics)))
# =============================================================================
# Point d'entrée
# =============================================================================
# =============================================================================
# Helpers affichage résultats
# =============================================================================

def print_full_results_table(out, metrics=("accuracy", "f1_macro"), sort_by="accuracy"):
    strat_out = out["strategies"]
    names = list(strat_out.keys())

    def get_auc(name, metric):
        s = strat_out[name]
        if hasattr(s, "auc_norm"):          # cas objet LearningCurveResult
            return float(s.auc_norm.get(metric, float("nan")))
        elif isinstance(s, dict):           # cas dict
            return float(s.get("auc_norm", {}).get(metric, float("nan")))
        return float("nan")

    def get_final(name, metric):
        s = strat_out[name]
        if hasattr(s, "curve"):
            curve = s.curve
        elif isinstance(s, dict):
            curve = s.get("curve", [])
        else:
            curve = []
        if not curve:
            return float("nan")
        last = curve[-1]
        if hasattr(last, "metrics"):
            return float(last.metrics.get(metric, float("nan")))
        elif isinstance(last, dict):
            return float(last.get("metrics", {}).get(metric, float("nan")))
        return float("nan")

    # tri
    names = sorted(
        names,
        key=lambda n: get_auc(n, sort_by),
        reverse=True,
    )

    line_len = 26 + 22 * len(metrics)
    print("\n" + "-" * line_len)

    header = f"{'strategy':>24} "
    for m in metrics:
        header += f"| AULC({m})  | final({m}) "
    print(header)
    print("-" * line_len)

    for name in names:
        row = f"{name:>24} "
        for m in metrics:
            auc_val = get_auc(name, m)
            final_val = get_final(name, m)
            row += f"| {auc_val:>10.4f} | {final_val:>10.4f} "
        print(row)

    print("-" * line_len)
    
def _aulc_norm_from_history(history: list[dict], metric: str, n_init: int) -> float:
    """
    Calcule une AULC normalisée à partir de history (liste de rounds).
    On utilise x = n_selected = n_labeled - n_init.
    AULC_norm = AUC / (x_max * 1.0)  (bornée [0,1] si metric bornée).
    """
    if not history:
        return float("nan")

    xs = []
    ys = []
    for h in history:
        if not isinstance(h, dict):
            continue
        if "n_labeled" not in h:
            continue
        x = int(h["n_labeled"]) - int(n_init)
        y = h.get(metric, None)
        if y is None:
            continue
        xs.append(x)
        ys.append(float(y))

    if len(xs) < 2:
        return float("nan")

    # tri par x au cas où
    order = np.argsort(xs)
    xs = np.asarray(xs, dtype=float)[order]
    ys = np.asarray(ys, dtype=float)[order]

    x_max = float(np.max(xs))
    if x_max <= 0:
        return float("nan")

    auc = float(np.trapz(ys, xs))
    return auc / x_max

def plot_curves_from_api_out(out, metric="accuracy", show=None, save_path=None, do_show=True):
    import matplotlib.pyplot as plt

    strat_out = out.get("strategies", {})
    setup = out.get("setup", {})
    n_init = int(setup.get("n_init", 0))

    names = list(strat_out.keys()) if show is None else show

    plt.figure(figsize=(10, 6))
    plotted = 0

    for name in names:
        s = strat_out.get(name, {})
        hist = s.get("history", [])
        if not hist:
            continue

        xs, ys = [], []
        for h in hist:
            if "n_labeled" not in h:
                continue
            x = int(h["n_labeled"]) - int(n_init)
            y = h.get(metric, None)
            if y is None:
                continue
            xs.append(x)
            ys.append(float(y))

        if xs:
            plt.plot(xs, ys, label=name)
            plotted += 1

    plt.xlabel("Nombre de labels sélectionnés")
    plt.ylabel(metric)
    plt.title(f"Learning curves (history) — {metric}")
    plt.grid(True)
    if plotted:
        plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    if save_path:
        from pathlib import Path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)

    if do_show:
        plt.show()
    else:
        plt.close()

    if plotted == 0:
        print(f"[WARN] Aucune courbe plotable trouvée pour metric={metric}.")
def print_results_table_from_api_out(out, metrics=("accuracy", "f1_macro"), sort_by="accuracy"):
    """
    Affiche:
      - AULC_norm calculée depuis history
      - final = metrics_al
    Compatible avec ton out_it (dicts avec keys: metrics_al, history).
    """
    strat_out = out.get("strategies", {})
    setup = out.get("setup", {})
    n_init = int(setup.get("n_init", 0))  # dans ton print tu avais n_init=20

    names = list(strat_out.keys())

    def sort_key(name: str) -> float:
        s = strat_out[name]
        hist = s.get("history", [])
        return _aulc_norm_from_history(hist, sort_by, n_init)

    names = sorted(names, key=sort_key, reverse=True)

    line_len = 26 + 22 * len(metrics)
    print("\n" + "-" * line_len)
    header = f"{'strategy':>24} "
    for m in metrics:
        header += f"| AULC({m})  | final({m}) "
    print(header)
    print("-" * line_len)

    for name in names:
        s = strat_out[name]
        hist = s.get("history", [])
        final_metrics = s.get("metrics_al", {})

        row = f"{name:>24} "
        for m in metrics:
            aulc = _aulc_norm_from_history(hist, m, n_init)
            finalv = float(final_metrics.get(m, float("nan")))
            row += f"| {aulc:>10.4f} | {finalv:>10.4f} "
        print(row)

    print("-" * line_len)



def _aulc_from_history(history, metric="accuracy", x_key="n_labeled"):
    """AULC normalisée sur l'axe n_labeled (ou autre)."""
    if not history:
        return np.nan
    xs = np.array([h.get(x_key, np.nan) for h in history], dtype=float)
    ys = np.array([h.get(metric, np.nan) for h in history], dtype=float)

    # retire NaN
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    if len(xs) < 2:
        return float(ys[-1]) if len(ys) else np.nan

    # tri au cas où
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    area = np.trapz(ys, xs)
    width = xs[-1] - xs[0]
    return float(area / width) if width > 0 else float(area)

def _metric_at_n_labeled(history, metric="accuracy", target_n=100, x_key="n_labeled"):
    """
    Retourne metric au round où n_labeled >= target_n (première occurrence).
    Si jamais atteint, retourne la dernière valeur.
    """
    if not history:
        return np.nan
    # history est déjà ordonnée par rounds
    for h in history:
        n = h.get(x_key, None)
        if n is not None and n >= target_n and metric in h:
            return float(h[metric])
    # fallback dernière valeur
    last = history[-1]
    return float(last.get(metric, np.nan))

def extract_stats_from_out(out_it,
                           metrics=("accuracy", "f1_macro"),
                           early_n_labeled=100):
    """
    Retourne un DataFrame "long" (une ligne par stratégie) avec :
    - final metrics
    - aulc metrics
    - early metrics @ early_n_labeled
    """
    rows = []
    strategies = out_it["strategies"]

    for name, info in strategies.items():
        hist = info.get("history", [])
        mal = info.get("metrics_al", {})  # final

        row = {"strategy": name}

        for m in metrics:
            row[f"{m}_final"] = float(mal.get(m, np.nan))
            row[f"{m}_aulc"] = _aulc_from_history(hist, metric=m, x_key="n_labeled")
            row[f"{m}@{early_n_labeled}"] = _metric_at_n_labeled(hist, metric=m, target_n=early_n_labeled)

        rows.append(row)

    return pd.DataFrame(rows)

def aggregate_across_seeds(per_seed_dfs,
                           metrics=("accuracy", "f1_macro"),
                           early_n_labeled=100,
                           rank_by="accuracy@100",   # ou "accuracy_final" ou "accuracy_aulc"
                           robust=True):             # mean-std

    df_all = []
    for seed, df in per_seed_dfs:
        d = df.copy()
        d["seed"] = seed
        df_all.append(d)
    df_all = pd.concat(df_all, ignore_index=True)

    # colonnes à agréger
    cols = []
    for m in metrics:
        cols += [f"{m}_final", f"{m}_aulc", f"{m}@{early_n_labeled}"]

    # groupby
    agg = df_all.groupby("strategy")[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()

    # score de ranking
    mean_col = f"{rank_by}_mean"
    std_col  = f"{rank_by}_std"

    if robust and std_col in agg.columns:
        agg["rank_score"] = agg[mean_col] - agg[std_col].fillna(0.0)
        agg["rank_rule"] = f"{rank_by}: mean - std"
    else:
        agg["rank_score"] = agg[mean_col]
        agg["rank_rule"] = f"{rank_by}: mean"

    agg = agg.sort_values("rank_score", ascending=False).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg) + 1)

    return df_all, agg

def print_funder_table(agg_df, early_n_labeled=100):
    print("\n" + "="*90)
    print(f"Rank | Strat | E{early_n_labeled}(μ±σ) | F(μ±σ) | A(μ±σ) | R")

    for _, r in agg_df.iterrows():
        print(
            f"{int(r['rank']):>4} | "
            f"{r['strategy'][:25]:<25} | "
            f"{r.get(f'accuracy@{early_n_labeled}_mean', float('nan')):.3f}"
            f"±{r.get(f'accuracy@{early_n_labeled}_std', 0.0):.3f} | "
            f"{r.get('accuracy_final_mean', float('nan')):.3f}"
            f"±{r.get('accuracy_final_std', 0.0):.3f} | "
            f"{r.get('accuracy_aulc_mean', float('nan')):.3f}"
            f"±{r.get('accuracy_aulc_std', 0.0):.3f} | "
            f"{r.get('rank_score', float('nan')):.3f}"
        )
# =============================================================================
# MAIN COMPLET
# =============================================================================

from pathlib import Path

def One_scenario(random_state=42, max_budget=200, dataset_params=None, 
                 strategy_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                 strategy_names: Optional[List[str]] = None,
                 display=True):

    if strategy_names is None:
        strategy_names = ["active_pseudolabel",  "dbal", "rank2022"]

    out_it = run_iterative_all_strategies(
        parameterconfig=False,
        budget_step=10,
        max_budget=max_budget,
        strategy_names=strategy_names,          # ✅ ici
        dataset_params=dataset_params,
        strategy_overrides=strategy_overrides,
        n_init=20,
        print_progress=display,
        rng_seed=random_state,
        split_random_state=random_state,
        init_random_state=123 + random_state,
    )
    return out_it


def CompareStrategy_constantbudget(strategy_names = ["ActivePseudoLabelV3","ActivePseudoLabelV4",
                                                "dbal"],max_budget = 200, class_sep=1, n_features=2, 
                                   n_splits=5):
    max_budget = max_budget
    seeds = list(range(0, n_splits))
    early_n = 200
    
    strategy_overrides = {
        "active_pseudolabel": {
            "k_neighbors": 10,
            "lambda_prop": 1.0,
            "alpha_decay":  2.0,
            "weighted_kmeans" : True,
        }
    }
    
    per_seed = []
    for rs in seeds:
        dataset_params = dict(
            n_samples=2000, n_features=n_features, n_informative=n_features,
            n_classes=3, class_sep=class_sep, flip_y=0.02,
            random_state=rs
        )
        out_it = One_scenario(random_state=rs, max_budget=max_budget,
                              dataset_params=dataset_params, 
                              strategy_overrides = strategy_overrides,
                              strategy_names = strategy_names,
                              display=False)

        df_seed = extract_stats_from_out(out_it, metrics=("accuracy", "f1_macro"), early_n_labeled=early_n)
        per_seed.append((rs, df_seed))
    '''
    df_all, agg = aggregate_across_seeds(
        per_seed,
        metrics=("accuracy", "f1_macro"),
        early_n_labeled=early_n,
        rank_by=f"accuracy@{early_n}",  # classement “early”
        robust=True
    )

    print_funder_table(agg, early_n_labeled=early_n)
    '''
    # Optionnel: aussi un classement long-terme
    _, agg_long = aggregate_across_seeds(per_seed, early_n_labeled=early_n,
                                         rank_by="accuracy_final", robust=True)
    print_funder_table(agg_long, early_n_labeled=early_n)
    

def CompareStrategySweep(
    *,
    target_strategy: str = "active_pseudolabel",
    # grille des paramètres à tester (dict param -> liste de valeurs)
    param_grid: Dict[str, List[Any]] = None,
    # seeds
    seeds: List[int] = None,
    # data / AL setup
    max_budget: int = 200,
    early_n: int = 100,
    dataset_base_params: Optional[Dict[str, Any]] = None,
    # ranking
    rank_by: Optional[str] = None,          # ex f"accuracy@{early_n}" ou "accuracy_final"
    robust: bool = True,
    # affichage
    display_runs: bool = False,             # print_progress dans run_iterative...
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Balaye une grille de paramètres pour `target_strategy` via strategy_overrides,
    agrège sur plusieurs seeds, puis renvoie un DF classé des configs.
    """

    if param_grid is None:
        param_grid = {"k_neighbors": [5, 10, 15], "lambda_prop": [0.0, 0.5, 1.0, 2.0]}

    if seeds is None:
        seeds = list(range(0, 5))

    if dataset_base_params is None:
        dataset_base_params = dict(
            n_samples=5000, n_features=20, n_informative=20,
            n_classes=3, class_sep=1., flip_y=0.02
        )

    if rank_by is None:
        rank_by = f"accuracy@{early_n}"

    # construit la liste des configs (produit cartésien)
    keys = list(param_grid.keys())
    values_lists = [param_grid[k] for k in keys]
    configs = [dict(zip(keys, vals)) for vals in product(*values_lists)]

    rows_cfg = []
    iteration=0
    for cfg in configs:
        print("config:",iteration)
        # overrides pour cette config
        strategy_overrides = {target_strategy: cfg}

        per_seed = []
        for rs in seeds:
            dataset_params = dict(dataset_base_params)
            dataset_params["random_state"] = rs
            out_it = One_scenario(
                    random_state=rs,
                    max_budget=max_budget,
                    dataset_params=dataset_params,
                    strategy_overrides=strategy_overrides,
                    strategy_names=[target_strategy],   # ✅ une seule stratégie
                    display=display_runs
                    )   
            

            df_seed = extract_stats_from_out(out_it, metrics=("accuracy", "f1_macro"), early_n_labeled=early_n)
            per_seed.append((rs, df_seed))

        # agrège sur seeds (comme CompareStrategy)
        _, agg = aggregate_across_seeds(
            per_seed,
            metrics=("accuracy", "f1_macro"),
            early_n_labeled=early_n,
            rank_by=rank_by,
            robust=robust
        )

        # on ne garde que la ligne de la stratégie cible (celle qu'on tune)
        row = agg[agg["strategy"] == target_strategy].copy()
        if row.empty:
            # si pas trouvé, on passe
            continue

        # ajoute un identifiant config lisible
        cfg_id = ", ".join(f"{k}={cfg[k]}" for k in keys)
        row["config_id"] = cfg_id

        rows_cfg.append(row)
        iteration = iteration + 1
    if not rows_cfg:
        raise RuntimeError("Aucune config évaluée (vérifie target_strategy ou la présence dans agg).")

    cfg_df = pd.concat(rows_cfg, ignore_index=True)

    # classement configs
    cfg_df = cfg_df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    cfg_df["rank_cfg"] = np.arange(1, len(cfg_df) + 1)

    print("\n" + "="*110)
    print(f"TOP {top_k} configs | {target_strategy} | rank_by={rank_by}")

    for _, r in cfg_df.head(top_k).iterrows():
        print(
        f"{int(r['rank_cfg']):>3} | "
        f"{r['config_id']:<35} | "
        f"E={r.get(f'accuracy@{early_n}_mean', float('nan')):.3f}"
        f"±{r.get(f'accuracy@{early_n}_std', 0.0):.3f} | "
        f"F={r.get('accuracy_final_mean', float('nan')):.3f}"
        f"±{r.get('accuracy_final_std', 0.0):.3f} | "
        f"A={r.get('accuracy_aulc_mean', float('nan')):.3f}"
        f"±{r.get('accuracy_aulc_std', 0.0):.3f} | "
        f"R={r.get('rank_score', float('nan')):.3f}"
    )

    print("\nBest config:", cfg_df.iloc[0]["config_id"])

    return cfg_df

def comparesweep():
    dataset_base_params = dict(
                n_samples=5000, n_features=20, n_informative=20,
                n_classes=3, class_sep=0.5, flip_y=0.1)
    
        
    CompareStrategySweep(
        target_strategy="ActivePseudoLabelV2",
        param_grid={
            "k_neighbors": [10,15],
            "lambda_prop": [0.2],
             "alpha_decay": [2.0],
            "gamma_disagreement": [0,0.1, 0.2],
        },
        seeds=list(range(0, 30)),
        dataset_base_params = dataset_base_params,
        max_budget=200,
        early_n=100,
        rank_by="accuracy_final",
        robust=True,
        display_runs=False
    )

if __name__ == "__main__":
    
    n_splits = 5
    max_budget = 400
    strategy_names = ["active_pseudolabel","ActivePseudoLabelV2",
                      "ActivePseudoLabelV3","ActivePseudoLabelV4"]
    for class_sep in [0.6, 0.8, 1]:
        #for features in [2,5,10,15,20,25,30,35,40]:
        for features in [2,5,10,15,20,25,30,35,40]:
            max_budget = 400
            print("\n results:",class_sep, features)
            if (features >= 30): 
                max_budget = 600
            CompareStrategy_constantbudget(strategy_names = strategy_names,
                                           max_budget = max_budget,class_sep=class_sep,
                                           n_features=features, n_splits=n_splits)
#    comparesweep()