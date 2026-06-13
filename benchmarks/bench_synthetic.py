from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ─── Path setup ─────────────────────────────────────────────────────────
this_file = Path(__file__).resolve()
project_root = this_file.parent

while project_root != project_root.parent:
    src = project_root / "src"
    if src.exists():
        break
    project_root = project_root.parent

if str(src) not in sys.path:
    sys.path.insert(0, str(src))
'''
import alframework.strategies.strategy_v58 as m
print('fichier chargé :', m.__file__)
print('classe         :', m.ActivePseudoLabelV58)
print('bases          :', m.ActivePseudoLabelV58.__bases__)
'''
# ─── Force registry population ─────────────────────────────────────────
def _ensure_strategies_registered() -> None:

    must_have = [
    "alframework.strategies.random",
    "alframework.strategies.typiclust",
    "alframework.strategies.uncertainty",
    "alframework.strategies.dbal",
    "alframework.strategies.rank2022",
    "alframework.strategies.qbc",
    "alframework.strategies.tri_committee",
    "alframework.strategies.coreset",
    "alframework.strategies.robust_qbc",
    "alframework.strategies.active_pseudolabelv4",
    #"alframework.strategies.active_pseudolabelv44",
    #"alframework.strategies.active_pseudolabelv45",
    #"alframework.strategies.active_pseudolabelv46",
    #"alframework.strategies.active_pseudolabelv47",
    #"alframework.strategies.active_pseudolabelv48",
    #"alframework.strategies.active_pseudolabelv5",
    #"alframework.strategies.active_pseudolabelv51",
    #"alframework.strategies.active_pseudolabelv52",
     #"alframework.strategies.active_pseudolabelv53",
     "alframework.strategies.active_pseudolabelv53",
    "alframework.strategies.active_pseudolabelv54",
    "alframework.strategies.active_pseudolabelv55",
    "alframework.strategies.active_pseudolabelv56",
    "alframework.strategies.active_pseudolabelv57",
     "alframework.strategies.strategy_v58",
    "alframework.strategies.probcover",
    "alframework.strategies.badge",
    "alframework.strategies.bait_simple",
    "alframework.strategies.adaptive_disagreement",
    "alframework.strategies.diversity_optimized_batch",
    "alframework.strategies.selftrain_acq",
    ]

    for mod_path in must_have:
        try:
            __import__(mod_path)
        except ImportError as e:
            print(f"[FATAL] could not import {mod_path}: {e}")
            sys.exit(1)


# ─── Scenarios ─────────────────────────────────────────────────────────
SCENARIOS: Dict[str, Dict] = {
    # Original ablation scenarios
    "easy": dict(
        n_samples=1500,
        n_classes=3,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        class_sep=1.5,
        flip_y=0.01,
    ),

    "medium": dict(
        n_samples=1500,
        n_classes=3,
        n_features=20,
        n_informative=10,
        n_redundant=6,
        class_sep=1.2,
        flip_y=0.02,
    ),

    "hard": dict(
        n_samples=1500,
        n_classes=3,
        n_features=30,
        n_informative=10,
        n_redundant=15,
        class_sep=0.8,
        flip_y=0.05,
    ),

   "imbalanced": dict(
    n_samples=1500,
    n_classes=3,
    n_features=30,
    n_informative=12,
    n_redundant=8,
    class_sep=1.0,
    flip_y=0.03,
    weights=[0.70, 0.20, 0.10],
    ),

    "imbalanced_hard": dict(
    n_samples=1500,
    n_classes=3,
    n_features=30,
    n_informative=10,
    n_redundant=10,
    class_sep=0.8,
    flip_y=0.05,
    weights=[0.75, 0.18, 0.07],
    ),
    
    # Additional controlled cases, not too extreme
    "clean20": dict(
        n_samples=1500,
        n_classes=3,
        n_features=20,
        n_informative=20,
        n_redundant=0,
        class_sep=1.0,
        flip_y=0.01,
    ),

    "redundant20": dict(
        n_samples=1500,
        n_classes=3,
        n_features=20,
        n_informative=8,
        n_redundant=8,
        class_sep=1.0,
        flip_y=0.02,
    ),

    "noisy_medium": dict(
        n_samples=1500,
        n_classes=3,
        n_features=30,
        n_informative=12,
        n_redundant=8,
        class_sep=1.0,
        flip_y=0.06,
    ),

    "medium5c": dict(
        n_samples=1500,
        n_classes=5,
        n_features=30,
        n_informative=15,
        n_redundant=8,
        class_sep=1.0,
        flip_y=0.03,
    ),
    
    "low_signal": dict(
    n_samples=1500,
    n_classes=3,
    n_features=30,
    n_informative=8,
    n_redundant=8,
    class_sep=0.6,
    flip_y=0.03,
    ),
    
    "many_classes": dict(
    n_samples=1500,
    n_classes=8,
    n_features=40,
    n_informative=20,
    n_redundant=10,
    class_sep=1.0,
    flip_y=0.03,
),
    
    "clustered_imbalanced": dict(
    n_samples=2000,
    n_classes=4,
    n_features=40,
    n_informative=14,
    n_redundant=10,
    n_repeated=2,
    n_clusters_per_class=3,
    class_sep=0.9,
    flip_y=0.03,
    weights=[0.55, 0.25, 0.15, 0.05],
),
    
    "highdim_sparse": dict(
    n_samples=2000,
    n_classes=3,
    n_features=200,
    n_informative=12,
    n_redundant=20,
    n_repeated=0,
    class_sep=1.0,
    flip_y=0.02,
),
    "local_overlap": dict(
    n_samples=1800,
    n_classes=3,
    n_features=35,
    n_informative=14,
    n_redundant=10,
    n_clusters_per_class=4,
    class_sep=0.7,
    flip_y=0.02,
),
    "extreme_redundancy": dict(
    n_samples=1800,
    n_classes=3,
    n_features=80,
    n_informative=8,
    n_redundant=50,
    class_sep=1.0,
    flip_y=0.02,
),
  
    "rare_class": dict(
    n_samples=2500,
    n_classes=4,
    n_features=40,
    n_informative=15,
    n_redundant=10,
    class_sep=1.0,
    flip_y=0.01,
    weights=[0.80, 0.12, 0.06, 0.02],
),
}
  


# ─── Strategy panel ─────────────────────────────────────────────────────
'''
PANEL: Dict[str, Tuple[str, Dict]] = {
    "random": ("random", {}),
    "margin": ("margin", {}),
    "entropy": ("entropy", {}),
    "least_confident": ("least_confident", {}),

    "dbal": ("dbal", {
        "method": "margin",
        "dbal_factor": 5,
    }),

    "rank2022": ("rank2022", {}),
    "qbc": ("qbc", {}),
    "coreset": ("coreset_greedy", {}),

    "V4": ("ActivePseudoLabelV4", {}),
    "V4.4": ("ActivePseudoLabelV44", {}),

    "V4.7": ("ActivePseudoLabelV47", {
        "candidate_multiplier": 5.0,
        "diversity_weight": 0.35,
        "density_weight": 0.10,
        "adaptive_diversity": True,
        "lam_threshold": 0.15,
    }),

    "V4.8": ("ActivePseudoLabelV48", {
        "candidate_multiplier": 5.0,
        "diversity_base": 0.18,
        "diversity_max": 0.65,
        "lam_weakness_weight": 0.35,
        "multiclass_weight": 0.15,
        "imbalance_weight": 0.15,
        "density_weight": 0.10,
        "adaptive_diversity": True,
        "lam_threshold": 0.15,
        "random_state": 0,
    }),
   
    "V5": ("ActivePseudoLabelV5", {
    "candidate_multiplier": 5.0,
    "diversity_base": 0.10,
    "diversity_max": 0.50,
    "density_weight": 0.05,
    "lam_threshold": 0.15,
    "lam_weakness_weight": 0.20,
    "flatness_weight": 0.18,
    "imbalance_weight": 0.10,
    "contrast_guard_weight": 0.18,
    "highdim_guard_weight": 0.15,
    "adaptive_diversity": True,
    "adaptive_representative": True,
    "random_state": 0,
}),

    # ── V5 improvement attempts (Piste 1 / Piste 2) ──────────────────
    # Honest (cross-validated) propagation is now built into the V5
    # family itself, so V5 / V5.1 / V5.2 genuinely diverge.
    "V5.1": ("ActivePseudoLabelV51", {
        "candidate_multiplier": 5.0,
        "diversity_base": 0.10,
        "diversity_max": 0.50,
        "density_weight": 0.05,
        "lam_threshold": 0.15,
        "lam_weakness_weight": 0.20,
        "flatness_weight": 0.18,
        "imbalance_weight": 0.10,
        "contrast_guard_weight": 0.18,
        "highdim_guard_weight": 0.15,
        "adaptive_diversity": True,
        "adaptive_representative": True,
        "random_state": 0,
        "cold_threshold": 0.05,
        "transition_smoothing": False,
    }),

    "V5.2": ("ActivePseudoLabelV52", {
        "candidate_multiplier": 5.0,
        "diversity_base": 0.10,
        "diversity_max": 0.50,
        "density_weight": 0.05,
        "lam_threshold": 0.15,
        "lam_weakness_weight": 0.20,
        "flatness_weight": 0.18,
        "imbalance_weight": 0.10,
        "contrast_guard_weight": 0.18,
        "highdim_guard_weight": 0.15,
        "adaptive_diversity": True,
        "adaptive_representative": True,
        "random_state": 0,
    }),
    
    "V5.4": ("ActivePseudoLabelV54", {
    "lambda_prop": 0.0,
    "source_policy": "auto_fast",
    "max_u_contrast_for_source": 0.28,
    "min_u_flatness_for_source": 0.88,
    "max_selection_pool": 250,
    "adaptive_representative": True,
    "stag_window": 3,
    "stag_threshold": 0.15 ,
    "stag_delta_ref": 0.03,
    "epsilon": 0.20,
    "explore_low_pct": 20.0,
    "random_state": 0,
}
),
}
'''
'''
PANEL: Dict[str, Tuple[str, Dict]] = {
    "random": ("random", {}),
    "margin": ("margin", {}),
   

    "dbal": ("dbal", {
        "method": "margin",
        "dbal_factor": 5,
    }),

   
   "V5.2": ("ActivePseudoLabelV52", {
    "lambda_prop": 0.0,
    "adaptive_representative": True,
    "random_state": 0,
}),
}'''
'''
PANEL: Dict[str, Tuple[str, Dict]] = {
    "random": ("random", {}),
    "margin": ("margin", {}),
    "entropy": ("entropy", {}),
    "least_confident": ("least_confident", {}),
    "tri_committee": ("tri_committee", {}),
    "typiclust": ("typiclust", {
        "neighbors": 5,
        "random_state": 0,
    }),
    "coreset": ("coreset_greedy", {}),
    "dbal": ("dbal", {
        "method": "margin",
        "dbal_factor": 5,
    }),

    "rank2022": ("rank2022", {}),
    "qbc": ("qbc", {}),
    
    #"V4": ("ActivePseudoLabelV4", {}),
    "V5.3": ("ActivePseudoLabelV53", {
        "lambda_prop": 0.0,
        "source_policy": "auto_fast",
        "max_u_contrast_for_source": 0.28,
        "min_u_flatness_for_source": 0.88,
        "max_selection_pool": 250,
        "adaptive_representative": True,
        "random_state": 0,
    }),

    "V5.4": ("ActivePseudoLabelV54", {
        "lambda_prop": 0.0,
        "source_policy": "auto_fast",
        "max_u_contrast_for_source": 0.28,
        "min_u_flatness_for_source": 0.88,
        "max_selection_pool": 250,
        "adaptive_representative": True,
        "random_state": 0,
    }),

    "V5.5": ("ActivePseudoLabelV55", {
        "lambda_prop": 0.0,
        "source_policy": "auto_fast",
        "max_u_contrast_for_source": 0.28,
        "min_u_flatness_for_source": 0.88,
        "max_selection_pool": 250,
        "adaptive_representative": True,
        "random_state": 0,
    }),
"V5.5": ("ActivePseudoLabelV55", {
    # ── Héritage V5.3 — identique ─────────────────────
    "lambda_prop": 0.0,
    "source_policy": "auto_fast",
    "max_u_contrast_for_source": 0.28,
    "min_u_flatness_for_source": 0.88,
    "max_selection_pool": 250,
    "adaptive_representative": True,
    "random_state": 0,
    # ── Smart V5.4 — hérité ───────────────────────────
    "smart_enabled": True,
    # ── Routing clustering V5.5 — defaults suffisants ─
    "ward_enabled": True,     # Ward si LR-like, KMeans si RF-like
    "pm_flat": 8,             # pool×8  pour classifieur plat  (RF)
    "pm_disc": 10,            # pool×20 pour classifieur discr. (LR)
    "ward_linkage": "ward",   # linkage Ward
}),
  
  "ActivePseudoLabelV56": ("ActivePseudoLabelV56", {
    "lambda_prop": 0.0,
    "source_policy": "auto_fast",
    "max_u_contrast_for_source": 0.28,
    "min_u_flatness_for_source": 0.88,
    "max_selection_pool": 250,
    "adaptive_representative": True,
    "random_state": 0,
    "smart_enabled": True,
    "ward_enabled": True,
    "pm_flat": 8,
    "pm_disc": 20,
    "ward_linkage": "ward",
    # V5.6 sat-adapt
    "gain_window": 3,
    "gain_thr": 0.005,
    "p_plateau": 1.4,
    "debug_sat": True,
}),
    "ActivePseudoLabelV57": ("ActivePseudoLabelV57", {
        "lambda_prop": 0.0, "source_policy": "auto_fast",
        "max_u_contrast_for_source": 0.28, "min_u_flatness_for_source": 0.88,
        "max_selection_pool": 250, "adaptive_representative": True,
        "random_state": 0, "smart_enabled": True, "ward_enabled": True,
        "route_round": 1, "multiclass_thr": 5,
        "dbal_pool_mult": 5, "r22_pool_mult": 10, "debug_route": False,
    }),
}
'''

   
# =====================================================================
# PANEL complet — toutes les méthodes du stock, regroupées par école.
# Collez ce dict à la place du PANEL actif (le DERNIER du fichier).
# Pour un test 2-seeds, lancez avec --seeds 2 (ou réglez SEEDS=2).
# =====================================================================
PANEL: Dict[str, Tuple[str, Dict]] = {

    # ── Baselines / incertitude pure ─────────────────────────────────
    "random":          ("random", {}),
    "margin":          ("margin", {}),
    "entropy":         ("entropy", {}),
    "least_confident": ("least_confident", {}),

    # ── Diversité pure ───────────────────────────────────────────────
    "coreset_greedy":   ("coreset_greedy", {}),
    "coreset_kmeanspp": ("coreset_kmeanspp", {}),
    "typiclust":        ("typiclust", {"neighbors": 5, "random_state": 0}),
    "probcover":        ("probcover", {"delta_quantile": 0.10, "knn_k": 10, "random_state": 0}),

    # ── Hybrides incertitude + diversité ─────────────────────────────
    "badge_approx":        ("badge_approx", {"random_state": 0}),
    "unc_feature_kmeans":  ("unc_feature_kmeans", {"uncertainty_metric": "margin", "n_clusters_factor": 3}),
    "diversity_opt_batch": ("diversity_optimized_batch", {"uncertainty_metric": "margin", "lambda_diversity": 0.5}),

    # ── Comité / désaccord ───────────────────────────────────────────
    "qbc":                   ("qbc", {"n_committee": 5, "bootstrap_ratio": 1.0, "metric": "vote_entropy"}),
    "robust_qbc":            ("robust_qbc", {"n_committee": 5, "bootstrap_ratio": 0.8, "metric": "confidence_weighted"}),
    "tri_committee":         ("tri_committee", {"bootstrap_ratio": 1.0, "mode": "two_vs_one"}),
    "adaptive_disagreement": ("adaptive_disagreement", {"n_committee": 5, "min_disagreement": 0.2}),

    # ── Famille unifiée (uncertainty + clustering) ───────────────────
    "dbal":     ("dbal", {"method": "margin", "dbal_factor": 5, "random_state": 0}),
    "rank2022": ("rank2022", {"s_factor": 10, "linkage": "ward", "loss": "margin"}),
    "V58B":     ("ActivePseudoLabelV58", {
        "variant": "V58b", "multiclass_thr": 5, "u_flat_trigger": 0.50,
        "eff_dim_thr": 12.0, "peak_lo": 0.58, "route_round": 1,
        "correction_round": 3, "hysteresis": 0.05, "kmeans_n_init": "auto",
        "random_state": 0, "debug_route": False,
    }),

    # ── À MANIER AVEC PRUDENCE ───────────────────────────────────────
    # selftrain_acq n'est PAS une acquisition AL à budget fixe : en mode
    # near_threshold elle ne renvoie que les points dans une bande étroite
    # autour de tau, donc souvent < budget points. Elle fausserait une
    # comparaison à budget constant. Décommentez en connaissance de cause.
    # "selftrain_acq": ("selftrain_acq", {"tau": 0.9, "window": 0.05, "mode": "near_threshold"}),
}

 


# ─── Helpers ────────────────────────────────────────────────────────────
def auc_curve(accs: List[float]) -> float:
    """Normalized trapezoidal AUC of an accuracy curve."""
    a = np.asarray(accs, dtype=float)
    trap = np.trapezoid(a) if hasattr(np, "trapezoid") else np.trapz(a)
    return float(trap / max(1, len(a) - 1))


def _make_model(model_kind: str, seed: int):
    if model_kind == "rf":
        return RandomForestClassifier(n_estimators=50, random_state=seed)

    if model_kind == "lr":
        return LogisticRegression(max_iter=2000, random_state=seed)

    raise ValueError(f"unknown model_kind: {model_kind}")


def run_one(
    variant_label: str,
    seed: int,
    scenario: str,
    model_kind: str,
    *,
    n_init: int = 30,
    n_rounds: int = 6,
    budget_per_round: int = 10,
    init_mode: str = "stratified",
    histories_dir: str = None,
    split_idx: int = 0,
) -> List[float]:

    from alframework.core.state import ALState
    from alframework.core.runner import active_learning_loop
    from alframework.core.labeler import ArrayLabeler
    from alframework.utils.seed import seed_everything
    from alframework.core.registry import STRATEGIES

    rng = seed_everything(seed)
    p = dict(SCENARIOS[scenario])

    n_samples = int(p.pop("n_samples", 1500))
    n_classes = int(p.pop("n_classes", 3))

    X, y = make_classification(
        n_samples=n_samples,
        n_classes=n_classes,
        random_state=seed,
        **p,
    )

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )

    # ------------------------------------------------------------------
    # Initial labeled set
    # ------------------------------------------------------------------
    if n_init <= 0:
        raise ValueError(f"n_init must be positive, got {n_init}")

    if n_init > len(Xtr):
        raise ValueError(
            f"n_init={n_init} cannot exceed train size={len(Xtr)}"
        )

    if init_mode == "random":
        # Pure random initialization.
        # Can miss rare classes on imbalanced datasets.
        init_indices = rng.choice(
            len(Xtr),
            size=n_init,
            replace=False,
        ).astype(int)

    elif init_mode in ["stratified", "random_safe"]:
        # stratified:
        #   controlled protocol, at least one sample per class.
        #
        # random_safe:
        #   same class coverage guarantee, but remaining points are random.
        #   Useful for imbalanced datasets without metric crashes.
        if n_init < n_classes:
            raise ValueError(
                f"n_init={n_init} must be >= n_classes={n_classes} "
                f"when init_mode={init_mode!r}"
            )

        init_indices = []

        # Ensure at least one sample per class.
        for c in range(n_classes):
            candidates = np.where(ytr == c)[0]

            if candidates.size == 0:
                raise ValueError(f"No sample found for class {c} in train split")

            init_indices.append(int(rng.choice(candidates)))

        remaining_pool = np.setdiff1d(
            np.arange(len(Xtr)),
            np.asarray(init_indices, dtype=int),
            assume_unique=False,
        )

        n_remaining = n_init - len(init_indices)

        if n_remaining > 0:
            extra = rng.choice(
                remaining_pool,
                size=n_remaining,
                replace=False,
            )
            init_indices.extend([int(i) for i in extra])

        init_indices = np.asarray(init_indices, dtype=int)

    else:
        raise ValueError(
            f"unknown init_mode={init_mode!r}; "
            f"expected 'random', 'random_safe' or 'stratified'"
        )

    mask = np.zeros(len(Xtr), dtype=bool)
    mask[init_indices] = True

    X_l, y_l = Xtr[mask], ytr[mask]
    X_u, y_u = Xtr[~mask], ytr[~mask]

    model = _make_model(model_kind, seed)

    state = ALState(
        X_labeled=X_l,
        y_labeled=y_l,
        X_unlabeled=X_u,
        model=model,
        rng=rng,
        X_test=Xte,
        y_test=yte,
    )

    # Metadata useful for QBC / robust_qbc / safe probability mapping
    state.n_classes = n_classes
    state.labels = np.arange(n_classes)
    state.all_labels = np.arange(n_classes)
    state.y_pool = ytr

    labeler = ArrayLabeler(y_u)

    reg_name, kwargs = PANEL[variant_label]

    if reg_name not in STRATEGIES:
        raise KeyError(
            f"strategy '{reg_name}' not found in registry. "
            f"Available: {sorted(STRATEGIES.keys())}"
        )

    # Reproducibility: inject the run seed into random_state for strategies that
    # accept it, so each seed produces a genuinely different (yet reproducible)
    # internal randomization. Strategies without a random_state parameter are
    # left untouched (their variability already comes from the seeded data/model).
    import inspect as _inspect
    _kwargs = dict(kwargs)
    try:
        _params = _inspect.signature(STRATEGIES[reg_name].__init__).parameters
        if "random_state" in _params:
            _kwargs["random_state"] = int(seed)
    except (ValueError, TypeError):
        pass
    strategy = STRATEGIES[reg_name](**_kwargs)

    hist = active_learning_loop(
        state,
        strategy,
        labeler,
        n_rounds=n_rounds,
        budget=budget_per_round,
    )

    # ── Sauvegarde history JSON (compatible analyze_histories.py) ──────
    if histories_dir is not None:
        import json as _json
        os.makedirs(histories_dir, exist_ok=True)
        # Format : history_{model}_{scenario}_{split_idx}_{variant_label}.json
        safe_label = variant_label.replace("/", "_").replace(".", "_")
        fname = f"history_{model_kind}_{scenario}_split{split_idx}_{safe_label}.json"
        fpath = os.path.join(histories_dir, fname)
        # Chaque step : n_labeled + métriques
        steps = []
        n_lab = n_init
        for i, h in enumerate(hist):
            step = {
                "n_labeled":          n_lab,
                "accuracy":           float(h.get("accuracy", 0.0)),
                "f1_macro":           float(h.get("f1_macro", h.get("accuracy", 0.0))),
                "balanced_accuracy":  float(h.get("balanced_accuracy", h.get("accuracy", 0.0))),
            }
            steps.append(step)
            if i < len(hist) - 1:
                n_lab += budget_per_round
        with open(fpath, "w", encoding="utf-8") as _f:
            _json.dump(steps, _f, indent=2)

    return {
        "accuracy": [h["accuracy"]                                    for h in hist],
        "f1_macro": [h.get("f1_macro", h.get("accuracy", 0.0))       for h in hist],
        "balanced_accuracy": [h.get("balanced_accuracy",
                                    h.get("accuracy", 0.0))          for h in hist],
    }
# ─── Main benchmark ─────────────────────────────────────────────────────
# STRICT_SEEDS=True : tout seed qui échoue stoppe le run avec
# la stack trace (à utiliser pour les runs de validation).
# False : la cellule touchée est silencieusement abandonnée
# (jamais faussement appariée).
STRICT_SEEDS = False




# ─── Per-seed dump (ajouté par make_fixed_benchmark) ────────────────────
def write_per_seed_csv(results: Dict, path: str) -> None:
    """
    Dump LONG : une ligne par (model, scenario, variant, seed_index).

    Prérequis de toute analyse APPARIÉE. seed_index est positionnel ;
    grâce au correctif d'appariement (cellule vidée si un seed échoue),
    une position N correspond au même seed (42 + N*17) pour toutes les
    méthodes d'une même cellule (model, scenario).
    """
    fieldnames = ["model", "scenario", "variant",
                  "seed_index",
                  "auc_acc",   "final_acc",
                  "auc_f1",    "final_f1",
                  "auc",       "final"]   # aliases legacy
    n_rows = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        w.writeheader()
        for (model, scenario, variant), r in sorted(results.items()):
            aucs_acc   = r.get("aucs_acc",   r.get("aucs",   []))
            finals_acc = r.get("finals_acc", r.get("finals", [None]*len(aucs_acc)))
            aucs_f1    = r.get("aucs_f1",    [None]*len(aucs_acc))
            finals_f1  = r.get("finals_f1",  [None]*len(aucs_acc))
            for s_idx, auc_val in enumerate(aucs_acc):
                fa  = finals_acc[s_idx] if s_idx < len(finals_acc) else None
                af1 = aucs_f1[s_idx]    if s_idx < len(aucs_f1)   else None
                ff1 = finals_f1[s_idx]  if s_idx < len(finals_f1)  else None
                def _fmt(v): return "" if v is None else f"{float(v):.6f}"
                w.writerow({
                    "model": model, "scenario": scenario, "variant": variant,
                    "seed_index": s_idx,
                    "auc_acc":   _fmt(auc_val), "final_acc": _fmt(fa),
                    "auc_f1":    _fmt(af1),     "final_f1":  _fmt(ff1),
                    "auc":       _fmt(auc_val), "final":     _fmt(fa),
                })
                n_rows += 1
    print(f"[per_seed_dump] wrote {n_rows} rows -> {path}")
# ────────────────────────────────────────────────────────────────────────


def run_benchmark(
    *,
    seeds: int,
    models: List[str],
    scenarios: List[str],
    panel: List[str],
    init_mode: str,
    n_init: int = 30,
    batch_size: int = 10,
    n_rounds: int = 6,
    max_budget: int = None,
    histories_dir: str = None,
) -> Dict:
    results: Dict = {}
    n_cells = len(models) * len(scenarios) * len(panel)
    cell_idx = 0

    for model_kind in models:
        for sc in scenarios:
            for v in panel:
                cell_idx += 1
                ts = time.time()
                finals_acc, aucs_acc = [], []
                finals_f1,  aucs_f1  = [], []

                for s_idx in range(seeds):
                    try:
                        _n_rounds = n_rounds
                        if max_budget is not None:
                            _n_rounds = max(1, (max_budget - n_init) // batch_size)

                        curves = run_one(
                            variant_label=v,
                            seed=42 + s_idx * 17,
                            scenario=sc,
                            model_kind=model_kind,
                            init_mode=init_mode,
                            n_init=n_init,
                            n_rounds=_n_rounds,
                            budget_per_round=batch_size,
                            histories_dir=histories_dir,
                            split_idx=s_idx,
                        )

                        acc_curve = curves["accuracy"]
                        f1_curve  = curves["f1_macro"]

                        finals_acc.append(acc_curve[-1])
                        aucs_acc.append(auc_curve(acc_curve))
                        finals_f1.append(f1_curve[-1])
                        aucs_f1.append(auc_curve(f1_curve))

                    except KeyError as e:
                        print(f"  [{cell_idx}/{n_cells}] SKIP {v}: {e}")
                        break

                    except Exception as e:
                        print(
                            f"  [{cell_idx}/{n_cells}] FAIL "
                            f"{model_kind}/{sc}/{v} seed {s_idx}: "
                            f"{type(e).__name__}: {str(e)[:80]}"
                        )
                        if STRICT_SEEDS:
                            raise
                        finals_acc, aucs_acc = [], []
                        finals_f1,  aucs_f1  = [], []
                        break

                if finals_acc and len(finals_acc) == seeds:
                    results[(model_kind, sc, v)] = {
                        "finals":     np.array(finals_acc),   # alias legacy
                        "aucs":       np.array(aucs_acc),     # alias legacy
                        "finals_acc": np.array(finals_acc),
                        "aucs_acc":   np.array(aucs_acc),
                        "finals_f1":  np.array(finals_f1),
                        "aucs_f1":    np.array(aucs_f1),
                    }

                    print(
                        f"  [{cell_idx}/{n_cells}] "
                        f"{model_kind:<3}/{sc:<14}/{v:<7}  "
                        f"AUC = {np.mean(aucs_acc):.4f} ± {np.std(aucs_acc):.4f}  "
                        f"ACC = {np.mean(finals_acc):.4f}  "
                        f"F1 = {np.mean(finals_f1):.4f}  "
                        f"({time.time() - ts:.1f}s)"
                    )

    return results


# ─── Output emission ────────────────────────────────────────────────────
def write_csv(results: Dict, path: str) -> None:
    fieldnames = [
        "model", "scenario", "variant", "n_seeds",
        "acc_final_mean", "acc_final_std",
        "auc_acc_mean",   "auc_acc_std",
        "f1_final_mean",  "f1_final_std",
        "auc_f1_mean",    "auc_f1_std",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        w.writeheader()
        for (m, s, v), r in sorted(results.items()):
            fa  = r.get("finals_acc", r.get("finals", np.array([])))
            aa  = r.get("aucs_acc",   r.get("aucs",   np.array([])))
            ff  = r.get("finals_f1",  np.array([]))
            af  = r.get("aucs_f1",    np.array([]))
            def _ms(a): return (f"{a.mean():.4f}", f"{a.std():.4f}") if len(a) else ("","")
            w.writerow({
                "model": m, "scenario": s, "variant": v, "n_seeds": len(fa),
                "acc_final_mean": _ms(fa)[0], "acc_final_std": _ms(fa)[1],
                "auc_acc_mean":   _ms(aa)[0], "auc_acc_std":   _ms(aa)[1],
                "f1_final_mean":  _ms(ff)[0], "f1_final_std":  _ms(ff)[1],
                "auc_f1_mean":    _ms(af)[0], "auc_f1_std":    _ms(af)[1],
            })


def emit_latex_table(
    results: Dict,
    panel: List[str],
    scenarios: List[str],
    model_kind: str,
    path: str,
) -> None:
    model_name = {
        "rf": "RandomForest",
        "lr": "LogReg",
    }.get(model_kind, model_kind)

    means: Dict[Tuple[str, str], float] = {}

    for sc in scenarios:
        for v in panel:
            if (model_kind, sc, v) in results:
                means[(sc, v)] = float(
                    results[(model_kind, sc, v)]["aucs"].mean()
                )

    best, second = {}, {}

    for sc in scenarios:
        col = [(v, means[(sc, v)]) for v in panel if (sc, v) in means]
        col.sort(key=lambda kv: -kv[1])

        if col:
            best[sc] = col[0][0]

            if len(col) > 1:
                second[sc] = col[1][0]

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Active learning strategy ablation. Learning-curve AUC on "
        + model_name
        + r", mean over seeds. Best per scenario in "
        + r"\textbf{bold}, second-best \underline{underlined}.}"
    )
    lines.append(r"\label{tab:ablation_v4_" + model_kind + r"}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "c" * len(scenarios) + r"}")
    lines.append(r"\toprule")

    header = "Variant"
    for sc in scenarios:
        header += f" & {sc}"

    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for v in panel:
        row = v

        for sc in scenarios:
            if (sc, v) not in means:
                row += " & --"
                continue

            txt = f"{means[(sc, v)]:.4f}"

            if best.get(sc) == v:
                txt = r"\textbf{" + txt + r"}"
            elif second.get(sc) == v:
                txt = r"\underline{" + txt + r"}"

            row += f" & {txt}"

        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def emit_summary(
    results: Dict,
    panel: List[str],
    models: List[str],
    scenarios: List[str],
    path: str,
    seeds: int,
) -> None:
    """Human-readable summary with ranks, wins/losses, t-tests, Wilcoxon and Friedman."""
    lines: List[str] = []

    lines.append("# Ablation summary — active learning strategies")
    lines.append("")
    lines.append(
        f"Configuration: {seeds} seeds per cell, "
        f"models = {models}, scenarios = {scenarios}, "
        f"panel = {panel}"
    )
    lines.append("")

    # ------------------------------------------------------------------
    # AUC matrix
    # ------------------------------------------------------------------
    for model_kind in models:
        lines.append(f"## Learning-curve AUC — {model_kind}")
        lines.append("")

        header = "| Variant | " + " | ".join(scenarios) + " |"
        sep = "|" + "---|" * (len(scenarios) + 1)

        lines.append(header)
        lines.append(sep)

        for v in panel:
            row = f"| {v} |"

            for sc in scenarios:
                key = (model_kind, sc, v)
                if key in results:
                    row += f" {results[key]['aucs'].mean():.4f} ± {results[key]['aucs'].std():.4f} |"
                else:
                    row += " -- |"

            lines.append(row)

        lines.append("")

    # ------------------------------------------------------------------
    # Average ranks
    # ------------------------------------------------------------------
    lines.append("## Average ranks")
    lines.append("")
    lines.append("Lower is better. Ranks are computed per model/scenario cell using mean AUC.")
    lines.append("")
    lines.append("| Variant | Avg rank | Best cells |")
    lines.append("|---|---:|---:|")

    rank_sum = {v: 0.0 for v in panel}
    rank_count = {v: 0 for v in panel}
    best_count = {v: 0 for v in panel}

    for model_kind in models:
        for sc in scenarios:
            vals = []
            for v in panel:
                key = (model_kind, sc, v)
                if key in results:
                    vals.append((v, float(results[key]["aucs"].mean())))

            vals.sort(key=lambda x: -x[1])

            if not vals:
                continue

            best_val = vals[0][1]
            for v, val in vals:
                if np.isclose(val, best_val, atol=1e-12):
                    best_count[v] += 1

            # simple competition ranking with average ties
            sorted_values = np.array([x[1] for x in vals])
            names = [x[0] for x in vals]

            used = np.zeros(len(vals), dtype=bool)
            for i in range(len(vals)):
                if used[i]:
                    continue

                tied = np.where(np.isclose(sorted_values, sorted_values[i], atol=1e-12))[0]
                tied = [j for j in tied if not used[j]]

                rank = 1.0 + np.mean(tied)

                for j in tied:
                    rank_sum[names[j]] += rank
                    rank_count[names[j]] += 1
                    used[j] = True

    avg_ranks = {}
    for v in panel:
        if rank_count[v] > 0:
            avg_ranks[v] = rank_sum[v] / rank_count[v]
        else:
            avg_ranks[v] = np.nan

    for v in sorted(panel, key=lambda x: avg_ranks.get(x, np.inf)):
        lines.append(f"| {v} | {avg_ranks[v]:.3f} | {best_count[v]} |")

    lines.append("")

    # ------------------------------------------------------------------
    # Wins / ties / losses vs V4
    # ------------------------------------------------------------------
    baseline = "V4"

    if baseline in panel:
        lines.append("## Wins / ties / losses vs V4")
        lines.append("")
        lines.append("| Variant | Wins | Ties | Losses | Mean delta |")
        lines.append("|---|---:|---:|---:|---:|")

        for v in panel:
            if v == baseline:
                continue

            wins = ties = losses = 0
            deltas = []

            for model_kind in models:
                for sc in scenarios:
                    k_base = (model_kind, sc, baseline)
                    k = (model_kind, sc, v)

                    if k_base not in results or k not in results:
                        continue

                    delta = float(results[k]["aucs"].mean() - results[k_base]["aucs"].mean())
                    deltas.append(delta)

                    if delta > 1e-4:
                        wins += 1
                    elif delta < -1e-4:
                        losses += 1
                    else:
                        ties += 1

            mean_delta = float(np.mean(deltas)) if deltas else np.nan
            lines.append(f"| {v} | {wins} | {ties} | {losses} | {mean_delta:+.4f} |")

        lines.append("")

    # ------------------------------------------------------------------
    # Paired t-test and Wilcoxon vs V4
    # ------------------------------------------------------------------
    if baseline in panel:
        compared_variants = [v for v in panel if v != baseline]

        lines.append("## Paired tests on AUC vs V4")
        lines.append("")
        lines.append("| Model | Scenario | Variant | Delta | paired t p | Wilcoxon p | Sig |")
        lines.append("|---|---|---|---:|---:|---:|---|")

        for model_kind in models:
            for sc in scenarios:
                k_base = (model_kind, sc, baseline)

                if k_base not in results:
                    continue

                a_base = results[k_base]["aucs"]

                for v in compared_variants:
                    k = (model_kind, sc, v)

                    if k not in results:
                        continue

                    a = results[k]["aucs"]

                    if len(a) != len(a_base) or len(a) <= 1:
                        continue

                    delta = float(a.mean() - a_base.mean())

                    try:
                        _, p_t = stats.ttest_rel(a, a_base)
                    except Exception:
                        p_t = np.nan

                    try:
                        # Wilcoxon can fail if all differences are exactly zero
                        _, p_w = stats.wilcoxon(a, a_base)
                    except Exception:
                        p_w = np.nan

                    p_ref = p_w if not np.isnan(p_w) else p_t

                    sig = ""
                    if not np.isnan(p_ref):
                        if p_ref < 0.01:
                            sig = "***"
                        elif p_ref < 0.05:
                            sig = "**"
                        elif p_ref < 0.10:
                            sig = "*"

                    lines.append(
                        f"| {model_kind} | {sc} | {v} | "
                        f"{delta:+.4f} | {p_t:.4f} | {p_w:.4f} | {sig} |"
                    )

        lines.append("")
        lines.append("Significance based preferably on Wilcoxon p-value: `***` p<0.01, `**` p<0.05, `*` p<0.10")
        lines.append("")

    # ------------------------------------------------------------------
    # Friedman global test per model/scenario
    # ------------------------------------------------------------------
    lines.append("## Friedman tests across variants")
    lines.append("")
    lines.append("Friedman is computed within each model/scenario cell across variants, using seed-wise AUC.")
    lines.append("")
    lines.append("| Model | Scenario | Variants included | Friedman statistic | p-value |")
    lines.append("|---|---|---:|---:|---:|")

    for model_kind in models:
        for sc in scenarios:
            arrays = []
            included = []

            for v in panel:
                key = (model_kind, sc, v)
                if key in results:
                    arrays.append(results[key]["aucs"])
                    included.append(v)

            if len(arrays) < 3:
                continue

            lengths = [len(a) for a in arrays]
            if len(set(lengths)) != 1 or lengths[0] <= 1:
                continue

            try:
                stat, p = stats.friedmanchisquare(*arrays)
                lines.append(
                    f"| {model_kind} | {sc} | {len(included)} | "
                    f"{stat:.4f} | {p:.4f} |"
                )
            except Exception:
                lines.append(
                    f"| {model_kind} | {sc} | {len(included)} | "
                    f"nan | nan |"
                )

    lines.append("")

    # ------------------------------------------------------------------
    # Compact recommendation
    # ------------------------------------------------------------------
    lines.append("## Compact interpretation guide")
    lines.append("")
    lines.append("- `Avg rank` summarizes global robustness across cells.")
    lines.append("- `Best cells` counts how often a method is first or tied first.")
    lines.append("- `Wins / ties / losses vs V4` measures direct improvement over the baseline V4.")
    lines.append("- Wilcoxon is more robust than paired t-test when seed-wise differences are non-Gaussian.")
    lines.append("- Friedman tests whether there is a global difference among variants within a cell.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ─── CLI ────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Active learning ablation benchmark across baseline and proposed strategies"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=20,
        help="seeds per cell; use 3 for smoke",
    )

    parser.add_argument(
        "--out",
        default="ablation_v5_results",
        help="output directory",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf", "lr"],
        choices=["rf", "lr"],
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SCENARIOS.keys()),
        choices=list(SCENARIOS.keys()),
    )

    parser.add_argument(
        "--panel",
        nargs="+",
        default=list(PANEL.keys()),
        choices=list(PANEL.keys()),
    )
   
    parser.add_argument(
        "--init-mode",
        choices=["random", "random_safe", "stratified"],
        default="random_safe",
        help="Mode de construction du labeled set initial",
    )

    parser.add_argument(
        "--n-init",
        type=int,
        default=20,
        help="Nombre de points initiaux labelisés (défaut: 30)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Taille du batch AL à chaque itération (défaut: 10)",
    )

    parser.add_argument(
        "--max-budget",
        type=int,
        default=500,
        help="Budget total maximum (labels). "
             "Si None, utilise n_rounds × batch_size (défaut: None)",
    )

    parser.add_argument(
        "--n-rounds",
        type=int,
        default=3,
        help="Nombre de rounds AL si max-budget non spécifié (défaut: 6)",
    )

    parser.add_argument(
        "--histories-dir",
        type=str,
        default='True',
        help="Dossier de sauvegarde des histories JSON "
             "(compatible analyze_histories.py). "
             "Si None, pas de sauvegarde JSON (défaut: None). "
             "Exemple : --histories-dir ./histories/synth",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.out = f"{args.out}_{args.init_mode}"

    _ensure_strategies_registered()

    # Calcul du budget effectif pour affichage
    _n_rounds_eff = args.n_rounds
    if args.max_budget is not None:
        _n_rounds_eff = max(1, (args.max_budget - args.n_init) // args.batch_size)

    print("Active learning ablation benchmark:")
    print(f"  seeds       = {args.seeds}")
    print(f"  models      = {args.models}")
    print(f"  scenarios   = {args.scenarios}")
    print(f"  panel       = {args.panel}")
    print(f"  init_mode   = {args.init_mode}")
    print(f"  n_init      = {args.n_init}")
    print(f"  batch_size  = {args.batch_size}")
    if args.max_budget is not None:
        print(f"  max_budget  = {args.max_budget}  →  n_rounds effectif = {_n_rounds_eff}")
    else:
        print(f"  n_rounds    = {args.n_rounds}  →  budget max = {args.n_init + args.n_rounds * args.batch_size}")
    print()

    t0 = time.time()

    results = run_benchmark(
        seeds=args.seeds,
        models=args.models,
        scenarios=args.scenarios,
        panel=args.panel,
        init_mode=args.init_mode,
        n_init=args.n_init,
        batch_size=args.batch_size,
        n_rounds=args.n_rounds,
        max_budget=args.max_budget,
        histories_dir=args.histories_dir,
    )

    if args.histories_dir:
        print(f"\nHistories JSON sauvegardées dans : {args.histories_dir}")

    print(f"\nTotal benchmark time: {time.time() - t0:.1f}s")

    os.makedirs(args.out, exist_ok=True)

    write_csv(
        results,
        os.path.join(args.out, "per_cell_results.csv"),
    )

    write_per_seed_csv(
        results,
        os.path.join(args.out, "per_seed_results.csv"),
    )

    for model_kind in args.models:
        emit_latex_table(
            results,
            panel=args.panel,
            scenarios=args.scenarios,
            model_kind=model_kind,
            path=os.path.join(
                args.out,
                f"ablation_table_{model_kind}.tex",
            ),
        )

    emit_summary(
        results,
        panel=args.panel,
        models=args.models,
        scenarios=args.scenarios,
        path=os.path.join(args.out, "summary.md"),
        seeds=args.seeds,
    )

    print(f"\nOutputs written to: {args.out}/")

    for fname in sorted(os.listdir(args.out)):
        print(f"  {fname}")


if __name__ == "__main__":
    main()