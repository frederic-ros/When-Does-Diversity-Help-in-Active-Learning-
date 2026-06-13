# -*- coding: utf-8 -*-
"""
Config + factory pour instancier les stratégies AL via le registry.

Objectif:
- Les clés de STRATEGY_SPECS correspondent EXACTEMENT aux @register("...").
- Les init_kwargs contiennent les paramètres spécifiques des __init__ (contrôlables).
- make_strategy(...) instancie via STRATEGIES[name](**kwargs) avec validation.

Important:
- STRATEGIES est rempli seulement si tu as importé tes modules de stratégies AVANT
  d'appeler make_strategy / make_all_enabled_strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional
import inspect

from alframework.core.registry import STRATEGIES


@dataclass(frozen=True)
class StrategySpec:
    """
    enabled: active/désactive la stratégie dans tes runs/benchmarks
    init_kwargs: paramètres passés au constructeur __init__(**init_kwargs)
    """
    enabled: bool = False
    init_kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# CONFIG : 1 entrée PAR STRATEGIE (nom EXACT @register("..."))
# ---------------------------------------------------------------------
STRATEGY_SPECS: Dict[str, StrategySpec] = {
    # -----------------------------
    # Random
    # -----------------------------
    "random": StrategySpec(
        enabled=True,
        init_kwargs={},
    ),

    # -----------------------------
    # Uncertainty (3 stratégies distinctes)
    # -----------------------------
    "least_confident": StrategySpec(
        enabled=False,
        init_kwargs={},
    ),
    "entropy": StrategySpec(
        enabled=False,
        init_kwargs={},
    ),
    "margin": StrategySpec(
        enabled=False,
        init_kwargs={},
    ),

    # -----------------------------
    # Coreset (2 stratégies distinctes)
    # -----------------------------
    "coreset_greedy": StrategySpec(
        enabled=False,
        init_kwargs={},
    ),
    "coreset_kmeanspp": StrategySpec(
        enabled=False,
        init_kwargs={},
    ),

    # -----------------------------
    # Typiclust
    # -----------------------------
    "typiclust": StrategySpec(
        enabled=False,
        init_kwargs={
            "neighbors": 5,
            "random_state": 0,
        },
    ),

    # -----------------------------
    # ProbCover
    # -----------------------------
    "probcover": StrategySpec(
        enabled=False,
        init_kwargs={},
    ),

    # -----------------------------
    # DBAL
    # -----------------------------
    "dbal": StrategySpec(
        enabled=False,
        init_kwargs={
            "method": "margin",
            "dbal_factor": 5,
            "random_state": 0,
        },
    ),

    # -----------------------------
    # BADGE (approx)
    # -----------------------------
    "badge_approx": StrategySpec(
        enabled=False,
        init_kwargs={
            "random_state": 0,
        },
    ),

    # -----------------------------
    # Rank2022
    # -----------------------------
    "rank2022": StrategySpec(
        enabled=False,
        init_kwargs={
            "s_factor": 10,
            "linkage": "ward",
            "loss": "margin",
        },
    ),

    # -----------------------------
    # Active Pseudolabel
    # -----------------------------
    "active_pseudolabel": StrategySpec(
        enabled=False,
        init_kwargs={
            "k_neighbors": 10,
            "lambda_prop": 0.2,
            "alpha_decay": 2.0,
            "weighted_kmeans":True,
            "weight_power" : 1.0,     # optionnel : accentue ou adoucit l'effet
            "weight_eps": 1e-8,      # évite poids nuls
        },
    ),

    # -----------------------------
    # Active Pseudolabel V2
    # -----------------------------
    

    "ActivePseudoLabelV2": StrategySpec(
        enabled=False,
        init_kwargs={
            "k_neighbors":  10,
            "lambda_prop":  0.2,
            "random_state": 42,
            },
        ),
    
    "ActivePseudoLabelV3": StrategySpec(
    enabled=False,
    init_kwargs={
        # Base uncertainty
        "mix_entropy": True,
        "entropy_weight": 0.25,

        # Propagation
        "k_neighbors": 10,
        "lambda_prop": 0.30,
        "alpha_decay": 2.0,
        "conf_error_weight": True,

        # Diversification
        "weighted_kmeans": True,
        "weight_power": 1.0,
        "random_state": 0,

        # Normalization
        "normalize_once_at_end": True,

        # Numerical stability
        "eps": 1e-8,
    },
),
    
    "ActivePseudoLabelV4": StrategySpec(
    enabled=False,
    init_kwargs={
        "mix_entropy": True,
        "entropy_weight": 0.20,

        # Geometry / propagation
        "k_neighbors": 10,
        "lambda_prop": 0.40,
        "alpha_decay": 2.0,
        "min_source_score": 0.15,
        "adaptive_lambda": True,
        "credible_errors_tau": 8.0,

        # Diversification
        "weighted_kmeans": True,
        "weight_power": 1.0,
        "pool_multiplier": 1.5,
        "random_state": 0,

        # Stability
        "normalize_once_at_end": True,
        "eps": 1e-8,
    },
),
    

    "ActivePseudoLabelV44": StrategySpec(
        enabled=False,
        init_kwargs={
            # V4 base parameters (inherited)
            "mix_entropy": True,
            "entropy_weight": 0.20,
            "k_neighbors": 10,
            "lambda_prop": 0.40,
            "alpha_decay": 2.0,
            "min_source_score": 0.15,
            "adaptive_lambda": True,
            "credible_errors_tau": 8.0,
            "weighted_kmeans": True,
            "weight_power": 1.0,
            "pool_multiplier": 1.5,
            "random_state": 0,
            "normalize_once_at_end": True,
            "eps": 1e-8,

            # V4.4 specific parameters
            "adaptive_representative": True,
            "lam_threshold": 0.15,
        },
    ),

    "ActivePseudoLabelV45": StrategySpec(
        enabled=False,
        init_kwargs={
            # V4 base parameters (inherited via V4.4)
            "mix_entropy": True,
            "entropy_weight": 0.20,
            "k_neighbors": 10,
            "lambda_prop": 0.40,
            "alpha_decay": 2.0,
            "min_source_score": 0.15,
            "adaptive_lambda": True,
            "credible_errors_tau": 8.0,
            "weighted_kmeans": True,
            "weight_power": 1.0,
            "pool_multiplier": 1.5,
            "random_state": 0,
            "normalize_once_at_end": True,
            "eps": 1e-8,

            # V4.4 specific (inherited)
            "adaptive_representative": True,
            "lam_threshold": 0.15,

            # V4.5 specific parameters
            "use_cluster_quality_gate": True,
            "quality_low": 0.40,
            "quality_high": 0.50,
        },
    ),

    # -----------------------------
    # Active Pseudolabel V5 (V4.7 + V4.4 intra-cluster + V4.8 controller)
    # -----------------------------
    "ActivePseudoLabelV5": StrategySpec(
        enabled=False,
        init_kwargs={
            # V4 base (inherited via V4.7)
            "mix_entropy": True,
            "entropy_weight": 0.20,
            "k_neighbors": 10,
            "lambda_prop": 0.40,
            "alpha_decay": 2.0,
            "min_source_score": 0.15,
            "adaptive_lambda": True,
            "credible_errors_tau": 8.0,
            "weighted_kmeans": True,
            "weight_power": 1.0,
            "pool_multiplier": 1.5,
            "normalize_once_at_end": True,
            "eps": 1e-8,

            # V4.7 base
            "candidate_multiplier": 5.0,
            "density_weight": 0.05,
            "lam_threshold": 0.15,
            "random_state": 0,

            # V5 diversity controller
            "diversity_base": 0.10,
            "diversity_max": 0.50,
            "lam_weakness_weight": 0.20,
            "flatness_weight": 0.18,
            "imbalance_weight": 0.10,
            "contrast_guard_weight": 0.18,
            "highdim_guard_weight": 0.15,
            "adaptive_diversity": True,

            # Intra-cluster selection
            "adaptive_representative": True,
        },
    ),

    # -----------------------------
    # V5.1 — Hard regime-based switching (Piste 1)
    # Inherits V5; replaces continuous smoothstep by 3-regime hard switch.
    # New parameter: cold_threshold (below which selection is pure V4.4).
    # -----------------------------
    "ActivePseudoLabelV51": StrategySpec(
        enabled=False,
        init_kwargs={
            # V4 base
            "mix_entropy": True,
            "entropy_weight": 0.20,
            "k_neighbors": 10,
            "lambda_prop": 0.40,
            "alpha_decay": 2.0,
            "min_source_score": 0.15,
            "adaptive_lambda": True,
            "credible_errors_tau": 8.0,
            "weighted_kmeans": True,
            "weight_power": 1.0,
            "pool_multiplier": 1.5,
            "normalize_once_at_end": True,
            "eps": 1e-8,

            # V4.7 base
            "candidate_multiplier": 5.0,
            "density_weight": 0.05,
            "lam_threshold": 0.15,
            "random_state": 0,

            # V5 controller (preserved)
            "diversity_base": 0.10,
            "diversity_max": 0.50,
            "lam_weakness_weight": 0.20,
            "flatness_weight": 0.18,
            "imbalance_weight": 0.10,
            "contrast_guard_weight": 0.18,
            "highdim_guard_weight": 0.15,
            "adaptive_diversity": True,
            "adaptive_representative": True,

            # V5.1 specific
            "cold_threshold": 0.05,
            "transition_smoothing": False,
        },
    ),

    # -----------------------------
    # V5.2 — Signal decoupling (Piste 2)
    # Inherits V5; preserves V4.4 smoothstep but removes V4.7 outer
    # diversification and V4.8 controller from intra-cluster selection.
    # No new hyperparameter (structural simplification only).
    # -----------------------------
    "ActivePseudoLabelV52": StrategySpec(
        enabled=False,
        init_kwargs={
            # V4 base
            "mix_entropy": True,
            "entropy_weight": 0.20,
            "k_neighbors": 10,
            "lambda_prop": 0.40,
            "alpha_decay": 2.0,
            "min_source_score": 0.15,
            "adaptive_lambda": True,
            "credible_errors_tau": 8.0,
            "weighted_kmeans": True,
            "weight_power": 1.0,
            "pool_multiplier": 1.5,
            "normalize_once_at_end": True,
            "eps": 1e-8,

            # V4.7 base
            "candidate_multiplier": 5.0,
            "density_weight": 0.05,
            "lam_threshold": 0.15,
            "random_state": 0,

            # V5 controller params kept for parameter parity with V5;
            # NOT actively used by V5.2's overridden batch selection.
            "diversity_base": 0.10,
            "diversity_max": 0.50,
            "lam_weakness_weight": 0.20,
            "flatness_weight": 0.18,
            "imbalance_weight": 0.10,
            "contrast_guard_weight": 0.18,
            "highdim_guard_weight": 0.15,
            "adaptive_diversity": True,
            "adaptive_representative": True,
        },
    ),


    # -----------------------------
    # QBC / Tri-committee / Selftrain acquisition
    # -----------------------------
    "qbc": StrategySpec(
        enabled=False,
        init_kwargs={
            "n_committee": 5,
            "bootstrap_ratio": 1.0,
            "metric": "vote_entropy",
        },
    ),
    "tri_committee": StrategySpec(
        enabled=False,
        init_kwargs={
            "bootstrap_ratio": 1.0,
            "mode": "two_vs_one",
        },
    ),
    "selftrain_acq": StrategySpec(
        enabled=False,
        init_kwargs={
            "tau": 0.9,
            "window": 0.05,
            "mode": "near_threshold",
        },
    ),

    # -----------------------------
    # Robust QBC / Adaptive disagreement
    # -----------------------------
    "robust_qbc": StrategySpec(
        enabled=True,
        init_kwargs={
            "n_committee": 5,
            "bootstrap_ratio": 0.8,
            "metric": "confidence_weighted",
        },
    ),
    "adaptive_disagreement": StrategySpec(
        enabled=False,
        init_kwargs={
            "n_committee": 5,
            "min_disagreement": 0.2,
        },
    ),

    # -----------------------------
    # Diversity optimized batch / BAIT simple
    # -----------------------------
    "diversity_optimized_batch": StrategySpec(
        enabled=False,
        init_kwargs={
            "uncertainty_metric": "margin",
            "lambda_diversity": 0.5,
        },
    ),
    "bait_simple": StrategySpec(
        enabled=False,
        init_kwargs={
            "uncertainty_metric": "margin",
            "n_clusters_factor": 3,
            "uncertainty_weight": 1.0,
        },
    ),
}


def enabled_strategy_names() -> list[str]:
    """Liste des stratégies activées dans la config."""
    return [name for name, spec in STRATEGY_SPECS.items() if spec.enabled]


def validate_config_against_registry(*, strict: bool = True) -> None:
    """
    Vérifie que la config est alignée avec le registry.

    strict=True : erreur si une clé de STRATEGY_SPECS n'existe pas dans STRATEGIES.
    strict=False : imprime seulement un warning.
    """
    cfg = set(STRATEGY_SPECS.keys())
    reg = set(STRATEGIES.keys())

    extra = sorted(cfg - reg)
    missing = sorted(reg - cfg)

    if extra:
        msg = f"Noms dans STRATEGY_SPECS mais absents du registry STRATEGIES: {extra}"
        if strict:
            raise AssertionError(msg)
        print("⚠️", msg)

    if missing:
        # pas forcément fatal, mais utile
        print(f"⚠️ Stratégies présentes dans STRATEGIES mais pas dans STRATEGY_SPECS: {missing}")


def _validate_init_kwargs(cls: type, kwargs: Mapping[str, Any]) -> None:
    """Validation stricte : empêche de passer des kwargs inconnus au __init__."""
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    # Si le constructeur accepte **kwargs, pas besoin de valider strictement.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return

    allowed = {name for name in params.keys() if name != "self"}
    unknown = set(kwargs.keys()) - allowed
    if unknown:
        raise TypeError(
            f"Paramètres inconnus pour {cls.__name__}: {sorted(unknown)}. "
            f"Paramètres acceptés: {sorted(allowed)}"
        )


def make_strategy(
    name: str,
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    validate: bool = True,
):
    """
    Instancie une stratégie enregistrée dans STRATEGIES, en prenant les params
    du config + overrides optionnels.
    """
    if name not in STRATEGIES:
        available = sorted(STRATEGIES.keys())
        raise KeyError(f"Stratégie '{name}' inconnue dans STRATEGIES. Disponibles: {available}")

    cls = STRATEGIES[name]

    spec = STRATEGY_SPECS.get(name)
    if spec is None:
        raise KeyError(
            f"Stratégie '{name}' absente de STRATEGY_SPECS. "
            f"Ajoute-la dans config_strategies.py."
        )

    kwargs: Dict[str, Any] = dict(spec.init_kwargs)
    if overrides:
        kwargs.update(dict(overrides))  # overrides > config

    if validate:
        _validate_init_kwargs(cls, kwargs)

    return cls(**kwargs)


def make_all_enabled_strategies(*, validate: bool = True) -> Dict[str, Any]:
    """Instancie toutes les stratégies enabled=True."""
    out: Dict[str, Any] = {}
    for name in enabled_strategy_names():
        out[name] = make_strategy(name, validate=validate)
    return out