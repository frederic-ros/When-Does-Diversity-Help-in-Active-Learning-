# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 09:34:10 2026

@author: frederic.ros
"""

# src/alframework/config/strategies_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class StrategySpec:
    """
    enabled: active/désactive la stratégie dans tes expés
    init_kwargs: paramètres passés à cls(**init_kwargs) au moment de l'instanciation
    """
    enabled: bool = False
    init_kwargs: Dict[str, Any] = field(default_factory=dict)


# IMPORTANT:
# Les clés doivent correspondre EXACTEMENT aux noms utilisés dans @register("...")
# Les init_kwargs ci-dessous reprennent les DEFAULTS vus dans tes classes.
# Tu modifies ensuite ces valeurs pour "contrôler" ton protocole expérimental.
STRATEGY_SPECS: Dict[str, StrategySpec] = {
    # -----------------------------
    # Uncertainty (3 stratégies)
    # -----------------------------
    "least_confident": StrategySpec(
        enabled=False,
        init_kwargs={},  # pas de __init__ spécifique :contentReference[oaicite:5]{index=5}
    ),
    "entropy": StrategySpec(
        enabled=False,
        init_kwargs={},  # pas de __init__ spécifique :contentReference[oaicite:6]{index=6}
    ),
    "margin": StrategySpec(
        enabled=False,
        init_kwargs={},  # pas de __init__ spécifique :contentReference[oaicite:7]{index=7}
    ),

    # -----------------------------
    # Random
    # -----------------------------
    "random": StrategySpec(
        enabled=False,
        init_kwargs={},  # pas de __init__ spécifique :contentReference[oaicite:8]{index=8}
    ),

    # -----------------------------
    # Coreset
    # -----------------------------
    "coreset_greedy": StrategySpec(
        enabled=False,
        init_kwargs={},  # pas de __init__ spécifique :contentReference[oaicite:9]{index=9}
    ),
    "coreset_kmeanspp": StrategySpec(
        enabled=False,
        init_kwargs={},  # pas de __init__ spécifique :contentReference[oaicite:10]{index=10}
    ),

    # -----------------------------
    # TypiClust
    # -----------------------------
    "typiclust": StrategySpec(
        enabled=False,
        init_kwargs={
            "neighbors": 5,
            "random_state": 0,
        },  # :contentReference[oaicite:11]{index=11}
    ),

    # -----------------------------
    # ProbCover (dépendance optionnelle)
    # -----------------------------
    "probcover": StrategySpec(
        enabled=False,
        init_kwargs={},  # pas de __init__ spécifique :contentReference[oaicite:12]{index=12}
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
        },  # :contentReference[oaicite:13]{index=13}
    ),

    # -----------------------------
    # BADGE (approx)
    # -----------------------------
    "badge_approx": StrategySpec(
        enabled=False,
        init_kwargs={
            "random_state": 0,
        },  # :contentReference[oaicite:14]{index=14}
    ),

    # -----------------------------
    # RANK2022
    # -----------------------------
    "rank2022": StrategySpec(
        enabled=False,
        init_kwargs={
            "s_factor": 10,
            "linkage": "ward",
            "loss": "margin",
        },  # :contentReference[oaicite:15]{index=15}
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
             "weighted_kmeans": True,
        },  # :contentReference[oaicite:16]{index=16}
    ),

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
    init_kwargs={},
),

    "ActivePseudoLabelV45": StrategySpec(
    enabled=False,
    init_kwargs={
        "quality_low": 0.40,
        "quality_high": 0.50,
    },
),

    "ActivePseudoLabelV5": StrategySpec(
    enabled=False,
    init_kwargs={},
),

    "ActivePseudoLabelV51": StrategySpec(
    enabled=False,
    init_kwargs={
        "cold_threshold": 0.05,
        "transition_smoothing": False,
    },
),

    "ActivePseudoLabelV52": StrategySpec(
    enabled=False,
    init_kwargs={},
),

    # -----------------------------
    # QBC / Tri-committee / Selftrain-aware
    # -----------------------------
    "qbc": StrategySpec(
        enabled=False,
        init_kwargs={
            "n_committee": 5,
            "bootstrap_ratio": 1.0,
            "metric": "vote_entropy",  # ou "kl"
        },  # :contentReference[oaicite:17]{index=17}
    ),
    "tri_committee": StrategySpec(
        enabled=False,
        init_kwargs={
            "bootstrap_ratio": 1.0,
            "mode": "two_vs_one",  # ou "vote_entropy"
        },  # :contentReference[oaicite:18]{index=18}
    ),
    "selftrain_acq": StrategySpec(
        enabled=False,
        init_kwargs={
            "tau": 0.9,
            "window": 0.05,
            "mode": "near_threshold",  # ou "low_confidence"
        },  # :contentReference[oaicite:19]{index=19}
    ),

    # -----------------------------
    # Robust QBC / Adaptive disagreement
    # -----------------------------
    "robust_qbc": StrategySpec(
        enabled=False,
        init_kwargs={
            "n_committee": 5,
            "bootstrap_ratio": 0.8,
            "metric": "confidence_weighted",  # ou "vote_entropy"
        },  # :contentReference[oaicite:20]{index=20}
    ),
    "adaptive_disagreement": StrategySpec(
        enabled=False,
        init_kwargs={
            "n_committee": 5,
            "min_disagreement": 0.2,
        },  # :contentReference[oaicite:21]{index=21}
    ),

    # -----------------------------
    # Batch diversification
    # -----------------------------
    "diversity_optimized_batch": StrategySpec(
        enabled=False,
        init_kwargs={
            "uncertainty_metric": "margin",  # ou "entropy"
            "lambda_diversity": 0.5,
        },  # :contentReference[oaicite:22]{index=22}
    ),
    "bait_simple": StrategySpec(
        enabled=False,
        init_kwargs={
            "uncertainty_metric": "margin",  # ou "entropy"
            "n_clusters_factor": 3,
            "uncertainty_weight": 1.0,
        },  # :contentReference[oaicite:23]{index=23}
    ),
}