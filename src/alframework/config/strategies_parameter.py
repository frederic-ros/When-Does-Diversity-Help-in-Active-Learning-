# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 09:34:10 2026

@author: frederic.ros
"""

# src/alframework/config/strategies_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import inspect
from alframework.core.registry import STRATEGIES


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
    
    "probcover": StrategySpec(
        enabled=True,
        init_kwargs={
            "delta": None,           # ou une valeur float fixe si tu veux
            "delta_quantile": 0.10,  # 10% => plutôt “local”
            "knn_k": 10,
            "metric": "euclidean",
            "random_state": 0,
            },
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
            "k_neighbors": 5,
            "lambda_prop": 0.2,
            "alpha_decay": 2.0,
            "weighted_kmeans": True
        },  # :contentReference[oaicite:16]{index=16}
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




def validate_strategy_config(verbose: bool = True, strict: bool = True) -> None:
    """
    Vérifie que STRATEGY_SPECS est cohérent avec STRATEGIES.

    - strict=True : lève une erreur si incohérence
    - strict=False : affiche seulement des warnings
    """

    cfg_names = set(STRATEGY_SPECS.keys())
    reg_names = set(STRATEGIES.keys())

    extra_in_cfg = sorted(cfg_names - reg_names)
    missing_in_cfg = sorted(reg_names - cfg_names)

    # 1️⃣ Vérification noms
    if extra_in_cfg:
        msg = f"Noms dans STRATEGY_SPECS mais absents du registry: {extra_in_cfg}"
        if strict:
            raise AssertionError(msg)
        print("⚠️", msg)

    if missing_in_cfg and verbose:
        print(f"⚠️ Stratégies présentes dans STRATEGIES mais absentes de STRATEGY_SPECS: {missing_in_cfg}")

    # 2️⃣ Vérification des paramètres __init__
    for name, spec in STRATEGY_SPECS.items():
        if name not in STRATEGIES:
            continue

        cls = STRATEGIES[name]
        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        # ignore si **kwargs accepté
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue

        allowed = {p for p in params.keys() if p != "self"}
        unknown = set(spec.init_kwargs.keys()) - allowed

        if unknown:
            msg = f"Paramètres inconnus pour {name}: {sorted(unknown)} (acceptés: {sorted(allowed)})"
            if strict:
                raise AssertionError(msg)
            print("⚠️", msg)

    if verbose:
        print("✅ Validation STRATEGY_SPECS OK.")
        print(f"   - {len(cfg_names)} stratégies dans la config")
        print(f"   - {len(reg_names)} stratégies dans le registry")
        
        
from typing import Optional, Mapping

def get_strategy_spec(name: str) -> StrategySpec:
    """
    Retourne le StrategySpec (enabled + init_kwargs) d'une stratégie.
    Lève KeyError si la stratégie n'existe pas dans STRATEGY_SPECS.
    """
    if name not in STRATEGY_SPECS:
        available = sorted(STRATEGY_SPECS.keys())
        raise KeyError(f"'{name}' absent de STRATEGY_SPECS. Disponibles: {available}")
    return STRATEGY_SPECS[name]


def get_strategy_kwargs(name: str) -> Dict[str, Any]:
    """
    Retourne une COPIE des kwargs de config pour une stratégie.
    """
    spec = get_strategy_spec(name)
    return dict(spec.init_kwargs)


def make_strategy_from_config(
    name: str,
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    validate: bool = True,
):
    """
    Instancie une stratégie en injectant automatiquement les paramètres
    définis dans STRATEGY_SPECS[name].init_kwargs.

    overrides (optionnel) permet d’écraser certains paramètres à la volée.
    validate=True : vérifie que les kwargs existent dans le __init__.
    """
    if name not in STRATEGIES:
        available = sorted(STRATEGIES.keys())
        raise KeyError(f"Stratégie '{name}' inconnue dans le registry. Disponibles: {available}")

    cls = STRATEGIES[name]

    kwargs = get_strategy_kwargs(name)
    if overrides:
        kwargs.update(dict(overrides))  # overrides > config

    if validate:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        # Si **kwargs accepté, pas besoin de valider strictement
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            allowed = {p for p in params.keys() if p != "self"}
            unknown = set(kwargs.keys()) - allowed
            if unknown:
                raise TypeError(
                    f"Paramètres inconnus pour {name}: {sorted(unknown)}. "
                    f"Acceptés: {sorted(allowed)}"
                )

    return cls(**kwargs)


def enabled_strategy_names() -> list:
    """
    Liste des stratégies activées (enabled=True) dans STRATEGY_SPECS.
    """
    return [name for name, spec in STRATEGY_SPECS.items() if spec.enabled]


def make_enabled_strategies_from_config(*, validate: bool = True) -> Dict[str, Any]:
    """
    Instancie toutes les stratégies enabled=True avec leurs paramètres de config.
    Retourne un dict {name: instance}.
    """
    out: Dict[str, Any] = {}
    for name in enabled_strategy_names():
        out[name] = make_strategy_from_config(name, validate=validate)
    return out