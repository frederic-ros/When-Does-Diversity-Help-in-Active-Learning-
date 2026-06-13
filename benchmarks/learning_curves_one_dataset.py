# -*- coding: utf-8 -*-
"""
learning_curves_one_dataset.py  —  shim de compatibilité
=========================================================
Ce fichier réexporte depuis alframework_clean les symboles attendus
par bench_real.py, qui historiquement importait un module du même nom
présent dans l'ancienne arborescence (tests/).

Symboles exposés
----------------
DEFAULT_TRACKED_METRICS  : tuple des métriques tracées par défaut
ArrayLabeler             : étiqueteur à partir d'un tableau numpy
active_learning_loop     : boucle AL principale
CurvePoint               : point de courbe (n_selected + dict métriques)
LearningCurveResult      : résultat agrégé d'une courbe
evaluate                 : évaluation modèle → dict métriques
_compute_auc             : calcul AUC + AUC normalisée d'une courbe
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# ── Réexports directs depuis alframework_clean ───────────────────────────────
from alframework.core.labeler    import ArrayLabeler          # noqa: F401
from alframework.core.runner     import active_learning_loop  # noqa: F401
from alframework.core.metrics    import evaluate               # noqa: F401
from alframework.utils.curve_utils import CurvePoint          # noqa: F401

# ── Métriques tracées par défaut ─────────────────────────────────────────────
DEFAULT_TRACKED_METRICS: Tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
)

# ── LearningCurveResult ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class LearningCurveResult:
    """Résultat d'une courbe d'apprentissage pour une stratégie."""
    curve            : List[CurvePoint]
    auc              : Dict[str, float]     = field(default_factory=dict)
    auc_norm         : Dict[str, float]     = field(default_factory=dict)
    selected_indices : Any                  = field(default=None)   # np.ndarray optionnel
    params           : Dict[str, Any]       = field(default_factory=dict)
    strategy_name    : str                  = ""
    meta             : Dict[str, Any]       = field(default_factory=dict)

# ── _compute_auc ──────────────────────────────────────────────────────────────
def _compute_auc(
    curve: Sequence[CurvePoint],
    tracked_metrics: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calcule l'AUC (intégrale trapézoïdale) et l'AUC normalisée pour chaque
    métrique suivie, à partir d'une liste de CurvePoint.

    Returns
    -------
    auc      : dict  métrique → AUC brute (aire sous la courbe)
    auc_norm : dict  métrique → AUC normalisée ∈ [0, 1]
               (divisée par l'AUC d'un classifieur parfait = n_selected_max)
    """
    if not curve:
        empty = {m: 0.0 for m in tracked_metrics}
        return empty, empty

    xs = np.array([p.n_selected for p in curve], dtype=float)
    x_min, x_max = float(xs[0]), float(xs[-1])
    x_range = x_max - x_min if x_max > x_min else 1.0

    auc: Dict[str, float] = {}
    auc_norm: Dict[str, float] = {}

    for metric in tracked_metrics:
        ys = np.array(
            [float(p.metrics.get(metric, np.nan)) for p in curve],
            dtype=float,
        )
        # Ignorer les NaN
        valid = ~np.isnan(ys)
        if valid.sum() < 2:
            auc[metric] = float(np.nanmean(ys)) if valid.any() else 0.0
            auc_norm[metric] = auc[metric]
            continue
        auc_val = float(np.trapz(ys[valid], xs[valid]) / x_range)
        auc[metric] = auc_val
        auc_norm[metric] = auc_val  # déjà normalisé par x_range ∈ [0,1]

    return auc, auc_norm
