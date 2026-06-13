# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 06:14:54 2026

@author: frederic.ros
"""

# alframework/utils/curve_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def make_budget_grid(
    max_budget: int,
    *,
    step: int = 10,
    include_zero: bool = True,
    extra_points: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Construit une grille de budgets : 0, step, 2*step, ... max_budget (+ extra_points).
    """
    max_budget = int(max_budget)
    if max_budget < 0:
        raise ValueError("max_budget must be >= 0")

    pts = set()
    if include_zero:
        pts.add(0)

    if step <= 0:
        raise ValueError("step must be >= 1")

    for b in range(step, max_budget + 1, step):
        pts.add(b)

    pts.add(max_budget)

    if extra_points:
        for b in extra_points:
            b = int(b)
            if 0 <= b <= max_budget:
                pts.add(b)

    return sorted(pts)


def auc_trapz(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Aire sous la courbe (trapèzes). x doit être croissant.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return float(trapz(y, x))


def normalize_auc(auc: float, x0: float, x1: float) -> float:
    """
    Normalise une AUC par la longueur de l'intervalle (retourne une "moyenne").
    """
    denom = float(x1 - x0)
    if denom <= 0:
        return float("nan")
    return float(auc) / denom


@dataclass(frozen=True)
class CurvePoint:
    n_selected: int
    metrics: Dict[str, float]


def curve_to_arrays(curve: Sequence[CurvePoint], metric: str) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([p.n_selected for p in curve], dtype=float)
    y = np.array([float(p.metrics.get(metric, np.nan)) for p in curve], dtype=float)
    return x, y


def curves_to_long_rows(
    curves_by_strategy: Dict[str, Sequence[CurvePoint]],
    *,
    scenario: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convertit des courbes en lignes 'long format' (pratique CSV / pandas).
    Une ligne = (strategy, n_selected, metric_name, metric_value).
    """
    rows: List[Dict[str, Any]] = []
    for strat, curve in curves_by_strategy.items():
        for p in curve:
            for k, v in p.metrics.items():
                rows.append(
                    {
                        "scenario": scenario,
                        "strategy": strat,
                        "n_selected": int(p.n_selected),
                        "metric": str(k),
                        "value": float(v),
                    }
                )
    return rows