# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 07:41:52 2026

@author: frederic.ros
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np

from alframework.utils.learning_curve_metrics import compute_auc_metrics, CurveAUC


def _get_attr(obj: Any, name: str, default=None):
    return getattr(obj, name, default)


def _is_seq(x: Any) -> bool:
    return isinstance(x, (list, tuple, np.ndarray))


def _point_get(p: Any, key: str, default=None):
    """Récupère p[key] si dict, sinon getattr(p,key)."""
    if isinstance(p, dict):
        return p.get(key, default)
    return getattr(p, key, default)


def _extract_from_points(points: Any, metric_key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    points: list[dict] ou list[obj]
      chaque point doit contenir un x (n_selected/n_labels/budget/...) + des métriques
    """
    if not _is_seq(points) or len(points) == 0:
        return None

    # ✅ IMPORTANT: inclure n_selected (ton CurvePoint)
    x_keys = (
        "n_selected",     # <-- clé de CurvePoint dans ton code
        "n_labels",
        "n_labeled",
        "budget",
        "k",
        "step",
        "x",
    )

    xs = []
    ys = []

    for p in points:
        # x
        x_val = None
        for k in x_keys:
            x_val = _point_get(p, k, None)
            if x_val is not None:
                break

        # y
        metrics = _point_get(p, "metrics", None)
        if isinstance(metrics, dict) and metric_key in metrics:
            y_val = metrics[metric_key]
        else:
            # métrique directement stockée au niveau du point (p[metric_key] ou p.metric_key)
            y_val = _point_get(p, metric_key, None)

        if x_val is None or y_val is None:
            continue

        xs.append(float(x_val))
        ys.append(float(y_val))

    if len(xs) < 2:
        # au moins 2 points pour AUC (trapz) + éviter edge cases
        return None

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    # tri par x (au cas où)
    order = np.argsort(x)
    return x[order], y[order]


def _find_curve_xy(strat_out: Any, *, metric_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrait (x,y) depuis un LearningCurveResult (dataclass/objet) ou dict.

    Formats supportés:
    - strat_out.curve / points / history : list de points {n_selected|... , metrics:{...}}
    - strat_out.metrics[metric_key] + strat_out.x (moins utilisé chez toi)
    - dict avec clés analogues
    """

    # ----------------------------
    # 1) Cas dict
    # ----------------------------
    if isinstance(strat_out, dict):
        for key in ("curve", "points", "history", "records"):
            if key in strat_out:
                got = _extract_from_points(strat_out[key], metric_key)
                if got is not None:
                    return got

        metrics = strat_out.get("metrics", None)
        if isinstance(metrics, dict) and metric_key in metrics:
            y = np.asarray(metrics[metric_key], dtype=float)
            for xk in ("x", "n_selected", "n_labels", "budgets", "n_labeled", "steps"):
                if xk in strat_out:
                    x = np.asarray(strat_out[xk], dtype=float)
                    if len(x) != len(y):
                        raise ValueError(f"len(x)={len(x)} != len(y)={len(y)}")
                    return x, y

        raise TypeError("Impossible d'extraire la courbe depuis un dict (structure inattendue).")

    # ----------------------------
    # 2) Cas objet/dataclass
    # ----------------------------
    d = getattr(strat_out, "__dict__", {})
    if not isinstance(d, dict):
        d = {}

    # 2.a) courbe sous forme de liste de points
    for key in ("curve", "points", "history", "records"):
        pts = getattr(strat_out, key, None)
        got = _extract_from_points(pts, metric_key)
        if got is not None:
            return got
        if key in d:
            got = _extract_from_points(d[key], metric_key)
            if got is not None:
                return got

    # 2.b) format "metrics dict + x"
    metrics = getattr(strat_out, "metrics", None)
    if isinstance(metrics, dict) and metric_key in metrics:
        y = np.asarray(metrics[metric_key], dtype=float)
        for x_attr in ("x", "n_selected", "n_labels", "budgets", "n_labeled", "steps"):
            x = getattr(strat_out, x_attr, None)
            if x is not None:
                x = np.asarray(x, dtype=float)
                if len(x) != len(y):
                    raise ValueError(f"Courbe incohérente: len(x)={len(x)} != len(y)={len(y)}")
                return x, y

    keys = list(d.keys())[:40]
    raise TypeError(
        f"Impossible d'extraire la courbe (x,y) pour metric_key='{metric_key}'. "
        f"type(strat_out)={type(strat_out)}. "
        f"Attributs visibles (max 40): {keys}. "
        "Attendu: strat_out.curve (list de points avec n_selected + metrics)."
    )


def add_auc_to_out(
    out: Dict[str, Any],
    *,
    metric_key: str = "accuracy",
    early_budget: Optional[int] = 100,
    x_min_from_setup: bool = True,
) -> Dict[str, Any]:
    """
    Ajoute AUC (brut/norm/early) pour chaque stratégie dans out["strategies"][name].

    Stockage:
      - si strat_out est dict: strat_out["auc"][metric_key] = {...}
      - sinon (objet/dataclass): strat_out.auc[metric_key] = {...}
    """
    if "strategies" not in out:
        raise KeyError("out ne contient pas 'strategies'.")

    setup = out.get("setup", {})
    n_init = int(setup.get("n_init", 0))

    for name, strat_out in out["strategies"].items():
        x, y = _find_curve_xy(strat_out, metric_key=metric_key)

        x_min = n_init if x_min_from_setup else None
        early_x_max = (n_init + int(early_budget)) if (early_budget is not None) else None

        auc_obj: CurveAUC = compute_auc_metrics(
            x, y,
            x_min=x_min,
            early_x_max=early_x_max,
        )

        payload = {
            "auc": auc_obj.auc,
            "auc_norm": auc_obj.auc_norm,
            "early_auc": auc_obj.early_auc,
            "early_auc_norm": auc_obj.early_auc_norm,
            "x_min": auc_obj.x_min,
            "x_max": auc_obj.x_max,
            "early_x_max": auc_obj.early_x_max,
        }

        # ⚠️ ton LearningCurveResult est frozen=True => on ne peut PAS setattr,
        # mais on peut muter le dict interne 'auc' (c'est un objet mutable).
        if isinstance(strat_out, dict):
            strat_out.setdefault("auc", {})
            strat_out["auc"][metric_key] = payload
        else:
            if not hasattr(strat_out, "auc") or getattr(strat_out, "auc") is None:
                raise TypeError(
                    f"{name}: strat_out n'a pas d'attribut 'auc' mutable. "
                    "Astuce: mets auc=dict(...) dans le dataclass (ce que tu as déjà)."
                )
            strat_out.auc[metric_key] = payload

    return out