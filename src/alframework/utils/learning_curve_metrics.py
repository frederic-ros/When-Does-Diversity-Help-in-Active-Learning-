# alframework/utils/learning_curve_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Dict, Any, Tuple, List
import numpy as np


@dataclass(frozen=True)
class CurveAUC:
    """Résumé AUC pour une courbe metric(label_count)."""
    auc: float                 # aire brute (trapz)
    auc_norm: float            # auc / (x_max - x_min)  (moyenne sur l'intervalle)
    early_auc: Optional[float] = None
    early_auc_norm: Optional[float] = None
    x_min: float = 0.0
    x_max: float = 0.0
    early_x_max: Optional[float] = None


def _as_sorted_xy(x: Sequence[float], y: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x et y doivent avoir la même taille: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] < 2:
        raise ValueError("Il faut au moins 2 points pour intégrer (AUC).")

    # Tri par x croissant
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Vérif monotone (strict ou non)
    if np.any(np.diff(x) < 0):
        raise ValueError("x doit être croissant après tri (erreur interne).")

    return x, y


def _integrate_trapz(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def _clip_curve_to_xmax(x: np.ndarray, y: np.ndarray, xmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tronque la courbe sur [x[0], xmax] en ajoutant un point interpolé à xmax si nécessaire.
    Pré-requis: x trié croissant.
    """
    if xmax <= x[0]:
        # intervalle vide ou quasi vide => impossible d'intégrer correctement
        return np.array([x[0], xmax], dtype=float), np.array([y[0], y[0]], dtype=float)

    if xmax >= x[-1]:
        return x, y

    # indices des points <= xmax
    mask = x <= xmax
    x_keep = x[mask]
    y_keep = y[mask]

    # si xmax tombe exactement sur un x existant
    if x_keep.size > 0 and x_keep[-1] == xmax:
        return x_keep, y_keep

    # sinon interpolation linéaire entre le dernier point <= xmax et le premier > xmax
    i_right = np.searchsorted(x, xmax, side="right")
    i_left = i_right - 1

    x0, y0 = x[i_left], y[i_left]
    x1, y1 = x[i_right], y[i_right]

    # interpolation
    t = (xmax - x0) / (x1 - x0)
    yx = y0 + t * (y1 - y0)

    x_new = np.concatenate([x_keep, [xmax]])
    y_new = np.concatenate([y_keep, [yx]])
    return x_new, y_new


def compute_auc_metrics(
    x: Sequence[float],
    y: Sequence[float],
    *,
    early_x_max: Optional[float] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> CurveAUC:
    """
    Calcule:
      - AUC brute (trapèzes)
      - AUC normalisée = AUC / (x_max - x_min)
      - Early AUC (sur [x_min, early_x_max]) + normalisée

    Notes:
      - Normaliser par la largeur en x revient à une "moyenne" de la métrique.
      - early_x_max peut être un int (ex 100 labels) ou None.
      - si tu veux commencer à n_init plutôt qu'à 0, passe x_min=n_init.
    """
    xs, ys = _as_sorted_xy(x, y)

    # fenêtre principale
    x0 = float(xs[0]) if x_min is None else float(x_min)
    x1 = float(xs[-1]) if x_max is None else float(x_max)

    # si l'utilisateur force x_min/x_max, on tronque/interpole aux bornes
    # 1) tronquer à x1
    xs1, ys1 = _clip_curve_to_xmax(xs, ys, x1)
    # 2) gérer x_min: on "décale" en tronquant sur [x_min, ...]
    if x0 > xs1[0]:
        # On découpe via masque + interpolation à x0
        # astuce: on renverse le problème en tronquant sur xmax puis en coupant le début
        # -> ici on construit un point à x0 si nécessaire
        if x0 >= xs1[-1]:
            # plus rien
            xs_main = np.array([x0, x1], dtype=float)
            ys_main = np.array([ys1[-1], ys1[-1]], dtype=float)
        else:
            # construire point à x0 par interpolation
            i_right = np.searchsorted(xs1, x0, side="right")
            i_left = i_right - 1
            xL, yL = xs1[i_left], ys1[i_left]
            xR, yR = xs1[i_right], ys1[i_right]
            t = (x0 - xL) / (xR - xL)
            y0_interp = yL + t * (yR - yL)

            xs_tail = xs1[i_right:]
            ys_tail = ys1[i_right:]
            xs_main = np.concatenate([[x0], xs_tail])
            ys_main = np.concatenate([[y0_interp], ys_tail])
    else:
        xs_main, ys_main = xs1, ys1

    width = float(xs_main[-1] - xs_main[0])
    if width <= 0:
        raise ValueError(f"Intervalle x invalide: x_min={xs_main[0]} x_max={xs_main[-1]}")

    auc = _integrate_trapz(xs_main, ys_main)
    auc_norm = auc / width

    # early
    e_auc = None
    e_auc_norm = None
    e_xmax = None
    if early_x_max is not None:
        e_xmax = float(early_x_max)
        # early_x_max ne peut pas dépasser la borne max utilisée
        e_xmax = min(e_xmax, float(xs_main[-1]))

        xs_e, ys_e = _clip_curve_to_xmax(xs_main, ys_main, e_xmax)
        e_width = float(xs_e[-1] - xs_e[0])
        if e_width > 0:
            e_auc = _integrate_trapz(xs_e, ys_e)
            e_auc_norm = e_auc / e_width
        else:
            e_auc = 0.0
            e_auc_norm = float(ys_e[0])

    return CurveAUC(
        auc=auc,
        auc_norm=auc_norm,
        early_auc=e_auc,
        early_auc_norm=e_auc_norm,
        x_min=float(xs_main[0]),
        x_max=float(xs_main[-1]),
        early_x_max=e_xmax,
    )
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 07:26:17 2026

@author: frederic.ros
"""

