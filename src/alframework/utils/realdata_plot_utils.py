# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 10:21:00 2026

@author: frederic.ros
"""

# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping, Tuple

import numpy as np


def plot_curves_and_save(
    out: Dict[str, Any],
    *,
    metric: str = "accuracy",
    show: Optional[List[str]] = None,
    title: Optional[str] = None,
    xlabel: str = "Nombre de labels sélectionnés",
    ylabel: Optional[str] = None,
    baseline: Optional[Tuple[List[int], List[float], str]] = None,
    save_path: Optional[Path | str] = None,
    dpi: int = 160,
    do_show: bool = False,
) -> Optional[str]:
    """
    Plot des learning curves (comme ta fonction plot_curves), avec sauvegarde.

    Parameters
    ----------
    out:
        Sortie de compute_learning_curves_*: out["strategies"][name].curve doit exister.
    metric:
        "accuracy", "f1_macro", etc.
    show:
        Liste des stratégies à afficher. None => toutes.
    baseline:
        Optionnel: (x_baseline, y_baseline, label) pour ajouter une courbe baseline.
        Exemple: baseline=([0, max_budget], [acc0, acc0], "supervised@init")
    save_path:
        Chemin .png à écrire. Si None, pas de sauvegarde.
    do_show:
        True => plt.show() (sinon on ferme la figure après save).
    """
    import matplotlib.pyplot as plt

    strat_out = out["strategies"]
    names = list(strat_out.keys()) if show is None else show

    plt.figure(figsize=(10, 6))

    for name in names:
        curve = strat_out[name].curve
        x = [int(p.n_selected) for p in curve]
        y = [float(p.metrics.get(metric, np.nan)) for p in curve]
        plt.plot(x, y, label=name)

    if baseline is not None:
        xb, yb, lbl = baseline
        plt.plot(xb, yb, linestyle="--", label=lbl)

    mode = out.get("setup", {}).get("mode", "?")
    plt.xlabel(xlabel)
    plt.ylabel(metric if ylabel is None else ylabel)
    plt.title(title if title is not None else f"Learning curves — {metric} [{mode}]")
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    saved = None
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
        saved = str(save_path)

    if do_show:
        plt.show()
    else:
        plt.close()

    return saved


def build_constant_baseline_from_out(
    out: Dict[str, Any],
    *,
    metric: str = "accuracy",
    label: str = "supervised@init",
) -> Tuple[List[int], List[float], str]:
    """
    Baseline constante: prend le 1er point (n_selected=0) d'une stratégie de référence
    ou, si tu veux, tu peux le calculer ailleurs.
    Ici on prend la première stratégie du dict (arbitraire mais pratique).

    Retourne (x, y, label) utilisable dans plot_curves_and_save(baseline=...).
    """
    strat_out = out["strategies"]
    if not strat_out:
        return ([0], [np.nan], label)

    first_name = next(iter(strat_out.keys()))
    curve = strat_out[first_name].curve
    if not curve:
        return ([0], [np.nan], label)

    x0 = int(curve[0].n_selected)
    y0 = float(curve[0].metrics.get(metric, np.nan))

    # on étend sur l’axe X jusqu’au max de la courbe pour faire une droite
    xmax = int(max(int(p.n_selected) for p in curve))
    return ([x0, xmax], [y0, y0], label)