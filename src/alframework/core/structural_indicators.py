"""
structural_indicators.py
========================
Indicateurs structurels a priori pour PRÉDIRE l'utilité de la diversité
en active learning (réponse à la question : "quand le clustering aide-t-il
vs la simple incertitude ?").

Deux indicateurs, calculés à un round PRÉCOCE (round 1-3) sur le pool non
labellisé, à partir du modèle courant et des features. Aucun ne regarde la
performance future -> pas de circularité.

  1. low_margin_fraction : proportion de points "frontière" (faible marge).
     Proxy du VOLUME de l'incertitude.
  2. uncertain_eff_dim   : dimension effective (participation ratio du spectre
     PCA) du SOUS-ENSEMBLE le plus incertain. Proxy de la MULTI-MODALITÉ /
     étalement de la zone de confusion -> c'est là que la diversité paie.

Cible à prédire (calculée ailleurs, depuis les courbes) :
     Delta = AULC(best_clustering) - AULC(margin)
"""
from __future__ import annotations
import numpy as np


def margin_uncertainty(proba: np.ndarray) -> np.ndarray:
    """1 - (p_top1 - p_top2). Grand = incertain."""
    p = np.sort(proba, axis=1)
    if p.shape[1] < 2:
        return np.zeros(p.shape[0])
    return 1.0 - (p[:, -1] - p[:, -2])


def effective_dim(X: np.ndarray, center: bool = True) -> float:
    """Participation ratio du spectre des valeurs propres de la covariance :
        eff_dim = (sum lambda_i)^2 / sum(lambda_i^2)
    Vaut ~1 si une seule direction domine, ~d si variance uniforme sur d dims.
    """
    if X.shape[0] < 2:
        return 1.0
    Xc = X - X.mean(axis=0, keepdims=True) if center else X
    # valeurs singulières -> lambda_i = s_i^2
    try:
        s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        return float(min(X.shape))
    lam = s ** 2
    denom = np.sum(lam ** 2)
    if denom <= 0:
        return 1.0
    return float((np.sum(lam) ** 2) / denom)


def compute_indicators(
    proba_pool: np.ndarray,
    X_pool: np.ndarray,
    uncertain_quantile: float = 0.10,
    margin_threshold: float = 0.5,
) -> dict:
    """Calcule les deux indicateurs sur le pool à un round donné.

    Parameters
    ----------
    proba_pool : (n, C) predict_proba du modèle courant sur le pool non labellisé
    X_pool     : (n, d) features du pool (mêmes lignes que proba_pool)
    uncertain_quantile : fraction du pool considérée "incertaine" (top-q par marge)
    margin_threshold   : seuil d'incertitude pour la fraction faible-marge

    Returns
    -------
    dict avec low_margin_fraction, uncertain_eff_dim, et métadonnées.
    """
    n = len(proba_pool)
    unc = margin_uncertainty(proba_pool)

    # 1) fraction de points à faible marge (= incertitude au-dessus du seuil)
    low_margin_fraction = float(np.mean(unc >= margin_threshold))

    # 2) eff_dim du sous-ensemble le plus incertain
    k = max(2, int(np.ceil(uncertain_quantile * n)))
    top_idx = np.argsort(-unc)[:k]
    uncertain_eff_dim = effective_dim(X_pool[top_idx])

    # normalisation optionnelle par la dim ambiante (comparable entre datasets)
    eff_dim_ratio = uncertain_eff_dim / X_pool.shape[1] if X_pool.shape[1] > 0 else np.nan

    return {
        "low_margin_fraction": low_margin_fraction,
        "uncertain_eff_dim": uncertain_eff_dim,
        "uncertain_eff_dim_ratio": float(eff_dim_ratio),
        "mean_uncertainty": float(np.mean(unc)),
        "n_pool": int(n),
        "n_uncertain": int(k),
        "ambient_dim": int(X_pool.shape[1]),
    }
