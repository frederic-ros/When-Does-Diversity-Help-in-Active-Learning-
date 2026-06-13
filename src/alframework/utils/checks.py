# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 06:09:42 2026

@author: frederic.ros
"""

from __future__ import annotations
from typing import Any, Dict, Tuple
import inspect
from sklearn.datasets import make_classification



from typing import Dict, Any
import math

def check_make_classification_params(p: Dict[str, Any]) -> None:
    """
    Vérifie les contraintes principales de sklearn.make_classification.
    Lève ValueError si incohérent.
    """

    # --- Extraction avec valeurs par défaut sklearn ---
    n_features = int(p.get("n_features", 20))
    n_informative = int(p.get("n_informative", 2))
    n_redundant = int(p.get("n_redundant", 2))
    n_repeated = int(p.get("n_repeated", 0))
    n_classes = int(p.get("n_classes", 2))
    n_clusters_per_class = int(p.get("n_clusters_per_class", 2))
    weights = p.get("weights", None)

    # --- Vérifications simples ---
    if n_features <= 0:
        raise ValueError("n_features doit être > 0")

    if n_informative <= 0:
        raise ValueError("n_informative doit être > 0")

    if n_redundant < 0 or n_repeated < 0:
        raise ValueError("n_redundant et n_repeated doivent être >= 0")

    if n_classes < 2:
        raise ValueError("n_classes doit être >= 2")

    if n_clusters_per_class < 1:
        raise ValueError("n_clusters_per_class doit être >= 1")

    # --- Contrainte fondamentale ---
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError(
            f"Invalid: n_informative({n_informative}) + "
            f"n_redundant({n_redundant}) + "
            f"n_repeated({n_repeated}) > n_features({n_features})"
        )

    # --- Contrainte structurelle sklearn ---
    # sklearn impose que 2**n_informative >= n_classes * n_clusters_per_class
    required = n_classes * n_clusters_per_class
    if 2 ** n_informative < required:
        min_inf = math.ceil(math.log2(required))
        raise ValueError(
            f"n_informative={n_informative} insuffisant pour "
            f"{n_classes} classes × {n_clusters_per_class} clusters. "
            f"Minimum requis: {min_inf}"
        )

    # --- Vérification des weights si présents ---
    if weights is not None:
        if len(weights) != n_classes:
            raise ValueError(
                f"weights doit avoir longueur n_classes ({n_classes}), "
                f"reçu {len(weights)}"
            )
        if not math.isclose(sum(weights), 1.0, rel_tol=1e-6):
            raise ValueError("La somme des weights doit être 1.0")

    # --- Vérification samples vs classes ---
    n_samples = p.get("n_samples", None)
    if n_samples is not None:
        if n_samples < n_classes:
            raise ValueError("n_samples doit être >= n_classes")

    # Si on arrive ici → OK