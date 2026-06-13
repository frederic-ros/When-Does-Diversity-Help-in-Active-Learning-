# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 22:04:56 2026

@author: frederic.ros
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

@dataclass
class AlignedProbaClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper sklearn: garantit que predict_proba renvoie toujours (n, C_total)
    en alignant sur classes_all.
    """
    base_estimator: BaseEstimator
    classes_all: Sequence[int]

    def fit(self, X, y):
        self.est_ = clone(self.base_estimator)
        self.est_.fit(X, y)
        # classes vues par CE modèle entraîné
        self.classes_ = np.array(getattr(self.est_, "classes_", np.unique(y)))
        self.classes_all_ = np.array(self.classes_all)
        return self

    def predict(self, X):
        return self.est_.predict(X)

    def predict_proba(self, X):
        p = self.est_.predict_proba(X)  # (n, C_seen)
        return _align_proba(p, self.classes_, self.classes_all_)

def _align_proba(p: np.ndarray, classes_seen: np.ndarray, classes_all: np.ndarray) -> np.ndarray:
    """
    p: (n, C_seen), classes_seen: labels correspondant aux colonnes de p
    classes_all: labels globaux [0..C-1] (ou liste explicite)
    """
    p = np.asarray(p, dtype=float)
    n = p.shape[0]
    C = len(classes_all)
    out = np.zeros((n, C), dtype=float)

    # mapping label -> colonne globale
    pos = {int(c): j for j, c in enumerate(classes_all)}
    for j_seen, c in enumerate(classes_seen):
        cj = pos.get(int(c), None)
        if cj is not None:
            out[:, cj] = p[:, j_seen]

    # renormalisation pour éviter des sommes != 1
    s = out.sum(axis=1, keepdims=True)
    s[s == 0.0] = 1.0
    out = out / s
    return out