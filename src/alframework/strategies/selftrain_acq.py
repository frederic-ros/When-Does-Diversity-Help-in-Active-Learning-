from __future__ import annotations

import numpy as np

from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.base import QueryStrategy


def _max_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return np.max(p, axis=1)
    return np.full(X.shape[0], 0.0, dtype=float)


@register("selftrain_acq")
class SelfTrainingAwareAcquisition(QueryStrategy):
    """A-only 'self-training aware' acquisition.

    Intuition:
    - In self-training, points with very high confidence would be pseudo-labeled.
    - Here, we DO NOT pseudo-label; we query points that block safe expansion:
        * "near_threshold": points with confidence close to tau
        * "low_confidence": points with lowest confidence (classic uncertainty)

    Parameters
    ----------
    tau : float
        confidence threshold (typical self-training threshold, e.g., 0.9)
    window : float
        for "near_threshold": select points with |conf - tau| small
    mode : {"near_threshold","low_confidence"}
    """

    def __init__(self, tau: float = 0.9, window: float = 0.05, mode: str = "near_threshold"):
        self.tau = float(tau)
        self.window = float(window)
        self.mode = str(mode)

    def select(self, state: ALState, budget: int) -> np.ndarray:
        X_L, y_L, X_U = state.X_labeled, state.y_labeled, state.X_unlabeled
        nU = len(X_U)
        k = min(int(budget), nU)
        if k <= 0:
            return np.array([], dtype=int)

        state.model.fit(X_L, y_L)
        conf = _max_proba(state.model, X_U)

        if self.mode == "near_threshold":
            score = -np.abs(conf - self.tau)
            if self.window > 0:
                band = np.abs(conf - self.tau) <= self.window
                if np.any(band):
                    idx_band = np.where(band)[0]
                    idx_sorted = idx_band[np.argsort(-score[idx_band])]
                    return idx_sorted[:k]
            return np.argsort(-score)[:k]

        score = -conf
        return np.argsort(-score)[:k]
