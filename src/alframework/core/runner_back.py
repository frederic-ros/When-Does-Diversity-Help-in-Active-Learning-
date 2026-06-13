from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from alframework.core.state import ALState
from alframework.core.metrics import evaluate
from alframework.core.labeler import Labeler

def _take_from_unlabeled(
    X_unlabeled: np.ndarray,
    y_new: np.ndarray,
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move selected indices from unlabeled -> labeled, returning (X_new, X_rest, idx_sorted)."""
    idx = np.asarray(indices, dtype=int)
    # Sort descending for robust deletion (keeps semantics clear)
    idx_sorted = np.sort(idx)[::-1]
    X_new = X_unlabeled[idx]
    X_rest = np.delete(X_unlabeled, idx_sorted, axis=0)
    return X_new, X_rest, idx_sorted

def active_learning_loop(
    state: ALState,
    strategy: Any,
    labeler: Labeler,
    n_rounds: int,
    budget: int,
) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for r in range(n_rounds):
        state.model.fit(state.X_labeled, state.y_labeled)
        metrics = evaluate(state.model, state.X_test, state.y_test)

        selected = strategy.select(state, budget=budget)
        selected = np.asarray(selected, dtype=int)
        y_new = labeler.label(selected)

        X_new, X_rest, idx_sorted = _take_from_unlabeled(state.X_unlabeled, y_new, selected)

        # Update labeled set
        state.X_labeled = np.concatenate([state.X_labeled, X_new], axis=0)
        state.y_labeled = np.concatenate([state.y_labeled, y_new], axis=0)

        # Update unlabeled + labeler if it's array-based
        state.X_unlabeled = X_rest
        if hasattr(labeler, "y_unlabeled"):
            labeler.y_unlabeled = np.delete(labeler.y_unlabeled, idx_sorted, axis=0)

        history.append(
            {
                "round": r,
                "n_labeled": int(len(state.X_labeled)),
                "n_unlabeled": int(len(state.X_unlabeled)),
                "selected": selected.tolist(),
                **metrics,
            }
        )
    return history
