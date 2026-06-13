from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class Labeler(ABC):
    @abstractmethod
    def label(self, indices: np.ndarray) -> np.ndarray:
        """Return ground-truth labels for given indices in the *current* unlabeled pool."""
        raise NotImplementedError

class ArrayLabeler(Labeler):
    """Simulation labeler: ground-truth labels are stored alongside the unlabeled pool."""
    def __init__(self, y_unlabeled: np.ndarray):
        self.y_unlabeled = y_unlabeled

    def label(self, indices: np.ndarray) -> np.ndarray:
        return self.y_unlabeled[indices]
