from __future__ import annotations
import os
import random
from dataclasses import dataclass
import numpy as np

def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and (optionally) other libs. Returns a NumPy Generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
