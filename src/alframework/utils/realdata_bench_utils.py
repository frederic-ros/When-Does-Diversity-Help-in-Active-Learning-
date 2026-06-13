# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 06:33:30 2026

@author: frederic.ros
"""

# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def find_dataset_files(
    root_dir: Path | str,
    *,
    pattern: str = "*.txt",
    recursive: bool = False,
) -> List[Path]:
    """
    Liste les fichiers dataset sous root_dir.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    it = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted([p for p in it if p.is_file()])


def load_tabular_txt(
    path: Path | str,
    *,
    delimiter: str = "\t",
    dtype=float,
    label_dtype=int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge un fichier .txt tabulé: features en colonnes, dernière colonne = label (0,1,...).
    Hypothèse: pas d'en-tête, pas de commentaires, données propres.
    """
    p = Path(path)
    arr = np.genfromtxt(p, delimiter=delimiter)

    if arr.ndim == 1:
        # dataset 1 seule ligne
        arr = arr.reshape(1, -1)

    if arr.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns (X + y). Got {arr.shape[1]} in {p}")

    X = arr[:, :-1].astype(dtype, copy=False)
    y = arr[:, -1].astype(label_dtype, copy=False)

    if np.any(np.isnan(y)):
        raise ValueError(f"NaN found in labels in {p}")
    return X, y


def _class_counts(y: np.ndarray, idx: np.ndarray) -> Dict[int, int]:
    vals, cnts = np.unique(y[idx], return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}


def make_stratified_splits(
    y: np.ndarray,
    *,
    n_splits: int,
    test_size: float,
    seed: int,
    min_train_per_class: int = 1,
    max_tries: int = 200,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Produit n_splits (train_idx, test_idx) stratifiés.
    Garantie: au moins min_train_per_class exemples par classe dans le TRAIN.
    Reproductible via seed.

    Note: on splitte "à la main" pour éviter dépendances sklearn si tu veux.
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")

    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)

    if np.any(counts < (min_train_per_class + 1)):
        raise ValueError(
            "Some classes are too small for splitting given min_train_per_class. "
            f"counts={dict(zip(classes.tolist(), counts.tolist()))}"
        )

    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for split_id in range(n_splits):
        ok = False

        for attempt in range(max_tries):
            local_seed = int(seed + 10_000 * split_id + attempt)
            rng = np.random.default_rng(local_seed)

            train_parts = []
            test_parts = []

            for c in classes:
                idx_c = np.flatnonzero(y == c)
                rng.shuffle(idx_c)

                n_test_c = int(np.round(test_size * len(idx_c)))
                n_test_c = max(1, n_test_c)  # au moins 1 test/ classe
                n_train_c = len(idx_c) - n_test_c

                if n_train_c < min_train_per_class:
                    break

                test_parts.append(idx_c[:n_test_c])
                train_parts.append(idx_c[n_test_c:])

            else:
                train_idx = np.concatenate(train_parts)
                test_idx = np.concatenate(test_parts)
                rng.shuffle(train_idx)
                rng.shuffle(test_idx)

                train_counts = _class_counts(y, train_idx)
                if all(train_counts.get(int(c), 0) >= min_train_per_class for c in classes):
                    splits.append((train_idx, test_idx))
                    ok = True
                    break

        if not ok:
            raise RuntimeError(
                f"Could not build valid stratified split split_id={split_id} "
                f"after {max_tries} attempts (seed={seed}, test_size={test_size})."
            )

    return splits


def aggregate_metric_list(values: List[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(v)),
        "std": float(np.nanstd(v)),
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
        "n": int(np.sum(~np.isnan(v))),
    }


def aggregate_scores(all_scores: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    all_scores[strategy][metric] = [v1, v2, ...]
    -> stats[strategy][metric] = {"mean":..., "std":..., "min":..., "max":..., "n":...}
    """
    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for strat, per_metric in all_scores.items():
        stats[strat] = {}
        for metric, vals in per_metric.items():
            stats[strat][metric] = aggregate_metric_list(vals)
    return stats