# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 06:01:24 2026

@author: frederic.ros
"""

# tests/test_synth_builder.py
import sys
from pathlib import Path

# --- PATH RELATIF + FIX SPYDER (comme tes autres tests) ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
SRC = PROJECT_ROOT / "src"

while not SRC.exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
    SRC = PROJECT_ROOT / "src"

if not SRC.exists():
    raise RuntimeError(f"Impossible de trouver le dossier 'src' depuis {THIS_FILE}")

sys.path.insert(0, str(SRC))

# purge des imports partiels (Spyder)
for k in list(sys.modules.keys()):
    if k == "alframework" or k.startswith("alframework."):
        del sys.modules[k]

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from alframework.data.synth_builder import build_synth_state_and_testset


def _default_dataset_params(seed: int = 0):
    return dict(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_classes=3,
        class_sep=1.2,
        flip_y=0.02,
        random_state=seed,
    )


def test_build_synth_state_shapes_ninit_20():
    params = _default_dataset_params(seed=0)

    state, y_unl_true, X_test, y_test = build_synth_state_and_testset(
        dataset_params=params,
        test_size=0.3,
        split_random_state=42,
        n_init=20,
        init_random_state=123,
        rng_seed=0,
        model=RandomForestClassifier(n_estimators=50, random_state=0),
    )

    # Basics
    assert state.X_labeled.shape[0] == 20
    assert state.y_labeled.shape[0] == 20
    assert state.X_unlabeled.shape[0] == len(y_unl_true)

    # Train = labeled + unlabeled (le tout vient du train split)
    n_train = state.X_labeled.shape[0] + state.X_unlabeled.shape[0]
    assert X_test.shape[0] + n_train == params["n_samples"]

    # Same feature dimension everywhere
    assert state.X_labeled.shape[1] == state.X_unlabeled.shape[1] == X_test.shape[1] == params["n_features"]

    # Labels shape
    assert y_test.shape[0] == X_test.shape[0]

    # sanity: classes exist
    assert len(np.unique(y_test)) <= params["n_classes"]


def test_build_synth_state_supports_ninit_0():
    params = _default_dataset_params(seed=1)

    state, y_unl_true, X_test, y_test = build_synth_state_and_testset(
        dataset_params=params,
        n_init=0,
        split_random_state=42,
        init_random_state=999,
        rng_seed=0,
        model=RandomForestClassifier(n_estimators=10, random_state=0),
    )

    # n_init=0 => labeled vide
    assert state.X_labeled.shape[0] == 0
    assert state.y_labeled.shape[0] == 0

    # tout le train est unlabeled
    assert state.X_unlabeled.shape[0] == y_unl_true.shape[0]
    assert state.X_unlabeled.shape[1] == params["n_features"]
    assert X_test.shape[1] == params["n_features"]
    assert y_test.shape[0] == X_test.shape[0]


def test_initial_labeled_is_randomized_by_seed():
    params = _default_dataset_params(seed=2)

    state1, _, _, _ = build_synth_state_and_testset(
        dataset_params=params,
        n_init=20,
        init_random_state=123,   # seed A
        split_random_state=42,
        rng_seed=0,
        model=RandomForestClassifier(n_estimators=10, random_state=0),
    )

    state2, _, _, _ = build_synth_state_and_testset(
        dataset_params=params,
        n_init=20,
        init_random_state=456,   # seed B (différent)
        split_random_state=42,
        rng_seed=0,
        model=RandomForestClassifier(n_estimators=10, random_state=0),
    )

    # Les labeled sets devraient être différents (très probable)
    # On compare les "empreintes" des X_labeled
    fp1 = np.round(state1.X_labeled, 8).tobytes()
    fp2 = np.round(state2.X_labeled, 8).tobytes()
    assert fp1 != fp2, "Les initial labeled sets sont identiques alors que init_random_state change."


def test_test_split_is_stable_with_split_seed():
    params = _default_dataset_params(seed=3)

    _, _, X_test1, y_test1 = build_synth_state_and_testset(
        dataset_params=params,
        n_init=20,
        split_random_state=42,
        init_random_state=123,
        rng_seed=0,
        model=RandomForestClassifier(n_estimators=10, random_state=0),
    )

    _, _, X_test2, y_test2 = build_synth_state_and_testset(
        dataset_params=params,
        n_init=20,
        split_random_state=42,  # même seed => même test split
        init_random_state=999,  # différent, mais ne doit pas affecter X_test/y_test
        rng_seed=0,
        model=RandomForestClassifier(n_estimators=10, random_state=0),
    )

    assert np.allclose(X_test1, X_test2)
    assert np.array_equal(y_test1, y_test2)


if __name__ == "__main__":
    # Run simple (Spyder)
    test_build_synth_state_shapes_ninit_20()
    test_build_synth_state_supports_ninit_0()
    test_initial_labeled_is_randomized_by_seed()
    test_test_split_is_stable_with_split_seed()
    print("✅ Synth builder tests OK")