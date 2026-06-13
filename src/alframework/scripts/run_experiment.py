from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from alframework.core.registry import STRATEGIES
from alframework.core.state import ALState
from alframework.core.runner import active_learning_loop
from alframework.core.labeler import ArrayLabeler
from alframework.utils.seed import seed_everything
from alframework.utils.logging import ResultLogger

# Import strategies so they register
from alframework.strategies import random as _random  # noqa: F401
from alframework.strategies import uncertainty as _unc  # noqa: F401
from alframework.strategies import coreset as _core  # noqa: F401
from alframework.strategies import typiclust as _typ  # noqa: F401
from alframework.strategies import probcover as _pc  # noqa: F401
from alframework.strategies import dbal as _dbal  # noqa: F401
from alframework.strategies import badge as _badge  # noqa: F401
from alframework.strategies import rank2022 as _rank  # noqa: F401
from alframework.strategies import active_pseudolabel as _apl  # noqa: F401
from alframework.strategies import active_pseudolabel_v2 as _aplv2  # noqa: F401
from alframework.strategies import qbc as _qbc  # noqa: F401
from alframework.strategies import tri_committee as _tri  # noqa: F401
from alframework.strategies import selftrain_acq as _sta  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", type=str, default="random", choices=sorted(STRATEGIES.keys()))
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="runs/demo")
    args = ap.parse_args()

    rng = seed_everything(args.seed)

    X, y = make_classification(n_samples=1500, n_features=20, n_informative=10, n_classes=3, random_state=args.seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)

    # Init labeled/unlabeled split
    n0 = 30
    idx0 = rng.choice(len(X_train), size=n0, replace=False)
    mask = np.zeros(len(X_train), dtype=bool)
    mask[idx0] = True

    X_l, y_l = X_train[mask], y_train[mask]
    X_u, y_u = X_train[~mask], y_train[~mask]

    model = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    state = ALState(X_labeled=X_l, y_labeled=y_l, X_unlabeled=X_u, model=model, rng=rng, X_test=X_test, y_test=y_test)
    labeler = ArrayLabeler(y_u)

    StrategyCls = STRATEGIES[args.strategy]
    strategy = StrategyCls()  # pass params here if needed

    hist = active_learning_loop(state, strategy, labeler, n_rounds=args.rounds, budget=args.budget)

    out_dir = Path(args.out) / args.strategy
    logger = ResultLogger(out_dir)
    logger.save_config(vars(args))
    logger.save_history(hist)

    print(f"Saved run to: {out_dir}")

if __name__ == "__main__":
    main()
