"""
step1_load.py — parse every history JSON into one long table.

Walks RESULTS_DIR, parses filenames of the form
  history_{lr|rf}_{dataset}_split{n}_{strategy}.json
across all four history folders, normalizes strategy names, tags the regime,
and writes one row per (context, strategy, split, budget-point).

Output: out/cache/long.parquet
Run standalone:  python step1_load.py
"""
import os
import re
import json
import pandas as pd

import config as C

_FNAME = re.compile(r"history_(lr|rf)_(.+?)_split(\d+)_(.+)\.json")


def _regime_of(path: str) -> str:
    if "BenchSynthetique" in path:
        return "synthetic"
    if "tabulaire" in path:
        return "tabular"
    if "latent" in path.lower():
        return "latent"
    return "unknown"


def run() -> pd.DataFrame:
    rows = []
    for dirpath, _, files in os.walk(C.RESULTS_DIR):
        for f in files:
            if not f.endswith(".json"):
                continue
            m = _FNAME.match(f)
            if not m:
                continue
            clf, ds, split, strat = m.groups()
            strat = C.norm_strat(strat)
            regime = _regime_of(dirpath)
            try:
                hist = json.load(open(os.path.join(dirpath, f)))
            except Exception:
                continue
            if not hist:
                continue
            for step in hist:
                rows.append({
                    "regime": regime, "clf": clf, "dataset": ds,
                    "split": int(split), "strategy": strat,
                    "family": C.FAMILY.get(strat, "other"),
                    "n_labeled": step.get("n_labeled"),
                    "accuracy": step.get("accuracy"),
                    "f1_macro": step.get("f1_macro"),
                    "balanced_accuracy": step.get("balanced_accuracy"),
                    "f1_weighted": step.get("f1_weighted"),
                })
    df = pd.DataFrame(rows)
    out = C.CACHE / "long.parquet"
    df.to_parquet(out)
    print(f"[step1] {len(df):,} rows -> {out}")
    print(f"[step1] regimes: {df.groupby('regime')['strategy'].count().to_dict()}")
    print(f"[step1] strategies ({df.strategy.nunique()}): {sorted(df.strategy.unique())}")
    return df


if __name__ == "__main__":
    run()
