#!/usr/bin/env python3
"""
run_all.py — master script reproducing the FULL analysis behind the paper.

Usage:
    python run_all.py                 # run every step in order
    python run_all.py --from 4        # resume from step 4 (uses cached parquet)
    python run_all.py --only 8        # run only step 8
    python run_all.py --skip-structure  # skip step 5 (slow: regenerates pools);
                                        # steps 6/10 then skip their structure parts

Steps and the paper artifacts they produce:
    1  load          parse history JSONs -> long.parquet
    2  saturation    ante-saturation windows -> saturation.parquet
    3  aulc          split-level windowed AULC -> aulc_split.parquet
    4  leaderboard   Table 1; leaderboard/regret/flatness/router stats
    5  structure     Table 6 inputs; regenerate pools, structural indicators
    6  figures       Figures 1-5 (spread, curves, structure, showcase, risk)
    7  tables        Tables 1,2,3,6 LaTeX fragments
    8  difficulty    Table 4; intrinsic-difficulty stratification
    9  classifier    Table 5; LR-vs-RF rank shift
   10  trend         Sec 4.3 robustness (partial Spearman, mixed-effects, LORO)

Edit paths in config.py first. Outputs land under ./out (cache/figures/tables).
"""
import argparse
import sys
import time

import config as C

import step1_load
import step2_saturation
import step3_aulc
import step4_leaderboard
import step5_structure
import step6_figures
import step7_tables
import step8_difficulty
import step9_classifier
import step10_trend_robustness

STEPS = [
    (1, "load", step1_load.run),
    (2, "saturation", step2_saturation.run),
    (3, "aulc", step3_aulc.run),
    (4, "leaderboard", step4_leaderboard.run),
    (5, "structure", step5_structure.run),
    (6, "figures", step6_figures.run),
    (7, "tables", step7_tables.run),
    (8, "difficulty", step8_difficulty.run),
    (9, "classifier", step9_classifier.run),
    (10, "trend", step10_trend_robustness.run),
]


def _check_data():
    missing = []
    if not C.RESULTS_DIR.exists():
        missing.append(str(C.RESULTS_DIR))
    if not C.TAB_DIR.exists():
        missing.append(str(C.TAB_DIR))
    if not C.LATENT_ROOT.exists():
        missing.append(str(C.LATENT_ROOT))
    if missing:
        print("[run_all] WARNING — these data paths do not exist:")
        for m in missing:
            print("   ", m)
        print("   Edit DATA_ROOT (and sub-paths) in config.py. Steps needing them will fail.")


def main():
    ap = argparse.ArgumentParser(description="Reproduce the full active-learning analysis.")
    ap.add_argument("--from", dest="start", type=int, default=1, help="resume from this step")
    ap.add_argument("--only", type=int, default=None, help="run only this step")
    ap.add_argument("--skip-structure", action="store_true",
                    help="skip step 5 (pool regeneration is the slowest part)")
    args = ap.parse_args()

    _check_data()
    print(f"[run_all] data root : {C.DATA_ROOT}")
    print(f"[run_all] output dir: {C.OUT_ROOT}\n")

    for num, name, fn in STEPS:
        if args.only is not None and num != args.only:
            continue
        if args.only is None and num < args.start:
            continue
        if args.skip_structure and num == 5:
            print(f"[run_all] skipping step {num} ({name}) per --skip-structure\n")
            continue
        print(f"{'='*72}\n[run_all] STEP {num}: {name}\n{'='*72}")
        t0 = time.time()
        try:
            fn()
        except Exception as e:
            print(f"[run_all] STEP {num} ({name}) FAILED: {type(e).__name__}: {e}")
            sys.exit(1)
        print(f"[run_all] step {num} done in {time.time()-t0:.1f}s\n")

    print("[run_all] complete. See out/cache, out/figures, out/tables.")


if __name__ == "__main__":
    main()
