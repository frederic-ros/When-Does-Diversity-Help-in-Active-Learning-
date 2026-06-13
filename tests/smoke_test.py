# -*- coding: utf-8 -*-
"""Smoke test : valide patch + analyse sur données synthétiques contrôlées."""
import csv, subprocess, sys
from pathlib import Path
import numpy as np

W = Path("/home/claude/work")
rng = np.random.default_rng(0)

# 3 régimes contrôlés pour vérifier que les verdicts sont CORRECTS :
#  - scenario "true_gain"  : challenger réellement > baseline (+0.02)
#  - scenario "noise_only" : challenger == baseline en espérance (Δ~0)
#  - scenario "tiny_gain"  : Δ réel +0.001 < plancher 0.005 (doit être rejeté)
rows = []
N_SEED = 8
for model in ["rf"]:
    for sc, shift in [("true_gain", 0.02), ("noise_only", 0.0), ("tiny_gain", 0.001)]:
        base = rng.uniform(0.60, 0.75)
        for s in range(N_SEED):
            common_noise = rng.normal(0, 0.03)  # effet "même dataset" partagé
            for v, extra in [("V4.4", shift), ("V4", shift*0.5),
                             ("V5", shift*0.3), ("V5.1", 0.0),
                             ("V5.2", -0.002),
                             ("dbal", 0.0), ("qbc", -0.002),
                             ("margin", -0.005), ("random", -0.02)]:
                auc = base + common_noise + extra + rng.normal(0, 0.004)
                rows.append([model, sc, v, s, f"{auc:.6f}", ""])

csv_path = W / "synthetic_per_seed.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["model", "scenario", "variant", "seed_index", "auc", "final_acc"])
    w.writerows(rows)
print(f"[smoke] wrote {len(rows)} rows -> {csv_path}")

# lance l'analyse multi-challengers (n=8 < min-seeds 20 => SOUS-PUISSÉ)
r = subprocess.run(
    [sys.executable, str(W / "analyze_paired.py"), str(csv_path),
     "--out", str(W / "smoke_out"), "--mode", "challenger",
     "--challengers", "V4", "V4.4", "V5", "V5.1", "V5.2",
     "--baselines", "dbal", "qbc", "--delta-floor", "0.005",
     "--min-seeds", "20"],
    capture_output=True, text=True)
print("STDOUT:", r.stdout)
print("STDERR:", r.stderr[-500:] if r.stderr else "(none)")
print("RC:", r.returncode)

print("\n===== REPORT =====")
print((W / "smoke_out" / "paired_report.md").read_text())
