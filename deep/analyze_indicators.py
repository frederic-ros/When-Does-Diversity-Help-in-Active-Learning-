"""
analyze_indicators.py
=====================
Teste la LOI PRÉDICTIVE : l'indicateur structurel a priori (mesuré aux rounds
précoces) prédit-il le gain de la diversité sur l'incertitude pure ?

Entrée  : dossier d'historiques enrichis (produits par les bench_*_deep.py,
          avec log_indicators=True) -> fichiers history_{clf}_{dataset}_split{n}_{strat}.json
Sortie  : table (dataset x classifieur) avec indicateurs + Delta, corrélation de
          Spearman, et figure scatter indicateur vs Delta colorée par famille.

Delta = AULC(meilleure stratégie de clustering) - AULC(margin)
Indicateur = moyenne (sur seeds/splits) de l'indicateur au round précoce choisi.

Aucune circularité : Delta vient des courbes complètes, l'indicateur des
premiers rounds uniquement.
"""
from __future__ import annotations
import os, glob, json, re
import numpy as np

# --- Configuration ---------------------------------------------------------
HIST_DIR = "."                  # dossier des history_*.json
CLUSTERING_METHODS = [          # la "famille diversité/clustering"
    "dbal", "rank2022", "ActivePseudoLabelV58",
    "coreset_greedy", "coreset_kmeanspp", "typiclust", "probcover",
    "badge_approx", "unc_feature_kmeans", "diversity_optimized_batch",
]
MARGIN_METHOD = "margin"
METRIC = "f1_macro"             # métrique pour l'AULC
INDICATOR_KEY = "ind_uncertain_eff_dim"   # ou "ind_low_margin_fraction"
INDICATOR_ROUND = 0            # round précoce où lire l'indicateur
FAMILY_FROM_PATH = True         # déduire la famille (synth/real/latent) du chemin


def _parse(fname):
    b = os.path.basename(fname)[:-5]
    m = re.match(r"history_(rf|lr)_(.+?)_split(\d+)_(.+)$", b)
    if not m:
        m = re.match(r"history_(.+?)_split(\d+)_(.+)$", b)
        if not m: return None
        return {"clf": "na", "dataset": m.group(1), "split": int(m.group(2)), "strat": m.group(3)}
    return {"clf": m.group(1), "dataset": m.group(2), "split": int(m.group(3)), "strat": m.group(4)}


def _aulc(curve):
    vals = [r.get(METRIC, np.nan) for r in curve]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals[1:])) if len(vals) > 1 else np.nan


def _indicator(curve, key, rnd):
    for r in curve:
        if r.get("round") == rnd and key in r:
            return float(r[key])
    # fallback : 1er round où la clé existe
    for r in curve:
        if key in r:
            return float(r[key])
    return np.nan


def load_all(hist_dir=HIST_DIR):
    rows = {}  # (clf,dataset,strat) -> list of (aulc, indicator)
    for f in glob.glob(os.path.join(hist_dir, "**", "history_*.json"), recursive=True):
        meta = _parse(f)
        if meta is None: continue
        try:
            curve = json.load(open(f))
        except Exception:
            continue
        a = _aulc(curve)
        ind = _indicator(curve, INDICATOR_KEY, INDICATOR_ROUND)
        key = (meta["clf"], meta["dataset"], meta["strat"])
        rows.setdefault(key, []).append((a, ind))
    return rows


def build_table(rows):
    # agrège par (clf, dataset, strat) sur les splits/seeds
    agg = {}
    for (clf, ds, strat), lst in rows.items():
        a = np.nanmean([x[0] for x in lst])
        ind = np.nanmean([x[1] for x in lst])
        agg[(clf, ds, strat)] = (a, ind)
    # par (clf, dataset) : Delta = best_clustering_AULC - margin_AULC ; indicateur = celui de margin (ou moyenne)
    configs = sorted(set((clf, ds) for (clf, ds, _) in agg))
    table = []
    for (clf, ds) in configs:
        margin_a = agg.get((clf, ds, MARGIN_METHOD), (np.nan, np.nan))[0]
        clus = [agg[(clf, ds, s)][0] for s in CLUSTERING_METHODS if (clf, ds, s) in agg]
        if not clus or margin_a != margin_a:
            continue
        best_clus = np.nanmax(clus)
        delta = best_clus - margin_a
        # indicateur : on prend celui mesuré sous margin (état initial identique entre stratégies)
        ind = agg.get((clf, ds, MARGIN_METHOD), (np.nan, np.nan))[1]
        if ind != ind:  # si margin n'a pas d'indicateur, moyenne sur les stratégies dispo
            inds = [agg[(clf, ds, s)][1] for (c, d, s) in agg if c == clf and d == ds]
            ind = np.nanmean(inds)
        table.append({"clf": clf, "dataset": ds, "indicator": ind,
                      "delta": delta, "margin_aulc": margin_a, "best_clustering_aulc": best_clus})
    return table


def family_of(dataset):
    d = dataset.lower()
    if any(k in d for k in ["cifar", "mnist", "fashion", "pca"]): return "latent"
    if any(k in d for k in ["easy", "medium", "hard", "imbalanced", "overlap",
                             "redundant", "signal", "class", "noisy", "clean"]): return "synthetic"
    return "tabular"


def main():
    rows = load_all()
    table = build_table(rows)
    if not table:
        print("Aucune config exploitable. Vérifie HIST_DIR, INDICATOR_KEY et le panel.")
        return
    ind = np.array([t["indicator"] for t in table])
    delta = np.array([t["delta"] for t in table])
    mask = np.isfinite(ind) & np.isfinite(delta)
    ind, delta = ind[mask], delta[mask]
    table = [t for t, m in zip(table, mask) if m]

    # corrélation de Spearman
    try:
        from scipy.stats import spearmanr, pearsonr
        rho, p = spearmanr(ind, delta)
        r, pr = pearsonr(ind, delta)
        print(f"Spearman rho = {rho:.3f} (p={p:.2e})   Pearson r = {r:.3f} (p={pr:.2e})")
    except Exception:
        rho = np.corrcoef(ind, delta)[0, 1]
        print(f"corr (numpy) = {rho:.3f}")

    print(f"\nN configs = {len(table)}")
    print(f"{'family':<10}{'clf':<5}{'dataset':<26}{'indicator':>11}{'delta':>10}")
    for t in sorted(table, key=lambda x: x["indicator"]):
        print(f"{family_of(t['dataset']):<10}{t['clf']:<5}{t['dataset']:<26}"
              f"{t['indicator']:>11.3f}{t['delta']:>+10.4f}")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fams = [family_of(t["dataset"]) for t in table]
        colors = {"synthetic": "tab:blue", "tabular": "tab:green", "latent": "tab:red"}
        plt.figure(figsize=(7, 5))
        for fam in set(fams):
            xs = [t["indicator"] for t, f in zip(table, fams) if f == fam]
            ys = [t["delta"] for t, f in zip(table, fams) if f == fam]
            plt.scatter(xs, ys, label=fam, c=colors.get(fam, "gray"), alpha=0.7, s=40)
        # tendance
        z = np.polyfit(ind, delta, 1)
        xx = np.linspace(ind.min(), ind.max(), 50)
        plt.plot(xx, np.polyval(z, xx), "k--", alpha=0.5, label="tendance")
        plt.axhline(0, color="gray", lw=0.8)
        plt.xlabel(INDICATOR_KEY + f" (round {INDICATOR_ROUND})")
        plt.ylabel("Delta = AULC(best clustering) - AULC(margin)")
        plt.title("Predictivite de l'indicateur structurel sur le gain de diversite")
        plt.legend()
        plt.tight_layout()
        out = "indicator_vs_delta.png"
        plt.savefig(out, dpi=140)
        print(f"\nFigure sauvegardee : {out}")
    except Exception as e:
        print(f"[figure non generee: {e}]")


if __name__ == "__main__":
    main()
