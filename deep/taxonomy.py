"""
taxonomy.py
===========
Construit une TAXONOMIE des datasets selon leur POUVOIR DISCRIMINANT pour
comparer des stratégies d'active learning, calculée pour acc ET f1_macro.

Idée : sur un dataset où random atteint déjà ~le full-supervision, aucune
stratégie ne peut se distinguer (dataset "trivial"). On ne devrait pas le
laisser diluer l'analyse. On mesure donc l'espace de gain disponible.

Pour chaque (dataset, classifieur, métrique) :
  - perf_full   : modèle entraîné sur TOUT le train (full supervision)
  - perf_random : AULC (ou perf finale) de la stratégie 'random'
  - perf_best   : AULC (ou perf finale) de la meilleure stratégie
  - discrim     = perf_full - perf_random      (espace de gain ; petit => trivial)
  - coverage(s) = (perf_s - perf_random) / (perf_full - perf_random)
                  fraction de l'écart comblée par la stratégie s

Classification :
  - trivial      : discrim < eps  (random ~ full ; rien à apprendre)
  - discriminant : sinon
Le seuil eps est exprimé en fraction de perf_full (robuste, sans unité).

NOTE : le full-supervision est calculé ICI pour le SYNTHÉTIQUE (scénarios
make_classification reproductibles). Pour tabulaire/latent, fournir une table
full_supervision externe (voir compute_full_supervision_* ci-dessous) ou
brancher les loaders.
"""
from __future__ import annotations
import glob, re, json, os
import numpy as np
from collections import defaultdict

USE_FINAL = False   # False = AULC (aire sous la courbe) ; True = perf finale au budget max
METRICS = ["accuracy", "f1_macro"]


# ----------------------------------------------------------------------
# 1) Lecture des historiques : perf par (clf, dataset, strat, split, métrique)
# ----------------------------------------------------------------------
def _perf_from_curve(curve, metric):
    vals = [r.get(metric) for r in curve if r.get(metric) is not None]
    if not vals:
        return np.nan
    if USE_FINAL:
        return float(vals[-1])
    return float(np.mean(vals[1:])) if len(vals) > 1 else float(vals[0])


def load_histories(hist_dir):
    # perf[(clf,ds,strat,metric)] = {split: value}
    perf = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(hist_dir, "**", "history_*.json"), recursive=True):
        b = os.path.basename(f)[:-5]
        m = re.match(r"history_(rf|lr)_(.+?)_split(\d+)_(.+)$", b)
        if not m:
            continue
        clf, ds, split, strat = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        try:
            curve = json.load(open(f))
        except Exception:
            continue
        for metric in METRICS:
            perf[(clf, ds, metric)][strat][split] = _perf_from_curve(curve, metric)
    return perf


# ----------------------------------------------------------------------
# 2) Full-supervision pour le SYNTHÉTIQUE (reproduit make_classification)
# ----------------------------------------------------------------------
def compute_full_supervision_synthetic(scenarios, seeds, clfs=("rf", "lr"),
                                       test_size=0.25):
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    def make_model(kind, seed):
        if kind == "rf":
            return RandomForestClassifier(n_estimators=50, random_state=seed)
        return LogisticRegression(max_iter=2000, random_state=seed)

    full = defaultdict(lambda: defaultdict(dict))  # (clf,ds,metric)->{split:val}
    for ds, p in scenarios.items():
        p = dict(p)
        n_samples = int(p.pop("n_samples", 1500))
        n_classes = int(p.pop("n_classes", 3))
        for seed in seeds:
            X, y = make_classification(n_samples=n_samples, n_classes=n_classes,
                                       random_state=seed, **p)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y)
            for clf in clfs:
                mdl = make_model(clf, seed).fit(Xtr, ytr)
                yp = mdl.predict(Xte)
                full[(clf, ds, "accuracy")][seed] = accuracy_score(yte, yp)
                full[(clf, ds, "f1_macro")][seed] = f1_score(yte, yp, average="macro")
    return full


# ----------------------------------------------------------------------
# 3) Construire la taxonomie
# ----------------------------------------------------------------------
def build_taxonomy(perf, full, trivial_eps_frac=0.02):
    """Retourne une liste de dicts par (clf, ds, metric)."""
    out = []
    keys = sorted(set(perf.keys()) & set(full.keys()))
    for (clf, ds, metric) in keys:
        strat_perf = perf[(clf, ds, metric)]
        full_vals = full[(clf, ds, metric)]
        # apparier par split commun
        common_splits = set.intersection(
            *[set(d.keys()) for d in strat_perf.values()]
        ) if strat_perf else set()
        common_splits &= set(full_vals.keys())
        if not common_splits:
            continue
        common = sorted(common_splits)
        # moyennes sur splits
        def mean_strat(s):
            return np.mean([strat_perf[s][sp] for sp in common if sp in strat_perf.get(s, {})])
        p_full = np.mean([full_vals[sp] for sp in common])
        p_rand = mean_strat("random") if "random" in strat_perf else np.nan
        best_strat = max(strat_perf, key=lambda s: mean_strat(s))
        p_best = mean_strat(best_strat)
        discrim = p_full - p_rand
        denom = discrim if abs(discrim) > 1e-9 else np.nan
        cov_best = (p_best - p_rand) / denom if denom == denom else np.nan
        is_trivial = (discrim < trivial_eps_frac * p_full)
        out.append({
            "clf": clf, "dataset": ds, "metric": metric,
            "perf_full": p_full, "perf_random": p_rand, "perf_best": p_best,
            "best_strategy": best_strat,
            "discrim": discrim, "coverage_best": cov_best,
            "class": "trivial" if is_trivial else "discriminant",
            "n_splits": len(common),
        })
    return out


def print_taxonomy(tax):
    for metric in METRICS:
        rows = [t for t in tax if t["metric"] == metric]
        rows.sort(key=lambda t: t["discrim"])
        print(f"\n{'='*78}\nTAXONOMIE — métrique = {metric}  (USE_FINAL={USE_FINAL})\n{'='*78}")
        print(f"{'clf':<4}{'dataset':<22}{'full':>8}{'random':>8}{'best':>8}"
              f"{'discrim':>9}{'class':>14}")
        for t in rows:
            print(f"{t['clf']:<4}{t['dataset']:<22}{t['perf_full']:>8.3f}"
                  f"{t['perf_random']:>8.3f}{t['perf_best']:>8.3f}"
                  f"{t['discrim']:>+9.3f}{t['class']:>14}")
        ntriv = sum(1 for t in rows if t["class"] == "trivial")
        print(f"  -> {ntriv}/{len(rows)} triviaux (random ~ full), "
              f"{len(rows)-ntriv} discriminants")


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True, help="dossier des history_*.json")
    ap.add_argument("--bench_synthetic", default=None,
                    help="chemin vers bench_synthetic.py (pour importer SCENARIOS et calculer le full-supervision synthétique)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--final", action="store_true", help="utiliser la perf finale au lieu de l'AULC")
    args = ap.parse_args()

    if args.final:
        globals()["USE_FINAL"] = True

    perf = load_histories(args.hist)

    full = defaultdict(dict)
    if args.bench_synthetic:
        sys.path.insert(0, os.path.dirname(os.path.abspath(args.bench_synthetic)))
        import importlib.util
        spec = importlib.util.spec_from_file_location("bsy", args.bench_synthetic)
        bsy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bsy)
        full = compute_full_supervision_synthetic(bsy.SCENARIOS, range(args.seeds))

    tax = build_taxonomy(perf, full)
    if not tax:
        print("Aucune taxonomie construite. As-tu fourni --bench_synthetic pour le full-supervision ?")
    else:
        print_taxonomy(tax)
