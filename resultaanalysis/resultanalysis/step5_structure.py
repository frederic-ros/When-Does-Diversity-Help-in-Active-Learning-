"""
step5_structure.py — regenerate pools and compute structural indicators of the
uncertain region for all three regimes, then correlate with diversity gain.

For each (regime, clf, dataset) it reproduces that regime's preprocessing,
fits the classifier on a small stratified init, identifies the lowest-margin 20%
of the pool (the "uncertain region"), and measures:
  low_margin_frac, mean_margin, eff_dim_unc (participation ratio),
  eff_dim_ratio, clusterability, n_classes.
It then computes, per cell, the gain over margin of the pure-diversity branch and
of the integrated clustering branch (using the windowed AULC from step3), and
reports Spearman correlations per regime and pooled (with within-regime control).

Inputs:  out/cache/aulc_split.parquet  (for the deltas)
         DATA: synthetic regenerated in-code; TAB_DIR / LATENT_ROOT .txt files
Outputs: out/cache/structure_indicators.parquet
         out/cache/delta_structure.parquet
Run standalone:  python step5_structure.py
"""
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

import config as C

warnings.filterwarnings("ignore")


# ----------------------------- shared helpers -----------------------------
def _model(kind, seed):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=50, random_state=seed)
    return LogisticRegression(max_iter=2000, random_state=seed)


def _eff_dim(X):
    if len(X) < 3:
        return np.nan
    ev = np.linalg.eigvalsh(np.cov(X.T))
    ev = ev[ev > 1e-12]
    return float((ev.sum() ** 2) / np.square(ev).sum()) if len(ev) else np.nan


def _geom(Xpool, P):
    Ps = np.sort(P, axis=1)
    margin = Ps[:, -1] - Ps[:, -2]
    k = max(10, int(0.2 * len(Xpool)))
    unc = Xpool[np.argsort(margin)[:k]]
    ed = _eff_dim(unc)
    try:
        nk = min(4, max(2, len(unc) // 50))
        km = KMeans(n_clusters=nk, n_init=3, random_state=0).fit(unc)
        within = km.inertia_ / len(unc)
        overall = np.square(unc - unc.mean(0)).sum() / len(unc)
        clu = 1 - within / overall if overall > 0 else 0.0
    except Exception:
        clu = np.nan
    return dict(low_margin_frac=float((margin < 0.1).mean()),
                mean_margin=float(margin.mean()),
                eff_dim_unc=ed, eff_dim_ratio=ed / Xpool.shape[1],
                clusterability=clu)


def _stratified_init(ytr, n_init, seed):
    rng = np.random.default_rng(seed)
    cl = np.unique(ytr)
    per = max(1, n_init // len(cl))
    idx = []
    for c in cl:
        ci = np.flatnonzero(ytr == c)
        idx += list(rng.choice(ci, size=min(per, len(ci)), replace=False))
    return np.array(idx)


def _load_txt(fp):
    """Tab-separated, last column = label (used for tabular & latent)."""
    df = pd.read_csv(fp, sep="\t", header=None, dtype=str, engine="python")
    df = df.dropna(how="all").reset_index(drop=True)
    # compatible pandas <2.1 (applymap) et >=2.1 (map)
    _elementwise = df.map if hasattr(df, "map") else df.applymap
    df = _elementwise(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.replace({"": np.nan, "nan": np.nan, "NaN": np.nan,
                     "None": np.nan, "NULL": np.nan, "?": np.nan})
    lab = df.shape[1] - 1
    y_raw = df.iloc[:, lab].astype(str).str.strip()
    bad = y_raw.isna() | (y_raw == "") | y_raw.str.lower().isin(["nan", "none", "null", "?"])
    df = df.loc[~bad].reset_index(drop=True)
    y_raw = y_raw.loc[~bad].reset_index(drop=True)
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(y_raw.to_numpy()).astype(int)
    feats = [c for c in range(df.shape[1]) if c != lab]
    parts = []
    for c in feats:
        s = df[c].astype(str).str.strip().str.replace(",", ".", regex=False)
        num = pd.to_numeric(s, errors="coerce")
        nz = s.replace({"nan": np.nan}).notna()
        if num.notna().sum() / max(1, int(nz.sum())) >= 0.80:
            parts.append(num.to_numpy(float).reshape(-1, 1))
        else:
            codes, _ = pd.factorize(s.fillna("__M__"), sort=True)
            parts.append(codes.astype(float).reshape(-1, 1))
    X = np.hstack(parts).astype(float)
    X = SimpleImputer(strategy="median").fit_transform(X)
    return X, y


def _make_strata(y, n_strates, size, seed):
    rng = np.random.default_rng(seed)
    cl, cnt = np.unique(y, return_counts=True)
    size = min(size, len(y))
    raw = (cnt / cnt.sum()) * size
    cs = np.floor(raw).astype(int)
    for j in np.argsort(-(raw - cs))[: size - cs.sum()]:
        cs[j] += 1
    out = []
    for _ in range(n_strates):
        idx = []
        for c, n in zip(cl, cs):
            if n <= 0:
                continue
            cand = np.flatnonzero(y == c)
            idx.append(rng.choice(cand, size=int(n), replace=int(n) > len(cand)))
        i = np.concatenate(idx)
        rng.shuffle(i)
        out.append(i)
    return out


# ----------------------------- per-regime extractors -----------------------------
def _synthetic_rows(seeds=range(0, 6)):
    p = C.PROTO["synthetic"]
    rows = []
    for sc, cfg in C.SYNTH_SCENARIOS.items():
        for kind in ["lr", "rf"]:
            recs = []
            for s in seeds:
                seed = 42 + s * 17
                X, y = make_classification(random_state=seed, **cfg)
                Xtr, _, ytr, _ = train_test_split(
                    X, y, test_size=p["test_size"], random_state=seed, stratify=y)
                init = _stratified_init(ytr, p["n_init"], seed)
                mask = np.ones(len(Xtr), bool)
                mask[init] = False
                m = _model(kind, seed).fit(Xtr[init], ytr[init])
                recs.append(_geom(Xtr[mask], m.predict_proba(Xtr[mask])))
            row = pd.DataFrame(recs).mean(numeric_only=True).to_dict()
            row.update(regime="synthetic", dataset=sc, clf=kind,
                       n_classes=cfg["n_classes"], n_features=cfg["n_features"])
            rows.append(row)
    return rows


def _txt_rows(regime, files, n_strates, strate_size, n_init, n_splits=2):
    p = C.PROTO[regime]
    base = p["base_seed"]
    rows = []
    for fp in files:
        name = Path(fp).stem
        X, y = _load_txt(fp)
        ncls = len(np.unique(y))
        strata = _make_strata(y, n_strates, strate_size, base)
        for kind in ["lr", "rf"]:
            recs = []
            for sid, sidx in enumerate(strata):
                Xs, ys = X[sidx], y[sidx]
                sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=p["test_size"],
                                             random_state=base + 10000 * sid)
                for spid, (tr, _te) in enumerate(sss.split(Xs, ys)):
                    sc = StandardScaler()
                    Xtr = sc.fit_transform(Xs[tr])
                    ytr = ys[tr]
                    init = _stratified_init(ytr, n_init, base + 10000 * sid + 2000 * spid + 123)
                    mask = np.ones(len(Xtr), bool)
                    mask[init] = False
                    m = _model(kind, base).fit(Xtr[init], ytr[init])
                    recs.append(_geom(Xtr[mask], m.predict_proba(Xtr[mask])))
            row = pd.DataFrame(recs).mean(numeric_only=True).to_dict()
            row.update(regime=regime, dataset=name, clf=kind,
                       n_classes=ncls, n_features=X.shape[1])
            rows.append(row)
    return rows


def _build_indicators():
    rows = []
    print("[step5] synthetic pools ...")
    rows += _synthetic_rows()
    print("[step5] tabular pools ...")
    tab_files = sorted(glob.glob(str(C.TAB_DIR / "*.txt")))
    rows += _txt_rows("tabular", tab_files, n_strates=3, strate_size=3000,
                      n_init=20, n_splits=2)
    print("[step5] latent pools ...")
    lat_files = []
    for sub in C.LATENT_SUBDIRS:
        lat_files += sorted(glob.glob(str(C.LATENT_ROOT / sub / "*.txt")))
    rows += _txt_rows("latent", lat_files, n_strates=3, strate_size=5000,
                      n_init=50, n_splits=1)
    ST = pd.DataFrame(rows)
    ST.to_parquet(C.CACHE / "structure_indicators.parquet")
    return ST


def _build_deltas(ST):
    S = pd.read_parquet(C.CACHE / "aulc_split.parquet")
    out = []
    for regime in ["synthetic", "tabular", "latent"]:
        sub = S[(S.regime == regime) & (S.metric == "f1_macro")]
        cell = (sub.groupby(["dataset", "clf", "strategy"])["aulc_adaptive"]
                .mean().unstack("strategy"))
        clus = cell[C.CLUSTERING_BRANCH].max(axis=1)
        divp = cell[C.PUREDIV_BRANCH].max(axis=1)
        d = pd.DataFrame({"delta_clustering": clus - cell["margin"],
                          "delta_purediv": divp - cell["margin"]}).reset_index()
        d["regime"] = regime
        out.append(d)
    D = pd.concat(out, ignore_index=True).merge(
        ST, on=["regime", "dataset", "clf"], how="inner")
    D.to_parquet(C.CACHE / "delta_structure.parquet")
    return D


def _report(D):
    feats = ["low_margin_frac", "mean_margin", "eff_dim_unc", "clusterability", "n_classes"]

    def sp(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return ("--", 1.0)
        r, p = spearmanr(x, y)
        return (f"{r:+.2f}", p)

    print("\n[step5] Pure-diversity gain vs structure (Spearman rho), per regime + pooled:")
    print(f"   {'indicator':>16} | synth   tab     latent  pooled")
    for f in feats:
        cells = []
        for rg in ["synthetic", "tabular", "latent"]:
            s = D[D.regime == rg]
            cells.append(sp(s[f], s["delta_purediv"])[0])
        pooled = sp(D[f], D["delta_purediv"])[0]
        print(f"   {f:>16} | {cells[0]:>6} {cells[1]:>6} {cells[2]:>6}  {pooled:>6}")

    # within-regime standardized pooled (the robust law)
    def wz(df, col):
        return df.groupby("regime")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    Dz = D.copy()
    print("\n[step5] Within-regime standardized pooled (pure-diversity):")
    for f in feats:
        if D.groupby("regime")[f].std().min() == 0:
            print(f"   {f:>16}: -- (constant within a regime)")
            continue
        rho, p = spearmanr(wz(Dz, f), wz(Dz, "delta_purediv"))
        print(f"   {f:>16}: rho={rho:+.3f}  p={p:.3f}")


def run():
    ST = _build_indicators()
    print(f"[step5] indicators for {len(ST)} cells -> structure_indicators.parquet")
    D = _build_deltas(ST)
    print(f"[step5] deltas merged for {len(D)} cells -> delta_structure.parquet")
    _report(D)
    return ST, D


if __name__ == "__main__":
    run()
