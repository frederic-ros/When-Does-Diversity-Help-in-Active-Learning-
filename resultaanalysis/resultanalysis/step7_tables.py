"""
step7_tables.py — generate LaTeX table fragments used by paper.tex.

  tab1_leaderboard.tex      : global leaderboard (18 strategies)
  tab2_perregime.tex        : mean rank per regime
  tab3_collapse.tex         : collapse rate per regime
  tab6_struct_purediv.tex   : pure-diversity gain vs structure (Spearman)
  tab6b_struct_clustering.tex: clustering-family gain vs structure (Spearman)

Inputs:  out/cache/aulc_split.parquet, global_summary.parquet, delta_structure.parquet
Outputs: out/tables/*.tex
Run standalone:  python step7_tables.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C

WIN = "aulc_adaptive"


def _f(x, d=2):
    return f"{x:.{d}f}"


def tab1_leaderboard():
    G = pd.read_parquet(C.CACHE / "global_summary.parquet").copy()
    G["name"] = G["strategy"].map(C.PRETTY)
    G["fam"] = G["family"].map(C.FAM_SHORT)
    G = G.sort_values("rank_f1")
    rows = [
        f"{r.name} & {r.fam} & {_f(r.rank_f1)} & {_f(r.rank_ba)} & "
        f"{_f(r.win_f1*100,0)}\\% & {_f(r.reg_f1,3)} & {_f(r.p90_f1,3)} & "
        f"{_f(r.coll_f1*100,0)}\\% \\\\"
        for r in G.itertuples()
    ]
    (C.TABDIR / "tab1_leaderboard.tex").write_text("\n".join(rows))


def _cellrank(S, metric):
    c = (S[S.metric == metric]
         .groupby(["regime", "clf", "dataset", "strategy"])[WIN].mean().reset_index())
    grp = c.groupby(["regime", "clf", "dataset"])
    c["rank"] = grp[WIN].rank(ascending=False)
    c["worst3"] = grp[WIN].rank(ascending=True) <= 3
    return c


def tab2_perregime():
    S = pd.read_parquet(C.CACHE / "aulc_split.parquet")
    c = _cellrank(S, "f1_macro")
    piv = c.groupby(["strategy", "regime"])["rank"].mean().unstack("regime")
    piv["overall"] = piv.mean(axis=1)
    piv = piv.sort_values("overall")
    piv.index = [C.PRETTY[i] for i in piv.index]
    rows = [f"{i} & {_f(r.synthetic)} & {_f(r.tabular)} & {_f(r.latent)} & {_f(r.overall)} \\\\"
            for i, r in piv.iterrows()]
    (C.TABDIR / "tab2_perregime.tex").write_text("\n".join(rows))


def tab3_collapse():
    S = pd.read_parquet(C.CACHE / "aulc_split.parquet")
    c = _cellrank(S, "f1_macro")
    cr = c.groupby(["strategy", "regime"])["worst3"].mean().unstack("regime")
    cr["overall"] = cr.mean(axis=1)
    cr = cr.sort_values("overall")
    cr.index = [C.PRETTY[i] for i in cr.index]
    rows = [f"{i} & {_f(r.synthetic*100,0)}\\% & {_f(r.tabular*100,0)}\\% & "
            f"{_f(r.latent*100,0)}\\% & {_f(r.overall*100,0)}\\% \\\\"
            for i, r in cr.iterrows()]
    (C.TABDIR / "tab3_collapse.tex").write_text("\n".join(rows))


def _struct_table(target, fname):
    fp = C.CACHE / "delta_structure.parquet"
    if not fp.exists():
        print(f"[step7] skipping {fname} (run step 5 first)")
        return
    M = pd.read_parquet(fp)

    def sp(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return "--"
        r, p = spearmanr(x, y)
        star = "" if p >= 0.05 else ("*" if p >= 0.01 else ("**" if p >= 0.001 else "***"))
        return f"{r:+.2f}{star}"

    feats = [("low_margin_frac", "Low-margin frac."), ("mean_margin", "Mean margin"),
             ("eff_dim_unc", "Eff. dim (unc.)"), ("clusterability", "Clusterability"),
             ("n_classes", "\\#classes")]
    rows = []
    for key, lbl in feats:
        syn, tab, lat = (M[M.regime == r] for r in ["synthetic", "tabular", "latent"])
        rows.append(f"{lbl} & {sp(syn[key], syn[target])} & {sp(tab[key], tab[target])} & "
                    f"{sp(lat[key], lat[target])} & {sp(M[key], M[target])} \\\\")
    (C.TABDIR / fname).write_text("\n".join(rows))


def run():
    tab1_leaderboard()
    tab2_perregime()
    tab3_collapse()
    _struct_table("delta_purediv", "tab6_struct_purediv.tex")
    _struct_table("delta_clustering", "tab6b_struct_clustering.tex")
    print(f"[step7] wrote 5 LaTeX table fragments -> {C.TABDIR}")


if __name__ == "__main__":
    run()
