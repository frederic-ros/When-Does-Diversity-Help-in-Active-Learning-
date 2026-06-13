"""
step6_figures.py — generate the three paper figures.

  fig_spread.png    : across-strategy spread vs budget fraction (ante-saturation)
  fig_curves.png    : representative learning curves with the window shaded
  fig_structure.png : pure-diversity gain vs structural indicators (3 regimes)
  fig_risk.png      : risk-vs-performance summary (mean rank vs collapse rate)
Inputs:  out/cache/long.parquet, saturation.parquet, delta_structure.parquet
Outputs: out/figures/*.png
Run standalone:  python step6_figures.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import config as C

plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
REGIME_COL = {"synthetic": "#2ca02c", "tabular": "#1f77b4", "latent": "#d62728"}


def fig_spread():
    df = pd.read_parquet(C.CACHE / "long.parquet")
    sep = []
    for (regime, clf, ds), g in df.groupby(["regime", "clf", "dataset"]):
        m = g.groupby(["strategy", "n_labeled"])[C.PRIMARY_METRIC].mean()
        bmax = g["n_labeled"].max()
        for n, sub in m.groupby(level="n_labeled"):
            sep.append({"regime": regime, "bf": n / bmax,
                        "spread": sub.values.max() - sub.values.min()})
    sep = pd.DataFrame(sep)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for r, c in REGIME_COL.items():
        s = sep[sep.regime == r]
        b = pd.cut(s.bf, np.linspace(0, 1, 11))
        gp = s.groupby(b, observed=True)["spread"].mean()
        ax.plot([iv.mid for iv in gp.index], gp.values, "o-", color=c, label=r, lw=2, ms=6)
    ax.set_xlabel("budget fraction", fontsize=12)
    ax.set_ylabel(f"mean across-strategy spread ({C.PRIMARY_METRIC})", fontsize=12)
    ax.set_title("Strategy separation peaks early, decays toward saturation",
                 fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.FIGDIR / "fig_spread.png", bbox_inches="tight")
    plt.close(fig)
    print("[step6] fig_spread.png")


def fig_curves():
    df = pd.read_parquet(C.CACHE / "long.parquet")
    sat = pd.read_parquet(C.CACHE / "saturation.parquet")
    groups = {"margin": ("#1f77b4", "-"), "dbal": ("#d62728", "-"),
              "rank2022": ("#ff7f0e", "-"), "V58": ("#2ca02c", "-"),
              "random": ("#999999", "--"), "probcover": ("#9467bd", ":"),
              "robust_qbc": ("#8c564b", ":")}

    def panel(ax, regime, clf, ds):
        g = df[(df.regime == regime) & (df.clf == clf) & (df.dataset == ds)]
        if g.empty:
            ax.set_visible(False); return
        for st, (col, ls) in groups.items():
            s = g[g.strategy == st].groupby("n_labeled")[C.PRIMARY_METRIC].mean()
            if s.empty:
                continue
            ax.plot(s.index, s.values, color=col, ls=ls, label=st, lw=1.6)
        row = sat[(sat.regime == regime) & (sat.clf == clf) & (sat.dataset == ds)]
        if len(row) and row.n95.values[0]:
            ax.axvline(row.n95.values[0], color="k", ls="--", alpha=0.4, lw=1)
            ax.axvspan(g.n_labeled.min(), row.n95.values[0], color="gold", alpha=0.07)
        ax.set_title(f"{regime}/{clf}/{ds}", fontsize=8)
        ax.set_xlabel("# labeled"); ax.grid(alpha=0.3)

    # pick representative cells that exist in the data
    candidates = [("synthetic", "rf", "many_classes"), ("synthetic", "lr", "clean20"),
                  ("tabular", "rf", "letter_recognition_original"), ("tabular", "lr", "rice_original"),
                  ("latent", "lr", "cifar10trainpca100"), ("latent", "rf", "minsttrainpca10")]
    present = set(map(tuple, df[["regime", "clf", "dataset"]].drop_duplicates().values))
    cells = [c for c in candidates if c in present][:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, cell in zip(axes.flat, cells):
        panel(ax, *cell)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(); fig.savefig(C.FIGDIR / "fig_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("[step6] fig_curves.png")


def fig_structure():
    fp = C.CACHE / "delta_structure.parquet"
    if not fp.exists():
        print("[step6] skipping fig_structure (run step 5 first)")
        return
    M = pd.read_parquet(fp)
    panels = [("low_margin_frac", "Pure-diversity gain vs\nlow-margin fraction"),
              ("mean_margin", "Pure-diversity gain vs\nmean margin"),
              ("clusterability", "Pure-diversity gain vs\nclusterability of uncertain region")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (xf, title) in zip(axes, panels):
        for r, c in REGIME_COL.items():
            s = M[M.regime == r]
            ax.scatter(s[xf], s["delta_purediv"], c=c, label=r, alpha=0.75, s=34)
        rho, p = spearmanr(M[xf], M["delta_purediv"])
        z = np.polyfit(M[xf], M["delta_purediv"], 1)
        xs = np.linspace(M[xf].min(), M[xf].max(), 50)
        ax.plot(xs, np.polyval(z, xs), "k--", alpha=0.5, lw=1)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlabel(xf, fontsize=12); ax.set_ylabel("delta_purediv", fontsize=12)
        ax.set_title(f"{title}\nSpearman rho={rho:+.2f} (p={p:.3f})",
                     fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(C.FIGDIR / "fig_structure.png", bbox_inches="tight")
    plt.close(fig)
    print("[step6] fig_structure.png")


def fig_risk():
    """THE summary figure: risk vs performance. x=mean rank, y=collapse rate,
    size=mean regret, colour=family. Bottom-left = good AND safe."""
    from matplotlib.lines import Line2D
    G = pd.read_parquet(C.CACHE / "global_summary.parquet")
    FAMC = {"uncertainty": "#1f77b4", "diversity": "#d62728", "hybrid": "#9467bd",
            "committee": "#8c564b", "clustering": "#2ca02c", "router": "#2ca02c",
            "random": "#7f7f7f"}
    FAML = {"uncertainty": "Uncertainty", "diversity": "Pure diversity",
            "hybrid": "Hybrid", "committee": "Committee",
            "clustering": "Integrated clustering", "router": "Router (ours)",
            "random": "Random"}
    OFF = {"dbal": (0.25, 0.025), "rank2022": (0.0, -0.04), "V58": (-0.25, 0.04),
           "diversity_optimized_batch": (-0.25, -0.035), "margin": (0, -0.045),
           "unc_feature_kmeans": (-0.25, 0.03), "qbc": (0.25, -0.025),
           "tri_committee": (0.25, 0.028), "badge_approx": (-0.25, 0.04),
           "least_confident": (0.25, 0.035), "coreset_kmeanspp": (0.25, -0.025),
           "entropy": (0.25, 0.018), "random": (0.25, -0.02), "typiclust": (0.25, 0.025),
           "adaptive_disagreement": (-0.25, -0.045), "coreset_greedy": (0.25, -0.015),
           "probcover": (-0.3, 0.045), "robust_qbc": (0.0, -0.06)}
    fig, ax = plt.subplots(figsize=(9, 6.5))
    xsplit, ysplit = 9.0, 0.25
    ax.fill_between([2, xsplit], -0.05, ysplit, color="#2ca02c", alpha=0.06, zorder=0)
    ax.fill_between([xsplit, 17], ysplit, 0.85, color="#d62728", alpha=0.06, zorder=0)
    for r in G.itertuples():
        ax.scatter(r.rank_f1, r.coll_f1, s=40 + r.reg_f1 * 9000,
                   color=FAMC.get(r.family, "#333"), alpha=0.8,
                   edgecolor="white", linewidth=1.2, zorder=3)
    for r in G.itertuples():
        dx, dy = OFF.get(r.strategy, (0.2, 0.02))
        ax.annotate(C.PRETTY[r.strategy], (r.rank_f1, r.coll_f1),
                    xytext=(r.rank_f1 + dx, r.coll_f1 + dy), fontsize=8,
                    ha="left" if dx >= 0 else "right", va="center", color="#222")
    ax.text(3.0, 0.02, "SAFE & STRONG", fontsize=9, color="#2ca02c",
            fontweight="bold", alpha=0.7)
    ax.text(16.7, 0.80, "RISKY & WEAK", fontsize=9, color="#d62728",
            fontweight="bold", alpha=0.7, ha="right")
    ax.set_xlabel(r"Mean rank  (performance, lower is better) $\rightarrow$ worse", fontsize=10.5)
    ax.set_ylabel(r"Collapse rate  (worst-3 of 18) $\rightarrow$ riskier", fontsize=10.5)
    ax.set_title("Risk vs Performance across 86 contexts\n(point size $\\propto$ mean regret)",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlim(2.5, 17); ax.set_ylim(-0.05, 0.85); ax.grid(alpha=0.25, zorder=0)
    fam_handles = [Line2D([0], [0], marker="o", ls="", color=FAMC[k], label=FAML[k],
                          ms=9, mec="white")
                   for k in ["clustering", "uncertainty", "hybrid", "committee",
                             "diversity", "random"]]
    leg1 = ax.legend(handles=fam_handles, loc="upper left", fontsize=8.5,
                     title="Family", title_fontsize=9, framealpha=0.9)
    ax.add_artist(leg1)
    size_handles = [Line2D([0], [0], marker="o", ls="", color="#999", mec="white",
                    ms=np.sqrt((40 + v * 9000) / np.pi) * 0.8, label=f"{v:.02f}")
                    for v in [0.006, 0.03, 0.08]]
    ax.legend(handles=size_handles, loc="lower right", fontsize=8.5,
              title="Mean regret", title_fontsize=9, labelspacing=1.4,
              framealpha=0.9, borderpad=1.0)
    fig.tight_layout(); fig.savefig(C.FIGDIR / "fig_risk.png", bbox_inches="tight")
    plt.close(fig)
    print("[step6] fig_risk.png")


def fig_showcase():
    """Paper Figure 3: 6 representative learning curves, one per message.
    Skips any panel whose context cell is absent from the data."""
    from matplotlib.lines import Line2D
    df = pd.read_parquet(C.CACHE / "long.parquet")
    sat = pd.read_parquet(C.CACHE / "saturation.parquet")
    STYLE = {"margin": ("#1f77b4", "-", 2.0, "Margin (uncertainty)"),
             "dbal": ("#d62728", "-", 1.7, "DBAL"),
             "rank2022": ("#ff7f0e", "-", 1.7, "Rank-2022"),
             "V58": ("#2ca02c", "-", 1.7, "Router (ours)"),
             "random": ("#7f7f7f", "--", 1.5, "Random"),
             "probcover": ("#9467bd", ":", 1.7, "ProbCover"),
             "robust_qbc": ("#8c564b", ":", 1.7, "Robust QBC"),
             "typiclust": ("#e377c2", ":", 1.7, "TypiClust")}
    # (regime, clf, dataset, title, message, extra strategies to emphasize)
    CASES = [
        ("tabular", "rf", "covertype_original", "Tabular / RF / covertype",
         "Diversity helps: clustering clearly above margin", ["typiclust"]),
        ("tabular", "rf", "letter_recognition_original", "Tabular / RF / letter",
         "Many classes: clustering pulls ahead, no saturation", ["typiclust"]),
        ("latent", "rf", "cifar10trainpca100", "Latent / RF / CIFAR-10 PCA-100",
         "Total collapse of density/coverage methods", ["probcover", "robust_qbc"]),
        ("latent", "rf", "minsttrainpca100", "Latent / RF / MNIST PCA-100",
         "ProbCover & Robust QBC fall below random", ["probcover", "robust_qbc"]),
        ("synthetic", "lr", "noisy_medium", "Synthetic / LR / noisy_medium",
         "Flat top: clustering == margin (within noise)", []),
        ("tabular", "lr", "rice_original", "Tabular / LR / rice",
         "Saturated: AL barely beats random", []),
    ]
    present = set(map(tuple, df[["regime", "clf", "dataset"]].drop_duplicates().values))

    def panel(ax, regime, clf, ds, title, msg, emph):
        if (regime, clf, ds) not in present:
            ax.set_visible(False)
            return
        g = df[(df.regime == regime) & (df.clf == clf) & (df.dataset == ds)]
        for st in ["margin", "dbal", "rank2022", "V58", "random"] + emph:
            if st not in STYLE:
                continue
            s = g[g.strategy == st].groupby("n_labeled")[C.PRIMARY_METRIC].agg(["mean", "std"])
            if s.empty:
                continue
            col, ls, lw, _ = STYLE[st]
            ax.plot(s.index, s["mean"], color=col, ls=ls, lw=lw)
            if st in ("margin",) + tuple(emph):
                ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"],
                                color=col, alpha=0.08)
        row = sat[(sat.regime == regime) & (sat.clf == clf) & (sat.dataset == ds)]
        if len(row) and row.n95.values[0]:
            ax.axvspan(g.n_labeled.min(), row.n95.values[0], color="gold", alpha=0.08)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=22)
        ax.annotate(msg, xy=(0.5, 1.02), xycoords="axes fraction", ha="center",
                    va="bottom", fontsize=10, style="italic", color="#444")
        ax.set_xlabel("# labeled", fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5))
    for ax, cs in zip(axes.flat, CASES):
        panel(ax, *cs)
    axes.flat[0].set_ylabel("macro-F1", fontsize=11)
    axes.flat[3].set_ylabel("macro-F1", fontsize=11)
    handles = [Line2D([0], [0], color=STYLE[s][0], ls=STYLE[s][1], lw=2, label=STYLE[s][3])
               for s in ["margin", "dbal", "rank2022", "V58", "random",
                         "probcover", "robust_qbc", "typiclust"]]
    fig.legend(handles=handles, loc="upper center", ncol=8, fontsize=10,
               bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.suptitle("Representative learning curves: when diversity helps, and when it collapses",
                 fontsize=15, fontweight="bold", y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=3.0)
    fig.savefig(C.FIGDIR / "fig_showcase.png", bbox_inches="tight")
    plt.close(fig)
    print("[step6] fig_showcase.png")


def run():
    fig_spread()
    fig_curves()
    fig_structure()
    fig_showcase()
    fig_risk()


if __name__ == "__main__":
    run()
