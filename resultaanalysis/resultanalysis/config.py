"""
config.py — shared paths, constants, and strategy metadata.

Edit DATA_ROOT (and the three sub-paths if your layout differs) to point at the
extracted benchmark data, then run `python run_all.py`.

Expected layout under DATA_ROOT (after extracting the provided .7z archives):

    DATA_ROOT/
      results/
        BenchSynthetique06062026/     history_{lr|rf}_{scenario}_split{n}_{strategy}.json
        historiesltabulaire/          history_{lr|rf}_{dataset}_split{n}_{strategy}.json
        historieslatentRF/<sub>/      history_rf_{dataset}_split{n}_{strategy}.json
        histroieslatetLR/<sub>/       history_lr_{dataset}_split{n}_{strategy}.json
      tab/reeltabulaire/              {dataset}.txt          (tab-separated, label = last col)
      latent/latent_pca{10,20,50,100}H3/   {dataset}.txt     (same format)

Every output is written under OUT_ROOT (cache parquet, figures, LaTeX tables).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS — edit these to match where you extracted the data.
# ---------------------------------------------------------------------------
DATA_ROOT = Path("./data").resolve()          # root of the extracted archives
RESULTS_DIR = DATA_ROOT / "results"           # the four history folders
TAB_DIR = DATA_ROOT / "tab" / "reeltabulaire"  # tabular .txt feature files
LATENT_ROOT = DATA_ROOT / "latent"             # contains latent_pca{N}H3/ subdirs

OUT_ROOT = Path("./out").resolve()
CACHE = OUT_ROOT / "cache"        # intermediate parquet tables
FIGDIR = OUT_ROOT / "figures"     # PNG figures
TABDIR = OUT_ROOT / "tables"      # LaTeX table fragments
for _d in (CACHE, FIGDIR, TABDIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ANALYSIS CONSTANTS
# ---------------------------------------------------------------------------
METRICS = ["f1_macro", "balanced_accuracy"]
PRIMARY_METRIC = "f1_macro"

# per-regime benchmark protocol (used only to regenerate pools for the
# structural analysis — the leaderboard/regret analysis reads histories only).
PROTO = {
    "synthetic": dict(n_init=20, test_size=0.30),
    "tabular":   dict(n_init=20, test_size=0.30, strate_size=3000, base_seed=42),
    "latent":    dict(n_init=50, test_size=0.30, strate_size=5000, base_seed=42),
}

# Strategy-name normalization across regimes (filenames differ slightly).
STRAT_MAP = {
    "V58B": "V58", "ActivePseudoLabelV58": "V58",
    "diversity_opt_batch": "diversity_optimized_batch",
}

def norm_strat(s: str) -> str:
    return STRAT_MAP.get(s, s)

# Strategy -> family.
FAMILY = {
    "random": "random",
    "margin": "uncertainty", "entropy": "uncertainty", "least_confident": "uncertainty",
    "coreset_greedy": "diversity", "coreset_kmeanspp": "diversity",
    "typiclust": "diversity", "probcover": "diversity",
    "badge_approx": "hybrid", "unc_feature_kmeans": "hybrid",
    "diversity_optimized_batch": "hybrid",
    "qbc": "committee", "robust_qbc": "committee",
    "tri_committee": "committee", "adaptive_disagreement": "committee",
    "dbal": "clustering", "rank2022": "clustering", "V58": "router",
}

# Pretty names + short family labels (for tables/figures).
PRETTY = {
    "margin": "Margin", "entropy": "Entropy", "least_confident": "Least-confident",
    "random": "Random", "coreset_greedy": "Core-set (greedy)",
    "coreset_kmeanspp": "Core-set (k-means++)", "typiclust": "TypiClust",
    "probcover": "ProbCover", "badge_approx": "BADGE",
    "unc_feature_kmeans": "Unc.+feat. k-means", "diversity_optimized_batch": "Div.-opt. batch",
    "qbc": "QBC", "robust_qbc": "Robust QBC", "tri_committee": "Tri-committee",
    "adaptive_disagreement": "Adapt. disagree", "dbal": "DBAL",
    "rank2022": "Rank-2022", "V58": "Router (ours)",
}
FAM_SHORT = {
    "uncertainty": "Unc.", "diversity": "Div.", "hybrid": "Hybrid",
    "committee": "Comm.", "clustering": "Clust.", "router": "Router", "random": "Rand.",
}

# The two diversity branches compared against margin in the structural analysis.
CLUSTERING_BRANCH = ["dbal", "rank2022", "V58"]
PUREDIV_BRANCH = ["coreset_greedy", "coreset_kmeanspp", "typiclust", "probcover"]

# Synthetic scenario configs (verbatim from bench_synthetic.py) — used to
# regenerate synthetic pools for the structural analysis.
SYNTH_SCENARIOS = {
 "easy": dict(n_samples=1500, n_classes=3, n_features=20, n_informative=12, n_redundant=4, class_sep=1.5, flip_y=0.01),
 "medium": dict(n_samples=1500, n_classes=3, n_features=20, n_informative=10, n_redundant=6, class_sep=1.2, flip_y=0.02),
 "hard": dict(n_samples=1500, n_classes=3, n_features=30, n_informative=10, n_redundant=15, class_sep=0.8, flip_y=0.05),
 "imbalanced": dict(n_samples=1500, n_classes=3, n_features=30, n_informative=12, n_redundant=8, class_sep=1.0, flip_y=0.03, weights=[0.70, 0.20, 0.10]),
 "imbalanced_hard": dict(n_samples=1500, n_classes=3, n_features=30, n_informative=10, n_redundant=10, class_sep=0.8, flip_y=0.05, weights=[0.75, 0.18, 0.07]),
 "clean20": dict(n_samples=1500, n_classes=3, n_features=20, n_informative=20, n_redundant=0, class_sep=1.0, flip_y=0.01),
 "redundant20": dict(n_samples=1500, n_classes=3, n_features=20, n_informative=8, n_redundant=8, class_sep=1.0, flip_y=0.02),
 "noisy_medium": dict(n_samples=1500, n_classes=3, n_features=30, n_informative=12, n_redundant=8, class_sep=1.0, flip_y=0.06),
 "medium5c": dict(n_samples=1500, n_classes=5, n_features=30, n_informative=15, n_redundant=8, class_sep=1.0, flip_y=0.03),
 "low_signal": dict(n_samples=1500, n_classes=3, n_features=30, n_informative=8, n_redundant=8, class_sep=0.6, flip_y=0.03),
 "many_classes": dict(n_samples=1500, n_classes=8, n_features=40, n_informative=20, n_redundant=10, class_sep=1.0, flip_y=0.03),
 "clustered_imbalanced": dict(n_samples=2000, n_classes=4, n_features=40, n_informative=14, n_redundant=10, n_repeated=2, n_clusters_per_class=3, class_sep=0.9, flip_y=0.03, weights=[0.55, 0.25, 0.15, 0.05]),
 "highdim_sparse": dict(n_samples=2000, n_classes=3, n_features=200, n_informative=12, n_redundant=20, n_repeated=0, class_sep=1.0, flip_y=0.02),
 "local_overlap": dict(n_samples=1800, n_classes=3, n_features=35, n_informative=14, n_redundant=10, n_clusters_per_class=4, class_sep=0.7, flip_y=0.02),
 "extreme_redundancy": dict(n_samples=1800, n_classes=3, n_features=80, n_informative=8, n_redundant=50, class_sep=1.0, flip_y=0.02),
 "rare_class": dict(n_samples=2500, n_classes=4, n_features=40, n_informative=15, n_redundant=10, class_sep=1.0, flip_y=0.01, weights=[0.80, 0.12, 0.06, 0.02]),
}

# latent PCA subdir -> dimension
LATENT_SUBDIRS = {"latent_pca10H3": 10, "latent_pca20H3": 20,
                  "latent_pca50H3": 50, "latent_pca100H3": 100}
