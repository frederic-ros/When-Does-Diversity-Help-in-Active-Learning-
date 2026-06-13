# alframework

**A modular framework for Active Learning.**
Reproducible benchmarks on synthetic, real tabular, and deep latent representations,
together with a complete analysis pipeline that regenerates every table and figure
of the paper *"When Does Diversity Help in Active Learning? A Large-Scale Empirical
Study of Uncertainty, Diversity, and Regret."*

The project has **two complementary parts**:

1. **Benchmarks** (`benchmarks/`) — produce the active-learning *run histories*
   (one JSON per context), across the three regimes.
2. **Analysis pipeline** (`resultaanalysis/`) — consumes those histories and
   regenerates the leaderboard, the regret/collapse analysis, the structural trend,
   and all figures and tables.

The **heavy data** (generated histories + datasets) is **not** versioned here; it is
archived on Zenodo (see [Data](#data-zenodo)).

---

## Repository layout

```
alframework/
├── src/alframework/         # Main package (installable)
│   ├── core/                # Runner, state, registry, labeler
│   ├── data/                # Synthetic data generator
│   ├── strategies/          # The 18 AL strategies (+ V4x–V5x variants)
│   ├── utils/               # Curves, metrics, seeds
│   └── config/              # Strategy configuration
├── benchmarks/              # (1) PRODUCE the histories
│   ├── bench_synthetic.py   # Synthetic regime (16 scenarios)
│   ├── bench_real.py        # Real tabular regime (15 datasets)
│   ├── bench_latent.py      # Latent regime (ResNet-18 → PCA 10/20/50/100)
│   └── learning_curves_one_dataset.py
├── resultaanalysis/
│   └── resultanalysis/      # (2) ANALYZE the histories
│       ├── config.py        # Paths + constants (EDIT THIS)
│       ├── run_all.py       # Master script (orchestrates step1 → step10)
│       ├── step1_load.py    # parse histories → long.parquet
│       ├── step2_saturation.py
│       ├── step3_aulc.py
│       ├── step4_leaderboard.py   # Table 1
│       ├── step5_structure.py     # structural indicators (Table 6)
│       ├── step6_figures.py       # Figures 1–5
│       ├── step7_tables.py        # Tables 1,2,3,6 (LaTeX)
│       ├── step8_difficulty.py    # Table 4
│       ├── step9_classifier.py    # Table 5
│       └── step10_trend_robustness.py
├── tools/                   # Auxiliary analysis/visualization
├── tests/                   # Tests + small example datasets
├── legacy/                  # Historical code (reference)
├── pyproject.toml
├── CITATION.cff
└── LICENSE                  # MIT
```

---

## Installation

```bash
git clone https://github.com/<your-account>/alframework.git
cd alframework
pip install -e .
# with development dependencies (tests):
pip install -e ".[dev]"
```

Python ≥ 3.9. Main dependencies: numpy, pandas, scipy, scikit-learn, matplotlib,
pyarrow, statsmodels. The latent regime assumes ResNet-18-encoded features
(PyTorch); these features are provided directly via Zenodo, so **PyTorch is not
required to reproduce the analysis** from the histories.

---

## The three benchmarks

Each benchmark writes one history file per context, named
`history_{lr|rf}_{dataset}_split{N}_{strategy}.json`. The 18 strategies span five
families: uncertainty (margin, entropy, least_confident), pure diversity
(coreset_greedy, coreset_kmeanspp, typiclust, probcover), hybrid (badge_approx,
unc_feature_kmeans, diversity_optimized_batch), committee/disagreement (qbc,
robust_qbc, tri_committee, adaptive_disagreement), and integrated clustering (dbal,
rank2022, ActivePseudoLabelV58 — the router).

### (1) Synthetic — `bench_synthetic.py`

16 `make_classification` scenarios (separability, redundancy, imbalance,
multimodality, noise), two classifiers, budget 480 in steps of 20.

```bash
python benchmarks/bench_synthetic.py \
    --out ./results/synthetic \
    --seeds 20 \
    --models rf lr \
    --n-init 20 --batch-size 20 --max-budget 480
```

Useful arguments: `--seeds` (splits per cell; `3` for a smoke test), `--scenarios`
and `--panel` (restrict to chosen scenarios / strategies),
`--init-mode {random,random_safe,stratified}`.

### (2) Real tabular — `bench_real.py`

15 tabular datasets (OpenML), balanced strata then stratified splits, budget 400 in
steps of 20. Paths and parameters (data directory, number of splits, budget) are
defined in `main()`; by default the benchmark reads the `.txt` files under
`tests/benchmark_data/reeltabulaire/`.

```bash
python benchmarks/bench_real.py
```

To point at your own data, edit `bench_root` / the constants at the top of `main()`
(the `.txt` directory, `n_splits`, `max_budget`, `out`).

### (3) Latent — `bench_latent.py`

Latent representations of MNIST, Fashion-MNIST, and CIFAR-10 encoded with ResNet-18
and reduced by PCA (10/20/50/100 dimensions), budget 1200 in steps of 50. Like
`bench_real`, the configuration lives in `main()`.

```bash
python benchmarks/bench_latent.py
```

> The tabular and latent benchmarks are expensive. To merely **reproduce the
> paper's results**, you do not need to rerun them: use the provided histories on
> Zenodo directly (see below).

---

## Analyzing the results (reproducing the paper)

The analysis pipeline regenerates **all** tables and figures from the histories.

### 1. Get the data

Download the three archives from Zenodo (see [Data](#data-zenodo)) and extract them
into the following `data/` layout, at your run root:

```
data/
├── results/                 # histories (.json) — the 4 folders
│   ├── BenchSynthetique06062026/
│   ├── historiesltabulaire/
│   ├── historieslatentRF/
│   └── histroieslatetLR/
├── tab/reeltabulaire/       # tabular datasets (.txt)
└── latent/                  # latent datasets (.txt)
    ├── latent_pca10H3/  ├── latent_pca20H3/
    ├── latent_pca50H3/  └── latent_pca100H3/
```

### 2. Configure paths

Edit `resultaanalysis/resultanalysis/config.py` if your layout differs:

```python
DATA_ROOT   = Path("./data").resolve()
RESULTS_DIR = DATA_ROOT / "results"            # the 4 history folders
TAB_DIR     = DATA_ROOT / "tab" / "reeltabulaire"
LATENT_ROOT = DATA_ROOT / "latent"             # contains latent_pca{N}H3/
```

> **Windows:** use a raw string (`r"C:\..."`) or forward slashes (`/`) so that
> `\0`, `\t`, etc. do not break the paths.

### 3. Run

```bash
cd resultaanalysis/resultanalysis
python run_all.py                 # everything, in order (step1 → step10)
python run_all.py --skip-structure  # everything except step5 (the slowest part)
python run_all.py --from 5        # resume from a given step (reuses caches)
python run_all.py --only 8        # rerun a single step
```

Outputs are written under `out/`: `out/cache/` (intermediate parquet),
`out/figures/` (PNG), and `out/tables/` (LaTeX fragments).

### Step → paper artifact mapping

| Step | Produces | Artifact |
|------|----------|----------|
| 1 `step1_load`        | `long.parquet`        | (history parsing) |
| 2 `step2_saturation`  | `saturation.parquet`  | ante-saturation window |
| 3 `step3_aulc`        | `aulc_split.parquet`  | per-split AULC |
| 4 `step4_leaderboard` | `global_summary`      | **Table 1**, regret/flatness |
| 5 `step5_structure`   | `delta_structure`     | structural indicators |
| 6 `step6_figures`     | 5 PNGs                | **Figures 1–5** |
| 7 `step7_tables`      | `.tex`                | **Tables 1, 2, 3, 6** |
| 8 `step8_difficulty`  | `tab4_difficulty.tex` | **Table 4** |
| 9 `step9_classifier`  | `tab5_classifier.tex` | **Table 5** |
| 10 `step10_trend_robustness` | (console)      | Sec. 4.3 robustness |

### Reproducibility notes

- Steps **1–4 and 7–9 are exact** (they read the stored histories): leaderboard,
  regret, collapse, difficulty, and classifier effects reproduce identically.
- **Step 5** (and the structural parts of 6/10) **regenerates the pools** to measure
  the geometry of the uncertain region; it is reproducible *up to seeding*. The
  correlations and qualitative conclusions are stable.
- pandas compatibility: the pipeline works with pandas < 2.1 (`applymap`) and ≥ 2.1
  (`map`).

---

## Data (Zenodo)

The generated histories and the datasets are archived on Zenodo:

> **DOI: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)**

Three archives:

| Archive | Contents |
|---------|----------|
| `histories.zip` | All run histories (`.json`), one per (classifier, dataset, split, strategy) |
| `tabular_datasets.zip` | The 15 tabular datasets (`.txt`, label in the last column) |
| `latent_datasets.zip` | ResNet-18 → PCA 10/20/50/100 latent representations (`.txt`) |

Download all three, extract them into the `data/` layout above, then run
`run_all.py`.

---

## Citation

If you use this work, please cite both the paper and the data deposit. The
[`CITATION.cff`](CITATION.cff) file provides the software citation metadata.

```bibtex
@article{ros_diversity_active_learning,
  title   = {When Does Diversity Help in Active Learning? A Large-Scale Empirical
             Study of Uncertainty, Diversity, and Regret},
  author  = {Ros, Frederic and <co-authors>},
  journal = {Machine Learning (submitted)},
  year    = {2026}
}
```

## License

Code released under the **MIT** license (see [`LICENSE`](LICENSE)). Zenodo data
released under **CC-BY-4.0**.
