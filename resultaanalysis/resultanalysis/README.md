# Reproducibility package — *When Does Diversity Help in Active Learning?*

This folder reproduces **every table and figure in the paper** from the raw
benchmark histories and the dataset feature files. One master script
(`run_all.py`) orchestrates ten specialised, independently runnable modules.

## 1. Install

```bash
pip install -r requirements.txt
```

Tested with Python 3.10–3.12. Dependencies: numpy, pandas, scipy, scikit-learn,
matplotlib, pyarrow, statsmodels.

## 2. Place the data

Extract the provided archives so the layout under `./data` is:

```
data/
  results/
    BenchSynthetique06062026/     history_{lr|rf}_{scenario}_split{n}_{strategy}.json
    historiesltabulaire/          history_{lr|rf}_{dataset}_split{n}_{strategy}.json
    historieslatentRF/<sub>/       history_rf_{dataset}_split{n}_{strategy}.json
    histroieslatetLR/<sub>/        history_lr_{dataset}_split{n}_{strategy}.json
  tab/reeltabulaire/              {dataset}.txt          (tab-separated, label = last col)
  latent/latent_pca{10,20,50,100}H3/   {dataset}.txt
```

If your data lives elsewhere, edit `DATA_ROOT` (and the three sub-paths) at the top
of `config.py`. Nothing else needs editing.

Steps 1–4, 7–10 and most of step 6 need only the history JSONs. The `.txt` feature
files are needed only by step 5 (structural analysis), which regenerates the pools.

## 3. Run

```bash
python run_all.py                  # everything, in order (~3 min; step 5 dominates)
python run_all.py --skip-structure # everything except step 5 (the slow part)
python run_all.py --from 4         # resume from step 4 using cached parquet
python run_all.py --only 8         # re-run a single step
```

Each step is also runnable on its own, e.g. `python step8_difficulty.py`.

Outputs are written under `./out`:

```
out/cache/    intermediate parquet (long, aulc_split, global_summary, delta_structure, ...)
out/figures/  the 5 paper figures (PNG)
out/tables/   LaTeX table fragments
```

## 4. What maps to what in the paper

| Step | File | Produces | Paper artifact |
|------|------|----------|----------------|
| 1 | `step1_load.py` | `long.parquet` | (parse all histories) |
| 2 | `step2_saturation.py` | `saturation.parquet` | ante-saturation window (Sec. 3.3) |
| 3 | `step3_aulc.py` | `aulc_split.parquet` | windowed AULC, paired splits |
| 4 | `step4_leaderboard.py` | `global_summary.parquet` | **Table 1**, flatness/regret/router stats |
| 5 | `step5_structure.py` | `delta_structure.parquet`, `structure_indicators.parquet` | structural indicators (Sec. 4.3) |
| 6 | `step6_figures.py` | 5 PNGs | **Figs 1–5** (see below) |
| 7 | `step7_tables.py` | `tab1/2/3/6*.tex` | **Tables 1, 2, 3, 6** |
| 8 | `step8_difficulty.py` | `tab4_difficulty.tex`, `difficulty.parquet` | **Table 4** (difficulty tiers) |
| 9 | `step9_classifier.py` | `tab5_classifier.tex` | **Table 5** (LR vs RF) |
| 10 | `step10_trend_robustness.py` | console report | Sec. 4.3 robustness (partial Spearman / mixed-effects / leave-one-regime-out) |

Figures produced by step 6:

| File | Paper figure |
|------|--------------|
| `fig_spread.png`    | **Fig. 1** — across-strategy spread vs budget (ante-saturation) |
| `fig_risk.png`      | **Fig. 2** — risk vs performance summary (mean rank vs collapse) |
| `fig_curves.png`    | **Fig. 3** — representative learning curves (window shaded) |
| `fig_showcase.png`  | **Fig. 4** — 6 annotated curve panels, one per message |
| `fig_structure.png` | **Fig. 5** — pure-diversity gain vs structural indicators |

Table fragments produced (inlined or `\input`-able into the manuscript):

| Fragment | Paper table |
|----------|-------------|
| `tab1_leaderboard.tex`        | Table 1 — global leaderboard |
| `tab2_perregime.tex`          | Table 2 — mean rank per regime |
| `tab3_collapse.tex`           | Table 3 — collapse rate per regime |
| `tab4_difficulty.tex`         | Table 4 — mean rank by difficulty tier |
| `tab5_classifier.tex`         | Table 5 — mean rank by classifier (LR/RF) |
| `tab6_struct_purediv.tex`     | Table 6 — pure-diversity gain vs structure |
| `tab6b_struct_clustering.tex` | (clustering-branch counterpart; cited as "not shown" in text) |

## 5. Reproducibility notes

- **Steps 1–4 and 7–9 are exact**: they read the stored histories, so the
  leaderboard, per-regime ranks, collapse, regret, flatness, router, difficulty
  and classifier numbers reproduce bit-for-bit.
- **Step 5 (and the structure parts of steps 6/10) is reproduced, not bit-exact**:
  it regenerates the unlabeled pools to measure uncertain-region geometry.
  Synthetic pools use the exact `make_classification` configs and the
  `seed = 42 + 17·split` schedule; tabular/latent mirror the original
  preprocessing (stratified strata, `StandardScaler` on train, stratified init).
  The Spearman correlations and every qualitative conclusion are stable to this.
- **Step 10** prints the three robustness checks quoted in Sec. 4.3. The
  mixed-effects model may emit benign convergence warnings (only three regime
  groups); the partial Spearman and leave-one-regime-out carry the argument
  independently.
- The **pairing key** that licenses the paired tests: within a context cell the
  split index equals the seed index, fixed identically across strategies by the
  benchmark's seed schedule.

## 6. Wiring the outputs into the manuscript

The manuscript (`paper.tex` for elsarticle/DMKD, `paper_MLJ.tex` for Springer MLJ)
already inlines the table values, so you do not strictly need the `out/tables`
fragments — they are provided so you can regenerate/verify the numbers. The
figures in `out/figures` are the ones the manuscript `\includegraphics`-es; copy
them next to the `.tex` (or adjust the paths) before compiling.
