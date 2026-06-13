# -*- coding: utf-8 -*-
"""
Analyse un répertoire de fichiers history_*.json issus d'expériences d'active learning.

Compatible Spyder :
- paramètres à modifier en haut du fichier
- lancement direct avec Run
- export CSV + figures

NOUVEAUTES :
- résumé global et par difficulté pour une stratégie cible
- % win
- % top-2
- % top-3
- gestion correcte si seulement 1 ou 2 méthodes
- possibilité de charger un fichier difficulty_map.csv
- sinon génération automatique de la difficulté à partir des accuracies
- boxplots par budget
- courbes multi-budgets par méthode
- heatmap méthode vs budget
- win-rates par budget pour la stratégie focus
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


# ============================================================================
# PARAMETRES A MODIFIER DANS SPYDER
# ============================================================================
LEARNING_CURVE_USE_INTERPOLATION = True
BASE_DIR = Path(__file__).resolve().parent.parent  # tools/ → alframework_clean/

#dirset = "real2"
#dirset = "latent_pca100"
#dirset = "latent_pca100H3_dualselect_weights_hd_init10"
#dirset="latent_pca100H3_test"
#dirset="bench_results_real_strates_stratified/histories/testnewreel"
#dirset="bench_results_real_strates_stratified/histories/newreel"
dirset="tests/bench_results_real_strates_stratified/histories/newreelLR"
INPUT_DIR  = BASE_DIR / dirset
OUTPUT_DIR = BASE_DIR / dirset / "analyze_out_base"

FOCUS_STRATEGY = "dual_select"
BUDGETS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 
           1100, 1200, 1300, 1400, 1500, 1550]
#BUDGETS = [0, 100, 200, 300, 400, 500, 600]

LEARNING_CURVE_STRATEGIES = None

# Fichier optionnel de difficulté
# Si absent, un fichier auto sera généré à partir des accuracies moyennes
DIFFICULTY_CSV = BASE_DIR / "tests" / dirset / "difficulty_map.csv"


# ============================================================================
# OUTILS
# ============================================================================

FILENAME_RE = re.compile(
    r"^history_(?P<dataset>.+)_split(?P<split>\d+)_(?P<strategy>.+)\.json$"
)

def safe_name(text):
    return re.sub(r'[\\/*?:"<>| ]+', "_", str(text))


def filter_strategies(df, strategies=None):
    """
    Filtre les stratégies.
    - None -> garde tout
    - str -> garde uniquement cette stratégie
    - list/tuple/set -> garde uniquement celles de la collection
    """
    if strategies is None:
        return df.copy()

    if isinstance(strategies, str):
        strategies = [strategies]

    strategies = set(strategies)
    return df[df["strategy"].isin(strategies)].copy()


def safe_float(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


# ============================================================================
# SCAN
# ============================================================================

def scan_histories(root):
    rows = []
    root = Path(root)

    if not root.exists():
        raise ValueError(f"Le répertoire d'entrée n'existe pas : {root}")

    files = list(root.rglob("history_*.json"))
    print(f"Nb fichiers history_*.json trouvés : {len(files)}")

    for path in files:
        m = FILENAME_RE.match(path.name)
        if not m:
            continue

        dataset = m.group("dataset")
        split_id = int(m.group("split"))
        strategy = m.group("strategy")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] JSON illisible : {path} ({e})")
            continue

        if not isinstance(data, list):
            print(f"[WARN] JSON ignoré (pas une liste) : {path}")
            continue

        for rec in data:
            if not isinstance(rec, dict):
                continue

            n = (
                rec.get("n_labeled")
                or rec.get("nlabelled")
                or rec.get("budget")
                or rec.get("n_samples")
            )

            acc = (
                rec.get("accuracy")
                or rec.get("acc")
                or rec.get("acc_final")
            )

            n = safe_float(n)
            acc = safe_float(acc)

            if n is None or acc is None:
                continue

            rows.append({
                "dataset": dataset,
                "split_id": split_id,
                "strategy": strategy,
                "n_labeled": float(n),
                "value": float(acc)
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            "Aucune donnée exploitable trouvée.\n"
            f"Répertoire scanné : {root}"
        )

    print("Colonnes détectées :", list(df.columns))
    print("Nb lignes brutes :", len(df))
    return df


# ============================================================================
# INTERPOLATION AUX BUDGETS
# ============================================================================

def interpolate_to_budgets(df, target_budgets):
    """
    Interpole linéairement la valeur 'value' pour chaque
    (dataset, split_id, strategy) aux budgets cibles.

    - interpolation uniquement à l'intérieur de la plage observée
    - pas d'extrapolation
    """
    rows = []

    for (dataset, split_id, strategy), sub in df.groupby(["dataset", "split_id", "strategy"]):
        sub = sub.sort_values("n_labeled").drop_duplicates(subset="n_labeled", keep="last")

        x = sub["n_labeled"].to_numpy(dtype=float)
        y = sub["value"].to_numpy(dtype=float)

        if len(x) == 0:
            continue

        xmin, xmax = x.min(), x.max()

        for b in target_budgets:
            b = float(b)

            if b in x:
                yb = y[np.where(x == b)[0][0]]
            elif len(x) >= 2 and xmin <= b <= xmax:
                yb = np.interp(b, x, y)
            else:
                yb = np.nan

            rows.append({
                "dataset": dataset,
                "split_id": split_id,
                "strategy": strategy,
                "n_labeled": int(b),
                "value": yb,
                "is_interpolated": not (b in x) if not np.isnan(yb) else False
            })

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    return out


# ============================================================================
# DIFFICULTY
# ============================================================================

def load_difficulty_map(path):
    """
    Charge un fichier CSV de difficulté avec au minimum :
    - dataset
    - difficulty
    """
    path = Path(path)
    if not path.exists():
        print(f"[INFO] Aucun fichier de difficulté trouvé : {path}")
        return pd.DataFrame(columns=["dataset", "difficulty"])

    try:
        diff = pd.read_csv(path, sep=";")
    except Exception:
        diff = pd.read_csv(path)

    diff.columns = [c.strip().lower() for c in diff.columns]

    if "dataset" not in diff.columns or "difficulty" not in diff.columns:
        raise ValueError(
            f"Le fichier de difficulté doit contenir les colonnes "
            f"'dataset' et 'difficulty' : {path}"
        )

    diff = diff[["dataset", "difficulty"]].copy()
    diff["dataset"] = diff["dataset"].astype(str).str.strip()
    diff["difficulty"] = diff["difficulty"].astype(str).str.strip().str.lower()

    return diff.drop_duplicates(subset=["dataset"])


def build_difficulty_from_data(dataset_summary, outdir):
    """
    Construit automatiquement un fichier de difficulté
    basé sur l'accuracy moyenne par dataset.

    Les datasets sont répartis en 3 groupes :
    - hard   : tiers inférieur
    - medium : tiers intermédiaire
    - easy   : tiers supérieur
    """
    df = (
        dataset_summary.groupby("dataset")["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "mean_acc"})
    )

    if df.empty:
        return pd.DataFrame(columns=["dataset", "difficulty"])

    df = df.sort_values("mean_acc").reset_index(drop=True)

    q1 = df["mean_acc"].quantile(0.33)
    q2 = df["mean_acc"].quantile(0.66)

    def assign_difficulty(x):
        if x <= q1:
            return "hard"
        elif x <= q2:
            return "medium"
        else:
            return "easy"

    df["difficulty"] = df["mean_acc"].apply(assign_difficulty)

    out_path = outdir / "difficulty_map_auto.csv"
    df[["dataset", "difficulty"]].to_csv(
        out_path, sep=";", index=False, encoding="utf-8-sig"
    )

    print(f"[INFO] difficulty_map_auto.csv généré : {out_path}")

    return df[["dataset", "difficulty"]]


def attach_difficulty(dataset_summary, difficulty_df):
    """
    Ajoute la colonne difficulty à dataset_summary.
    Si aucune difficulté n'est fournie, met 'unknown'.
    """
    out = dataset_summary.copy()

    if difficulty_df is None or difficulty_df.empty:
        out["difficulty"] = "unknown"
        return out

    out["dataset"] = out["dataset"].astype(str).str.strip()
    merged = out.merge(difficulty_df, on="dataset", how="left")
    merged["difficulty"] = merged["difficulty"].fillna("unknown")

    return merged


# ============================================================================
# STATS
# ============================================================================

def compute_budget_summary(df):
    return (
        df.groupby(["strategy", "n_labeled"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["n_labeled", "mean"], ascending=[True, False])
    )



def compute_dataset_summary(df):
    return (
        df.groupby(["dataset", "strategy", "n_labeled"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "value"})
    )

def compute_rank_distribution(dataset_summary):
    """
    Calcule, pour chaque méthode ET pour chaque budget,
    la distribution des rangs sur l'ensemble des datasets.
    """
    rows = []

    for (dataset, budget), sub in dataset_summary.groupby(["dataset", "n_labeled"]):
        sub = sub.sort_values("value", ascending=False).reset_index(drop=True)

        for i, row in sub.iterrows():
            rows.append({
                "dataset": dataset,
                "n_labeled": budget,
                "strategy": row["strategy"],
                "rank": i + 1
            })

    rank_df = pd.DataFrame(rows)

    if rank_df.empty:
        return pd.DataFrame()

    out = (
        rank_df.groupby(["strategy", "n_labeled"])
        .agg(
            rank1_rate=("rank", lambda x: (x == 1).mean() * 100),
            rank2_rate=("rank", lambda x: (x == 2).mean() * 100),
            rank3_rate=("rank", lambda x: (x == 3).mean() * 100),
            above3_rate=("rank", lambda x: (x > 3).mean() * 100),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            n_cases=("rank", "count")
        )
        .reset_index()
        .sort_values(["n_labeled", "mean_rank", "rank1_rate"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    return out


# ============================================================================
# FOCUS ANALYSIS
# ============================================================================

def compute_focus_advantage(dataset_summary, focus_strategy, eps=1e-12):
    """
    Calcule les indicateurs de position de la stratégie cible par
    (dataset, budget), avec gestion robuste des ties et du nombre
    réel de méthodes présentes.
    """
    rows = []

    has_difficulty = "difficulty" in dataset_summary.columns

    group_cols = ["dataset", "n_labeled"]
    if has_difficulty:
        group_cols.append("difficulty")

    for keys, sub in dataset_summary.groupby(group_cols):
        if has_difficulty:
            dataset, budget, difficulty = keys
        else:
            dataset, budget = keys
            difficulty = "unknown"

        sub = sub.copy().sort_values("value", ascending=False).reset_index(drop=True)
        n_methods = len(sub)

        focus = sub[sub["strategy"] == focus_strategy]
        if focus.empty:
            continue

        focus_val = float(focus["value"].iloc[0])

        # rang avec gestion des ex aequo
        sub["rank"] = sub["value"].rank(method="min", ascending=False).astype(int)
        rank = int(sub.loc[sub["strategy"] == focus_strategy, "rank"].iloc[0])

        if n_methods == 1:
            gap = 0.0
            win = True
            strong = True
        else:
            others = sub[sub["strategy"] != focus_strategy]
            best_other_val = float(others["value"].max())
            gap = focus_val - best_other_val

            win = gap > eps
            strong = gap >= 0.01

        rows.append({
            "dataset": dataset,
            "difficulty": difficulty,
            "n_labeled": budget,
            "n_methods": n_methods,
            "rank": rank,
            "gap": gap,
            "top1": rank <= min(1, n_methods),
            "top2": rank <= min(2, n_methods),
            "top3": rank <= min(3, n_methods),
            "win": win,
            "strong_win": strong
        })

    return pd.DataFrame(rows)


# ============================================================================
# AGGREGATION
# ============================================================================

def compute_focus_summary(df):
    """
    Résumé global par budget.
    """
    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby("n_labeled")
        .agg(
            win_rate=("win", "mean"),
            top1_rate=("top1", "mean"),
            top2_rate=("top2", "mean"),
            top3_rate=("top3", "mean"),
            strong_win_rate=("strong_win", "mean"),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            mean_gap=("gap", "mean"),
            n_cases=("dataset", "count"),
            mean_n_methods=("n_methods", "mean")
        )
        .reset_index()
        .sort_values("n_labeled")
        .reset_index(drop=True)
    )

    for c in out.columns:
        if c.endswith("_rate"):
            out[c] *= 100

    return out


def compute_focus_summary_by_difficulty(df):
    """
    Résumé par difficulté ET budget.
    """
    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby(["difficulty", "n_labeled"])
        .agg(
            win_rate=("win", "mean"),
            top1_rate=("top1", "mean"),
            top2_rate=("top2", "mean"),
            top3_rate=("top3", "mean"),
            strong_win_rate=("strong_win", "mean"),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            mean_gap=("gap", "mean"),
            n_cases=("dataset", "count"),
            mean_n_methods=("n_methods", "mean")
        )
        .reset_index()
        .sort_values(["difficulty", "n_labeled"])
        .reset_index(drop=True)
    )

    for c in out.columns:
        if c.endswith("_rate"):
            out[c] *= 100

    return out


# ============================================================================
# LEARNING CURVES
# ============================================================================

def generate_learning_curves(raw_df, outdir, strategies=None,
                             use_interpolation=False, target_budgets=None):
    if raw_df.empty:
        return pd.DataFrame()

    df = filter_strategies(raw_df, strategies)

    if df.empty:
        print("[WARN] Aucune donnée après filtrage des stratégies pour les learning curves.")
        return pd.DataFrame()

    if use_interpolation:
        if target_budgets is None:
            raise ValueError("target_budgets doit être fourni si use_interpolation=True")
        df = interpolate_to_budgets(df, target_budgets)

    lc_dir = outdir / "learning_curves"
    lc_dir.mkdir(parents=True, exist_ok=True)

    dataset_curve_summary = (
        df.groupby(["dataset", "strategy", "n_labeled"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["dataset", "strategy", "n_labeled"])
        .reset_index(drop=True)
    )

    global_curve_summary = (
        df.groupby(["strategy", "n_labeled"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["strategy", "n_labeled"])
        .reset_index(drop=True)
    )

    plt.figure(figsize=(10, 6))

    for strategy, sub in global_curve_summary.groupby("strategy"):
        sub = sub.sort_values("n_labeled")
        x = sub["n_labeled"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        s = sub["std"].fillna(0).to_numpy(dtype=float)

        plt.plot(x, y, marker="o", markersize=2, linewidth=1, label=strategy)
        plt.fill_between(x, y - s, y + s, alpha=0.18)

    plt.xlabel("Budget (n_labeled)")
    plt.ylabel("Mean accuracy")
    plt.title("Global learning curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(lc_dir / "learning_curves_global.png", dpi=180)
    plt.close()

    for dataset, sub_ds in dataset_curve_summary.groupby("dataset"):
        plt.figure(figsize=(10, 6))

        for strategy, sub in sub_ds.groupby("strategy"):
            sub = sub.sort_values("n_labeled")
            x = sub["n_labeled"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            s = sub["std"].fillna(0).to_numpy(dtype=float)

            plt.plot(x, y, marker="o", markersize=2, linewidth=1, label=strategy)
            plt.fill_between(x, y - s, y + s, alpha=0.18)

        plt.xlabel("Budget (n_labeled)")
        plt.ylabel("Mean Accuracy")
        plt.title(f"Learning curves - {dataset}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(lc_dir / f"learning_curve_{safe_name(dataset)}.png", dpi=180)
        plt.close()

    dataset_curve_summary.to_csv(
        lc_dir / "learning_curves_by_dataset.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig"
    )

    global_curve_summary.to_csv(
        lc_dir / "learning_curves_global.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"[INFO] Learning curves générées dans : {lc_dir}")

    return {
        "dataset_curve_summary": dataset_curve_summary,
        "global_curve_summary": global_curve_summary,
    }


# ============================================================================
# PLOTS
# ============================================================================

def plot_global(summary, outdir):
    if summary.empty:
        return

    plt.figure(figsize=(10, 6))

    for strat, sub in summary.groupby("strategy"):
        sub = sub.sort_values("n_labeled")
        x = sub["n_labeled"]
        y = sub["mean"]
        s = sub["std"].fillna(0)

        plt.plot(x, y, marker="o", label=strat)
        plt.fill_between(x, y - s, y + s, alpha=0.2)

    plt.xlabel("Budget")
    plt.ylabel("Mean accuracy")
    plt.title("Global learning curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "global.png", dpi=180)
    plt.close()


def plot_focus(summary, outdir):
    if summary.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(summary["n_labeled"], summary["win_rate"], marker="o", label="win")
    plt.plot(summary["n_labeled"], summary["top2_rate"], marker="o", label="top2")
    plt.plot(summary["n_labeled"], summary["top3_rate"], marker="o", label="top3")

    plt.xlabel("Budget")
    plt.ylabel("Percentage of datasets")
    plt.title(f"{FOCUS_STRATEGY} : win / top-2 / top-3")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "focus.png", dpi=180)
    plt.close()


def plot_focus_by_difficulty(summary, outdir):
    if summary.empty:
        return

    for difficulty, sub in summary.groupby("difficulty"):
        plt.figure(figsize=(8, 5))
        sub = sub.sort_values("n_labeled")

        plt.plot(sub["n_labeled"], sub["win_rate"], marker="o", label="win")
        plt.plot(sub["n_labeled"], sub["top2_rate"], marker="o", label="top2")
        plt.plot(sub["n_labeled"], sub["top3_rate"], marker="o", label="top3")

        plt.xlabel("Budget")
        plt.ylabel("Percentage of datasets")
        plt.title(f"{FOCUS_STRATEGY} - {difficulty}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"focus_{difficulty}.png", dpi=180)
        plt.close()


def generate_boxplots_per_budget(dataset_summary, outdir, box_width=0.25):
    """
    Génère un boxplot-like (mean ± std + min/max) pour chaque budget.
    Basé sur dataset_summary :
    - dataset
    - strategy
    - n_labeled
    - value
    """
    if dataset_summary.empty:
        print("[WARN] dataset_summary vide -> pas de boxplots")
        return

    box_dir = outdir / "boxplots_budget"
    box_dir.mkdir(parents=True, exist_ok=True)

    for budget, sub in dataset_summary.groupby("n_labeled"):
        stats = (
            sub.groupby("strategy")["value"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )

        methods = stats["strategy"].values
        mean = stats["mean"].values
        std = stats["std"].fillna(0).values
        min_v = stats["min"].values
        max_v = stats["max"].values

        x = np.arange(len(methods))

        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        for i in range(len(x)):
            ax.plot([x[i], x[i]], [min_v[i], max_v[i]], linewidth=2)

        for i in range(len(x)):
            y_low = mean[i] - std[i]
            height = 2 * std[i]

            rect = Rectangle(
                (x[i] - box_width / 2, y_low),
                box_width,
                height,
                alpha=0.4
            )
            ax.add_patch(rect)

        ax.scatter(x, mean, s=90, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Strategy")
        ax.set_title(f"Budget {int(budget)} — mean ± std (min-max)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = box_dir / f"boxplot_budget_{int(budget)}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

        print("[BOX] Saved:", save_path)


def generate_method_budget_curves(dataset_summary, outdir):
    """
    Courbes multi-budgets par méthode :
    - X = budget
    - Y = moyenne sur datasets
    - bande = std inter-datasets
    """
    if dataset_summary.empty:
        print("[WARN] dataset_summary vide -> pas de courbes multi-budgets")
        return

    curves_dir = outdir / "method_budget_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    stats = (
        dataset_summary.groupby(["strategy", "n_labeled"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["strategy", "n_labeled"])
    )

    plt.figure(figsize=(10, 6))

    for strategy, sub in stats.groupby("strategy"):
        sub = sub.sort_values("n_labeled")
        x = sub["n_labeled"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        s = sub["std"].fillna(0).to_numpy(dtype=float)

        plt.plot(x, y, marker="o", linewidth=1.5, label=strategy)
        plt.fill_between(x, y - s, y + s, alpha=0.18)

    plt.xlabel("Budget")
    plt.ylabel("Mean accuracy")
    plt.title("Multi-budget curves by method")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curves_dir / "method_budget_curves.png", dpi=180)
    plt.close()

    stats.to_csv(
        curves_dir / "method_budget_curves.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig"
    )

    print("[CURVES] Saved:", curves_dir / "method_budget_curves.png")


def generate_heatmap_method_budget(dataset_summary, outdir):
    """
    Heatmap méthode vs budget, valeur = accuracy moyenne sur datasets.
    """
    if dataset_summary.empty:
        print("[WARN] dataset_summary vide -> pas de heatmap")
        return

    heatmap_dir = outdir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    pivot = (
        dataset_summary.groupby(["strategy", "n_labeled"])["value"]
        .mean()
        .reset_index()
        .pivot(index="strategy", columns="n_labeled", values="value")
    )

    if pivot.empty:
        return

    pivot = pivot.sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    plt.figure(figsize=(12, max(4, 0.45 * len(pivot.index))))
    im = plt.imshow(pivot.values, aspect="auto")

    plt.xticks(
        ticks=np.arange(len(pivot.columns)),
        labels=[str(int(c)) for c in pivot.columns],
        rotation=45
    )
    plt.yticks(
        ticks=np.arange(len(pivot.index)),
        labels=pivot.index
    )
    plt.xlabel("Budget")
    plt.ylabel("Strategy")
    plt.title("Heatmap méthode vs budget (accuracy moyenne)")
    plt.colorbar(im, label="Accuracy moyenne")
    plt.tight_layout()
    plt.savefig(heatmap_dir / "heatmap_method_budget.png", dpi=180)
    plt.close()

    pivot.to_csv(
        heatmap_dir / "heatmap_method_budget.csv",
        sep=";",
        encoding="utf-8-sig"
    )

    print("[HEATMAP] Saved:", heatmap_dir / "heatmap_method_budget.png")


def generate_focus_winrate_plots(focus_summary, focus_summary_by_difficulty, outdir):
    """
    Plots supplémentaires centrés sur la stratégie focus :
    - win-rate / top-k global
    - par difficulté
    """
    if focus_summary.empty:
        print("[WARN] focus_summary vide -> pas de plots focus supplémentaires")
        return

    focus_dir = outdir / "focus_extra"
    focus_dir.mkdir(parents=True, exist_ok=True)

    # global
    plt.figure(figsize=(9, 5))
    plt.plot(focus_summary["n_labeled"], focus_summary["win_rate"], marker="o", label="win")
    plt.plot(focus_summary["n_labeled"], focus_summary["top1_rate"], marker="o", label="top1")
    plt.plot(focus_summary["n_labeled"], focus_summary["top2_rate"], marker="o", label="top2")
    plt.plot(focus_summary["n_labeled"], focus_summary["top3_rate"], marker="o", label="top3")

    plt.xlabel("Budget")
    plt.ylabel("Percentage")
    plt.title(f"{FOCUS_STRATEGY} — win/top1/top2/top3")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(focus_dir / "focus_winrates_global.png", dpi=180)
    plt.close()

    print("[FOCUS] Saved:", focus_dir / "focus_winrates_global.png")

    # par difficulté
    if not focus_summary_by_difficulty.empty:
        for difficulty, sub in focus_summary_by_difficulty.groupby("difficulty"):
            sub = sub.sort_values("n_labeled")

            plt.figure(figsize=(9, 5))
            plt.plot(sub["n_labeled"], sub["win_rate"], marker="o", label="win")
            plt.plot(sub["n_labeled"], sub["top1_rate"], marker="o", label="top1")
            plt.plot(sub["n_labeled"], sub["top2_rate"], marker="o", label="top2")
            plt.plot(sub["n_labeled"], sub["top3_rate"], marker="o", label="top3")

            plt.xlabel("Budget")
            plt.ylabel("Percentage")
            plt.title(f"{FOCUS_STRATEGY} — {difficulty}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(focus_dir / f"focus_winrates_{difficulty}.png", dpi=180)
            plt.close()

            print("[FOCUS] Saved:", focus_dir / f"focus_winrates_{difficulty}.png")


# ============================================================================
# EXPORT CSV
# ============================================================================

def export_csv(outdir, **tables):
    d = outdir / "tables"
    d.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(d / f"{name}.csv", sep=";", index=False, encoding="utf-8-sig")


# ============================================================================
# MAIN
# ============================================================================

def run_analysis():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("INPUT_DIR :", INPUT_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    raw_df = scan_histories(INPUT_DIR)

    if BUDGETS is not None:
        df = interpolate_to_budgets(raw_df, BUDGETS)
    else:
        df = raw_df.copy()
        df["n_labeled"] = df["n_labeled"].astype(int)
        df["is_interpolated"] = False

    print("Datasets  :", df["dataset"].nunique())
    print("Strategies:", df["strategy"].nunique())
    print("Budgets   :", sorted(df["n_labeled"].unique()))

    budget_summary = compute_budget_summary(df)
    dataset_summary = compute_dataset_summary(df)

    # difficulté : fichier externe si présent, sinon génération auto
    difficulty_df = load_difficulty_map(DIFFICULTY_CSV)
    if difficulty_df.empty:
        print("[INFO] Génération automatique de la difficulté à partir des accuracies moyennes.")
        difficulty_df = build_difficulty_from_data(dataset_summary, outdir)
    else:
        print("[INFO] difficulty_map.csv trouvé et chargé.")

    dataset_summary = attach_difficulty(dataset_summary, difficulty_df)

    focus_adv = compute_focus_advantage(dataset_summary, FOCUS_STRATEGY)
    focus_summary = compute_focus_summary(focus_adv)
    focus_summary_by_difficulty = compute_focus_summary_by_difficulty(focus_adv)
    rank_distribution = compute_rank_distribution(dataset_summary)

    # plots existants
    plot_global(budget_summary, outdir)
    plot_focus(focus_summary, outdir)
    plot_focus_by_difficulty(focus_summary_by_difficulty, outdir)

    # nouveaux plots
    generate_boxplots_per_budget(dataset_summary, outdir)
    generate_method_budget_curves(dataset_summary, outdir)
    generate_heatmap_method_budget(dataset_summary, outdir)
    generate_focus_winrate_plots(focus_summary, focus_summary_by_difficulty, outdir)

    learning_curves = generate_learning_curves(
        raw_df,
        outdir,
        strategies=LEARNING_CURVE_STRATEGIES,
        use_interpolation=LEARNING_CURVE_USE_INTERPOLATION,
        target_budgets=BUDGETS
    )

    export_csv(
        outdir,
        raw_points=df,
        budget_summary=budget_summary,
        dataset_summary=dataset_summary,
        difficulty_map=difficulty_df,
        focus_adv=focus_adv,
        focus_summary=focus_summary,
        focus_summary_by_difficulty=focus_summary_by_difficulty,
        rank_distribution=rank_distribution
    )

    return {
        "raw_points": df,
        "budget_summary": budget_summary,
        "dataset_summary": dataset_summary,
        "difficulty_map": difficulty_df,
        "focus_adv": focus_adv,
        "focus_summary": focus_summary,
        "focus_summary_by_difficulty": focus_summary_by_difficulty,
        "rank_distribution": rank_distribution,
        "learning_curves": learning_curves,
    }


if __name__ == "__main__":
    results = run_analysis()