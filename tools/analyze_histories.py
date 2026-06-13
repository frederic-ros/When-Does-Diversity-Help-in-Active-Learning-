#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_histories.py
====================

Analyse complète d'un dossier de histories JSON générées par le benchmark.
Produit : figures (courbes, heatmap rangs, delta) + tables CSV de statistiques.

Usage
-----
    python analyze_histories.py [OPTIONS]

Options principales
-------------------
    --base      Dossier racine contenant les histories JSON
                (défaut : ./testnewreel)
    --out       Dossier de sortie pour les figures et CSV
                (défaut : ./analyze_out)
    --metric    Métrique à analyser : f1_macro | accuracy | balanced_accuracy
                (défaut : f1_macro)
    --methods   Sous-ensemble de méthodes à inclure, séparés par des virgules
                Ex: "ActivePseudoLabelV53,dbal,margin,random"
                (défaut : toutes les méthodes trouvées)
    --focus     Méthodes mises en avant dans les courbes (traits épais + CI)
                Ex: "ActivePseudoLabelV53,ActivePseudoLabelV54"
                (défaut : V5.3 et V5.4 si présents, sinon 2 premiers)
    --budgets   Budgets pour les tables stats, séparés par des virgules
                Ex: "100,200,300,500"
                (défaut : auto-détection sur 5 points équidistants + saturation)
    --sat_frac  Fraction du gain total pour détecter la saturation (défaut : 0.05)
    --no-curves Ne pas générer la figure de courbes
    --no-heatmap Ne pas générer la heatmap des rangs
    --no-delta  Ne pas générer la figure des deltas
    --no-csv    Ne pas générer les fichiers CSV

Exemples
--------
    # Analyse complète avec toutes les méthodes
    python analyze_histories.py --base ./testnewreel --out ./results

    # Focus sur V5.3 vs margin vs dbal, stats à 200/300/500 labels
    python analyze_histories.py \\
        --methods "ActivePseudoLabelV53,dbal,margin" \\
        --focus   "ActivePseudoLabelV53" \\
        --budgets "200,300,500"

    # Latent space : accuracy à la place de f1_macro
    python analyze_histories.py \\
        --base ./latent_pca50 --metric accuracy \\
        --methods "ActivePseudoLabelV53,ActivePseudoLabelV54,margin"
"""

import argparse
import json
import os
import sys
import glob
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib non disponible — figures désactivées")

# ══════════════════════════════════════════════════════════════
# Couleurs et styles par méthode
# ══════════════════════════════════════════════════════════════
METHOD_COLORS = {
    'ActivePseudoLabelV53': '#e63946',
    'ActivePseudoLabelV54': '#2196F3',
    'dbal':                 '#555555',
    'rank2022':             '#3F51B5',
    'margin':               '#9C27B0',
    'entropy':              '#FF9800',
    'least_confident':      '#009688',
    'qbc':                  '#795548',
    'typiclust':            '#4CAF50',
    'tri_committee':        '#F06292',
    'committee':            '#F06292',
    'random':               '#AAAAAA',
}
METHOD_LABELS = {
    'ActivePseudoLabelV53': 'V5.3',
    'ActivePseudoLabelV54': 'V5.4',
    'dbal':                 'DBAL',
    'rank2022':             'Rank2022',
    'margin':               'Margin',
    'entropy':              'Entropy',
    'least_confident':      'LeastConf',
    'qbc':                  'QBC',
    'typiclust':            'TypiClust',
    'tri_committee':        'Tri-Comm',
    'committee':            'Committee',
    'random':               'Random',
}
DEFAULT_PALETTE = [
    '#e63946','#2196F3','#555555','#3F51B5','#9C27B0',
    '#FF9800','#009688','#795548','#4CAF50','#F06292','#AAAAAA',
]


def get_color(m, idx=0):
    return METHOD_COLORS.get(m, DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)])


def get_label(m):
    return METHOD_LABELS.get(m, m)


# ══════════════════════════════════════════════════════════════
# Chargement des données
# ══════════════════════════════════════════════════════════════
def load_data(base_dir: str):
    """
    Charge tous les fichiers JSON dans base_dir.
    Retourne data[dataset][split][method] = list_of_steps
    et la liste NL des budgets.
    """
    data = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(base_dir, '*.json')):
        bn = os.path.basename(f).replace('history_', '').replace('.json', '')
        parts = bn.split('_')
        si = next(
            (i for i, p in enumerate(parts)
             if p.startswith('split') and p[5:].isdigit()),
            None
        )
        if si is None:
            continue
        dataset = '_'.join(parts[:si])
        split   = parts[si]
        method  = '_'.join(parts[si + 1:])
        with open(f) as fh:
            data[dataset][split][method] = json.load(fh)

    if not data:
        print(f"[ERROR] Aucun fichier JSON trouvé dans {base_dir}")
        sys.exit(1)

    # Déduire NL depuis le premier fichier valide
    for ds in data:
        for sp in data[ds]:
            for m in data[ds][sp]:
                sample = data[ds][sp][m]
                if isinstance(sample, list) and sample:
                    NL = [s['n_labeled'] for s in sample]
                    return data, NL

    print("[ERROR] Impossible de déduire les budgets depuis les JSON")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
# Courbe moyenne + std
# ══════════════════════════════════════════════════════════════
def get_curve(data, ds, method, metric='f1_macro'):
    """Retourne (mean_curve, std_curve) ou (None, None) si absent."""
    c = []
    # longueur de référence = n_steps du premier split disponible
    ref_len = None
    for sp, methods in data[ds].items():
        if method not in methods:
            continue
        steps = methods[method]
        if not isinstance(steps, list):
            continue
        if ref_len is None:
            ref_len = len(steps)
        if len(steps) != ref_len:
            continue
        try:
            c.append([s[metric] for s in steps])
        except (KeyError, TypeError):
            continue
    if not c:
        return None, None
    a = np.array(c)
    return a.mean(0), a.std(0)


def get_splits_auc(data, ds, method, steps, metric='f1_macro'):
    """Retourne les AUC pré-saturation pour chaque split (pour IQR)."""
    ref_len = None
    for sp, methods in data[ds].items():
        if method in methods and isinstance(methods[method], list):
            ref_len = len(methods[method])
            break
    result = []
    for sp, methods in data[ds].items():
        if method not in methods:
            continue
        s = methods[method]
        if not isinstance(s, list) or len(s) != ref_len:
            continue
        vals = [s[i][metric] for i in steps if i < len(s)]
        if vals:
            result.append(float(np.mean(vals)))
    return np.array(result) if result else None


# ══════════════════════════════════════════════════════════════
# Détection de saturation
# ══════════════════════════════════════════════════════════════
def detect_saturation(data, ds, methods_ref, NL, sat_frac=0.05, abs_thresh=0.002):
    """
    Saturation = premier step où le gain marginal tombe sous
    sat_frac × gain_total (poolé sur methods_ref).
    Retourne (sat_nl, sat_idx).
    """
    curves = []
    for m in methods_ref:
        mn, _ = get_curve(data, ds, m)
        if mn is not None:
            curves.append(mn)
    if not curves:
        return NL[-1], len(NL) - 1

    pooled = np.mean(curves, axis=0)
    gains  = np.diff(pooled)
    tr     = pooled.max() - pooled[1]
    if tr < 1e-6:
        return NL[1], 1

    thr = max(sat_frac * tr, abs_thresh)
    for i in range(1, len(gains) - 2):
        if all(abs(g) < thr for g in gains[i:i + 3]):
            return NL[i + 1], i + 1

    return NL[-1], len(NL) - 1


# ══════════════════════════════════════════════════════════════
# Tables statistiques
# ══════════════════════════════════════════════════════════════
def compute_stats_table(data, datasets, methods, NL, budgets, sat_info, metric):
    """
    Retourne une liste de dicts avec les stats par (dataset, budget, method).
    Colonnes : dataset, budget, method, mean, std, median, q25, q75,
               rank, delta_vs_best, delta_vs_margin (si margin présent),
               presat_auc (si budget == 'presat')
    """
    rows = []

    for ds in datasets:
        sat_nl, sat_idx = sat_info[ds]
        presat_steps = list(range(1, sat_idx + 1))

        # Calculer l'AUC pré-sat pour chaque méthode
        presat_aucs = {}
        for m in methods:
            sc = get_splits_auc(data, ds, m, presat_steps, metric)
            if sc is not None:
                presat_aucs[m] = sc

        # Stats par budget
        for budget in budgets + ['presat']:
            if budget == 'presat':
                idx   = None
                label = 'presat_AUC'
            else:
                # Trouver l'index le plus proche
                dists = [abs(nl - budget) for nl in NL]
                idx   = dists.index(min(dists))
                label = str(budget)
                if NL[idx] > sat_nl and budget != 'presat':
                    pass  # inclure quand même (post-sat utile pour table)

            row_vals = {}
            for m in methods:
                if budget == 'presat':
                    sc = presat_aucs.get(m)
                else:
                    sc = get_splits_auc(data, ds, m, [idx], metric)
                if sc is not None and len(sc) > 0:
                    row_vals[m] = sc

            if not row_vals:
                continue

            # Rangs par split (pour rang moyen robuste)
            means = {m: float(np.mean(sc)) for m, sc in row_vals.items()}
            ranked = sorted(means, key=means.get, reverse=True)

            margin_mean = means.get('margin', None)

            for m, sc in row_vals.items():
                mean_v = float(np.mean(sc))
                std_v  = float(np.std(sc))
                med_v  = float(np.median(sc))
                q25_v  = float(np.percentile(sc, 25))
                q75_v  = float(np.percentile(sc, 75))
                rank_v = ranked.index(m) + 1 if m in ranked else np.nan
                best_v = max(means.values())
                d_best = (mean_v - best_v) * 100

                row = {
                    'dataset':         ds,
                    'budget':          label,
                    'sat_nl':          sat_nl,
                    'method':          m,
                    'method_label':    get_label(m),
                    'mean':            round(mean_v, 5),
                    'std':             round(std_v,  5),
                    'median':          round(med_v,  5),
                    'q25':             round(q25_v,  5),
                    'q75':             round(q75_v,  5),
                    'rank':            rank_v,
                    'n_splits':        len(sc),
                    'delta_vs_best_pp':round(d_best, 3),
                }
                if margin_mean is not None and m != 'margin':
                    row['delta_vs_margin_pp'] = round((mean_v - margin_mean) * 100, 3)
                else:
                    row['delta_vs_margin_pp'] = None

                rows.append(row)

    return rows


# ══════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════
def plot_curves(data, datasets, methods, focus_methods, NL, sat_info,
                metric, out_dir, suffix=''):
    """Figure grille : courbes learning par dataset."""
    n  = len(datasets)
    nc = min(4, n)
    nr = (n + nc - 1) // nc + (1 if n % nc != 0 and n > nc else 0)
    nr = max(1, (n + nc - 1) // nc)

    fig, axes = plt.subplots(nr, nc, figsize=(6 * nc, 5 * nr))
    if nr * nc == 1:
        axes = np.array([[axes]])
    elif nr == 1:
        axes = axes.reshape(1, -1)
    elif nc == 1:
        axes = axes.reshape(-1, 1)

    title = (f'Learning curves — {metric}\n'
             f'Zone grisée = post-saturation | batch={NL[1]-NL[0]} labels')
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.999)

    handles_labels = {}
    for di, ds in enumerate(datasets):
        row, col = divmod(di, nc)
        ax = axes[row, col]
        sat_nl, _ = sat_info[ds]
        ax.axvspan(sat_nl, NL[-1] + (NL[1] - NL[0]),
                   color='#F0F0F0', alpha=0.6, zorder=0)
        ax.axvline(sat_nl, color='#666', lw=1.4, ls='--', zorder=4)

        # Méthodes non-focus en arrière-plan
        for mi, m in enumerate(methods):
            if m in focus_methods:
                continue
            mn, std = get_curve(data, ds, m, metric)
            if mn is None:
                continue
            c = get_color(m, mi)
            lbl = get_label(m)
            ln, = ax.plot(NL, mn, color=c, lw=1.2, ls='--',
                          alpha=0.7, zorder=2, label=lbl)
            handles_labels[lbl] = ln

        # Méthodes focus en avant-plan
        for mi, m in enumerate(methods):
            if m not in focus_methods:
                continue
            mn, std = get_curve(data, ds, m, metric)
            if mn is None:
                continue
            c = get_color(m, mi)
            lbl = get_label(m)
            ln, = ax.plot(NL, mn, color=c, lw=2.6, zorder=10, label=lbl)
            ax.fill_between(NL, mn - std, mn + std,
                            color=c, alpha=0.15, zorder=9)
            handles_labels[lbl] = ln

        ds_short = ds.replace('_original', '').replace('_', ' ')
        ax.set_title(f'{ds_short}  [sat@{sat_nl}]',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('# labeled', fontsize=8)
        ax.set_ylabel(metric, fontsize=8)
        ax.set_xlim(0, NL[-1] + (NL[1] - NL[0]))
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, lw=0.5)

    # Axes vides
    for di in range(len(datasets), nr * nc):
        r, c = divmod(di, nc)
        axes[r, c].axis('off')

    if handles_labels:
        fig.legend(
            list(handles_labels.values()),
            list(handles_labels.keys()),
            loc='lower right', bbox_to_anchor=(0.99, 0.01),
            fontsize=9, ncol=min(len(handles_labels), 5), framealpha=0.9
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(out_dir, f'curves{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [FIG] {out_path}')


def plot_heatmap(data, datasets, methods, NL, sat_info, metric, out_dir, suffix=''):
    """Heatmap des rangs AUC pré-saturation."""
    nm = len(methods)
    rm = np.zeros((len(datasets), nm))

    for di, ds in enumerate(datasets):
        sat_nl, sat_idx = sat_info[ds]
        steps = list(range(1, sat_idx + 1))
        row = {}
        for m in methods:
            sc = get_splits_auc(data, ds, m, steps, metric)
            if sc is not None:
                row[m] = float(np.mean(sc))
        if not row:
            continue
        sm = sorted(row, key=row.get, reverse=True)
        for ji, m in enumerate(methods):
            if m in sm:
                rm[di, ji] = sm.index(m) + 1

    fig, ax = plt.subplots(figsize=(max(8, 2 * nm), max(5, 0.7 * len(datasets))))
    im = ax.imshow(rm, aspect='auto', cmap='RdYlGn_r', vmin=1, vmax=nm)
    ax.set_xticks(range(nm))
    ax.set_xticklabels([get_label(m) for m in methods],
                       fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(
        [ds.replace('_original', '').replace('_', ' ') for ds in datasets],
        fontsize=10
    )
    ax.set_title(f'Rang AUC pré-saturation ({metric}) | 1=meilleur',
                 fontsize=12, fontweight='bold')

    for i in range(len(datasets)):
        for j in range(nm):
            v = int(rm[i, j])
            if v:
                ax.text(j, i, str(v), ha='center', va='center',
                        fontsize=12, fontweight='bold' if j == 0 else 'normal',
                        color='white' if v >= max(3, nm - 1) else 'black')

    plt.colorbar(im, ax=ax, fraction=0.03, label='Rang')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'heatmap_ranks{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [FIG] {out_path}')


def plot_delta(data, datasets, methods, NL, sat_info, metric,
               ref_method, out_dir, suffix=''):
    """
    Barplot horizontal : Δ AUC pré-saturation vs ref_method.
    Un sous-plot par méthode (sauf ref).
    """
    compare = [m for m in methods if m != ref_method]
    if not compare:
        return

    nc = min(3, len(compare))
    nr = (len(compare) + nc - 1) // nc

    fig, axes = plt.subplots(nr, nc, figsize=(7 * nc, 5 * nr))
    if nr * nc == 1:
        axes = np.array([[axes]])
    elif nr == 1:
        axes = axes.reshape(1, -1)
    elif nc == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        f'Δ {metric} vs {get_label(ref_method)} | AUC pré-saturation\n'
        f'médiane ± IQR | {NL[1]-NL[0]} labels/batch',
        fontsize=12, fontweight='bold'
    )

    for ci, m in enumerate(compare):
        r, c = divmod(ci, nc)
        ax = axes[r, c]

        ds_lbs, meds, lo, hi, cols = [], [], [], [], []
        for ds in datasets:
            sat_nl, sat_idx = sat_info[ds]
            steps = list(range(1, sat_idx + 1))
            sc_m   = get_splits_auc(data, ds, m,          steps, metric)
            sc_ref = get_splits_auc(data, ds, ref_method, steps, metric)
            if sc_m is None or sc_ref is None:
                continue
            ml = min(len(sc_m), len(sc_ref))
            g  = (sc_m[:ml] - sc_ref[:ml]) * 100
            q25, med, q75 = np.percentile(g, [25, 50, 75])
            ds_lbs.append(ds.replace('_original', '').replace('_', '\n'))
            meds.append(med)
            lo.append(med - q25)
            hi.append(q75 - med)
            cols.append(get_color(m) if med > 0 else '#e63946')

        y = np.arange(len(ds_lbs))
        ax.barh(y, meds, color=cols, alpha=0.82, height=0.7, edgecolor='white')
        ax.errorbar(meds, y, xerr=[lo, hi],
                    fmt='none', color='#333', lw=1.2, capsize=3)
        for yi, med_v in zip(y, meds):
            ax.text(
                med_v + (0.04 if med_v >= 0 else -0.04), yi,
                f'{med_v:+.2f}',
                va='center',
                ha='left' if med_v >= 0 else 'right',
                fontsize=8
            )
        ax.axvline(0, color='black', lw=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels(ds_lbs, fontsize=8)
        ax.set_xlabel('Δ (pp)', fontsize=9)
        ax.set_title(f'{get_label(m)} − {get_label(ref_method)}',
                     fontsize=11, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3, lw=0.5)

    # Masquer les axes vides
    for ci in range(len(compare), nr * nc):
        r, c = divmod(ci, nc)
        axes[r, c].axis('off')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'delta_vs_{ref_method}{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [FIG] {out_path}')


def plot_rank_summary(rank_data, methods, out_dir, suffix=''):
    """Barplot résumé des rangs moyens."""
    m_sorted = sorted(rank_data, key=lambda x: np.mean(rank_data[x]))
    ranks    = [np.mean(rank_data[m]) for m in m_sorted]
    wins     = [sum(1 for r in rank_data[m] if r == 1) for m in m_sorted]
    n        = len(rank_data[m_sorted[0]]) if rank_data else 1
    colors   = [get_color(m, i) for i, m in enumerate(m_sorted)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Rang moyen
    ax = axes[0]
    bars = ax.barh(range(len(m_sorted)),
                   ranks, color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(m_sorted)))
    ax.set_yticklabels([get_label(m) for m in m_sorted], fontsize=11)
    ax.set_xlabel('Rang moyen (1=meilleur)', fontsize=10)
    ax.set_title('Rang moyen AUC pré-saturation', fontsize=12, fontweight='bold')
    for i, (r, m) in enumerate(zip(ranks, m_sorted)):
        ax.text(r + 0.03, i, f'{r:.2f}  ({wins[i]}/{n} wins)',
                va='center', fontsize=9)
    ax.set_xlim(0, len(methods) + 0.5)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()

    # Wins
    ax = axes[1]
    ax.barh(range(len(m_sorted)), wins, color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(m_sorted)))
    ax.set_yticklabels([get_label(m) for m in m_sorted], fontsize=11)
    ax.set_xlabel('Nombre de wins (1er rang)', fontsize=10)
    ax.set_title('Win rate AUC pré-saturation', fontsize=12, fontweight='bold')
    ax.set_xlim(0, n + 0.5)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'rank_summary{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [FIG] {out_path}')


# ══════════════════════════════════════════════════════════════
# Export CSV
# ══════════════════════════════════════════════════════════════
def export_csv(rows, out_dir, suffix=''):
    """Exporte les stats dans deux CSV : détail et résumé rangs."""
    if not rows:
        return

    # CSV détaillé
    keys = list(rows[0].keys())
    detail_path = os.path.join(out_dir, f'stats_detail{suffix}.csv')
    with open(detail_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'  [CSV] {detail_path}')

    # CSV résumé : rang moyen par méthode
    from collections import Counter
    rank_data = defaultdict(list)
    for r in rows:
        if r['budget'] == 'presat_AUC':
            rank_data[r['method']].append(r['rank'])

    summary_path = os.path.join(out_dir, f'stats_summary{suffix}.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['method', 'method_label', 'mean_rank', 'wins',
                    'top3', 'n_evaluations'])
        for m in sorted(rank_data, key=lambda x: np.mean(rank_data[x])):
            ranks = rank_data[m]
            w.writerow([
                m, get_label(m),
                round(np.mean(ranks), 3),
                sum(1 for r in ranks if r == 1),
                sum(1 for r in ranks if r <= 3),
                len(ranks)
            ])
    print(f'  [CSV] {summary_path}')

    return rank_data


# ══════════════════════════════════════════════════════════════
# Résumé console
# ══════════════════════════════════════════════════════════════
def print_summary(data, datasets, methods, NL, sat_info, metric, budgets):
    """Affiche le tableau récapitulatif dans le terminal."""
    print("\n" + "=" * 100)
    print(f"AUC PRÉ-SATURATION ({metric}) — résumé")
    print("=" * 100)

    col_w = 9
    header = f"  {'Dataset':<28} {'sat@':>5}"
    for m in methods:
        header += f"  {get_label(m):>{col_w}}"
    print(header + "  Δ(best-V53)" if 'ActivePseudoLabelV53' in methods else header)
    print("-" * (35 + len(methods) * (col_w + 2) + 14))

    rank_sum = defaultdict(list); wins = defaultdict(int)
    for ds in datasets:
        sat_nl, sat_idx = sat_info[ds]
        steps = list(range(1, sat_idx + 1))
        row_vals = {}
        for m in methods:
            sc = get_splits_auc(data, ds, m, steps, metric)
            if sc is not None:
                row_vals[m] = float(np.mean(sc))
        if not row_vals:
            continue
        best = max(row_vals.values())
        sm   = sorted(row_vals, key=row_vals.get, reverse=True)
        for rank, m in enumerate(sm, 1):
            rank_sum[m].append(rank)
            if abs(row_vals[m] - best) < 1e-9:
                wins[m] += 1

        ds_short = ds.replace('_original', '').replace('_', ' ')
        line = f"  {ds_short:<28} {sat_nl:>5}"
        for m in methods:
            v = row_vals.get(m, np.nan)
            star = '★' if not np.isnan(v) and abs(v - best) < 1e-5 else ' '
            line += f"  {v:.5f}{star}" if not np.isnan(v) else f"  {'—':>{col_w}} "
        print(line)

    print()
    print(f"  {'Rang moyen':<28} {'':>5}", end="")
    for m in methods:
        r = np.mean(rank_sum[m]) if rank_sum[m] else np.nan
        print(f"  {r:>{col_w}.2f} ", end="")
    print()
    print(f"  {'Wins':<28} {'':>5}", end="")
    n = len(datasets)
    for m in methods:
        print(f"  {wins[m]:>{col_w}}/{n}", end="")
    print("\n")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    _here = Path(__file__).resolve().parent
    _root = _here.parent
    base = str(_root / 'tests' / 'bench_results_real_strates_stratified' / 'histories')
    parser = argparse.ArgumentParser(
        description='Analyse de histories AL et génération de figures/CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--base',     default=base + '/mamo',
                        help='Dossier des histories JSON')
    parser.add_argument('--out',      default=None,          # ← None au lieu de './analyze_out'
                        help='Dossier de sortie (défaut : <base>/analyze_out)')
    parser.add_argument('--metric',   default='f1_macro',
                        choices=['f1_macro', 'accuracy', 'balanced_accuracy', 'f1_weighted'],
                        help='Métrique à analyser')
    parser.add_argument('--methods',  default=None,
                        help='Méthodes à inclure (séparées par des virgules)')
    parser.add_argument('--focus',    default=None,
                        help='Méthodes focus dans les courbes (traits épais)')
    parser.add_argument('--ref',      default=None,
                        help='Méthode de référence pour les deltas (défaut: première méthode)')
    parser.add_argument('--budgets',  default=None,
                        help='Budgets pour les tables (séparés par des virgules)')
    parser.add_argument('--sat_frac', default=0.05, type=float,
                        help='Fraction du gain pour détecter la saturation')
    parser.add_argument('--suffix',   default='',
                        help='Suffixe pour les noms de fichiers sortie')
    parser.add_argument('--no-curves',  action='store_true')
    parser.add_argument('--no-heatmap', action='store_true')
    parser.add_argument('--no-delta',   action='store_true')
    parser.add_argument('--no-summary', action='store_true')
    parser.add_argument('--no-csv',     action='store_true')
    args = parser.parse_args()

    # Résoudre --out : sous-dossier analyze_out dans --base  ← AJOUT
    if args.out is None:
        args.out = str(Path(args.base) / 'analyze_out')
    # ── Chargement ────────────────────────────────────────────
    print(f"\n[LOAD] {args.base}")
    data, NL = load_data(args.base)
    datasets = sorted(data.keys())
    all_methods = sorted(set(
        m for ds in data.values()
        for sp in ds.values()
        for m in sp
    ))
    print(f"  {len(datasets)} datasets | {len(NL)} steps | "
          f"NL={NL[0]}→{NL[-1]} | batch={NL[1]-NL[0]}")
    print(f"  Méthodes trouvées : {all_methods}")

    # ── Sélection méthodes ────────────────────────────────────
    if args.methods:
        methods = [m.strip() for m in args.methods.split(',') if m.strip()]
        missing = [m for m in methods if m not in all_methods]
        if missing:
            print(f"  [WARN] Méthodes non trouvées : {missing}")
        methods = [m for m in methods if m in all_methods]
    else:
        methods = all_methods

    if not methods:
        print("[ERROR] Aucune méthode valide sélectionnée")
        sys.exit(1)

    print(f"  Méthodes analysées : {methods}")

    # ── Focus ─────────────────────────────────────────────────
    if args.focus:
        focus = [m.strip() for m in args.focus.split(',') if m.strip() in methods]
    else:
        # Auto : V5.3 + V5.4 si présents, sinon 2 premiers
        auto = [m for m in methods
                if 'V53' in m or 'V54' in m or 'pseudolabel' in m.lower()]
        focus = auto if auto else methods[:2]

    print(f"  Focus (traits épais) : {focus}")

    # ── Méthode de référence pour deltas ─────────────────────
    ref = args.ref if args.ref and args.ref in methods else methods[0]
    print(f"  Référence delta    : {ref}")

    # ── Budgets pour stats ────────────────────────────────────
    if args.budgets:
        budgets = [int(b.strip()) for b in args.budgets.split(',') if b.strip().isdigit()]
    else:
        # Auto : 5 points équidistants dans la plage NL[1:]
        n_b = min(5, len(NL) - 1)
        step = max(1, (len(NL) - 1) // n_b)
        budgets = [NL[i] for i in range(1, len(NL), step)][:n_b]

    print(f"  Budgets stats      : {budgets}")

    # ── Saturation ────────────────────────────────────────────
    sat_info = {}
    for ds in datasets:
        sat_nl, sat_idx = detect_saturation(
            data, ds, methods, NL, args.sat_frac
        )
        sat_info[ds] = (sat_nl, sat_idx)
    print(f"\n  Saturation détectée :")
    for ds, (sat_nl, _) in sat_info.items():
        print(f"    {ds.replace('_original','').replace('_',' '):<28}: @{sat_nl} labels")

    # ── Sortie ────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    suffix = f'_{args.suffix}' if args.suffix else ''

    # ── Résumé console ────────────────────────────────────────
    if not args.no_summary:
        print_summary(data, datasets, methods, NL, sat_info, args.metric, budgets)

    # ── Figures ───────────────────────────────────────────────
    if HAS_MPL:
        if not args.no_curves:
            print("\n[FIG] Courbes...")
            plot_curves(data, datasets, methods, focus, NL, sat_info,
                        args.metric, args.out, suffix)

        if not args.no_heatmap:
            print("[FIG] Heatmap rangs...")
            plot_heatmap(data, datasets, methods, NL, sat_info,
                         args.metric, args.out, suffix)

        if not args.no_delta:
            print("[FIG] Deltas...")
            plot_delta(data, datasets, methods, NL, sat_info,
                       args.metric, ref, args.out, suffix)

        # Résumé rangs
        rank_sum = defaultdict(list)
        wins_r   = defaultdict(int)
        for ds in datasets:
            sat_nl, sat_idx = sat_info[ds]
            steps = list(range(1, sat_idx + 1))
            row_vals = {}
            for m in methods:
                sc = get_splits_auc(data, ds, m, steps, args.metric)
                if sc is not None:
                    row_vals[m] = float(np.mean(sc))
            if not row_vals:
                continue
            best = max(row_vals.values())
            sm   = sorted(row_vals, key=row_vals.get, reverse=True)
            for rank, m in enumerate(sm, 1):
                rank_sum[m].append(rank)
                if abs(row_vals[m] - best) < 1e-9:
                    wins_r[m] += 1

        print("[FIG] Résumé rangs...")
        plot_rank_summary(rank_sum, methods, args.out, suffix)
    else:
        print("[SKIP] Figures (matplotlib absent)")

    # ── CSV ───────────────────────────────────────────────────
    if not args.no_csv:
        print("\n[CSV] Calcul des statistiques...")
        rows = compute_stats_table(
            data, datasets, methods, NL, budgets, sat_info, args.metric
        )
        rank_data = export_csv(rows, args.out, suffix)

    print(f"\n[DONE] Sorties dans : {args.out}\n")


if __name__ == '__main__':
    main()
