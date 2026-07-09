"""build_csd_estimation_notebook.py — Generate CSD estimation analysis notebook.

This script programmatically builds a Jupyter notebook under:
`notebooks/Ecoli_Analysis_Notebooks/estimation/`.

The notebook aggregates per-edge CSV outputs from:
`notebooks/Ecoli_Analysis_Notebooks/estimation/20260708_161623/csv`.

Run with:

    uv run python scripts/build_csd_estimation_notebook.py
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


parser = ArgumentParser()
parser.add_argument(
    "run_id",
    help="The run ID for parameter estimation you wish to generate a notebook for.",
)
args = parser.parse_args()
RUN_ID = args.run_id


INPUT_DIR = REPO / "notebooks" / "Ecoli_Analysis_Notebooks" / "estimation" / RUN_ID / "csv"

NB_PATH = (
    REPO
    / "notebooks"
    / "Ecoli_Analysis_Notebooks"
    / "estimation"
    / f"csd_estimation_analysis_{RUN_ID}.ipynb"
)

NB_DIR = NB_PATH.parent
VIZ_DIR = REPO / "notebooks" / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)


cells: list[dict] = []

cells.append(
    md(
        """\
# CSD estimation analysis — E. coli (Ecoli_Analysis_Notebooks/estimation)

This notebook aggregates per-edge CSV outputs from

`notebooks/Ecoli_Analysis_Notebooks/estimation/20260708_161623/csv`.

It produces plots showing:

1. **Estimated path coefficient** vs **ground-truth beta**
2. **Error metrics** (MAE / RMSE / bias) across simulation parameters
3. **Residual variance** changes with parameters, and its relationship to error
"""
    )
)

cells.append(
    code(
        f"""\
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from IPython.display import Image, display

REPO = Path().resolve()
while not (REPO / 'src' / 'nocap').exists() and REPO != REPO.parent:
    REPO = REPO.parent

INPUT_DIR = REPO / {str(INPUT_DIR)!r}
NB_DIR = INPUT_DIR.parents[1]  # notebooks/Ecoli_Analysis_Notebooks/estimation
VIZ_DIR = REPO / 'notebooks' / 'visualizations'
VIZ_DIR.mkdir(exist_ok=True)

print('Input CSV dir:', INPUT_DIR)
csv_paths = sorted(INPUT_DIR.glob('*.csv'))
print('Num CSV files:', len(csv_paths))
assert len(csv_paths) > 0, f'No CSVs found in: {INPUT_DIR}'
print('First CSV:', csv_paths[0].name)
"""
    )
)

cells.append(
    code(
        """\
# ============================================================
# Load and derive error columns
# ============================================================

dfs = []
for p in csv_paths:
    df = pd.read_csv(p)
    df['csv_file'] = p.name
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

print('Total rows:', f"{len(data):,}")
print('Status counts:')
print(data['status'].value_counts())

for col in [
    'estimated_path_coefficient',
    'stderr',
    'residual_variance',
    'ground_truth_beta',
]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Core derived metrics
data['error'] = data['estimated_path_coefficient'] - data['ground_truth_beta']
data['abs_error'] = data['error'].abs()
data['sq_error'] = data['error'] ** 2

eval_df = data[data['status'].isin(['identifiable', 'insufficient_data'])].copy()
print('Eval rows:', f"{len(eval_df):,}")


def summarize(df: pd.DataFrame) -> dict[str, float]:
    mae = float(df['abs_error'].mean())
    rmse = float(np.sqrt(df['sq_error'].mean()))
    bias = float(df['error'].mean())
    med_abs = float(df['abs_error'].median())
    return {'MAE': mae, 'RMSE': rmse, 'bias': bias, 'median_abs_error': med_abs}


overall = summarize(eval_df)
print('Overall estimation metrics (across all CSVs):')
for k, v in overall.items():
    print(f"  {k}: {v:.6g}")
"""
    )
)

cells.append(md("## 1) Estimated beta vs ground-truth beta"))

cells.append(
    code(
        """\
# Scatter plot + y=x reference line
plot_df = eval_df.dropna(subset=['ground_truth_beta', 'estimated_path_coefficient']).copy()

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_xscale('log')
ax.set_yscale('log')
ax.scatter(
    plot_df['ground_truth_beta'],
    plot_df['estimated_path_coefficient'],
    s=4,
    alpha=0.25,
    rasterized=True,
)

lims = [
    float(plot_df['ground_truth_beta'].min()),
    float(plot_df['ground_truth_beta'].max()),
]
pad = 0.05 * (lims[1] - lims[0] + 1e-12)
lo, hi = lims[0] - pad, lims[1] + pad
ax.plot([lo, hi], [lo, hi], color='black', linewidth=1, linestyle='--', alpha=0.7)

ax.set_xlabel('Ground-truth beta (log)')
ax.set_ylabel('Estimated path coefficient (log)')
ax.set_title('Estimated vs ground-truth beta (all parameter settings)')

note = '\\n'.join([f"{k}: {v:.3g}" for k, v in overall.items()])
ax.text(
    0.02,
    0.98,
    note,
    transform=ax.transAxes,
    va='top',
    ha='left',
    fontsize=9,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='0.8'),
)

plt.tight_layout()
out = VIZ_DIR / 'csd_estimation_beta_scatter.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()

display(Image(str(out)))
print(f'Saved: {out}')
"""
    )
)

cells.append(md("## 2) Error-metric breakdown across simulation parameters"))

cells.append(
    code(
        """\
# Grouped metrics
group_cols = [
    'missing_data_mechanism',
    'missing_data_rate',
    'missing_edge_rate',
    'n_samples',
]

g = (
    eval_df.groupby(group_cols, dropna=False)
    .agg(
        n=('error', 'size'),
        MAE=('abs_error', 'mean'),
        RMSE=('sq_error', lambda x: float(np.sqrt(np.mean(x)))),
        bias=('error', 'mean'),
        median_abs_error=('abs_error', 'median'),
    )
    .reset_index()
)

err_summary_csv = NB_DIR / 'csd_estimation_error_summary.csv'
g.to_csv(err_summary_csv, index=False)
print(f'Wrote error summary: {err_summary_csv}')


def plot_metric_heatmap(mechanism: str, data_rate: float, metric: str = 'RMSE'):
    sub = g[(g['missing_data_mechanism'] == mechanism) & (g['missing_data_rate'] == data_rate)]
    if len(sub) == 0:
        print('No data for:', mechanism, data_rate)
        return

    pivot = (
        sub.pivot_table(
            index='missing_edge_rate',
            columns='n_samples',
            values=f'{metric}',
            aggfunc='mean',
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(pivot.values, aspect='auto', interpolation='nearest')

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns])

    ax.set_xlabel('Sample size (n_samples)')
    ax.set_ylabel('Missing edge rate')
    ax.set_title(f'{metric} heatmap: {mechanism}, missing_data_rate={data_rate}')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{metric}')

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f'{val:.2g}', ha='center', va='center', fontsize=8, color='black')

    plt.tight_layout()
    out = VIZ_DIR / f'csd_estimation_{metric}_heatmap_{mechanism}_data_rate{data_rate}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    display(Image(str(out)))
    print(f'Saved: {out}')


mechs = sorted(eval_df['missing_data_mechanism'].unique())
data_rates = sorted(eval_df['missing_data_rate'].unique())

for mech in mechs:
    for dr in data_rates:
        plot_metric_heatmap(mech, float(dr), 'RMSE')
        plot_metric_heatmap(mech, float(dr), 'MAE')
"""
    )
)

cells.append(
    md(
        """\
## 2b) Adjustment set size affects error metrics (pipeline sparsity)

The `adjustment_set` column contains a pipe-separated list of adjustment nodes.
We compute its size and show how MAE/RMSE/bias (and residual variance) change as
the adjustment set gets larger.
"""
    )
)

cells.append(
    code(
        """\
# Compute adjustment-set size from the pipe-separated adjustment_set string
def adj_set_size(s) -> float:
    if pd.isna(s):
        return float('nan')
    parts = str(s).split('|')
    parts = [p for p in parts if p]
    return float(len(parts))

eval_tmp = eval_df.copy()
eval_tmp['adjustment_set_size'] = eval_tmp['adjustment_set'].apply(adj_set_size)

eval_tmp = eval_tmp.dropna(subset=['adjustment_set_size'])
eval_tmp['adjustment_set_size'] = eval_tmp['adjustment_set_size'].astype(int)

adj_summary = (
    eval_tmp.groupby('adjustment_set_size', dropna=False)
    .agg(
        n=('error', 'size'),
        MAE=('abs_error', 'mean'),
        RMSE=('sq_error', lambda x: float(np.sqrt(np.mean(x)))),
        bias=('error', 'mean'),
        residual_mean=('residual_variance', 'mean'),
        residual_median=('residual_variance', 'median'),
    )
    .reset_index()
    .sort_values('adjustment_set_size')
)

adj_csv = NB_DIR / 'csd_estimation_adjustment_set_size_summary.csv'
adj_summary.to_csv(adj_csv, index=False)
print(f'Wrote adjustment-set-size summary: {adj_csv}')

display(adj_summary.head(20))

# Plot MAE and residual variance vs adjustment set size
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

axes[0].plot(adj_summary['adjustment_set_size'], adj_summary['MAE'], marker='o', linewidth=2)
axes[0].set_xlabel('Adjustment-set size (# nodes)')
axes[0].set_ylabel('MAE')
axes[0].set_title('MAE vs adjustment-set size')
axes[0].grid(True, alpha=0.25)

axes[1].plot(adj_summary['adjustment_set_size'], adj_summary['residual_mean'], marker='o', linewidth=2, color='#d35400')
axes[1].set_xlabel('Adjustment-set size (# nodes)')
axes[1].set_ylabel('Residual variance (mean)')
axes[1].set_title('Residual variance vs adjustment-set size')
axes[1].grid(True, alpha=0.25)

plt.tight_layout()
out = VIZ_DIR / 'csd_estimation_mae_resid_vs_adjustment_set_size.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()

display(Image(str(out)))
print(f'Saved: {out}')
"""
    )
)

cells.append(
    code(
        """\
# Line plot: MAE vs n_samples for each missing_edge_rate (separate panels)

edge_rates = sorted(eval_df['missing_edge_rate'].unique())

mechs = sorted(eval_df['missing_data_mechanism'].unique())
data_rates = sorted(eval_df['missing_data_rate'].unique())

for mech in mechs:
    for dr in data_rates:
        sub = g[(g['missing_data_mechanism'] == mech) & (g['missing_data_rate'] == dr)].copy()
        if len(sub) == 0:
            continue

        sub = sub.sort_values(['missing_edge_rate', 'n_samples'])
        fig, axes = plt.subplots(
            1,
            len(edge_rates),
            figsize=(4.0 * len(edge_rates), 3.8),
            sharey=True,
        )
        if len(edge_rates) == 1:
            axes = [axes]

        for ax, er in zip(axes, edge_rates):
            s2 = sub[sub['missing_edge_rate'] == er]
            ax.plot(s2['n_samples'], s2['MAE'], marker='o', linewidth=2)
            ax.set_title(f'missing_edge_rate={er}')
            ax.set_xlabel('n_samples')
            ax.grid(True, alpha=0.25)

        axes[0].set_ylabel('MAE')
        fig.suptitle(f'MAE vs n_samples: {mech}, missing_data_rate={dr}')
        plt.tight_layout()

        out = VIZ_DIR / f'csd_estimation_mae_lines_{mech}_data_rate{dr}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()

        display(Image(str(out)))
        print(f'Saved: {out}')
"""
    )
)

cells.append(md("## 3) How residual variance changes with parameters"))

cells.append(
    code(
        """\
resid_group_cols = [
    'missing_data_mechanism',
    'missing_data_rate',
    'missing_edge_rate',
    'n_samples',
]

res = (
    eval_df.groupby(resid_group_cols, dropna=False)
    .agg(
        n=('residual_variance', 'size'),
        residual_mean=('residual_variance', 'mean'),
        residual_median=('residual_variance', 'median'),
        residual_std=('residual_variance', 'std'),
    )
    .reset_index()
)

res_csv = NB_DIR / 'csd_estimation_residual_variance_summary.csv'
res.to_csv(res_csv, index=False)
print(f'Wrote residual variance summary: {res_csv}')
"""
    )
)

cells.append(
    code(
        """\
# Line plot residual variance vs n_samples (residual mean), plus one boxplot per setting.
plot_df = eval_df.dropna(subset=['residual_variance', 'n_samples']).copy()

for mech in sorted(plot_df['missing_data_mechanism'].unique()):
    for dr in sorted(plot_df['missing_data_rate'].unique()):
        sub = plot_df[
            (plot_df['missing_data_mechanism'] == mech) & (plot_df['missing_data_rate'] == dr)
        ].copy()
        if len(sub) == 0:
            continue

        edge_rates_sorted = sorted(sub['missing_edge_rate'].unique())
        ns_sorted = sorted(sub['n_samples'].unique())

        # Mean residual variance lines
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for er in edge_rates_sorted:
            s2 = res[
                (res['missing_data_mechanism'] == mech)
                & (res['missing_data_rate'] == dr)
                & (res['missing_edge_rate'] == er)
            ].sort_values('n_samples')
            ax.plot(
                s2['n_samples'],
                s2['residual_mean'],
                marker='o',
                linewidth=2,
                label=f'edge_rate={er}',
            )

        ax.set_xlabel('n_samples')
        ax.set_ylabel('Residual variance (mean)')
        ax.set_title(f'Residual variance vs n_samples: {mech}, missing_data_rate={dr}')
        ax.grid(True, alpha=0.25)
        ax.legend()
        plt.tight_layout()

        out = VIZ_DIR / f'csd_estimation_residual_vs_n_{mech}_data_rate{dr}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        display(Image(str(out)))
        print(f'Saved: {out}')

        # Distribution via boxplot for the edge-rates
        for er in edge_rates:
            box_sub = sub[sub['missing_edge_rate'] == er]
            if len(box_sub) > 0:
                data_by_n = [
                    box_sub[box_sub['n_samples'] == n]['residual_variance'].dropna().values
                    for n in ns_sorted
                ]
                fig2, ax2 = plt.subplots(figsize=(9, 4.5))
                ax2.boxplot(data_by_n, labels=[str(n) for n in ns_sorted], showfliers=False)
                ax2.set_xlabel('n_samples')
                ax2.set_ylabel('Residual variance')
                ax2.set_title(
                    f'Residual variance distribution (edge_rate={er}): {mech}, missing_data_rate={dr}'
                )
                plt.tight_layout()
                out2 = VIZ_DIR / f'csd_estimation_residual_box_{mech}_data_rate{dr}_edge{er}.png'
                plt.savefig(out2, dpi=150, bbox_inches='tight')
                plt.close()
                display(Image(str(out2)))
                print(f'Saved: {out2}')
"""
    )
)

cells.append(md("## 4) Residual variance vs estimation error"))

cells.append(
    code(
        """\
plot_df = eval_df.dropna(subset=['residual_variance', 'abs_error', 'missing_data_mechanism']).copy()

fig, ax = plt.subplots(figsize=(7.5, 5.8))
for mech in sorted(plot_df['missing_data_mechanism'].unique()):
    s2 = plot_df[plot_df['missing_data_mechanism'] == mech]
    ax.scatter(
        s2['residual_variance'],
        s2['abs_error'],
        s=4,
        alpha=0.25,
        label=mech,
        rasterized=True,
    )

ax.set_xlabel('Residual variance')
ax.set_ylabel('Absolute error |estimated - ground truth|')
ax.set_title('Residual variance vs estimation absolute error')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', alpha=0.25)
ax.legend()

plt.tight_layout()
out = VIZ_DIR / 'csd_estimation_residual_vs_abs_error.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()

display(Image(str(out)))
print(f'Saved: {out}')
"""
    )
)

cells.append(
    code(
        """\
# Correlation summary per setting (helps interpret residual changes)
def corr_safe(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])

link = (
    eval_df.groupby(['missing_data_mechanism', 'missing_data_rate', 'missing_edge_rate', 'n_samples'], dropna=False)
    .apply(lambda d: pd.Series({
        'corr_resid_vs_abs_error': corr_safe(d['residual_variance'].fillna(0.0), d['abs_error'].fillna(0.0)),
        'residual_mean': float(d['residual_variance'].mean()),
        'MAE': float(d['abs_error'].mean()),
        'n': len(d),
    }))
    .reset_index()
)

link_csv = NB_DIR / 'csd_estimation_residual_vs_error_link_summary.csv'
link.to_csv(link_csv, index=False)
print(f'Wrote link summary: {link_csv}')

print('Top 10 settings by residual_mean:')
display(link.sort_values('residual_mean', ascending=False).head(10))
"""
    )
)


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": cells,
}


NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook written to: {NB_PATH}")
print(f"  {len(cells)} cells")
