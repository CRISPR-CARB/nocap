"""build_csd_estimation_notebook.py — Generate CSD estimation analysis notebook.

This script programmatically builds a Jupyter notebook under:
`notebooks/Ecoli_Analysis_Notebooks/estimation/`.

The notebook aggregates per-edge CSV outputs from:
`notebooks/Ecoli_Analysis_Notebooks/estimation/{RUN_ID}/csv`.

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
    / RUN_ID
    / f"csd_estimation_analysis_{RUN_ID}.ipynb"
)

NB_DIR = NB_PATH.parent
VIZ_DIR = REPO / NB_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)


cells: list[dict] = []

cells.append(
    md(
        """\
# CSD estimation analysis — E. coli (Ecoli_Analysis_Notebooks/estimation)

This notebook aggregates per-edge CSV outputs from

`notebooks/Ecoli_Analysis_Notebooks/estimation/{RUN_ID}/csv`.

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
NB_DIR = INPUT_DIR.parent
VIZ_DIR = REPO / NB_DIR / 'visualizations'
VIZ_DIR.mkdir(exist_ok=True)
RUN_METADATA = NB_DIR / 'run_metadata.json'
run_metadata = {{}}
if RUN_METADATA.exists():
    run_metadata = pd.read_json(RUN_METADATA, typ='series').to_dict()

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

# Relative slope error: absolute error normalized by ground-truth beta
# (guard against division by zero / non-finite values)
data['relative_slope_error'] = data['abs_error'] / data['ground_truth_beta']
data.loc[~np.isfinite(data['relative_slope_error']), 'relative_slope_error'] = np.nan

eval_df = data[data['status'].isin(['identifiable', 'insufficient_data'])].copy()
print('Eval rows:', f"{len(eval_df):,}")

print('Replicate intervals are empirical seed distributions, not row-bootstrap confidence intervals.')
print('Full mode includes SCM and observation variability; fixed-SCM mode is conditional on one SCM block.')
print('Full-mode bands describe end-to-end simulated performance; fixed-SCM bands describe conditional data-generation performance within each missing_edge_rate SCM block.')

# Replicate summaries use seed identities rather than treating edge rows as replicates.
for col in ['scm_seed', 'data_seed', 'seed']:
    if col not in data:
        data[col] = np.nan
    data[col] = pd.to_numeric(data[col], errors='coerce')
for col in [
    'n_samples', 'missing_edge_rate', 'missing_data_rate',
    'estimated_path_coefficient', 'stderr', 'residual_variance',
    'ground_truth_beta', 'scm_true_missing_edges_count', 'n_rows_used',
]:
    assert col in data.columns, f'Missing required column: {col}'
    data[col] = pd.to_numeric(data[col], errors='coerce')
legacy_seed_fallback = data['scm_seed'].isna() | data['data_seed'].isna()
if legacy_seed_fallback.any():
    data.loc[legacy_seed_fallback, 'scm_seed'] = data.loc[legacy_seed_fallback, 'seed']
    data.loc[legacy_seed_fallback, 'data_seed'] = data.loc[legacy_seed_fallback, 'seed']
    print('Legacy fallback: seed is used for both SCM and data identities.')
assert data[['scm_seed', 'data_seed']].notna().all().all(), 'Missing replicate seed identity'
assert data[['n_samples', 'missing_edge_rate', 'missing_data_rate']].notna().all().all(), 'Missing cell parameter'
eval_df = data[data['status'].isin(['identifiable', 'insufficient_data'])].copy()
metadata_mode = run_metadata.get('bootstrap_mode') if run_metadata.get('bootstrap') else None
design = metadata_mode or ('full' if any('full' in p.name for p in csv_paths) else ('fixed_scm' if any('fixed_scm' in p.name for p in csv_paths) else 'legacy'))

# missing_edge_rate defines the SCM block. In fixed-SCM mode, data seeds may
# be compared only within one edge-rate block: changing edge rate changes the
# generated SCM and therefore does not produce a conditional replicate.
scm_block_cols = ['missing_edge_rate']
if design == 'fixed_scm':
    block_seed_counts = (data.groupby(scm_block_cols, dropna=False)['scm_seed']
        .nunique(dropna=True))
    assert (block_seed_counts <= 1).all(), (
        'fixed_scm requires one scm_seed per missing_edge_rate SCM block: '
        f'{block_seed_counts.to_dict()}'
    )
    block_metadata = (data.groupby(scm_block_cols, dropna=False)
        .agg(scm_seed_count=('scm_seed', 'nunique'),
             missing_edge_counts=('scm_true_missing_edges_count', 'nunique')))
    print('Fixed-SCM blocks by missing_edge_rate:')
    display(block_metadata.reset_index())
    assert (block_metadata['scm_seed_count'] <= 1).all()
    beta_by_edge = (data.groupby(scm_block_cols + ['cause', 'effect'], dropna=False)
        .agg(beta_values=('ground_truth_beta', 'nunique')).reset_index())
    assert (beta_by_edge['beta_values'] <= 1).all(), (
        'fixed_scm beta metadata varies within an edge/SCM block'
    )
    assert (block_metadata['missing_edge_counts'] <= 1).all(), (
        'fixed_scm true-missing-edge metadata varies within an SCM block'
    )

# Include the SCM identity in the grouping even for fixed-SCM mode. The
# edge-rate column is intentionally retained so different SCM blocks cannot
# be pooled into one interval.
data['replicate_id'] = list(zip(data['scm_seed'], data['data_seed'])) if design != 'fixed_scm' else list(zip(data['missing_edge_rate'], data['data_seed']))
rep_cols = ['missing_data_mechanism', 'missing_data_rate', 'missing_edge_rate', 'n_samples', 'replicate_id']
replicate_metrics = (eval_df.groupby(rep_cols, dropna=False)
    .agg(MAE=('abs_error', 'mean'), RMSE=('sq_error', lambda x: float(np.sqrt(np.mean(x)))),
         bias=('error', 'mean'), median_abs_error=('abs_error', 'median')).reset_index())
ci_summary = (replicate_metrics.groupby(rep_cols[:-1], dropna=False)
    .agg(total_replicates=('replicate_id', 'nunique'), usable_replicates=('MAE', 'count'),
         mean_error=('bias', 'mean'), mean_MAE=('MAE', 'mean'), mean_RMSE=('RMSE', 'mean'), mean_bias=('bias', 'mean'),
         median_abs_error=('median_abs_error', 'mean'),
         error_lower=('bias', lambda x: x.quantile(.025)), error_upper=('bias', lambda x: x.quantile(.975)),
         MAE_lower=('MAE', lambda x: x.quantile(.025)), MAE_upper=('MAE', lambda x: x.quantile(.975)),
          RMSE_lower=('RMSE', lambda x: x.quantile(.025)), RMSE_upper=('RMSE', lambda x: x.quantile(.975))).reset_index())
assert not replicate_metrics.duplicated(rep_cols).any(), 'Duplicate rows counted as one replicate'
ci_summary['failed_replicates'] = ci_summary['total_replicates'] - ci_summary['usable_replicates']
ci_summary['unique_seed_count'] = ci_summary['usable_replicates']
ci_summary['scm_block_key'] = ci_summary['missing_edge_rate'].map(lambda x: f'missing_edge_rate={x}')
ci_summary['interval_interpretation'] = np.where(
    design == 'fixed_scm',
    'conditional on one SCM realization within this missing_edge_rate block',
    'end-to-end variability across SCM and observed-data generation')
ci_summary.to_csv(NB_DIR / f'csd_estimation_{design}_seed_ci_summary.csv', index=False)
print('Replicate design:', design)
print('Replicate completeness by cell:')
display(ci_summary[['missing_data_mechanism', 'missing_data_rate', 'missing_edge_rate', 'n_samples',
                    'total_replicates', 'usable_replicates', 'failed_replicates']])


def summarize(df: pd.DataFrame) -> dict[str, float]:
    mae = float(df['abs_error'].mean())
    rmse = float(np.sqrt(df['sq_error'].mean()))
    bias = float(df['error'].mean())
    med_abs = float(df['abs_error'].median())
    rel_mean = float(df['relative_slope_error'].mean())
    rel_median = float(df['relative_slope_error'].median())
    return {
        'MAE': mae,
        'RMSE': rmse,
        'bias': bias,
        'median_abs_error': med_abs,
        'relative_slope_error_mean': rel_mean,
        'relative_slope_error_median': rel_median,
    }


overall = summarize(eval_df)
print('Overall estimation metrics (across all CSVs):')
for k, v in overall.items():
    print(f"  {k}: {v:.6g}")
"""
    )
)

cells.append(md("""## Bootstrap modes and interval interpretation

The two modes repeat the simulation with independent random seeds, rather than
resampling rows from one observed dataset:

- **Full mode** assigns an independent `(scm_seed, data_seed)` pair to every
  parameter cell and replicate. Each replicate redraws the structural beta
  coefficients and the realized true missing-edge pattern, then redraws the
  observed data. Its empirical distribution measures end-to-end variation from
  both SCM construction and observation/data generation.
- **Fixed-SCM mode** assigns one `scm_seed` to each `missing_edge_rate` SCM
  block and varies only `data_seed`. Its empirical distribution is conditional
  on that one structural realization: it measures variation from sampling,
  UMI/count generation, library sizes, and missingness. Different
  `missing_edge_rate` values are different SCM blocks and are never pooled into
  one fixed-SCM interval.

The shaded regions are empirical 95% percentile bands over completed,
independent simulation replicates. They describe the simulated distribution
of the displayed performance metric, not a formal confidence interval for a
single real biological edge. In particular, a full-mode band includes
structural-model uncertainty represented by the simulation, while a
fixed-SCM band is conditional and does not generalize beyond its selected SCM
without rerunning additional SCM seeds. Neither mode is a nonparametric
row-bootstrap confidence interval, and neither resamples observations from one
dataset. With few replicates, percentile endpoints are descriptive and can be
unstable; cells with too few usable replicates are shown without bands.
"""))

cells.append(code("""
plot_summary = ci_summary.copy()
plot_summary['has_interval'] = (
    (plot_summary['usable_replicates'] >= 2)
    & np.isfinite(plot_summary['error_lower'])
    & np.isfinite(plot_summary['error_upper'])
)

def plot_seed_bands(metric, lower, upper, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, part in plot_summary.groupby(
        ['missing_data_mechanism', 'missing_data_rate', 'missing_edge_rate'],
        dropna=False):
        part = part.sort_values('n_samples')
        ax.plot(part['n_samples'], part[metric], marker='o',
                label=f'mech={key[0]}, data={key[1]}, edge={key[2]}')
        band = part[part['has_interval']]
        if not band.empty:
            ax.fill_between(band['n_samples'], band[lower], band[upper], alpha=0.15)
    ax.set_xscale('log')
    ax.set_xlabel('n_samples')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = VIZ_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    display(Image(str(out)))
    print(f'Saved: {out}')

plot_seed_bands('mean_error', 'error_lower', 'error_upper', 'Mean error',
                f'Mean error with empirical 95% intervals ({design})',
                f'csd_estimation_{design}_mean_error_ci_vs_n_samples.png')
plot_seed_bands('mean_MAE', 'MAE_lower', 'MAE_upper', 'Mean MAE',
                f'Mean MAE with empirical 95% intervals ({design})',
                f'csd_estimation_{design}_mae_ci_vs_n_samples.png')

for x, label, suffix in [
    ('missing_edge_rate', 'Missing-edge rate (separate SCM blocks)', 'missing_edge_rate'),
    ('missing_data_rate', 'Missing-data rate', 'missing_data_rate'),
]:
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, part in plot_summary.groupby(
        ['missing_data_mechanism', 'missing_data_rate', 'missing_edge_rate'],
        dropna=False):
        part = part.sort_values(x)
        ax.plot(part[x], part['mean_error'], marker='o',
                label=f'mech={key[0]}, data={key[1]}, edge={key[2]}')
        band = part[part['has_interval']]
        if not band.empty:
            ax.fill_between(band[x], band['error_lower'], band['error_upper'], alpha=0.15)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel(label)
    ax.set_ylabel('Mean error')
    ax.set_title(f'Mean error with empirical 95% intervals versus {label} ({design})')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = VIZ_DIR / f'csd_estimation_{design}_mean_error_ci_vs_{suffix}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    display(Image(str(out)))
    print(f'Saved: {out}')
"""))

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

cells.append(md("## 1b) Relative slope error (|estimated-true| / true) vs ground-truth beta"))

cells.append(
    code(
        """\
# Scatter plot: relative_slope_error vs ground-truth beta
plot_df = eval_df.dropna(subset=['ground_truth_beta', 'relative_slope_error']).copy()
plot_df = plot_df[np.isfinite(plot_df['relative_slope_error'])]

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_xscale('log')
ax.set_yscale('log')
ax.scatter(
    plot_df['ground_truth_beta'],
    plot_df['relative_slope_error'],
    s=4,
    alpha=0.25,
    rasterized=True,
)

ax.set_xlabel('Ground-truth beta (log)')
ax.set_ylabel('Relative slope error (|error| / beta) (log)')
ax.set_title('Relative slope error vs ground-truth beta (all parameter settings)')

note = '\\n'.join([f"{k}: {v:.3g}" for k, v in overall.items() if k.startswith('relative_slope_error')])
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
out = VIZ_DIR / 'csd_estimation_relative_slope_error_scatter.png'
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
        RelativeSlopeError=('relative_slope_error', 'mean'),
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
        plot_metric_heatmap(mech, float(dr), 'RelativeSlopeError')
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
# Filter to rows with no missing edges, no missing data, and sample size > 10000
eval_filtered = eval_df[
    (eval_df["missing_edge_rate"] == 0)
    & (eval_df["missing_data_rate"] == 0)
    & (eval_df["n_samples"] > 10000)
].copy()

print(f"Filtered rows: {len(eval_filtered):,}")

# Compute adjustment-set size (uses adj_set_size defined earlier)
eval_filtered["adjustment_set_size"] = eval_filtered["adjustment_set"].apply(adj_set_size)
eval_filtered = eval_filtered.dropna(subset=["adjustment_set_size"])
eval_filtered["adjustment_set_size"] = eval_filtered["adjustment_set_size"].astype(int)

# Aggregate metrics by adjustment_set_size
adj_summary = (
    eval_filtered.groupby("adjustment_set_size", dropna=False)
    .agg(
        n=("error", "size"),
        MAE=("abs_error", "mean"),
        RMSE=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
        bias=("error", "mean"),
        residual_mean=("residual_variance", "mean"),
        residual_median=("residual_variance", "median"),
    )
    .reset_index()
    .sort_values("adjustment_set_size")
)

adj_csv = NB_DIR / "csd_estimation_adjustment_set_size_summary_no_missing_n_gt10000.csv"
adj_summary.to_csv(adj_csv, index=False)
print(f"Wrote adjustment-set-size summary: {adj_csv}")

display(adj_summary.head(20))

# Plot MAE and residual variance vs adjustment set size
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

axes[0].plot(adj_summary["adjustment_set_size"], adj_summary["MAE"], marker="o", linewidth=2)
axes[0].set_xlabel("Adjustment-set size (# nodes)")
axes[0].set_ylabel("MAE")
axes[0].set_title("MAE vs adjustment-set size (no missing, n>10000)")
axes[0].grid(True, alpha=0.25)

axes[1].plot(
    adj_summary["adjustment_set_size"],
    adj_summary["residual_mean"],
    marker="o",
    linewidth=2,
    color="#d35400",
)
axes[1].set_xlabel("Adjustment-set size (# nodes)")
axes[1].set_ylabel("Residual variance (mean)")
axes[1].set_title("Residual variance vs adjustment-set size (no missing, n>10000)")
axes[1].grid(True, alpha=0.25)

plt.tight_layout()
out = VIZ_DIR / "csd_estimation_mae_resid_vs_adjustment_set_size_no_missing_n_gt10000.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()

display(Image(str(out)))
print(f"Saved: {out}")
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
            ax.set_xscale('log')
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
