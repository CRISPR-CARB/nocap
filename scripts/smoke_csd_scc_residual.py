"""smoke_csd_scc_residual.py — Verify §1c notebook cells and generate the figure.

Checks:
1. Notebook contains the §1c markdown cell (heading present).
2. Notebook contains the §1c code cell (csd_scc_residual.csv present).
3. Runs the figure-generating code and confirms PNG is saved.
4. Verifies key numbers in the results CSV.

Usage:
    uv run python scripts/smoke_csd_scc_residual.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
NB_PATH = REPO / "notebooks" / "Ecoli_Analysis_Notebooks" / "Cyclic_SingleDoor_Analysis.ipynb"
RESIDUAL_CSV = REPO / "results" / "cyclic_single_door" / "csd_scc_residual.csv"
VIZ_OUT = REPO / "notebooks" / "visualizations" / "csd_scc_size_by_status.png"

errors: list[str] = []


def check(condition: bool, msg: str) -> None:
    if condition:
        print(f"  OK  {msg}")
    else:
        print(f"  FAIL {msg}")
        errors.append(msg)


# ---------------------------------------------------------------------------
# 1. Notebook structure
# ---------------------------------------------------------------------------
print("=== Notebook structure ===")
nb = json.loads(NB_PATH.read_text())
cells = nb["cells"]
sources = ["".join(c.get("source", [])) for c in cells]

has_1c_md = any("1c" in s and "Residual" in s for s in sources)
has_1c_code = any("csd_scc_residual.csv" in s for s in sources)

check(has_1c_md, "§1c markdown cell present (heading + table)")
check(has_1c_code, "§1c code cell present (loads csd_scc_residual.csv)")
check(len(cells) >= 28, f"Notebook has >= 28 cells (has {len(cells)})")

# ---------------------------------------------------------------------------
# 2. Results CSV sanity
# ---------------------------------------------------------------------------
print("\n=== Results CSV ===")
check(RESIDUAL_CSV.exists(), f"csd_scc_residual.csv exists at {RESIDUAL_CSV}")

if RESIDUAL_CSV.exists():
    import pandas as pd  # type: ignore

    df = pd.read_csv(RESIDUAL_CSV)
    check(len(df) == 9211, f"CSV has 9,211 rows (has {len(df)})")

    n_selfloops = df["is_self_loop"].sum()
    check(n_selfloops == 174, f"174 self-loops flagged (found {n_selfloops})")

    non_self = df[~df["is_self_loop"]]
    n_timeout_giant = (
        (non_self["status"] == "timeout") & (non_self["scc_size_before"] >= 50)
    ).sum()
    n_timeout_total = (non_self["status"] == "timeout").sum()
    pct = 100.0 * n_timeout_giant / n_timeout_total if n_timeout_total else 0
    check(pct > 95, f">=95% of timeouts in giant SCC (got {pct:.1f}%)")

    n_ident_giant = (
        (non_self["status"] == "identifiable") & (non_self["scc_size_before"] >= 50)
    ).sum()
    check(n_ident_giant == 0, f"0 identifiable edges touch giant SCC (found {n_ident_giant})")

    # --- Regression guard: adjustment_set column must be present and populated ---
    check("adjustment_set" in df.columns, "residual CSV contains 'adjustment_set' column")
    check("same_scc" in df.columns, "residual CSV contains 'same_scc' column")
    check("timed_out" in df.columns, "residual CSV contains 'timed_out' column")

    ident = df[df["status"] == "identifiable"]
    if "adjustment_set" in df.columns and len(ident) > 0:
        n_with_adj = ident["adjustment_set"].notna().sum()
        check(
            n_with_adj > 0,
            f"at least one identifiable edge has a non-null adjustment_set "
            f"(found {n_with_adj}/{len(ident)})",
        )

# ---------------------------------------------------------------------------
# 3. Generate the figure
# ---------------------------------------------------------------------------
print("\n=== Figure generation ===")

fig_script = REPO / "scripts" / "_smoke_fig_tmp.py"
fig_script.write_text(
    f"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

REPO = Path(r'{REPO}')
residual_csv = REPO / 'results' / 'cyclic_single_door' / 'csd_scc_residual.csv'
viz_dir = REPO / 'notebooks' / 'visualizations'
viz_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(residual_csv)
non_self = df[~df['is_self_loop']].copy()

bin_edges = [0, 1, 2, 5, 10, 50, 10_000]
labels    = ['1', '2-4', '5-9', '10-49', '50-68\\n(giant)', 'large']
labels    = labels[: len(bin_edges) - 1]
non_self['scc_bucket'] = pd.cut(non_self['scc_size_before'], bins=bin_edges, labels=labels, right=True)

ct = pd.crosstab(non_self['scc_bucket'], non_self['status'])
col_order = [c for c in ['identifiable', 'unidentifiable', 'timeout'] if c in ct.columns]
ct = ct[col_order]

colours = {{'identifiable': '#2ca02c', 'unidentifiable': '#d62728', 'timeout': '#aec7e8'}}
bar_colours = [colours.get(c, '#888888') for c in ct.columns]

fig, ax = plt.subplots(figsize=(8, 5))
ct.plot(kind='bar', stacked=True, color=bar_colours, ax=ax, width=0.6)
ax.set_xlabel('SCC size before edge removal', fontsize=12)
ax.set_ylabel('Number of edges', fontsize=12)
ax.set_title('E. coli: edge count by SCC size and status (174 self-loops excluded)', fontsize=11)
ax.tick_params(axis='x', rotation=0)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{{int(x):,}}'))
ax.legend(title='Status', bbox_to_anchor=(1.02, 1), loc='upper left')
ax.spines[['top', 'right']].set_visible(False)

out = viz_dir / 'csd_scc_size_by_status.png'
fig.tight_layout()
fig.savefig(out, dpi=150, bbox_inches='tight')
print('Saved:', out)
"""
)

try:
    result = subprocess.run(
        [shutil.which("uv") or "uv", "run", "python", str(fig_script)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    if result.returncode == 0:
        check(VIZ_OUT.exists(), f"PNG saved to {VIZ_OUT}")
        print(f"  Figure output: {result.stdout.strip()}")
    else:
        errors.append(f"Figure script failed: {result.stderr[-500:]}")
        print(f"  FAIL figure script: {result.stderr[-300:]}")
finally:
    fig_script.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
if errors:
    print(f"FAIL: {len(errors)} check(s) failed:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("PASS: all §1c checks passed.")
