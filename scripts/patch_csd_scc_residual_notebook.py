"""patch_csd_scc_residual_notebook.py — Add §1c (Residual-SCC Analysis) to
Cyclic_SingleDoor_Analysis.ipynb.

Inserts two cells after the existing §1b (Computational Limitations) section:
  - A markdown cell with the §1c heading, crosstab table, and interpretation.
  - A code cell that generates the SCC-size-by-status stacked bar chart
    and saves it to notebooks/visualizations/csd_scc_size_by_status.png.

Usage:
    uv run python scripts/patch_csd_scc_residual_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).parent.parent
NB_PATH = REPO / "notebooks" / "Ecoli_Analysis_Notebooks" / "Cyclic_SingleDoor_Analysis.ipynb"
VIZ_DIR = REPO / "notebooks" / "visualizations"

# ---------------------------------------------------------------------------
# Cell content
# ---------------------------------------------------------------------------

MARKDOWN_CELL = {
    "cell_type": "markdown",
    "id": "csd-scc-residual-md",
    "metadata": {},
    "source": [
        "## §1c  Residual-SCC Analysis — Are endpoints still co-SCC after edge removal?\n",
        "\n",
        "To further characterise *why* the O-set search is intractable for 80.8 % of\n",
        "edges, we ran a cheap purely-structural diagnostic: for every edge $u \\to v$\n",
        "we asked whether $u$ and $v$ remain in the **same strongly-connected component**\n",
        "after that edge is deleted.\n",
        "\n",
        "**Method.** One Tarjan SCC pass on the full graph ($O(V + E)$, ≈ 15 ms), then\n",
        "for each same-SCC edge a recompute on the induced subgraph with the edge\n",
        "removed.  Cross-SCC edges are trivially `same_scc_after = False` with no\n",
        "recompute.  Self-loops (174 edges) are flagged as degenerate and excluded from\n",
        "the crosstabs.  Total wall time: **< 2 s** on the head node.\n",
        "\n",
        "### Key results\n",
        "\n",
        "| | identifiable | timeout | unidentifiable |\n",
        "|---|---|---|---|\n",
        "| `same_scc_after = False` | **1,439** | 7,164 | 148 |\n",
        "| `same_scc_after = True`  | 0 | 278 | 8 |\n",
        "\n",
        "**SCC-size distribution by status** (non-self-loops):\n",
        "\n",
        "| SCC size bucket | identifiable | timeout | unidentifiable |\n",
        "|---|---|---|---|\n",
        "| 1 (singleton) | 1,373 | 275 | 0 |\n",
        "| 2–4 | 16 | 0 | 74 |\n",
        "| 5–9 | 50 | 0 | 54 |\n",
        "| 50–68 (giant SCC) | **0** | **7,167** | 28 |\n",
        "\n",
        "**96.3 % of timeouts** involve at least one endpoint inside the giant 68-node\n",
        "SCC.  Zero identifiable edges touch the giant SCC.  This confirms the\n",
        "mechanism described in §1b: the O-set enumeration becomes intractable\n",
        "precisely because the giant feedback SCC is reachable from (or reachable to)\n",
        "nearly every other node in the network.\n",
        "\n",
        "### Interpretation\n",
        "\n",
        "* **Identifiability requires feedback-free paths.**  Every identified edge\n",
        "  spans two *different* SCCs in the original graph, meaning both endpoints\n",
        "  are topologically separated by acyclic structure that the O-set can\n",
        "  exploit.  No intra-SCC edge is ever identifiable by the single-door\n",
        "  criterion.\n",
        "\n",
        "* **The 68-node giant SCC is the computational bottleneck.**  It is not merely\n",
        "  that the question is theoretically undecidable — the O-set path enumeration\n",
        "  inside this dense feedback hub grows combinatorially and exhausts the\n",
        "  timeout before a decision is reached, even at 20 min/edge (see §1b).\n",
        "\n",
        "* **Self-loops (174 edges) are separately flagged** as degenerate; they are\n",
        "  excluded from all identifiability and SCC analyses above.\n",
    ],
}

CODE_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "id": "csd-scc-residual-fig",
    "metadata": {},
    "outputs": [],
    "source": [
        "# §1c figure — SCC-size distribution by status (stacked bar)\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.ticker as ticker\n",
        "from pathlib import Path\n",
        "\n",
        "REPO = Path('.').resolve().parents[1]\n",
        "residual_csv = REPO / 'results' / 'cyclic_single_door' / 'csd_scc_residual.csv'\n",
        "viz_dir = REPO / 'notebooks' / 'visualizations'\n",
        "viz_dir.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "df = pd.read_csv(residual_csv)\n",
        "non_self = df[~df['is_self_loop']].copy()\n",
        "\n",
        "bin_edges = [0, 1, 2, 5, 10, 50, 10_000]\n",
        "labels    = ['1', '2-4', '5-9', '10-49', '50-68\\n(giant)', 'large']\n",
        "labels    = labels[: len(bin_edges) - 1]\n",
        "non_self['scc_bucket'] = pd.cut(\n",
        "    non_self['scc_size_before'],\n",
        "    bins=bin_edges, labels=labels, right=True\n",
        ")\n",
        "\n",
        "ct = pd.crosstab(non_self['scc_bucket'], non_self['status'])\n",
        "# Reorder columns for visual clarity\n",
        "col_order = [c for c in ['identifiable', 'unidentifiable', 'timeout'] if c in ct.columns]\n",
        "ct = ct[col_order]\n",
        "\n",
        "colours = {'identifiable': '#2ca02c', 'unidentifiable': '#d62728', 'timeout': '#aec7e8'}\n",
        "bar_colours = [colours.get(c, '#888888') for c in ct.columns]\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(8, 5))\n",
        "ct.plot(kind='bar', stacked=True, color=bar_colours, ax=ax, width=0.6)\n",
        "\n",
        "ax.set_xlabel('SCC size before edge removal', fontsize=12)\n",
        "ax.set_ylabel('Number of edges', fontsize=12)\n",
        "ax.set_title(\n",
        "    'E. coli network: edge count by SCC size and identifiability status\\n'\n",
        "    '(174 self-loops excluded; giant SCC = 68 nodes)',\n",
        "    fontsize=11\n",
        ")\n",
        "ax.tick_params(axis='x', rotation=0)\n",
        "ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))\n",
        "ax.legend(title='Status', bbox_to_anchor=(1.02, 1), loc='upper left')\n",
        "ax.spines[['top', 'right']].set_visible(False)\n",
        "\n",
        "out = viz_dir / 'csd_scc_size_by_status.png'\n",
        "fig.tight_layout()\n",
        "fig.savefig(out, dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print(f'Saved: {out}')\n",
    ],
}


# ---------------------------------------------------------------------------
# Patch the notebook
# ---------------------------------------------------------------------------

def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]

    # Find the last §1b cell (the code cell with the scope summary / timeout table)
    # We look for the most recent cell whose source contains "1b" or "Computational"
    insert_after = -1
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        if "1b" in src or "Computational Limitations" in src or "calibrat" in src.lower():
            insert_after = i

    if insert_after == -1:
        # Fallback: append at end
        insert_after = len(cells) - 1
        print(f"WARNING: could not find §1b anchor — appending after cell {insert_after}")
    else:
        print(f"Inserting §1c after cell index {insert_after}")

    # Insert the two new cells after §1b
    cells.insert(insert_after + 1, MARKDOWN_CELL)
    cells.insert(insert_after + 2, CODE_CELL)

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"Patched: {NB_PATH}")
    print(f"  Total cells now: {len(cells)}")


if __name__ == "__main__":
    main()
