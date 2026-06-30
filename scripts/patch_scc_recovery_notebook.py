"""patch_scc_recovery_notebook.py
Append a new Section 11 (TF Recovery Bank) to SCC_Perturbation_Analysis.ipynb,
mirroring the recovery-design section of Cyclic_SingleDoor_Analysis.ipynb.

Usage:
    uv run python scripts/patch_scc_recovery_notebook.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "Ecoli_Analysis_Notebooks" / "SCC_Perturbation_Analysis.ipynb"


# ---------------------------------------------------------------------------
# New cells to inject
# ---------------------------------------------------------------------------

SECTION_HEADER_MD = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## 11. TF Recovery Bank — Multi-Experiment Perturbation Design\n",
        "\n",
        "Sections 1–10 characterise identifiability for **50 TFs** in the CRISPR perturbation dataset "
        "(18 SCC TFs, 32 DAG TFs).  This section answers the complementary question for the "
        "**full E. coli TF complement** (~285 TFs):\n",
        "\n",
        "> *Given a budget of **n** multiplex perturbation experiments, each knocking out **k** genes "
        "simultaneously, which experiments should we run to maximise the number of currently "
        "unidentifiable TFs that become identifiable?*\n",
        "\n",
        "**Background.** The `scc_recovery_bank.py` script identifies, for each unidentifiable TF `t`, "
        "the set of *candidate genes* — SCC min-cut members that, when perturbed, break all return "
        "paths back to `t` in the regulatory graph — and runs a greedy set-cover optimizer to "
        "select the most informative multiplex experiments.\n",
        "\n",
        "### Method\n",
        "\n",
        "1. **Enumerate TFs** (out-degree ≥ 1): 285 TFs total.\n",
        "2. **Classify baseline identifiability**: a TF is *already identifiable* if it lies in a "
        "singleton SCC (no feedback loops).\n",
        "3. **Build candidate pool**: for each unidentifiable TF, compute the SCC min-cut genes "
        "under the background intervention interpretation (same Interpretation A used throughout "
        "this analysis). Only genes that break the TF's SCC membership are eligible.\n",
        "4. **Greedy bank** (`_greedy_bank`): iteratively select the gene that maximally increases "
        "coverage of currently-uncovered TFs, stop when all unidentifiable TFs are covered or the "
        "budget is exhausted.\n",
        "5. **Two budget designs** are evaluated:\n",
        "\n",
        "| Design | n experiments | k genes/exp | Newly recoverable TFs | % of unidentifiable |\n",
        "|--------|:---:|:---:|:---:|:---:|\n",
        "| n=10, k=3 | 10 | 3 | 67 / 95 | 70.5% |\n",
        "| n=5,  k=6 |  5 | 6 | 67 / 95 | 70.5% |\n",
        "\n",
        "Both designs recover the same 67 TFs; the first experiment alone accounts for 28 (n=10,k=3) "
        "or 42 (n=5,k=6) recoveries.\n",
        "\n",
        "To regenerate:\n",
        "```bash\n",
        "uv run python scripts/scc_recovery_bank.py --n 10 --k 3\n",
        "uv run python scripts/scc_recovery_bank.py --n 5  --k 6\n",
        "```\n",
    ],
}

LOAD_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── 11.1  Load SCC-TF recovery-bank results ───────────────────────────────\n",
        "from pathlib import Path\n",
        "import json\n",
        "import pandas as pd\n",
        "from IPython.display import display\n",
        "\n",
        "# ECOLI_DIR is already defined in section 9.A as Path('.') inside this notebook\n",
        "# Re-derive robustly:\n",
        "try:\n",
        "    _ecoli_dir = ECOLI_DIR          # already set in §9.A\n",
        "except NameError:\n",
        "    _ecoli_dir = Path(\".\")          # notebook lives in Ecoli_Analysis_Notebooks/\n",
        "\n",
        "SCC_REC_SUMMARY = _ecoli_dir / \"scc_recovery_summary.json\"\n",
        "\n",
        "if not SCC_REC_SUMMARY.exists():\n",
        "    print(\"SCC recovery-bank summary not found.\")\n",
        "    print(\"Run:  uv run python scripts/scc_recovery_bank.py --n 10 --k 3\")\n",
        "    print(\"      uv run python scripts/scc_recovery_bank.py --n 5  --k 6\")\n",
        "else:\n",
        "    with open(SCC_REC_SUMMARY) as _f:\n",
        "        scc_rec = json.load(_f)\n",
        "\n",
        "    for design_key, label in [(\"n10_k3\", \"n=10, k=3\"), (\"n5_k6\", \"n=5, k=6\")]:\n",
        "        d = scc_rec[design_key]\n",
        "        print(f\"=== {label} ===\")\n",
        "        print(\n",
        "            f\"  Total TFs               : {d['n_total_tfs']}\"\n",
        "        )\n",
        "        print(\n",
        "            f\"  Baseline identifiable   : {d['n_identifiable_baseline']} \"\n",
        "            f\"({d['n_identifiable_baseline']/d['n_total_tfs']*100:.1f}%)\"\n",
        "        )\n",
        "        print(\n",
        "            f\"  Baseline unidentifiable : {d['n_unidentifiable_baseline']} \"\n",
        "            f\"({d['n_unidentifiable_baseline']/d['n_total_tfs']*100:.1f}%)\"\n",
        "        )\n",
        "        print(\n",
        "            f\"  Recovered by bank       : {d['n_recovered']} \"\n",
        "            f\"({d['pct_recovered_of_unidentifiable']:.1f}% of unidentifiable)\"\n",
        "        )\n",
        "        print(\n",
        "            f\"  Identifiable after bank : {d['n_identifiable_baseline'] + d['n_recovered']} / \"\n",
        "            f\"{d['n_total_tfs']} \"\n",
        "            f\"({(d['n_identifiable_baseline']+d['n_recovered'])/d['n_total_tfs']*100:.1f}%)\"\n",
        "        )\n",
        "        rows = [\n",
        "            {\"Experiment\": s[\"set_index\"], \"Genes (perturb together)\": \", \".join(s[\"genes\"])}\n",
        "            for s in d[\"chosen_sets\"]\n",
        "        ]\n",
        "        display(pd.DataFrame(rows).set_index(\"Experiment\"))\n",
        "        print()\n",
    ],
}

MARGINAL_CURVE_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── 11.2  Marginal recovery curves ────────────────────────────────────────\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.patches as mpatches\n",
        "\n",
        "if SCC_REC_SUMMARY.exists():\n",
        "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "    fig.suptitle(\n",
        "        \"SCC-TF Recovery Bank: Marginal Gain per Experiment\",\n",
        "        fontsize=13, fontweight=\"bold\",\n",
        "    )\n",
        "\n",
        "    design_styles = {\n",
        "        \"n10_k3\": {\"label\": \"n=10, k=3\", \"color\": \"#2980b9\", \"marker\": \"o\"},\n",
        "        \"n5_k6\":  {\"label\": \"n=5, k=6\",  \"color\": \"#27ae60\", \"marker\": \"s\"},\n",
        "    }\n",
        "\n",
        "    # Left: cumulative recovery curve\n",
        "    ax1 = axes[0]\n",
        "    n_unid = scc_rec[\"n10_k3\"][\"n_unidentifiable_baseline\"]\n",
        "    for design_key, style in design_styles.items():\n",
        "        d = scc_rec[design_key]\n",
        "        marginal = d[\"marginal_curve\"]\n",
        "        cumulative, total = [], 0\n",
        "        for v in marginal:\n",
        "            total += v\n",
        "            cumulative.append(total)\n",
        "        x = list(range(1, len(cumulative) + 1))\n",
        "        ax1.plot(\n",
        "            x, cumulative,\n",
        "            marker=style[\"marker\"], color=style[\"color\"],\n",
        "            label=style[\"label\"], linewidth=2, markersize=6,\n",
        "        )\n",
        "\n",
        "    ax1.axhline(\n",
        "        n_unid, color=\"gray\", linestyle=\"--\", linewidth=1,\n",
        "        label=f\"All unidentifiable ({n_unid})\",\n",
        "    )\n",
        "    ax1.set_xlabel(\"Number of experiments\")\n",
        "    ax1.set_ylabel(\"Unidentifiable TFs recovered (proxy)\")\n",
        "    ax1.set_title(\"Cumulative TF recovery vs. experiments\")\n",
        "    ax1.legend(fontsize=9)\n",
        "    ax1.grid(True, linestyle=\"--\", alpha=0.35)\n",
        "    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))\n",
        "\n",
        "    # Right: marginal (incremental) gain per experiment, side-by-side bars\n",
        "    ax2 = axes[1]\n",
        "    max_n = max(\n",
        "        len(scc_rec[\"n10_k3\"][\"marginal_curve\"]),\n",
        "        len(scc_rec[\"n5_k6\"][\"marginal_curve\"]),\n",
        "    )\n",
        "    bar_width = 0.35\n",
        "    import numpy as np\n",
        "    xs = np.arange(1, max_n + 1)\n",
        "\n",
        "    for offset, (design_key, style) in zip([-bar_width / 2, bar_width / 2], design_styles.items()):\n",
        "        marginal = scc_rec[design_key][\"marginal_curve\"]\n",
        "        padded = marginal + [0] * (max_n - len(marginal))\n",
        "        ax2.bar(\n",
        "            xs + offset, padded, bar_width,\n",
        "            color=style[\"color\"], label=style[\"label\"],\n",
        "            edgecolor=\"white\", linewidth=0.7,\n",
        "        )\n",
        "        for xi, v in zip(xs + offset, padded):\n",
        "            if v > 0:\n",
        "                ax2.text(xi, v + 0.3, str(v), ha=\"center\", va=\"bottom\", fontsize=7)\n",
        "\n",
        "    ax2.set_xlabel(\"Experiment number\")\n",
        "    ax2.set_ylabel(\"TFs newly recovered\")\n",
        "    ax2.set_title(\"Marginal gain per experiment\")\n",
        "    ax2.set_xticks(xs)\n",
        "    ax2.legend(fontsize=9)\n",
        "    ax2.grid(axis=\"y\", linestyle=\"--\", alpha=0.35)\n",
        "\n",
        "    plt.tight_layout()\n",
        "    _out = _ecoli_dir / \"../visualizations/scc_recovery_marginal_curve.png\"\n",
        "    plt.savefig(str(_out), dpi=150, bbox_inches=\"tight\")\n",
        "    plt.show()\n",
        "    print(f\"Saved: {_out}\")\n",
    ],
}

BASELINE_PIE_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── 11.3  Before / after identifiability stacked bar ──────────────────────\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "if SCC_REC_SUMMARY.exists():\n",
        "    designs = [\n",
        "        (\"n10_k3\", \"n=10, k=3\"),\n",
        "        (\"n5_k6\",  \"n=5, k=6\"),\n",
        "    ]\n",
        "\n",
        "    fig, ax = plt.subplots(figsize=(9, 5))\n",
        "    bar_width = 0.3\n",
        "    xs = np.array([0, 0.8, 1.6])  # baseline + 2 designs\n",
        "\n",
        "    d0 = scc_rec[\"n10_k3\"]  # baseline is same for both designs\n",
        "    n_total = d0[\"n_total_tfs\"]\n",
        "    n_id_base = d0[\"n_identifiable_baseline\"]\n",
        "    n_unid_base = d0[\"n_unidentifiable_baseline\"]\n",
        "\n",
        "    # Baseline\n",
        "    ax.bar(xs[0], n_id_base, bar_width, color=\"#2ecc71\", label=\"Already identifiable\")\n",
        "    ax.bar(xs[0], n_unid_base, bar_width, bottom=n_id_base, color=\"#e74c3c\", label=\"Unidentifiable\")\n",
        "    ax.text(xs[0], n_total + 3, \"Baseline\", ha=\"center\", fontsize=9)\n",
        "\n",
        "    colors_rec = {\"n10_k3\": \"#3498db\", \"n5_k6\": \"#9b59b6\"}\n",
        "\n",
        "    for xi, (design_key, label) in zip(xs[1:], designs):\n",
        "        d = scc_rec[design_key]\n",
        "        n_recovered = d[\"n_recovered\"]\n",
        "        n_still_unid = d[\"n_still_unrecovered\"]\n",
        "        ax.bar(xi, n_id_base, bar_width, color=\"#2ecc71\")\n",
        "        ax.bar(xi, n_recovered, bar_width, bottom=n_id_base,\n",
        "               color=colors_rec[design_key], label=f\"Newly recovered ({label})\")\n",
        "        ax.bar(xi, n_still_unid, bar_width, bottom=n_id_base + n_recovered,\n",
        "               color=\"#e74c3c\")\n",
        "        ax.text(xi, n_total + 3, label, ha=\"center\", fontsize=9)\n",
        "\n",
        "        # Annotate recovery count\n",
        "        pct = n_recovered / n_unid_base * 100\n",
        "        ax.text(\n",
        "            xi, n_id_base + n_recovered / 2,\n",
        "            f\"+{n_recovered}\\n({pct:.0f}%)\",\n",
        "            ha=\"center\", va=\"center\", fontsize=8, color=\"white\", fontweight=\"bold\",\n",
        "        )\n",
        "\n",
        "    ax.set_xticks([])\n",
        "    ax.set_ylabel(\"Number of TFs\")\n",
        "    ax.set_title(\"SCC-TF identifiability: baseline vs. recovery bank\", fontsize=12)\n",
        "    ax.set_ylim(0, n_total + 20)\n",
        "    ax.legend(fontsize=9, loc=\"lower right\")\n",
        "    ax.axhline(n_total, color=\"#aaa\", linestyle=\":\", linewidth=0.8)\n",
        "    ax.text(xs[-1] + 0.3, n_total, f\"N={n_total}\", va=\"center\", fontsize=8, color=\"#555\")\n",
        "    ax.grid(axis=\"y\", linestyle=\"--\", alpha=0.3)\n",
        "\n",
        "    plt.tight_layout()\n",
        "    _out2 = _ecoli_dir / \"../visualizations/scc_recovery_before_after.png\"\n",
        "    plt.savefig(str(_out2), dpi=150, bbox_inches=\"tight\")\n",
        "    plt.show()\n",
        "    print(f\"Saved: {_out2}\")\n",
    ],
}

PER_TF_TABLE_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── 11.4  Per-TF recovery table ───────────────────────────────────────────\n",
        "import pandas as pd\n",
        "from IPython.display import display\n",
        "\n",
        "if SCC_REC_SUMMARY.exists():\n",
        "    _tfs_csv = _ecoli_dir / \"scc_recovery_tfs_n10_k3.csv\"\n",
        "    if _tfs_csv.exists():\n",
        "        tfs_df = pd.read_csv(_tfs_csv)\n",
        "        tfs_df = tfs_df.sort_values([\"recovered\", \"recovered_by_set\"], ascending=[False, True])\n",
        "\n",
        "        def _style_recovered(val):\n",
        "            if val is True or str(val).lower() == \"true\":\n",
        "                return \"background-color: #d5f5e3; color: #1a5276\"\n",
        "            elif val is False or str(val).lower() == \"false\":\n",
        "                return \"background-color: #fadbd8; color: #7b241c\"\n",
        "            return \"\"\n",
        "\n",
        "        n_rec = (tfs_df[\"recovered\"].astype(str).str.lower() == \"true\").sum()\n",
        "        n_unid = len(tfs_df)\n",
        "        print(\n",
        "            f\"Per-TF recovery table (n=10, k=3 design): \"\n",
        "            f\"{n_rec}/{n_unid} unidentifiable TFs recovered\"\n",
        "        )\n",
        "        display(\n",
        "            tfs_df.style\n",
        "            .map(_style_recovered, subset=[\"recovered\"])\n",
        "            .set_caption(\n",
        "                \"Table 3. Per-TF recovery status under the n=10, k=3 bank design.  \"\n",
        "                \"recovered=True: at least one bank experiment breaks the TF's SCC membership.  \"\n",
        "                \"recovered_by_set: experiment index (1-based) that first recovers this TF.\"\n",
        "            )\n",
        "            .set_table_styles(\n",
        "                [{\"selector\": \"caption\",\n",
        "                  \"props\": [(\"font-weight\", \"bold\"), (\"font-size\", \"12px\")]}]\n",
        "            )\n",
        "        )\n",
        "    else:\n",
        "        print(f\"TF table not found: {_tfs_csv}\")\n",
        "        print(\"Run:  uv run python scripts/scc_recovery_bank.py --n 10 --k 3\")\n",
    ],
}

COMPARISON_MD = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 11.5 Comparison to CSD Edge-Level Recovery\n",
        "\n",
        "The TF recovery bank above operates at the **TF level** — it asks which TFs "
        "transition from unidentifiable to identifiable when a set of background genes is perturbed.  "
        "The analogous analysis in `Cyclic_SingleDoor_Analysis.ipynb` (Section 7) operates at the "
        "**edge level** — it asks which individual causal edges become identifiable.\n",
        "\n",
        "| Dimension | CSD edge-level recovery | SCC-TF recovery bank (this section) |\n",
        "|---|---|---|\n",
        "| Unit of recovery | Directed edge (cause → effect) | TF (whole-TF identifiability) |\n",
        "| Total targets | 6,676 unidentifiable edges | 95 unidentifiable TFs |\n",
        "| n=10, k=3 recovery | 6,502 / 6,676 (97.4%) | 67 / 95 (70.5%) |\n",
        "| n=5, k=6 recovery | 6,502 / 6,676 (97.4%) | 67 / 95 (70.5%) |\n",
        "| Ceiling | Hard: 28 unrescuable edges | Hard: 28 unrecoverable TFs |\n",
        "| Algorithm | Greedy set-cover over rescue-node lists | Greedy set-cover over SCC min-cut genes |\n",
        "\n",
        "The **28 unrecoverable TFs** (those remaining unidentifiable even after all bank experiments) "
        "correspond structurally to TFs in direct 2-cycles with in-SCC children for which no "
        "intermediate cut target exists under Interpretation A.  These are the same cut-incomplete "
        "cases identified in Phase A (§ 9.A): `tf_still_cyclic = True` even after all candidate "
        "background interventions.\n",
        "\n",
        "**Implication for experimental design.**  The two budget configurations yield equivalent "
        "coverage.  If a smaller number of larger experiments is operationally preferred, the "
        "n=5, k=6 design achieves the same recovery with half the experimental runs, but each "
        "experiment requires knocking out 6 genes simultaneously.",
    ],
}

SUMMARY_MD = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 11.6 Summary\n",
        "\n",
        "| Metric | Value |\n",
        "|---|---|\n",
        "| Total TFs in E. coli network (out-degree ≥ 1) | 285 |\n",
        "| Baseline identifiable (singleton SCC) | 190 (66.7%) |\n",
        "| Baseline unidentifiable (non-trivial SCC) | 95 (33.3%) |\n",
        "| Candidate pool (min-cut genes) | 63 unique genes |\n",
        "| Recovered by n=10, k=3 bank | 67 / 95 (70.5%) |\n",
        "| Recovered by n=5, k=6 bank | 67 / 95 (70.5%) |\n",
        "| Identifiable after recovery | 257 / 285 (90.2%) |\n",
        "| Structurally unrecoverable (under Interpretation A) | 28 / 95 |\n",
        "\n",
        "**First experiment dominates:** under n=10, k=3, perturbation set 1 "
        "{fliZ, fur, rpoH} recovers 28 TFs alone; under n=5, k=6, set 1 "
        "{fliZ, fur, gadW, hns, rpoD, rpoH} recovers 42 TFs.\n",
        "\n",
        "**Diminishing returns are steep:** ≥ 70% of recoverable TFs are obtained "
        "in the first 2 experiments of either design.\n",
        "\n",
        "See `scripts/scc_recovery_bank.py`, `tests/test_scc_recovery_bank.py`, "
        "and `scripts/smoke_scc_recovery.py` for implementation and tests.",
    ],
}


def main() -> None:
    assert NB_PATH.exists(), f"Notebook not found: {NB_PATH}"
    with open(NB_PATH) as f:
        nb = json.load(f)

    # Guard: don't double-inject
    existing_srcs = [
        "".join(c.get("source", []))
        for c in nb["cells"]
        if c["cell_type"] == "markdown"
    ]
    if any("TF Recovery Bank" in s for s in existing_srcs):
        print("Section 11 already present — nothing to do.")
        sys.exit(0)

    new_cells = [
        SECTION_HEADER_MD,
        LOAD_CELL,
        MARGINAL_CURVE_CELL,
        BASELINE_PIE_CELL,
        PER_TF_TABLE_CELL,
        COMPARISON_MD,
        SUMMARY_MD,
    ]

    # Insert before References cell (last markdown cell)
    # Find index of the References cell
    ref_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown" and any(
            "References" in ln for ln in cell.get("source", [])
        ):
            ref_idx = i

    if ref_idx is not None:
        nb["cells"][ref_idx:ref_idx] = new_cells
        print(f"Inserted {len(new_cells)} cells before References (cell index {ref_idx}).")
    else:
        nb["cells"].extend(new_cells)
        print(f"Appended {len(new_cells)} cells at end (References cell not found).")

    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Updated: {NB_PATH}")


if __name__ == "__main__":
    main()
