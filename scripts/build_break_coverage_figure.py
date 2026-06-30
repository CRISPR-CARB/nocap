"""build_break_coverage_figure.py — Slide-2 supporting figure for SLIDES.md.

Produces two panel figure:
  Left:  stacked bar — break-set size distribution for 330 unidentifiable edges
         (0 = no intervention needed, 1/2/3 = rescued within k=3, >3 = beyond budget)
  Right: break-set TF frequency table (top-15 hub TFs most often appearing in min break-sets)

Output: notebooks/visualizations/csd_break_coverage_curve.png
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).parent.parent
RESULTS_CSV = REPO / "notebooks/Ecoli_Analysis_Notebooks/csd_break_results.csv"
SUMMARY_JSON = REPO / "notebooks/Ecoli_Analysis_Notebooks/csd_break_summary.json"
OUT_PNG = REPO / "notebooks/visualizations/csd_break_coverage_curve.png"

assert RESULTS_CSV.exists(), f"PRE: {RESULTS_CSV} must exist"
assert SUMMARY_JSON.exists(), f"PRE: {SUMMARY_JSON} must exist"


def _load_results() -> list[dict]:
    rows = []
    with open(RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    assert len(rows) > 0, "PRE: results CSV must be non-empty"
    return rows


def _tf_frequency(rows: list[dict], top_n: int = 15) -> list[tuple[str, int]]:
    """Count how often each TF gene appears in any min_break_set."""
    counter: Counter = Counter()
    for row in rows:
        bs_raw = row.get("min_break_set", "[]")
        try:
            bs = json.loads(bs_raw)
        except (json.JSONDecodeError, TypeError):
            bs = []
        for tf in bs:
            if tf:
                counter[tf] += 1
    return counter.most_common(top_n)


def main() -> None:
    rows = _load_results()
    summary = json.loads(SUMMARY_JSON.read_text())

    n_total = summary["n_unidentifiable_edges"]
    break_dist = summary["break_size_distribution"]

    # --- Left panel data ---
    # Semantically correct categories:
    #   cat_0: needs_intervention=False (148): removing direct edge severs cycle entirely
    #   cat_oset: needs_intervention=True AND break_size=0 (105): residual SCC survives
    #             but O-set still identifies the edge
    #   cat_1/2/3: needs 1/2/3 node knockouts
    #   beyond: needs 4+ knockouts
    cat_0 = sum(1 for r in rows if r.get("needs_intervention") in ("False", False))
    cat_oset = sum(
        1
        for r in rows
        if r.get("needs_intervention") in ("True", True) and str(r.get("min_break_size", "")) == "0"
    )
    rescued_1 = int(break_dist.get("1", 0))
    rescued_2 = int(break_dist.get("2", 0))
    rescued_3 = int(break_dist.get("3", 0))
    beyond = sum(int(v) for k, v in break_dist.items() if int(k) > 3)

    assert cat_0 + cat_oset + rescued_1 + rescued_2 + rescued_3 + beyond == n_total, (
        f"POST: category counts must sum to {n_total}"
    )

    # --- Right panel data ---
    tf_freq = _tf_frequency(rows, top_n=15)
    tf_names = [t for t, _ in tf_freq]
    tf_counts = [c for _, c in tf_freq]

    assert len(tf_freq) > 0, "POST: must find at least one TF in break sets"

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("white")

    # ---- Left: grouped bar ----
    categories = [
        "Direct-edge removal\nbreaks cycle\n(no further do())",
        "Residual SCC\nsurvives; O-set\nstill identifies",
        "1 knockout\nbreaks SCC",
        "2 knockouts\nbreak SCC",
        "3 knockouts\nbreak SCC",
        "Beyond budget\n(|B| = 4 or 5)",
    ]
    values = [cat_0, cat_oset, rescued_1, rescued_2, rescued_3, beyond]
    colors = ["#4CAF50", "#81C784", "#66BB6A", "#AED6AC", "#D4EDDA", "#E57373"]

    bars = ax1.bar(categories, values, color=colors, edgecolor="white", linewidth=1.2, zorder=3)
    ax1.set_facecolor("#f8f9fa")
    ax1.grid(axis="y", color="white", linewidth=1.5, zorder=2)
    ax1.set_ylabel("Number of unidentifiable edges", fontsize=11)
    ax1.set_title(
        f"Minimal break-set sizes for {n_total} unidentifiable edges\n"
        f"(k \u2264 3 budget; 97% fully resolved)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax1.tick_params(axis="x", labelsize=8.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        if val > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Rescuable annotation — all categories except "beyond budget"
    rescuable = cat_0 + cat_oset + rescued_1 + rescued_2 + rescued_3
    ax1.axhline(y=rescuable, color="#1565C0", linewidth=1.5, linestyle="--", alpha=0.8, zorder=4)
    pct = 100.0 * rescuable / n_total
    ax1.text(
        4.55,
        rescuable + 2,
        f"{rescuable} resolved\n({pct:.0f}%)",
        color="#1565C0",
        fontsize=8.5,
        ha="right",
        va="bottom",
    )

    # ---- Right: horizontal bar — TF frequency ----
    y_pos = np.arange(len(tf_names))
    hbars = ax2.barh(y_pos, tf_counts, color="#5C9BD6", edgecolor="white", linewidth=0.8, zorder=3)
    ax2.set_facecolor("#f8f9fa")
    ax2.grid(axis="x", color="white", linewidth=1.5, zorder=2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(tf_names, fontsize=10)
    ax2.invert_yaxis()  # highest at top
    ax2.set_xlabel("Edges whose min-break-set includes this TF", fontsize=10)
    ax2.set_title(
        "Most-needed hub TFs\n(top-15 by break-set frequency)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )

    for bar, val in zip(hbars, tf_counts):
        ax2.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout(pad=2.5)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved figure -> {OUT_PNG}")
    print(f"Top-5 break-set TFs: {tf_freq[:5]}")
    print(f"Rescuable (<=k=3): {rescuable}/{n_total} ({100.0 * rescuable / n_total:.1f}%)")


if __name__ == "__main__":
    main()
