"""build_csd_notebook.py — Generate Cyclic_SingleDoor_Analysis.ipynb programmatically.

Run with:
    uv run python scripts/build_csd_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("notebooks/Ecoli_Analysis_Notebooks/Cyclic_SingleDoor_Analysis.ipynb")


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = []

# ============================================================
# Section 1 — Title / Methods
# ============================================================
cells.append(md("""\
# Cyclic Single-Door Criterion — E. coli Network Analysis

**Purpose:** Assess the identifiability of every directed edge in the *E. coli* transcriptional
regulatory network using the sigma-separation single-door criterion (Forre & Mooij 2018;
Rantanen et al. 2020), diagnose *why* edges are unidentifiable, and quantify how targeted
genetic interventions (`do(v)`) could rescue identifiability.

## Methods summary

- **sigma-single-door:** For each edge `cause -> effect`, we test whether a valid
  back-door adjustment set exists in the sigma-extension graph (Perkovic et al. 2018 O-set,
  which is *complete*: no enumeration can find a set the O-set misses).
- **Non-identifiability taxonomy:** Unidentifiable edges are classified into five structural
  categories based on graph topology (self-loop, 2-cycle, same-SCC long feedback,
  SCC-edge-dissolved, cross-SCC-blocked).
- **Intervention rescue (notebook-light):** Per-edge single-node `do()` rescue on a sample
  of unidentifiable edges; global greedy `maximize_identifiable_edges` curve for small k.
- **Full sweep (SLURM):** `scripts/slurm/submit_csd_rescue.sh` runs the complete rescue
  analysis on the cluster; results load automatically below if present.

## References
- Forre & Mooij (2018). *Constraint-based Causal Discovery for Non-linear SCMs.*
- Rantanen et al. (2020). *Learning Optimal Cyclic Causal Graphs.*
- Perkovic et al. (2018). *Complete Graphical Characterization of Adjustment Sets. JMLR 19(1).*
- Henckel et al. (2022). *Graphical criteria for efficient total effect estimation. JRSS-B 84(2).*\
"""))

# ============================================================
# Section 2 — Imports & paths
# ============================================================
cells.append(md("## 0. Imports and paths"))
cells.append(code("""\
from __future__ import annotations

import sys, json
from pathlib import Path
from collections import Counter

import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Repo root on sys.path so nocap is importable ---
REPO = Path().resolve()
while not (REPO / "src" / "nocap").exists() and REPO != REPO.parent:
    REPO = REPO.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from nocap.cyclic_single_door import (
    evaluate_all_edges,
    maximize_identifiable_edges,
    nx_digraph_to_y0,
)

NB_DIR = REPO / "notebooks" / "Ecoli_Analysis_Notebooks"
VIZ_DIR = REPO / "notebooks" / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

GRAPHML  = NB_DIR / "ecoli_full_network_no_small_rna.graphml"
CSV_PATH = NB_DIR / "csd_results.csv"
SUMMARY_PATH = NB_DIR / "csd_summary.json"

print(f"REPO: {REPO}")
print(f"GraphML: {GRAPHML.exists()}")
print(f"CSV: {CSV_PATH.exists()}")
print(f"Summary: {SUMMARY_PATH.exists()}")\
"""))

# ============================================================
# Section 3 — Load & validate
# ============================================================
cells.append(md("## 1. Load and validate results"))
cells.append(code("""\
df = pd.read_csv(CSV_PATH)
with open(SUMMARY_PATH) as f:
    summary = json.load(f)

# Normalise types
df["same_scc"] = df["same_scc"].astype(str).str.lower().isin(["true", "1"])
df["timed_out"] = df["timed_out"].fillna(False).astype(str).str.lower().isin(["true", "1"])

print(f"Total edges: {len(df):,}")
print(f"Identifiable: {(df.status=='identifiable').sum():,}  ({summary['pct_identifiable']}%)")
print(f"Unidentifiable: {(df.status=='unidentifiable').sum():,}")
print(f"Timed out: {(df.status=='timeout').sum():,}")
print(f"Same-SCC: {df.same_scc.sum():,}")
print()
print(df.head())\
"""))

# ============================================================
# Section 4 — Overall identifiability
# ============================================================
cells.append(md("## 2. Overall identifiability"))
cells.append(code("""\
status_counts = df["status"].value_counts()
colors = {"identifiable": "#2ecc71", "unidentifiable": "#e74c3c", "timeout": "#f39c12"}
color_list = [colors.get(s, "#95a5a6") for s in status_counts.index]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Pie
axes[0].pie(
    status_counts.values,
    labels=status_counts.index,
    colors=color_list,
    autopct="%1.1f%%",
    startangle=90,
)
axes[0].set_title("Edge Identifiability\\n(sigma-single-door criterion)")

# Bar
axes[1].bar(status_counts.index, status_counts.values, color=color_list, edgecolor="black")
axes[1].set_ylabel("Number of edges")
axes[1].set_title("Edge counts by status")
for i, (k, v) in enumerate(status_counts.items()):
    axes[1].text(i, v + 30, str(v), ha="center", fontsize=9)

plt.tight_layout()
out = VIZ_DIR / "csd_identifiability_overall.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out}")\
"""))

# ============================================================
# Section 5 — Adjustment-set analysis
# ============================================================
cells.append(md("## 3. Adjustment-set analysis"))
cells.append(code("""\
ident = df[df.status == "identifiable"].copy()
ident["adj_nodes"] = ident["adjustment_set"].fillna("").str.split("|")
ident["adj_size"] = ident["adj_nodes"].apply(lambda x: len(x) if x != [""] else 0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Adjustment set size distribution
size_counts = ident["adj_size"].value_counts().sort_index()
axes[0].bar(size_counts.index, size_counts.values, color="#3498db", edgecolor="black")
axes[0].set_xlabel("Adjustment set size")
axes[0].set_ylabel("Number of identifiable edges")
axes[0].set_title("Distribution of adjustment-set sizes")

# Top nodes appearing in adjustment sets
all_adj_nodes: list[str] = []
for nodes in ident["adj_nodes"]:
    all_adj_nodes.extend([n for n in nodes if n])
node_counter = Counter(all_adj_nodes)
top_n = 20
top_nodes = node_counter.most_common(top_n)
if top_nodes:
    names, cnts = zip(*top_nodes)
    axes[1].barh(list(names)[::-1], list(cnts)[::-1], color="#9b59b6")
    axes[1].set_xlabel("Frequency in adjustment sets")
    axes[1].set_title(f"Top {top_n} adjustment-set nodes")
else:
    axes[1].text(0.5, 0.5, "No adjustment sets", ha="center", va="center", transform=axes[1].transAxes)

plt.tight_layout()
out = VIZ_DIR / "csd_adjustment_set_sizes.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out}")
print(f"\\nMedian adjustment-set size: {ident['adj_size'].median():.1f}")
print(f"Max adjustment-set size: {ident['adj_size'].max()}")\
"""))

# ============================================================
# Section 6 — Same-SCC vs Cross-SCC
# ============================================================
cells.append(md("## 4. Same-SCC vs Cross-SCC identifiability"))
cells.append(code("""\
cross = df[~df.same_scc]
same  = df[df.same_scc]

def rates(sub):
    n = len(sub)
    if n == 0:
        return {"identifiable": 0, "unidentifiable": 0, "timeout": 0, "total": 0}
    return {s: (sub.status == s).sum() for s in ("identifiable", "unidentifiable", "timeout")} | {"total": n}

r_cross = rates(cross)
r_same  = rates(same)

cats = ("identifiable", "unidentifiable", "timeout")
x = [0, 1]
width = 0.25
fig, ax = plt.subplots(figsize=(8, 5))
for i, cat in enumerate(cats):
    c_vals = [r_cross[cat], r_same[cat]]
    ax.bar([xi + i * width for xi in x], c_vals, width, label=cat,
           color=colors.get(cat, "#95a5a6"), edgecolor="black")

ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels([
    f"Cross-SCC\\n(n={r_cross['total']:,})",
    f"Same-SCC\\n(n={r_same['total']:,})",
])
ax.set_ylabel("Number of edges")
ax.set_title("Identifiability: Same-SCC vs Cross-SCC edges")
ax.legend()
plt.tight_layout()
out = VIZ_DIR / "csd_same_vs_cross_scc.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out}")
print(f"\\nCross-SCC  identifiable rate: {100*r_cross['identifiable']/r_cross['total']:.1f}%")
print(f"Same-SCC   identifiable rate: {100*r_same['identifiable']/r_same['total']:.1f}%")\
"""))

# ============================================================
# Section 7 — Non-identifiability cause taxonomy (in-notebook)
# ============================================================
cells.append(md("""\
## 5. Non-identifiability cause taxonomy

For each unidentifiable edge we classify the *structural reason*:

| Category | Meaning |
|---|---|
| `self_loop` | cause == effect |
| `two_cycle` | reverse edge effect→cause also exists |
| `same_scc_long` | same SCC even after removing cause→effect (long feedback path) |
| `scc_edge_dissolved` | removing cause→effect breaks the SCC (edge *is* the link) |
| `cross_scc_blocked` | different SCCs but O-set blocked (descendant of effect in O-set) |

This runs fully in-notebook (graph-structure only; no sigma oracle needed).\
"""))
cells.append(code("""\
from csd_rescue_worker import classify_nonident_cause, CAUSE_CATEGORIES

g = nx.read_graphml(str(GRAPHML))
unident = df[df.status == "unidentifiable"][["cause", "effect"]].copy()
print(f"Diagnosing {len(unident):,} unidentifiable edges...")

cause_labels = []
for _, row in unident.iterrows():
    if g.has_edge(row.cause, row.effect):
        cause_labels.append(classify_nonident_cause(g, row.cause, row.effect))
    else:
        cause_labels.append("unknown")

unident = unident.copy()
unident["nonident_cause"] = cause_labels
cause_counts = unident["nonident_cause"].value_counts()
print(cause_counts)
unident.to_csv(NB_DIR / "csd_nonident_diagnosis.csv", index=False)
print("\\nSaved: csd_nonident_diagnosis.csv")\
"""))

cells.append(code("""\
CATEGORY_COLORS = {
    "self_loop": "#e74c3c",
    "two_cycle": "#e67e22",
    "same_scc_long": "#f1c40f",
    "scc_edge_dissolved": "#3498db",
    "cross_scc_blocked": "#9b59b6",
    "unknown": "#95a5a6",
}
CATEGORY_LABELS = {
    "self_loop": "Self-loop",
    "two_cycle": "2-cycle (A→B→A)",
    "same_scc_long": "Same-SCC, long feedback",
    "scc_edge_dissolved": "SCC edge dissolved",
    "cross_scc_blocked": "Cross-SCC blocked",
}

ordered = [c for c in CAUSE_CATEGORIES if c in cause_counts.index]
vals = [cause_counts.get(c, 0) for c in ordered]
clrs = [CATEGORY_COLORS.get(c, "#95a5a6") for c in ordered]
labs = [CATEGORY_LABELS.get(c, c) for c in ordered]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar chart
axes[0].bar(labs, vals, color=clrs, edgecolor="black")
axes[0].set_ylabel("Number of unidentifiable edges")
axes[0].set_title("Non-identifiability cause taxonomy")
axes[0].tick_params(axis="x", rotation=25)
for i, v in enumerate(vals):
    axes[0].text(i, v + 10, str(v), ha="center", fontsize=8)

# Pie chart
axes[1].pie(vals, labels=labs, colors=clrs, autopct="%1.1f%%", startangle=90)
axes[1].set_title("Cause breakdown (unidentifiable edges)")

plt.tight_layout()
out = VIZ_DIR / "csd_nonident_causes.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out}")\
"""))

# ============================================================
# Section 8 — Intervention rescue (notebook-light)
# ============================================================
cells.append(md("""\
## 6. Intervention rescue (notebook-light)

### 6a. Per-edge targeted rescue (sampled)
For a small sample of unidentifiable edges, test each candidate intervention node
(from the SCC min-cut pool) and record which ones flip the edge to identifiable.\
"""))
cells.append(code("""\
import random
from csd_rescue_worker import compute_rescue_nodes, _candidate_pool

random.seed(42)
SAMPLE_SIZE = 50  # keep fast; increase if you have time

# Filter to same-SCC unidentifiable edges (most interesting for rescue)
same_scc_unident = unident[unident.cause.map(lambda c: True)].copy()
# Use nonident_cause != self_loop (self-loops can't be rescued by do on other nodes)
rescuable_pool = unident[unident.nonident_cause != "self_loop"]
sample_edges = rescuable_pool.sample(min(SAMPLE_SIZE, len(rescuable_pool)), random_state=42)

print(f"Computing rescue nodes for {len(sample_edges)} sampled edges...")
candidates = _candidate_pool(g)
print(f"Candidate pool size: {len(candidates)} nodes")

rescue_records = []
for _, row in sample_edges.iterrows():
    if not g.has_edge(row.cause, row.effect):
        continue
    nodes = compute_rescue_nodes(g, row.cause, row.effect, candidates)
    rescue_records.append({
        "cause": row.cause,
        "effect": row.effect,
        "nonident_cause": row.nonident_cause,
        "rescue_nodes": nodes,
        "n_rescue_nodes": len(nodes),
    })

rescue_df = pd.DataFrame(rescue_records)
print(f"\\n{rescue_df['n_rescue_nodes'].describe().to_string()}")
print(f"\\nEdges with at least one rescue node: {(rescue_df.n_rescue_nodes > 0).sum()} / {len(rescue_df)}")\
"""))

cells.append(code("""\
# Most frequently effective intervention targets (in sample)
rescue_node_counter: Counter = Counter()
for nodes in rescue_df["rescue_nodes"]:
    rescue_node_counter.update(nodes)

top_rescue = rescue_node_counter.most_common(20)
if top_rescue:
    names, cnts = zip(*top_rescue)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(list(names)[::-1], list(cnts)[::-1], color="#e74c3c")
    ax.set_xlabel(f"# edges rescued (in sample of {len(rescue_df)})")
    ax.set_title("Top single-node do() intervention targets (sampled)")
    plt.tight_layout()
    out = VIZ_DIR / "csd_rescue_targets.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")
else:
    print("No rescue nodes found in sample.")\
"""))

cells.append(md("### 6b. Global greedy intervention curve"))
cells.append(code("""\
K_MAX = 5  # budget; keep small for notebook speed

print(f"Running maximize_identifiable_edges(graph, k={K_MAX})...")
print("(This may take several minutes on the full 9k-edge graph)")

greedy = maximize_identifiable_edges(g, K_MAX)

print(f"\\nBaseline identifiable: {greedy['n_identifiable_baseline']:,}")
print(f"Final identifiable:    {greedy['n_identifiable_final']:,}")
print(f"Chosen interventions:  {greedy['chosen_nodes']}")
print(f"Curve: {greedy['curve']}")\
"""))

cells.append(code("""\
curve = greedy["curve"]
xs = [c[0] for c in curve]
ys = [c[1] for c in curve]
pcts = [100.0 * y / len(df) for y in ys]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(xs, pcts, "o-", color="#2ecc71", linewidth=2, markersize=8)
ax.set_xlabel("Number of hard interventions do(v)")
ax.set_ylabel("% edges identifiable")
ax.set_title("Global greedy intervention curve\\n(sigma-single-door, E. coli network)")
ax.set_ylim(0, 100)
for x, y, p in zip(xs, ys, pcts):
    ax.annotate(f"{p:.1f}%", (x, p), textcoords="offset points", xytext=(5, 3), fontsize=8)

# Annotate chosen nodes
chosen = greedy["chosen_nodes"]
for i, node in enumerate(chosen, 1):
    ax.annotate(f"do({node})", (i, pcts[i]), textcoords="offset points", xytext=(5, -12), fontsize=7, color="darkblue")

plt.tight_layout()
out = VIZ_DIR / "csd_rescue_curve.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out}")\
"""))

# ============================================================
# Section 9 — Full-sweep results (load if present)
# ============================================================
cells.append(md("""\
## 7. Full-sweep rescue results (SLURM)

If the full rescue sweep has been run on the cluster, this section loads
`csd_rescue_results.csv` and visualizes the most effective intervention targets
across *all* unidentifiable edges.

To run the full sweep:
```bash
bash scripts/slurm/submit_csd_rescue.sh
# After completion:
uv run python scripts/csd_rescue_gather.py \\
    --input-dir results/cyclic_single_door/rescue_classified \\
    --output-csv notebooks/Ecoli_Analysis_Notebooks/csd_rescue_results.csv \\
    --output-summary notebooks/Ecoli_Analysis_Notebooks/csd_rescue_summary.json
```\
"""))
cells.append(code("""\
RESCUE_CSV = NB_DIR / "csd_rescue_results.csv"
RESCUE_SUMMARY = NB_DIR / "csd_rescue_summary.json"

if RESCUE_CSV.exists():
    rescue_full = pd.read_csv(RESCUE_CSV)
    with open(RESCUE_SUMMARY) as f:
        rescue_summary = json.load(f)

    print(f"Full sweep: {len(rescue_full):,} unidentifiable edges analysed")
    print(f"Rescuable: {rescue_summary['n_rescuable']:,} ({rescue_summary['pct_rescuable']}%)")
    print(f"\\nCause counts: {rescue_summary['cause_counts']}")
    print(f"\\nTop rescue nodes:")
    for row in rescue_summary["top_rescue_nodes"][:10]:
        print(f"  {row['node']}: {row['count']} edges")

    # Full-sweep top rescue targets bar chart
    top_nodes_full = rescue_summary["top_rescue_nodes"][:20]
    if top_nodes_full:
        names_f, cnts_f = zip(*[(r["node"], r["count"]) for r in top_nodes_full])
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(list(names_f)[::-1], list(cnts_f)[::-1], color="#c0392b")
        ax.set_xlabel("# unidentifiable edges rescued (full sweep)")
        ax.set_title("Top single-node do() intervention targets\\n(all unidentifiable edges)")
        plt.tight_layout()
        out = VIZ_DIR / "csd_rescue_targets_full.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {out}")

    # Full cause taxonomy breakdown
    cause_counts_full = rescue_summary["cause_counts"]
    ordered_f = [c for c in CAUSE_CATEGORIES if c in cause_counts_full]
    vals_f = [cause_counts_full.get(c, 0) for c in ordered_f]
    labs_f = [CATEGORY_LABELS.get(c, c) for c in ordered_f]
    clrs_f = [CATEGORY_COLORS.get(c, "#95a5a6") for c in ordered_f]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labs_f, vals_f, color=clrs_f, edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_title("Non-identifiability cause taxonomy (full sweep)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    out = VIZ_DIR / "csd_nonident_causes_full.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")
else:
    print("Full rescue sweep not yet available.")
    print("Run: bash scripts/slurm/submit_csd_rescue.sh")\
"""))

# ============================================================
# Section 10 — Export CSVs
# ============================================================
cells.append(md("## 8. Export final tables"))
cells.append(code("""\
# Export identifiable edges
ident_out = df[df.status == "identifiable"][["cause", "effect", "adjustment_set", "same_scc"]]
ident_out.to_csv(NB_DIR / "csd_identifiable_edges.csv", index=False)
print(f"Identifiable edges: {len(ident_out):,} -> csd_identifiable_edges.csv")

# Export non-identifiability diagnosis (if computed above)
diag_path = NB_DIR / "csd_nonident_diagnosis.csv"
if diag_path.exists():
    print(f"Non-identifiability diagnosis: {diag_path}")

print("\\nAll done.")\
"""))

# ============================================================
# Build and write the notebook
# ============================================================
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    },
    "cells": cells,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook written to: {NB_PATH}")
print(f"  {len(cells)} cells")
