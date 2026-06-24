"""smoke_csd_notebook.py — Fast smoke-test for Cyclic_SingleDoor_Analysis notebook cells.

Runs the non-slow sections: imports, load, summary stats, cause taxonomy.
Skips the greedy optimizer (section 6b) which is slow.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import Counter

import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from nocap.cyclic_single_door import nx_digraph_to_y0  # noqa: F401

NB_DIR = REPO / "notebooks" / "Ecoli_Analysis_Notebooks"
VIZ_DIR = REPO / "notebooks" / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

GRAPHML = NB_DIR / "ecoli_full_network_no_small_rna.graphml"
CSV_PATH = NB_DIR / "csd_results.csv"
SUMMARY_PATH = NB_DIR / "csd_summary.json"

# --- Section 1: Load ---
df = pd.read_csv(CSV_PATH)
with open(SUMMARY_PATH) as f:
    summary = json.load(f)

df["same_scc"] = df["same_scc"].astype(str).str.lower().isin(["true", "1"])
df["timed_out"] = df["timed_out"].fillna(False).astype(str).str.lower().isin(["true", "1"])

print(f"Total edges: {len(df):,}")
print(f"Identifiable: {(df.status=='identifiable').sum():,}  ({summary['pct_identifiable']}%)")
print(f"Unidentifiable: {(df.status=='unidentifiable').sum():,}")
print(f"Same-SCC: {df.same_scc.sum():,}")
assert len(df) > 0, "POST: df must be non-empty"
assert (df.status == "identifiable").sum() > 0, "POST: must have identifiable edges"

# --- Section 2: Identifiability pie ---
status_counts = df["status"].value_counts()
colors = {"identifiable": "#2ecc71", "unidentifiable": "#e74c3c", "timeout": "#f39c12"}
color_list = [colors.get(s, "#95a5a6") for s in status_counts.index]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].pie(status_counts.values, labels=status_counts.index, colors=color_list, autopct="%1.1f%%", startangle=90)
axes[0].set_title("Edge Identifiability\n(sigma-single-door criterion)")
axes[1].bar(status_counts.index, status_counts.values, color=color_list, edgecolor="black")
axes[1].set_ylabel("Number of edges")
axes[1].set_title("Edge counts by status")
for i, (k, v) in enumerate(status_counts.items()):
    axes[1].text(i, v + 30, str(v), ha="center", fontsize=9)
plt.tight_layout()
out = VIZ_DIR / "csd_identifiability_overall.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"Saved: {out}")

# --- Section 3: Adjustment-set sizes ---
ident = df[df.status == "identifiable"].copy()
ident["adj_nodes"] = ident["adjustment_set"].fillna("").str.split("|")
ident["adj_size"] = ident["adj_nodes"].apply(lambda x: len(x) if x != [""] else 0)
print(f"Median adj-set size: {ident['adj_size'].median():.1f}, max: {ident['adj_size'].max()}")

# --- Section 4: Same-SCC vs Cross-SCC ---
cross = df[~df.same_scc]
same  = df[df.same_scc]
n_cross = len(cross)
n_same  = len(same)
if n_cross > 0:
    print(f"Cross-SCC identifiable: {100*(cross.status=='identifiable').sum()/n_cross:.1f}%")
if n_same > 0:
    print(f"Same-SCC  identifiable: {100*(same.status=='identifiable').sum()/n_same:.1f}%")

out2 = VIZ_DIR / "csd_same_vs_cross_scc.png"
fig, ax = plt.subplots(figsize=(8, 5))
cats = ("identifiable", "unidentifiable", "timeout")
x = [0, 1]
width = 0.25
for i, cat in enumerate(cats):
    c_vals = [(cross.status == cat).sum(), (same.status == cat).sum()]
    ax.bar([xi + i * width for xi in x], c_vals, width, label=cat, color=colors.get(cat, "#95a5a6"), edgecolor="black")
ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels([f"Cross-SCC\n(n={n_cross:,})", f"Same-SCC\n(n={n_same:,})"])
ax.set_ylabel("Number of edges")
ax.set_title("Identifiability: Same-SCC vs Cross-SCC edges")
ax.legend()
plt.tight_layout()
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"Saved: {out2}")

# --- Section 5: Cause taxonomy ---
from csd_rescue_worker import classify_nonident_cause, CAUSE_CATEGORIES  # noqa: E402

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
print(cause_counts.to_string())
assert all(c in list(CAUSE_CATEGORIES) + ["unknown"] for c in cause_counts.index), "POST: all causes valid"

unident.to_csv(NB_DIR / "csd_nonident_diagnosis.csv", index=False)

CATEGORY_COLORS = {
    "self_loop": "#e74c3c", "two_cycle": "#e67e22", "same_scc_long": "#f1c40f",
    "scc_edge_dissolved": "#3498db", "cross_scc_blocked": "#9b59b6", "unknown": "#95a5a6",
}
CATEGORY_LABELS = {
    "self_loop": "Self-loop", "two_cycle": "2-cycle (A->B->A)",
    "same_scc_long": "Same-SCC, long feedback", "scc_edge_dissolved": "SCC edge dissolved",
    "cross_scc_blocked": "Cross-SCC blocked",
}
ordered = [c for c in CAUSE_CATEGORIES if c in cause_counts.index]
vals = [cause_counts.get(c, 0) for c in ordered]
clrs = [CATEGORY_COLORS.get(c, "#95a5a6") for c in ordered]
labs = [CATEGORY_LABELS.get(c, c) for c in ordered]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(labs, vals, color=clrs, edgecolor="black")
axes[0].set_ylabel("Number of unidentifiable edges")
axes[0].set_title("Non-identifiability cause taxonomy")
axes[0].tick_params(axis="x", rotation=25)
for i, v in enumerate(vals):
    axes[0].text(i, v + 10, str(v), ha="center", fontsize=8)
axes[1].pie(vals, labels=labs, colors=clrs, autopct="%1.1f%%", startangle=90)
axes[1].set_title("Cause breakdown (unidentifiable edges)")
plt.tight_layout()
out3 = VIZ_DIR / "csd_nonident_causes.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"Saved: {out3}")

print("\nSmoke test PASSED.")
