"""
classify_scc_states.py
======================
Classify each SCC-experiment TF into one of three states based on freshly
recomputed graph properties:

    identifiable   – joint cyclic_id query succeeded
    unidentifiable – joint query failed AND cut is complete (a real result)
    cut_incomplete – do(B(t)) did NOT sever every in-SCC child→t return path
                     (Interpretation A structural limitation, e.g. 2-cycles)

Writes:
    notebooks/Ecoli_Analysis_Notebooks/scc_three_state_classification.csv

All cut-completeness and cyclic properties are recomputed freshly from the
graph + manifest; no stale shard values are trusted.

Usage
-----
    uv run python scripts/classify_scc_states.py 2>&1 | tail -40
"""

from __future__ import annotations

import csv
import json
import os
import sys

import networkx as nx

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
MANIFEST = "notebooks/Ecoli_Analysis_Notebooks/scc_perturb_job.json"
SHARDS_DIR = "notebooks/Ecoli_Analysis_Notebooks/scc_perturb_shards"
GRAPHML = "notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml"
OUTPUT_CSV = "notebooks/Ecoli_Analysis_Notebooks/scc_three_state_classification.csv"

# ---------------------------------------------------------------------------
# Imports from library
# ---------------------------------------------------------------------------
try:
    from nocap.scc_perturb import (
        residual_cluster_size_distribution,
        residual_scc_analysis,
        verify_cut_complete,
    )
except ImportError as exc:
    sys.exit(
        f"ERROR: cannot import scc_perturb — run with `uv run python ...`\n{exc}"
    )

# ---------------------------------------------------------------------------
# Load graph
# ---------------------------------------------------------------------------
print("Loading graph ...", flush=True)
raw = nx.read_graphml(GRAPHML)
if not isinstance(raw, nx.DiGraph):
    raw = nx.DiGraph(raw)
print(f"  nodes={raw.number_of_nodes()}, edges={raw.number_of_edges()}", flush=True)

# ---------------------------------------------------------------------------
# Load manifest
# ---------------------------------------------------------------------------
print("Loading manifest ...", flush=True)
with open(MANIFEST) as f:
    manifest = json.load(f)
tasks: dict = {t["tf"]: t for t in manifest["tasks"]}
print(f"  {len(tasks)} tasks in manifest", flush=True)

# ---------------------------------------------------------------------------
# Load shards
# ---------------------------------------------------------------------------
print("Loading shards ...", flush=True)
shards: dict = {}
for fn in os.listdir(SHARDS_DIR):
    if fn.startswith("scc_perturb_shard_") and fn.endswith(".json"):
        tf_name = fn[len("scc_perturb_shard_"):-5]
        with open(os.path.join(SHARDS_DIR, fn)) as f:
            shards[tf_name] = json.load(f)
print(f"  {len(shards)} shards loaded", flush=True)

# ---------------------------------------------------------------------------
# Classify each TF
# ---------------------------------------------------------------------------
rows = []

print("\n{:<10}  {:<14}  {:<14}  {:<12}  {:<10}  {}".format(
    "TF", "state", "joint_ident", "cut_complete", "tf_cyclic", "surviving_children"
))
print("-" * 90)

for tf in sorted(shards.keys()):
    shard = shards[tf]
    task = tasks.get(tf, {})

    min_cut: list = task.get("min_cut", [])
    in_scc_children: list = task.get("in_scc_children", [])
    joint_identifiable: bool = bool(shard.get("joint_identifiable", False))
    children: list = shard.get("outcomes", [])
    per_gene: dict = shard.get("per_gene", {})

    # --- Freshly recompute cut completeness from graph + manifest ---
    cut_result = verify_cut_complete(tf, in_scc_children, min_cut, raw)
    cut_complete: bool = cut_result["complete"]
    tf_still_cyclic: bool = cut_result["tf_still_cyclic"]
    surviving: list = cut_result["surviving_children"]

    # --- Freshly recompute residual cluster info ---
    analysis = residual_scc_analysis(tf, children, min_cut, raw)
    dist = residual_cluster_size_distribution(analysis)

    n_pergene_identifiable = sum(1 for v in per_gene.values() if v) if per_gene else None

    # --- 3-state classification ---
    if not cut_complete:
        state = "cut_incomplete"
    elif joint_identifiable:
        state = "identifiable"
    else:
        state = "unidentifiable"

    # Console output
    surv_str = str(surviving[:3]) + ("..." if len(surviving) > 3 else "") if surviving else "[]"
    print(f"{tf:<10}  {state:<14}  {joint_identifiable!s:<14}  {cut_complete!s:<12}  {tf_still_cyclic!s:<10}  {surv_str}")

    rows.append({
        "tf": tf,
        "n_children": len(children),
        "n_in_scc_children": len(in_scc_children),
        "min_cut_size": len(min_cut),
        "min_cut": ",".join(sorted(min_cut)),
        "joint_identifiable": joint_identifiable,
        "cut_complete": cut_complete,
        "tf_still_cyclic": tf_still_cyclic,
        "surviving_children": ",".join(surviving),
        "n_residual_clusters": dist["n_clusters"],
        "max_residual_cluster_size": dist["max_size"],
        "n_pergene_identifiable": n_pergene_identifiable if n_pergene_identifiable is not None else "",
        "state": state,
    })

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
from collections import Counter

state_counts = Counter(r["state"] for r in rows)
print("\n--- 3-state summary ---")
for s in ["identifiable", "unidentifiable", "cut_incomplete"]:
    print(f"  {s:<16} : {state_counts[s]}")
print(f"  {'TOTAL':<16} : {len(rows)}")

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
fieldnames = [
    "tf", "state", "joint_identifiable", "cut_complete", "tf_still_cyclic",
    "n_children", "n_in_scc_children", "min_cut_size", "min_cut",
    "surviving_children", "n_residual_clusters", "max_residual_cluster_size",
    "n_pergene_identifiable",
]
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote: {OUTPUT_CSV}")
