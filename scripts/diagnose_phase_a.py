"""
diagnose_phase_a.py
===================
Quick diagnostic script for Phase A results.

1. For cut_verified=False TFs: show WHY child->tf path survives do(B(t)).
   The key question: does child reach tf directly (tf is child's child)?
   Or via a longer path through the still-live SCC?

2. For Check-2 violators (identifiable + has residual cluster):
   Show the residual cluster members and whether tf is still cyclic.

3. For Check-1 violators (unidentifiable + no residual cluster):
   marR: 2 children, both per_gene=False — the joint query fails and
         individuals also fail. No cluster but also not the "some succeed"
         pattern. This is a genuine falsification of the original hypothesis
         wording, but may suggest a tighter subhypothesis.
   uxuR: 10 children, 5 per_gene=True (fliC, gntP, uidA, uidB, uidC),
         5 per_gene=False. Also min_cut=[]. Both violate.
"""

from __future__ import annotations

import json
import os

import networkx as nx

from nocap.scc_perturb import (
    build_intervened_graph,
    residual_cluster_size_distribution,
    residual_scc_analysis,
)

MANIFEST = "notebooks/Ecoli_Analysis_Notebooks/scc_perturb_job.json"
SHARDS_DIR = "notebooks/Ecoli_Analysis_Notebooks/scc_perturb_shards"
GRAPHML = "notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml"

print("Loading graph...")
raw = nx.read_graphml(GRAPHML)
if not isinstance(raw, nx.DiGraph):
    raw = nx.DiGraph(raw)

with open(MANIFEST) as f:
    manifest = json.load(f)

tasks = {t["tf"]: t for t in manifest["tasks"]}
shards = {}
for fn in os.listdir(SHARDS_DIR):
    if fn.startswith("scc_perturb_shard_") and fn.endswith(".json"):
        tf = fn[len("scc_perturb_shard_") : -5]
        with open(os.path.join(SHARDS_DIR, fn)) as f:
            shards[tf] = json.load(f)

SEP = "-" * 65

# ---------------------------------------------------------------------------
# Section 1: cut_verified=False — why do children still reach tf?
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("SECTION 1: cut_verified=False — path analysis")
print(SEP)

for tf, shard in sorted(shards.items()):
    task = tasks.get(tf, {})
    min_cut = task.get("min_cut", [])
    children = shard.get("outcomes", [])

    g_do = build_intervened_graph(raw, min_cut)

    # Find children that can still reach tf
    reaching = []
    for c in children:
        if c in g_do and tf in g_do:
            try:
                if nx.has_path(g_do, c, tf):
                    reaching.append(c)
            except nx.NetworkXError:
                pass

    if reaching:
        print(f"\n  {tf}: {len(reaching)}/{len(children)} children can reach tf after do(B(t))")
        print(f"    min_cut = {min_cut}")
        # Show shortest path for up to 3 reaching children
        for c in reaching[:3]:
            try:
                path = nx.shortest_path(g_do, c, tf)
                print(
                    f"    Path {c}->...->tf: {path[:6]}{'...' if len(path) > 6 else ''} (len={len(path)})"
                )
            except nx.NetworkXNoPath:
                pass

# ---------------------------------------------------------------------------
# Section 2: Check-2 violators — identifiable TFs WITH residual clusters
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("SECTION 2: Check-2 violators (identifiable + has residual cluster)")
print(SEP)

check2_violators = ["cpxR", "cra", "glnG", "phoB", "torR"]
for tf in check2_violators:
    shard = shards[tf]
    task = tasks[tf]
    min_cut = task["min_cut"]
    children = shard["outcomes"]

    analysis = residual_scc_analysis(tf, children, min_cut, raw)
    dist = residual_cluster_size_distribution(analysis)

    print(f"\n  {tf}: joint=True, n_children={len(children)}, |B(t)|={len(min_cut)}")
    print(
        f"    tf_still_cyclic={analysis['tf_still_cyclic']}, "
        f"cut_verified={analysis['cut_verified']}"
    )
    print(f"    n_residual_clusters={dist['n_clusters']}, sizes={dist['sizes']}")
    for i, cluster in enumerate(analysis["residual_clusters"]):
        members = sorted(cluster)
        print(f"    Cluster {i}: {members}")

# ---------------------------------------------------------------------------
# Section 3: Check-1 violators — unidentifiable TFs with NO residual cluster
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("SECTION 3: Check-1 violators (unidentifiable + no residual cluster)")
print(SEP)

check1_violators = ["marR", "uxuR"]
for tf in check1_violators:
    shard = shards[tf]
    task = tasks[tf]
    min_cut = task["min_cut"]
    children = shard["outcomes"]
    per_gene = shard.get("per_gene", {})

    analysis = residual_scc_analysis(tf, children, min_cut, raw)

    print(f"\n  {tf}: joint=False, n_children={len(children)}, |B(t)|={len(min_cut)}")
    print(f"    in_scc_children: {task['in_scc_children']}")
    print(
        f"    tf_still_cyclic={analysis['tf_still_cyclic']}, "
        f"cut_verified={analysis['cut_verified']}"
    )
    print(f"    n_residual_clusters={len(analysis['residual_clusters'])}")
    print(f"    children_cyclic:  {analysis['children_cyclic']}")
    print(f"    children_acyclic: {analysis['children_acyclic']}")
    if per_gene:
        ident = [g for g, v in per_gene.items() if v]
        unident = [g for g, v in per_gene.items() if not v]
        print(f"    per_gene identifiable ({len(ident)}): {sorted(ident)}")
        print(f"    per_gene unidentifiable ({len(unident)}): {sorted(unident)}")
    # Check if tf itself is in the same SCC as any child in g_do
    g_do = build_intervened_graph(raw, min_cut)
    sccs = list(nx.strongly_connected_components(g_do))
    node_to_scc = {}
    for idx, scc in enumerate(sccs):
        for n in scc:
            node_to_scc[n] = idx
    if tf in node_to_scc:
        tf_scc_id = node_to_scc[tf]
        tf_scc = [scc for scc in sccs if tf in scc][0]
        children_in_tf_scc = [
            c for c in children if c in node_to_scc and node_to_scc[c] == tf_scc_id
        ]
        print(f"    SCC containing tf has {len(tf_scc)} nodes")
        print(f"    Children in same SCC as tf: {sorted(children_in_tf_scc)}")
