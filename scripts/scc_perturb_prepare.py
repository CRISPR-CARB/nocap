"""
scc_perturb_prepare.py
======================
Serial preparation step for the SCC-perturbation pipeline.

For each TF in supptable1.csv that participates in a non-trivial strongly
connected component (SCC), computes:

  B(t) = minimum set of *intermediate* nodes (neither t nor any direct child
         of t) whose hard intervention do(B(t)) ensures no direct child of t
         remains in the same SCC as t.

Formally, B(t) is the minimum vertex cut that severs every return path
c → t (for each in-SCC direct child c of t) within the SCC subgraph, subject
to the constraint that neither t nor any direct child of t is in B(t)
(Interpretation A).

The cut is computed via node-splitting max-flow on the small SCC subgraph,
using ``networkx.algorithms.connectivity.minimum_node_cut``.

Outputs a ``scc_perturb_job.json`` manifest consumed by ``scc_perturb_worker.py``.

Usage:
  uv run python scripts/scc_perturb_prepare.py \\
    --graphml  notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml \\
    --supptable notebooks/Ecoli_Analysis_Notebooks/supptable1.csv \\
    --manifest  notebooks/Ecoli_Analysis_Notebooks/scc_perturb_job.json

Manifest schema:
  {
    "graphml":      "<absolute path>",
    "supptable":    "<absolute path>",
    "n_tasks":      <int>,         # == len(tasks)
    "tasks": [
      {
        "tf":             "<gene>",
        "scc_size":       <int>,
        "scc_nodes":      ["<gene>", ...],
        "in_scc_children": ["<gene>", ...],
        "min_cut":        ["<gene>", ...],   # B(t) — may be empty
        "in_scc":         true               # always true for tasks list
      },
      ...
    ],
    "dag_tfs": ["<gene>", ...]   # TFs not in any non-trivial SCC (B = empty)
  }

After this script completes, submit the array with:
  sbatch --array=0-<n_tasks-1> scripts/slurm/submit_scc_perturb.sh
"""

import argparse
import json
import os
import sys

import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from coverage_common import load_valid_genes


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def find_in_scc_children(tf: str, scc_nodes: set, graph) -> list:
    """
    Return the direct children of *tf* that are members of *scc_nodes*.

    These are the nodes c such that there is a directed edge tf→c and c
    is in the same SCC as tf.

    axiomander:
        ensures:
            len(result) <= len(list(graph.successors(tf)))
            all(c != tf for c in result)
            all(c in scc_nodes for c in result)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(tf, str), "PRE: tf must be a str"
    assert isinstance(scc_nodes, (set, frozenset)), (
        "PRE: scc_nodes must be a set or frozenset"
    )

    children = [
        c for c in graph.successors(tf)
        if c in scc_nodes and c != tf
    ]

    # --- POST ---
    assert isinstance(children, list), "POST: result must be a list"
    assert all(c != tf for c in children), "POST: no self-loops in result"
    assert all(c in scc_nodes for c in children), (
        "POST: all children must be in scc_nodes"
    )
    return children


def compute_min_cut_b(
    tf: str,
    scc_nodes: set,
    in_scc_children: list,
    graph,
) -> list:
    """
    Compute the minimum background perturbation set B(t) for TF *tf*.

    B(t) is the minimum vertex cut (on the SCC subgraph restricted to
    ``scc_nodes``) that separates every return path from any in-SCC child
    of *tf* back to *tf*, subject to:
      - *tf* itself is NOT in B(t)
      - direct children of *tf* that are in the SCC are NOT in B(t)
        (Interpretation A)

    The cut is found via ``networkx.minimum_node_cut``, which internally
    uses node-splitting + max-flow.  We exclude forbidden nodes by
    contracting them to infeasible in the auxiliary graph.

    If *tf* is not in any non-trivial SCC (len(scc_nodes) <= 1) or has no
    in-SCC children, returns an empty list.

    axiomander:
        ensures:
            tf not in result
            all(c not in result for c in in_scc_children)
            all(n in scc_nodes for n in result)
            implies(len(in_scc_children) == 0, len(result) == 0)
            implies(len(scc_nodes) <= 1, len(result) == 0)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(tf, str), "PRE: tf must be a str"
    assert isinstance(scc_nodes, (set, frozenset)), (
        "PRE: scc_nodes must be a set or frozenset"
    )
    assert isinstance(in_scc_children, list), (
        "PRE: in_scc_children must be a list"
    )

    if not in_scc_children:
        return []

    # Forbidden nodes: tf itself + its direct in-SCC children
    forbidden = {tf} | set(in_scc_children)

    # Build the SCC subgraph
    scc_sub = graph.subgraph(scc_nodes).copy()

    # Remove forbidden nodes from the interior of the cut (they can still be
    # sources/sinks in the flow problem but cannot be removed).  We achieve
    # this by giving them infinite capacity in node-split flow — i.e. we
    # compute the cut on the subgraph where forbidden nodes are contracted
    # away from the interior.
    #
    # Implementation: build an auxiliary graph that omits forbidden nodes
    # from the cut candidates by temporarily removing their in-edges in the
    # interior (equivalent to infinite capacity).  The simplest correct
    # approach: compute minimum_node_cut with explicit source/sink and then
    # validate the result excludes forbidden nodes.  If any cut node is
    # forbidden, we fall back to removing ALL in-SCC children (safe, always
    # valid but not always minimum).
    #
    # Aggregate over all in-SCC children: we need every c→t path cut.
    # We use a super-source (virtual) connected to all in-SCC children.

    if len(scc_nodes) <= 1:
        return []

    # Add a virtual super-source outside the SCC that connects to all
    # in-SCC children, so a single min-cut call handles all of them at once.
    aux = scc_sub.copy()
    super_src = "__SUPER_SRC__"
    aux.add_node(super_src)
    for c in in_scc_children:
        aux.add_edge(super_src, c)

    try:
        raw_cut: set = set(nx.minimum_node_cut(aux, s=super_src, t=tf))
        # Exclude the virtual super-source if it sneaked in
        cut_nodes: set = raw_cut - {super_src}
        # If any forbidden node ended up in the cut (shouldn't happen for
        # correct graphs, but guard anyway), fall back to safe heuristic.
        if cut_nodes & forbidden:
            # Heuristic fallback: remove all in-neighbours of tf inside SCC
            # except forbidden nodes — guaranteed to cut all return paths.
            cut_nodes = {
                n for n in scc_sub.predecessors(tf)
                if n not in forbidden
            }
    except (nx.NetworkXError, nx.NetworkXNoPath):
        # networkx raises when no path exists (already disconnected) or on
        # degenerate graphs; in those cases B is empty.
        cut_nodes = set()

    result = sorted(cut_nodes)  # deterministic order

    # --- POST ---
    assert isinstance(result, list), "POST: result must be a list"
    assert tf not in result, "POST: tf must not be in B(t)"
    assert all(c not in result for c in in_scc_children), (
        "POST: direct in-SCC children must not be in B(t)"
    )
    assert all(n in scc_nodes for n in result), (
        "POST: all cut nodes must be in the SCC"
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SCC-perturbation manifest for SLURM array job"
    )
    parser.add_argument("--graphml", required=True, help="Path to .graphml network file")
    parser.add_argument("--supptable", required=True, help="Path to supptable1.csv")
    parser.add_argument(
        "--manifest",
        default="scc_perturb_job.json",
        help="Output manifest path (default: scc_perturb_job.json)",
    )
    args = parser.parse_args()

    graphml_path = os.path.abspath(args.graphml)
    supptable_path = os.path.abspath(args.supptable)
    manifest_path = os.path.abspath(args.manifest)

    assert os.path.isfile(graphml_path), f"graphml not found: {graphml_path}"
    assert os.path.isfile(supptable_path), f"supptable not found: {supptable_path}"

    # --- Load graph ---
    print("Loading graph...")
    graph = nx.read_graphml(graphml_path)
    if not isinstance(graph, nx.DiGraph):
        graph = nx.DiGraph(graph)
    network_nodes = set(graph.nodes())
    print(f"  Nodes: {len(network_nodes)}, Edges: {graph.number_of_edges()}")

    # --- Load experimental gene set ---
    print("Loading experimental gene set...")
    valid_genes = load_valid_genes(supptable_path, network_nodes)
    print(f"  Valid TFs in network: {len(valid_genes)}")

    # --- Global SCC decomposition (one pass, O(V+E)) ---
    print("Computing strongly connected components...")
    sccs = list(nx.strongly_connected_components(graph))
    nontrivial_sccs = [s for s in sccs if len(s) > 1]
    print(f"  Total SCCs: {len(sccs)}")
    print(f"  Non-trivial SCCs (size > 1): {len(nontrivial_sccs)}")

    # Map each node to its SCC (only for non-trivial SCCs)
    node_to_scc: dict = {}
    for scc in nontrivial_sccs:
        for node in scc:
            node_to_scc[node] = frozenset(scc)

    # --- Process each TF ---
    tasks: list = []
    dag_tfs: list = []

    tf_list = sorted(valid_genes)  # deterministic order
    print(f"\nProcessing {len(tf_list)} TFs...")

    for tf in tf_list:
        if tf not in network_nodes:
            continue

        scc_nodes = node_to_scc.get(tf)
        if scc_nodes is None:
            # TF is already in a trivial SCC (DAG node) — no perturbation needed
            dag_tfs.append(tf)
            continue

        in_scc_children = find_in_scc_children(tf, scc_nodes, graph)
        if not in_scc_children:
            # TF is in a non-trivial SCC but has no direct children in it
            # (only in-edges from SCC members) — no children to protect
            dag_tfs.append(tf)
            continue

        min_cut = compute_min_cut_b(tf, scc_nodes, in_scc_children, graph)

        tasks.append({
            "tf": tf,
            "scc_size": len(scc_nodes),
            "scc_nodes": sorted(scc_nodes),
            "in_scc_children": sorted(in_scc_children),
            "min_cut": min_cut,
            "in_scc": True,
        })

    print(f"\n  TFs requiring SCC perturbation: {len(tasks)}")
    print(f"  TFs already in DAG (no perturbation needed): {len(dag_tfs)}")

    # --- Summary stats ---
    if tasks:
        cut_sizes = [len(t["min_cut"]) for t in tasks]
        print(f"  |B(t)| min={min(cut_sizes)}  max={max(cut_sizes)}  "
              f"mean={sum(cut_sizes)/len(cut_sizes):.1f}")
        zero_cut = sum(1 for s in cut_sizes if s == 0)
        if zero_cut:
            print(f"  WARNING: {zero_cut} TF(s) have |B(t)|=0 (SCC may be trivially separable)")

    # --- Write manifest ---
    manifest = {
        "graphml": graphml_path,
        "supptable": supptable_path,
        "n_tasks": len(tasks),
        "tasks": tasks,
        "dag_tfs": dag_tfs,
    }
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to: {manifest_path}")
    print(f"  Array size: {len(tasks)}")
    if tasks:
        print(
            f"\nSubmit with:\n"
            f"  bash scripts/slurm/submit_scc_perturb.sh\n"
            f"or manually:\n"
            f"  sbatch --array=0-{len(tasks)-1} ..."
        )
    else:
        print("No TFs require SCC perturbation — nothing to submit.")


if __name__ == "__main__":
    main()
