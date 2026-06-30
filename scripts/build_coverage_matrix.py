r"""
build_coverage_matrix.py
========================
Builds a boolean coverage matrix M[candidate_tf][query] where:
  M[g][q] = True  if adding background do(g) makes unidentifiable query q identifiable
  M[g][q] = False otherwise

Inputs (relative to this script's location or passed as args):
  --graphml   path to ecoli_full_network_no_small_rna.graphml
  --supptable path to supptable1.csv
  --output    path to write coverage_matrix.csv  (default: coverage_matrix.csv)
  --checkpoint path to checkpoint JSON            (default: coverage_checkpoint.json)

The candidate TF set is every node with out-degree >= 1 in the graphml.
The query set is the 55 unidentifiable baseline queries derived from the
50-gene experimental set (same logic as Biology_Analysis.ipynb Phase 1).

Run from the repo root:
  uv run python scripts/build_coverage_matrix.py \\
    --graphml notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml \\
    --supptable notebooks/Ecoli_Analysis_Notebooks/supptable1.csv \\
    --output notebooks/Ecoli_Analysis_Notebooks/coverage_matrix.csv \\
    --checkpoint notebooks/Ecoli_Analysis_Notebooks/coverage_checkpoint.json

For HPC / SLURM parallel execution use the three-script pipeline instead:
  scripts/coverage_prepare.py  -- build manifest
  scripts/coverage_worker.py   -- SLURM array task (one per query)
  scripts/coverage_reduce.py   -- merge shards -> CSV
  scripts/slurm/submit_coverage.sh -- orchestrator
"""

import argparse
import csv
import json
import os
import sys

import networkx as nx
from y0.algorithm.identify.cyclic_id import cyclic_id
from y0.algorithm.identify.utils import Unidentifiable
from y0.algorithm.ioscm.utils import get_apt_order
from y0.dsl import P, Variable
from y0.graph import NxMixedGraph

sys.path.insert(0, os.path.dirname(__file__))

# All helpers now live in coverage_common; import them from there.
from coverage_common import (
    build_baseline_queries,
    get_candidate_tfs,
    load_valid_genes,
    rows_to_matrix,
    run_phase1,
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build single-background coverage matrix")
    parser.add_argument("--graphml", required=True)
    parser.add_argument("--supptable", required=True)
    parser.add_argument("--output", default="coverage_matrix.csv")
    parser.add_argument("--checkpoint", default="coverage_checkpoint.json")
    args = parser.parse_args()

    # --- Load graph ---
    print("Loading graph...")
    ecoli_graph = nx.read_graphml(args.graphml)
    network_nodes = set(ecoli_graph.nodes())
    print(f"  Nodes: {len(network_nodes)}, Edges: {ecoli_graph.number_of_edges()}")

    # --- Experimental gene set ---
    print("Loading experimental gene set...")
    valid_genes = load_valid_genes(args.supptable, network_nodes)

    # --- Baseline queries ---
    print("Building baseline query pairs...")
    query_pairs = build_baseline_queries(ecoli_graph, valid_genes)
    print(f"  Total query pairs: {len(query_pairs)}")

    # --- Build NxMixedGraph and apt-order ---
    print("Building NxMixedGraph and computing apt-order (this may take a moment)...")
    ecoli_mixed = NxMixedGraph.from_edges(directed=list(ecoli_graph.edges()))
    apt_order = get_apt_order(ecoli_mixed)
    print("  Done.")

    # --- Phase 1: identify baseline unidentifiable queries ---
    print("Running Phase 1 (baseline identifiability)...")
    identifiable, unidentifiable = run_phase1(ecoli_mixed, query_pairs, apt_order)
    print(f"  Identifiable:   {len(identifiable)}")
    print(f"  Unidentifiable: {len(unidentifiable)}")

    if not unidentifiable:
        print("All queries are identifiable at baseline. Nothing to optimize.")
        sys.exit(0)

    # --- Candidate TFs ---
    print("Deriving candidate TF set...")
    candidates = get_candidate_tfs(ecoli_graph, valid_genes)

    # --- Load checkpoint ---
    all_network_vars = {Variable(g) for g in network_nodes}
    checkpoint_path = args.checkpoint

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        results = ckpt["results"]  # list of [tf1, candidate, outcome, found]
        completed = set(tuple(x) for x in ckpt["completed"])
        print(f"Resuming from checkpoint: {len(completed)} pairs done, {len(results)} results")
    else:
        results = []
        completed = set()
        print("Starting fresh coverage matrix build.")

    total = len(unidentifiable) * len(candidates)
    done = len(completed)
    print(f"Total pairs to evaluate: {total}  (already done: {done})")

    # --- Main loop ---
    for q_idx, (tf1, outcome) in enumerate(unidentifiable):
        for c_idx, candidate in enumerate(candidates):
            # skip: candidate == tf1 (same gene), candidate == outcome
            if candidate == tf1 or candidate == outcome:
                continue
            key = (tf1, candidate, outcome)
            if key in completed:
                continue

            try:
                cyclic_id(
                    graph=ecoli_mixed,
                    outcomes={Variable(outcome)},
                    interventions={Variable(tf1)},
                    ordering=apt_order,
                    base_distribution=P[{Variable(candidate)}](all_network_vars),
                )
                found = True
            except Unidentifiable:
                found = False

            results.append([tf1, candidate, outcome, found])
            completed.add(key)
            done += 1

            # progress + checkpoint every 100 pairs
            if done % 100 == 0:
                pct = done / total * 100
                print(
                    f"  Progress: {done}/{total} ({pct:.1f}%) | "
                    f"query {q_idx + 1}/{len(unidentifiable)}: do({tf1})->{outcome}, "
                    f"candidate {c_idx + 1}/{len(candidates)}: {candidate}"
                )
                with open(checkpoint_path, "w") as f:
                    json.dump({"results": results, "completed": list(completed)}, f)

    # --- Final checkpoint save ---
    with open(checkpoint_path, "w") as f:
        json.dump({"results": results, "completed": list(completed)}, f)
    print(f"Checkpoint saved: {len(completed)} pairs evaluated.")

    # --- Write coverage matrix CSV ---
    query_labels = [f"{tf1}->{outcome}" for tf1, outcome in unidentifiable]
    candidate_set, query_labels_out, lookup = rows_to_matrix(results, query_labels)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_tf", *query_labels_out])
        for cand in candidate_set:
            row = [cand]
            for qlabel in query_labels_out:
                row.append(int(lookup.get((cand, qlabel), False)))
            writer.writerow(row)

    # Summary
    resolved_queries = set()
    for tf1, candidate, outcome, found in results:
        if found:
            resolved_queries.add(f"{tf1}->{outcome}")
    print("\n--- Coverage Matrix Summary ---")
    print(f"Unidentifiable queries:          {len(unidentifiable)}")
    print(f"Queries resolvable by >= 1 TF:   {len(resolved_queries)}")
    print(f"Queries with no rescue found:    {len(unidentifiable) - len(resolved_queries)}")
    print(f"Candidate TFs in matrix:         {len(candidate_set)}")
    print(f"Output written to:               {args.output}")


if __name__ == "__main__":
    main()
