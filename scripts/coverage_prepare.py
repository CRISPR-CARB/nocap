r"""
coverage_prepare.py
===================
Serial preparation step for the parallelised coverage-matrix pipeline.

Runs once before the SLURM array job.  Loads the graph, builds the
unidentifiable query list and ranked candidate list, then writes a
``coverage_job.json`` manifest that the worker tasks read.

Usage:
  uv run python scripts/coverage_prepare.py \\
    --graphml  notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml \\
    --supptable notebooks/Ecoli_Analysis_Notebooks/supptable1.csv \\
    --manifest  notebooks/Ecoli_Analysis_Notebooks/coverage_job.json

The manifest contains:
  {
    "graphml":       "<absolute path>",
    "supptable":     "<absolute path>",
    "unidentifiable": [["tf1", "outcome"], ...],
    "candidates":    ["gene1", "gene2", ...],
    "n_queries":     <int>,
    "n_candidates":  <int>
  }

After this script prints "Array size: N", submit the worker array with
  --array=0-<N-1>
"""

import argparse
import json
import os
import sys
from multiprocessing import cpu_count

import networkx as nx
from y0.algorithm.ioscm.utils import get_apt_order
from y0.graph import NxMixedGraph

sys.path.insert(0, os.path.dirname(__file__))
from coverage_common import (
    build_baseline_queries,
    get_candidate_tfs,
    load_valid_genes,
    run_phase1,
)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare coverage-matrix manifest for SLURM array job"
    )
    parser.add_argument("--graphml", required=True, help="Path to .graphml network file")
    parser.add_argument("--supptable", required=True, help="Path to supptable1.csv")
    parser.add_argument(
        "--manifest",
        default="coverage_job.json",
        help="Output manifest path (default: coverage_job.json)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=cpu_count(),
        help=(
            "Number of parallel workers for Phase 1 identifiability check "
            "(default: all available CPUs).  Set to 1 for serial execution."
        ),
    )
    args = parser.parse_args()

    # Resolve to absolute paths so worker tasks can find them from any cwd
    graphml_path = os.path.abspath(args.graphml)
    supptable_path = os.path.abspath(args.supptable)
    manifest_path = os.path.abspath(args.manifest)

    assert os.path.isfile(graphml_path), f"graphml not found: {graphml_path}"
    assert os.path.isfile(supptable_path), f"supptable not found: {supptable_path}"

    # --- Load graph ---
    print("Loading graph...")
    ecoli_graph = nx.read_graphml(graphml_path)
    network_nodes = set(ecoli_graph.nodes())
    print(f"  Nodes: {len(network_nodes)}, Edges: {ecoli_graph.number_of_edges()}")

    # --- Experimental gene set ---
    print("Loading experimental gene set...")
    valid_genes = load_valid_genes(supptable_path, network_nodes)

    # --- Baseline queries ---
    print("Building baseline query pairs...")
    query_pairs = build_baseline_queries(ecoli_graph, valid_genes)
    print(f"  Total query pairs: {len(query_pairs)}")

    # --- Build NxMixedGraph and apt-order ---
    print("Building NxMixedGraph and computing apt-order...")
    ecoli_mixed = NxMixedGraph.from_edges(directed=list(ecoli_graph.edges()))
    apt_order = get_apt_order(ecoli_mixed)
    print("  Done.")

    # --- Phase 1: identify baseline unidentifiable queries ---
    n_workers = args.n_workers
    print(f"Running Phase 1 (baseline identifiability, {n_workers} worker(s))...")
    identifiable, unidentifiable = run_phase1(
        ecoli_mixed, query_pairs, apt_order, n_workers=n_workers
    )
    print(f"  Identifiable:   {len(identifiable)}")
    print(f"  Unidentifiable: {len(unidentifiable)}")

    if not unidentifiable:
        print("All queries are identifiable at baseline. Nothing to optimise.")
        sys.exit(0)

    # --- Candidate TFs ---
    print("Deriving candidate TF set...")
    candidates = get_candidate_tfs(ecoli_graph, valid_genes)

    # --- Write manifest ---
    manifest = {
        "graphml": graphml_path,
        "supptable": supptable_path,
        "unidentifiable": [[tf1, outcome] for tf1, outcome in unidentifiable],
        "candidates": candidates,
        "n_queries": len(unidentifiable),
        "n_candidates": len(candidates),
    }
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to: {manifest_path}")
    print(f"  Unidentifiable queries: {len(unidentifiable)}")
    print(f"  Candidate TFs:          {len(candidates)}")
    print(f"  Total pairs to eval:    {len(unidentifiable) * len(candidates)}")
    print(f"\nArray size: {len(unidentifiable)}")
    print(
        f"Submit with: sbatch --array=0-{len(unidentifiable) - 1} scripts/slurm/submit_coverage.sh"
    )


if __name__ == "__main__":
    main()
