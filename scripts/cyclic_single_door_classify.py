"""cyclic_single_door_classify.py — Snakemake scatter worker + CLI entrypoint.

Subcommands
-----------
prepare     List all directed edges in a GraphML file, shard them into JSON
            files, and write a manifest.
classify    Classify all edges in one shard under the σ-single-door criterion.
preprocess  Run the greedy intervention rescue optimizer on the full graph.

Usage
-----
    # Prepare (called by Snakemake rule 'prepare')
    python scripts/cyclic_single_door_classify.py prepare \\
        --graphml path/to/graph.graphml \\
        --shard-dir results/shards \\
        --shard-size 500 \\
        --manifest results/shard_manifest.json

    # Classify one shard (called by Snakemake rule 'classify')
    python scripts/cyclic_single_door_classify.py classify \\
        --graphml path/to/graph.graphml \\
        --shard results/shards/shard_0.json \\
        --output results/classified/shard_0.json

    # Greedy rescue (called by Snakemake rule 'preprocess')
    python scripts/cyclic_single_door_classify.py preprocess \\
        --graphml path/to/graph.graphml \\
        --k 10 \\
        --output-csv results/rescue_curve.csv \\
        --output-json results/rescue_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Graph loading helper
# ---------------------------------------------------------------------------


def _load_graph(graphml_path: str):
    """Load a GraphML file and return an nx.DiGraph."""
    import networkx as nx

    g = nx.read_graphml(graphml_path)
    # Ensure we have a DiGraph (read_graphml may return MultiDiGraph)
    if not isinstance(g, nx.DiGraph):
        g = nx.DiGraph(g)
    return g


# ---------------------------------------------------------------------------
# prepare subcommand
# ---------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> None:
    """Shard all directed edges into JSON files and write a manifest."""
    import json

    g = _load_graph(args.graphml)
    edges = list(g.edges())
    shard_size = args.shard_size

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_ids: list[str] = []
    for i in range(0, len(edges), shard_size):
        chunk = edges[i : i + shard_size]
        shard_id = str(i // shard_size)
        shard_path = shard_dir / f"shard_{shard_id}.json"
        with open(shard_path, "w") as f:
            json.dump({"shard_id": shard_id, "edges": [[u, v] for u, v in chunk]}, f)
        shard_ids.append(shard_id)

    manifest = {
        "graphml": args.graphml,
        "shard_size": shard_size,
        "n_edges": len(edges),
        "n_shards": len(shard_ids),
        "shard_ids": shard_ids,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"prepare: {len(edges)} edges → {len(shard_ids)} shards "
        f"(size {shard_size}) in {shard_dir}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# classify subcommand
# ---------------------------------------------------------------------------


def cmd_classify(args: argparse.Namespace) -> None:
    """Classify all edges in one shard and write results to a JSON file."""
    import json

    from nocap.cyclic_single_door import evaluate_all_edges

    g = _load_graph(args.graphml)

    with open(args.shard) as f:
        shard = json.load(f)

    edges: list[tuple[str, str]] = [(u, v) for u, v in shard["edges"]]
    shard_id = shard["shard_id"]

    print(
        f"classify shard {shard_id}: {len(edges)} edges",
        file=sys.stderr,
    )

    results = evaluate_all_edges(g, restrict_edges=edges)

    # Serialise: frozenset → list for JSON
    serialisable = []
    for r in results:
        row = dict(r)
        if row["adjustment_set"] is not None:
            row["adjustment_set"] = sorted(row["adjustment_set"])
        serialisable.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"shard_id": shard_id, "results": serialisable}, f)

    n_ident = sum(1 for r in results if r["status"] == "identifiable")
    print(
        f"classify shard {shard_id}: {n_ident}/{len(results)} identifiable",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# preprocess subcommand
# ---------------------------------------------------------------------------


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Run greedy rescue and write rescue curve CSV + full result JSON."""
    import csv
    import json

    from nocap.cyclic_single_door import maximize_identifiable_edges

    g = _load_graph(args.graphml)

    print(
        f"preprocess: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, k={args.k}",
        file=sys.stderr,
    )

    result = maximize_identifiable_edges(g, k=args.k)

    # Write rescue curve CSV
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_interventions", "n_identifiable"])
        for n_int, n_ident in result["curve"]:
            writer.writerow([n_int, n_ident])

    # Write full result JSON (excluding the nx.DiGraph which is not serialisable)
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable_results = []
    for r in result["final_results"]:
        row = dict(r)
        if row["adjustment_set"] is not None:
            row["adjustment_set"] = sorted(row["adjustment_set"])
        serialisable_results.append(row)

    summary = {
        "curve": result["curve"],
        "chosen_nodes": result["chosen_nodes"],
        "n_identifiable_baseline": result["n_identifiable_baseline"],
        "n_identifiable_final": result["n_identifiable_final"],
        "final_results": serialisable_results,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"preprocess: baseline={result['n_identifiable_baseline']} "
        f"final={result['n_identifiable_final']} "
        f"chosen={result['chosen_nodes']}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="σ-separation single-door edge classifier (scatter worker)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # --- prepare ---
    p_prep = sub.add_parser("prepare", help="Shard edges into JSON files")
    p_prep.add_argument("--graphml", required=True, help="Input GraphML file")
    p_prep.add_argument("--shard-dir", required=True, help="Output shard directory")
    p_prep.add_argument("--shard-size", type=int, default=500, help="Edges per shard")
    p_prep.add_argument("--manifest", required=True, help="Output manifest JSON path")

    # --- classify ---
    p_cls = sub.add_parser("classify", help="Classify edges in one shard")
    p_cls.add_argument("--graphml", required=True, help="Input GraphML file")
    p_cls.add_argument("--shard", required=True, help="Input shard JSON file")
    p_cls.add_argument("--output", required=True, help="Output classified JSON file")

    # --- preprocess ---
    p_pre = sub.add_parser("preprocess", help="Greedy intervention rescue")
    p_pre.add_argument("--graphml", required=True, help="Input GraphML file")
    p_pre.add_argument("--k", type=int, default=10, help="Intervention budget")
    p_pre.add_argument("--output-csv", required=True, help="Output rescue curve CSV")
    p_pre.add_argument("--output-json", required=True, help="Output rescue result JSON")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.subcommand == "prepare":
        cmd_prepare(args)
    elif args.subcommand == "classify":
        cmd_classify(args)
    elif args.subcommand == "preprocess":
        cmd_preprocess(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
