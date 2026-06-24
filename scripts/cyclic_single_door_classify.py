"""cyclic_single_door_classify.py — Snakemake scatter worker + CLI entrypoint.

Subcommands
-----------
prepare     List all directed edges in a GraphML file, shard them into JSON
            files, and write a manifest.
classify    Classify all edges in one shard under the sigma-single-door criterion.
            Supports per-edge timeout and incremental checkpoint/resume.
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
        --output results/classified/shard_0.json \\
        --timeout 60

    # Greedy rescue (called by Snakemake rule 'preprocess')
    python scripts/cyclic_single_door_classify.py preprocess \\
        --graphml path/to/graph.graphml \\
        --k 10 \\
        --output-csv results/rescue_curve.csv \\
        --output-json results/rescue_result.json

Incremental checkpoint/resume
------------------------------
``classify`` writes each completed edge result as a JSON line to
``<output>.partial`` immediately after classification (or timeout).  If the
worker is killed and restarted, already-completed edges are loaded from the
partial file and skipped, so no work is duplicated.

When all edges are done the partial file is assembled into the final
``<output>`` JSON and then removed.
"""

from __future__ import annotations

import argparse
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


def _split_into_n_shards(
    edges: list, n_shards: int
) -> list[list]:
    """Split *edges* into exactly *n_shards* near-equal contiguous chunks.

    Uses array_split-style division: the first ``r`` shards receive ``q + 1``
    edges and the remaining shards receive ``q`` edges, where
    ``q, r = divmod(len(edges), n_shards)``.  Empty shards are never emitted
    (so the actual count may be less than *n_shards* when
    *n_shards* > *len(edges)*).

    PRE: n_shards >= 1
    PRE: len(edges) >= 0
    POST: len(result) == min(n_shards, len(edges)) or len(edges) == 0
    POST: sum(len(c) for c in result) == len(edges)
    """
    assert n_shards >= 1, "PRE: n_shards must be >= 1"
    assert len(edges) >= 0, "PRE: edges list must be non-negative length"

    n = len(edges)
    if n == 0:
        result: list[list] = []
        assert len(result) == 0, "POST: empty input yields no shards"
        assert sum(len(c) for c in result) == 0, "POST: total edges preserved"
        return result

    actual_shards = min(n_shards, n)
    q, r = divmod(n, actual_shards)
    chunks: list[list] = []
    start = 0
    for i in range(actual_shards):
        size = q + 1 if i < r else q
        chunks.append(edges[start : start + size])
        start += size

    assert len(chunks) == actual_shards, "POST: correct shard count"
    assert sum(len(c) for c in chunks) == n, "POST: all edges accounted for"
    return chunks


def cmd_prepare(args: argparse.Namespace) -> None:
    """Shard all directed edges into JSON files and write a manifest."""
    import json

    g = _load_graph(args.graphml)
    edges = list(g.edges())

    # Resolve sharding strategy: --n-shards takes priority over --shard-size.
    if args.n_shards is not None:
        chunks = _split_into_n_shards(edges, args.n_shards)
    else:
        shard_size = args.shard_size
        chunks = [
            edges[i : i + shard_size] for i in range(0, len(edges), shard_size)
        ]

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_ids: list[str] = []
    for idx, chunk in enumerate(chunks):
        shard_id = str(idx)
        shard_path = shard_dir / f"shard_{shard_id}.json"
        with open(shard_path, "w") as f:
            json.dump({"shard_id": shard_id, "edges": [[u, v] for u, v in chunk]}, f)
        shard_ids.append(shard_id)

    manifest = {
        "graphml": args.graphml,
        "n_shards": len(shard_ids),
        "n_edges": len(edges),
        "shard_ids": shard_ids,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"prepare: {len(edges)} edges → {len(shard_ids)} shards in {shard_dir}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# classify subcommand
# ---------------------------------------------------------------------------


def _row_to_jsonable(r: dict) -> dict:
    """Convert a classify_edge result dict to a JSON-serialisable dict.

    frozenset → sorted list; all other values are already JSON-safe.

    PRE: 'adjustment_set' key present
    POST: returned dict has no frozenset values
    """
    assert "adjustment_set" in r, "PRE: result dict must have adjustment_set key"
    row = dict(r)
    if row["adjustment_set"] is not None:
        row["adjustment_set"] = sorted(row["adjustment_set"])
    assert not any(
        isinstance(v, frozenset) for v in row.values()
    ), "POST: no frozenset values remain"
    return row


def cmd_classify(args: argparse.Namespace) -> None:
    """Classify all edges in one shard with per-edge timeout + incremental checkpoint/resume.

    Incremental checkpoint/resume
    ------------------------------
    Results are written one-per-line to ``<output>.partial`` immediately after
    each edge completes (or times out).  On restart the partial file is loaded
    and already-done edges are skipped, so no work is duplicated.

    When all edges are done the partial file is assembled into the final
    ``<output>`` JSON and the partial file is removed.

    PRE: args.timeout is None or a positive int
    PRE: args.shard is a valid JSON file with keys "shard_id" and "edges"
    POST: output JSON contains results for every edge in the shard
    """
    import json

    from nocap.cyclic_single_door import evaluate_all_edges

    # Treat --timeout 0 as "no per-edge timeout"
    timeout_s: int | None = args.timeout if args.timeout and args.timeout > 0 else None

    # --- PRE ---
    assert timeout_s is None or (
        isinstance(timeout_s, int) and timeout_s > 0
    ), "PRE: timeout must be None or a positive int"

    g = _load_graph(args.graphml)

    with open(args.shard) as f:
        shard = json.load(f)

    all_edges: list[tuple[str, str]] = [(u, v) for u, v in shard["edges"]]
    shard_id = shard["shard_id"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")

    # ----------------------------------------------------------------
    # Resume: load already-done edges from .partial checkpoint file
    # ----------------------------------------------------------------
    done_results: list[dict] = []
    done_edge_keys: set[tuple[str, str]] = set()

    if partial_path.exists():
        with open(partial_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    done_results.append(row)
                    done_edge_keys.add((row["cause"], row["effect"]))
                except json.JSONDecodeError:
                    pass  # skip corrupt lines

    remaining_edges = [
        (u, v) for (u, v) in all_edges if (u, v) not in done_edge_keys
    ]

    n_total = len(all_edges)
    n_already = len(done_results)
    n_todo = len(remaining_edges)

    print(
        f"classify shard {shard_id}: {n_total} edges total, "
        f"{n_already} resumed from checkpoint, {n_todo} remaining",
        file=sys.stderr,
        flush=True,
    )

    # ----------------------------------------------------------------
    # Classify remaining edges one at a time, writing each to .partial
    # ----------------------------------------------------------------
    with open(partial_path, "a") as partial_f:
        for i, (u, v) in enumerate(remaining_edges):
            rows = evaluate_all_edges(
                g,
                restrict_edges=[(u, v)],
                timeout_seconds=timeout_s,
            )
            assert len(rows) == 1, "POST invariant: one row per edge call"
            row = rows[0]
            json_row = _row_to_jsonable(row)
            partial_f.write(json.dumps(json_row) + "\n")
            partial_f.flush()

            status_tag = row["status"]
            print(
                f"  [{n_already + i + 1}/{n_total}] {u} -> {v}: {status_tag}",
                file=sys.stderr,
                flush=True,
            )

    # ----------------------------------------------------------------
    # Assemble final output from all results (done + new)
    # ----------------------------------------------------------------
    all_results: list[dict] = []
    edge_order = {(u, v): idx for idx, (u, v) in enumerate(all_edges)}

    # Reload full partial file to capture this run's new results too
    with open(partial_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # Sort back to original shard edge order
    all_results.sort(key=lambda r: edge_order.get((r["cause"], r["effect"]), 999999))

    # --- POST ---
    assert len(all_results) == n_total, (
        f"POST: expected {n_total} results, got {len(all_results)}"
    )

    with open(output_path, "w") as f:
        json.dump({"shard_id": shard_id, "results": all_results}, f)

    # Clean up the partial checkpoint file now that the final output is written
    partial_path.unlink(missing_ok=True)

    n_ident = sum(1 for r in all_results if r["status"] == "identifiable")
    n_timeout = sum(1 for r in all_results if r["status"] == "timeout")
    print(
        f"classify shard {shard_id}: {n_ident}/{n_total} identifiable, "
        f"{n_timeout} timed out",
        file=sys.stderr,
        flush=True,
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
        description="sigma-separation single-door edge classifier (scatter worker)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # --- prepare ---
    p_prep = sub.add_parser("prepare", help="Shard edges into JSON files")
    p_prep.add_argument("--graphml", required=True, help="Input GraphML file")
    p_prep.add_argument("--shard-dir", required=True, help="Output shard directory")
    p_prep.add_argument(
        "--n-shards",
        type=int,
        default=None,
        help="Number of shards (takes priority over --shard-size)",
    )
    p_prep.add_argument(
        "--shard-size",
        type=int,
        default=500,
        help="Edges per shard (fallback when --n-shards is not given)",
    )
    p_prep.add_argument("--manifest", required=True, help="Output manifest JSON path")

    # --- classify ---
    p_cls = sub.add_parser("classify", help="Classify edges in one shard")
    p_cls.add_argument("--graphml", required=True, help="Input GraphML file")
    p_cls.add_argument("--shard", required=True, help="Input shard JSON file")
    p_cls.add_argument("--output", required=True, help="Output classified JSON file")
    p_cls.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-edge timeout in seconds (POSIX SIGALRM); 0 disables (default: 60)",
    )

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
