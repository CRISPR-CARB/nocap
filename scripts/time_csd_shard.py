"""time_csd_shard.py — measure startup vs per-edge cost for csd classify.

Reports:
  - import time (networkx + nocap.cyclic_single_door)
  - graph load time (_load_graph on the ecoli graphml)
  - per-edge classify time over a small sample (with a hard timeout per edge)

Usage:
    uv run python scripts/time_csd_shard.py --graphml PATH --sample 12 --timeout 120
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphml", required=True)
    ap.add_argument("--shard", default=None, help="optional shard json to draw edges from")
    ap.add_argument("--sample", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    t0 = time.perf_counter()
    import networkx as nx

    from nocap.cyclic_single_door import classify_edge, evaluate_all_edges  # noqa: F401

    t_import = time.perf_counter() - t0
    print(f"import_time_s: {t_import:.3f}")

    t0 = time.perf_counter()
    g = nx.read_graphml(args.graphml)
    if not isinstance(g, nx.DiGraph):
        g = nx.DiGraph(g)
    t_load = time.perf_counter() - t0
    print(f"graph_load_time_s: {t_load:.3f}")
    print(f"graph_nodes: {g.number_of_nodes()}  graph_edges: {g.number_of_edges()}")

    # Pick a sample of edges
    if args.shard:
        import json

        shard = json.loads(Path(args.shard).read_text())
        edges = [(u, v) for u, v in shard["edges"]][: args.sample]
    else:
        edges = list(g.edges())[: args.sample]

    print(f"timing {len(edges)} edges (timeout={args.timeout}s each)...")
    t0 = time.perf_counter()
    results = evaluate_all_edges(g, restrict_edges=edges, timeout_seconds=args.timeout)
    t_eval = time.perf_counter() - t0

    n_timeout = sum(1 for r in results if r.get("status") == "timeout")
    n_done = len(results) - n_timeout
    print(f"eval_total_s: {t_eval:.3f}")
    print(f"edges: {len(edges)}  completed: {n_done}  timed_out: {n_timeout}")
    if len(edges):
        print(f"per_edge_avg_s: {t_eval / len(edges):.3f}")
    print(f"startup_overhead_s (import+load): {t_import + t_load:.3f}")


if __name__ == "__main__":
    main()
