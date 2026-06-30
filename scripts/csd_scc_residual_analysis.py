"""csd_scc_residual_analysis.py — Per-edge SCC membership before and after edge removal.

For every edge u->v in the E. coli network, computes:
  - is_self_loop       : u == v (degenerate SCC, flagged separately)
  - scc_id_before      : Tarjan SCC label of u (and v) in the full graph
  - scc_size_before    : size of u's SCC in the full graph
  - same_scc_before    : whether u and v were in the same SCC (sanity-check vs csd_results.csv)
  - same_scc_after     : whether u and v are STILL in the same SCC after removing u->v
  - scc_size_u_after   : size of u's SCC in G - {u->v}
  - scc_size_v_after   : size of v's SCC in G - {u->v}

Algorithm
---------
1. Run Tarjan once on G  ->  partition edges into groups by SCC.
2. Cross-SCC edges       ->  same_scc_after = False  (trivial, no recompute)
3. Self-loops            ->  flagged degenerate, same_scc_after = N/A
4. Same-SCC edges        ->  recompute SCCs on the SCC-induced subgraph with the
                             edge removed.  Only the giant SCC (130 nodes) is ever
                             large; even repeated passes over it are cheap.

Runtime:  < 60 s single-threaded on the head node.
Optional: srun --time=5:00 --mem=4G uv run python scripts/csd_scc_residual_analysis.py

Output:
  results/cyclic_single_door/csd_scc_residual.csv
  (also printed to stdout: crosstabs and SCC-size buckets by status)

Usage:
    uv run python scripts/csd_scc_residual_analysis.py
"""

from __future__ import annotations

import time
from pathlib import Path

import networkx as nx
import pandas as pd

REPO = Path(__file__).parent.parent
GRAPHML = (
    REPO / "notebooks" / "Ecoli_Analysis_Notebooks" / "ecoli_full_network_no_small_rna.graphml"
)
CSD_CSV = REPO / "notebooks" / "Ecoli_Analysis_Notebooks" / "csd_results.csv"
OUT_DIR = REPO / "results" / "cyclic_single_door"
OUT_CSV = OUT_DIR / "csd_scc_residual.csv"


# ---------------------------------------------------------------------------
# Core per-edge function
# ---------------------------------------------------------------------------


def residual_scc_info(
    g: nx.DiGraph,
    u: str,
    v: str,
    scc_map: dict[str, int],
    scc_sets: dict[int, set[str]],
    scc_sizes: dict[int, int],
) -> dict:
    """Return residual-SCC columns for edge u -> v.

    Parameters
    ----------
    g         : full graph (not mutated)
    u, v      : edge endpoints
    scc_map   : node -> SCC id in full graph
    scc_sets  : SCC id -> set of nodes
    scc_sizes : SCC id -> size
    """
    sid_u = scc_map[u]
    sid_v = scc_map[v]
    scc_sz = scc_sizes[sid_u]
    same_before = sid_u == sid_v
    is_self = u == v

    row: dict = {
        "is_self_loop": is_self,
        "scc_id_before": sid_u,
        "scc_size_before": scc_sz,
        "same_scc_before": same_before,
        "same_scc_after": None,
        "scc_size_u_after": None,
        "scc_size_v_after": None,
    }

    if is_self:
        # Self-loop: flagged degenerate.  After removal u is still in whatever
        # SCC it belongs to without the loop (which may still contain it if
        # other paths exist, or may be a singleton).
        sub = nx.DiGraph(g.subgraph(scc_sets[sid_u]))
        sub.remove_edge(u, v)
        sccs_after = {
            n: cid for cid, comp in enumerate(nx.strongly_connected_components(sub)) for n in comp
        }
        row["same_scc_after"] = False  # self-loop can't be "same" after removal
        row["scc_size_u_after"] = sum(1 for n in sub if sccs_after[n] == sccs_after[u])
        row["scc_size_v_after"] = row["scc_size_u_after"]  # same node
        return row

    if not same_before:
        # Cross-SCC edge: removing it cannot change SCC membership of u or v.
        row["same_scc_after"] = False
        row["scc_size_u_after"] = scc_sizes[sid_u]
        row["scc_size_v_after"] = scc_sizes[sid_v]
        return row

    # Same-SCC edge: recompute on the SCC-induced subgraph only.
    nodes_in_scc = scc_sets[sid_u]
    sub = nx.DiGraph(g.subgraph(nodes_in_scc))
    sub.remove_edge(u, v)
    sccs_after = {
        n: cid for cid, comp in enumerate(nx.strongly_connected_components(sub)) for n in comp
    }
    same_after = sccs_after[u] == sccs_after[v]
    row["same_scc_after"] = same_after
    row["scc_size_u_after"] = sum(1 for n in sub if sccs_after[n] == sccs_after[u])
    row["scc_size_v_after"] = sum(1 for n in sub if sccs_after[n] == sccs_after[v])
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()

    print("Loading graph ...", flush=True)
    g: nx.DiGraph = nx.read_graphml(str(GRAPHML))
    print(f"  {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")

    print("Loading CSD results ...", flush=True)
    df = pd.read_csv(CSD_CSV)
    print(f"  {len(df):,} edges in csd_results.csv")

    # ----- Step 1: single Tarjan pass on full graph -----
    print("Running Tarjan SCC on full graph ...", flush=True)
    scc_list = list(nx.strongly_connected_components(g))
    scc_map: dict[str, int] = {}
    scc_sets: dict[int, set[str]] = {}
    scc_sizes: dict[int, int] = {}
    for cid, comp in enumerate(scc_list):
        for n in comp:
            scc_map[n] = cid
        scc_sets[cid] = comp
        scc_sizes[cid] = len(comp)

    n_sccs = len(scc_list)
    giant_size = max(scc_sizes.values())
    print(f"  {n_sccs:,} SCCs, giant SCC size = {giant_size}")

    # ----- Step 2: per-edge analysis -----
    print("Computing residual SCC for all edges ...", flush=True)
    rows = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        u, v = row.cause, row.effect
        if u not in scc_map or v not in scc_map:
            # Node not in graphml (shouldn't happen, but handle gracefully)
            rows.append(
                {
                    "cause": u,
                    "effect": v,
                    "status": row.status,
                    "adjustment_set": row.adjustment_set,
                    "same_scc": row.same_scc,
                    "timed_out": row.timed_out,
                    "is_self_loop": str(u) == str(v),
                    "scc_id_before": -1,
                    "scc_size_before": 0,
                    "same_scc_before": False,
                    "same_scc_after": None,
                    "scc_size_u_after": None,
                    "scc_size_v_after": None,
                }
            )
            continue
        info = residual_scc_info(g, str(u), str(v), scc_map, scc_sets, scc_sizes)
        rows.append(
            {
                "cause": u,
                "effect": v,
                "status": row.status,
                "adjustment_set": row.adjustment_set,
                "same_scc": row.same_scc,
                "timed_out": row.timed_out,
                **info,
            }
        )
        if i % 1000 == 0:
            print(f"  {i:,}/{len(df):,} edges processed ...", flush=True)

    result = pd.DataFrame(rows)

    # ----- Step 3: save -----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    t1 = time.time()
    print(f"\nWrote {OUT_CSV}  ({len(result):,} rows, {t1 - t0:.1f}s)")

    # ----- Step 4: report -----
    _print_report(result)


def _print_report(result: pd.DataFrame) -> None:
    print()
    print("=" * 65)
    print("CSD Residual-SCC Analysis Report")
    print("=" * 65)

    self_loops = result[result.is_self_loop]
    non_self = result[~result.is_self_loop]

    print(f"\n  Self-loops (degenerate SCC): {len(self_loops):,}")

    # Crosstab 1: same_scc_after x status (non-self-loops only)
    print("\n  Crosstab: same_scc_after  x  status (excluding self-loops)")
    ct = pd.crosstab(non_self["same_scc_after"], non_self["status"])
    print(ct.to_string())

    # Crosstab 2: same_scc_before x same_scc_after (sanity check)
    print("\n  Crosstab: same_scc_before  x  same_scc_after (non-self)")
    ct2 = pd.crosstab(non_self["same_scc_before"], non_self["same_scc_after"])
    print(ct2.to_string())

    # SCC size buckets vs status
    # Giant SCC size = 68; bucket at >=50 captures it.
    print("\n  SCC-size-before distribution by status (non-self-loops)")
    bin_edges = [0, 1, 2, 5, 10, 50, 10_000]
    labels = ["1", "2-4", "5-9", "10-49", "50-68 (giant)", "large"]
    # Ensure labels == len(bin_edges)-1
    labels = labels[: len(bin_edges) - 1]
    non_self = non_self.copy()
    non_self["scc_bucket"] = pd.cut(
        non_self["scc_size_before"],
        bins=bin_edges,
        labels=labels,
        right=True,
    )
    ct3 = pd.crosstab(non_self["scc_bucket"], non_self["status"])
    print(ct3.to_string())

    # Key finding: timeout concentration in large SCCs (giant = 68 nodes)
    GIANT_THRESH = 50
    print()
    for status in ["identifiable", "unidentifiable", "timeout"]:
        sub = non_self[non_self.status == status]
        n_giant = (sub.scc_size_before >= GIANT_THRESH).sum()
        print(
            f"  {status:15s}: {len(sub):5,} total, "
            f"{n_giant:5,} with scc_size_before>={GIANT_THRESH}  "
            f"({100.0 * n_giant / len(sub) if len(sub) else 0:.1f}%)"
        )

    print("=" * 65)


if __name__ == "__main__":
    main()
