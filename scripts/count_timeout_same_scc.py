"""count_timeout_same_scc.py — How many timed-out edges have cause & effect
in the same SCC of G' = G - {cause->effect}?

Those are exactly the edges the csd_break SCC-break sweep can act on:
a residual cycle survives the removal of the direct edge, so an
intervention may dissolve it and make the edge single-door identifiable.
"""

from __future__ import annotations

import csv
from pathlib import Path

import networkx as nx

REPO = Path(__file__).parent.parent
GRAPHML = REPO / "notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml"
RESULTS_CSV = REPO / "notebooks/Ecoli_Analysis_Notebooks/csd_results.csv"
OUT_CSV = REPO / "results/cyclic_single_door/timeout_same_scc_edges.csv"


def main() -> None:
    g = nx.read_graphml(str(GRAPHML))

    timeouts: list[tuple[str, str]] = []
    with open(RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "timeout":
                timeouts.append((row["cause"], row["effect"]))

    print(f"timed-out edges: {len(timeouts)}")

    # Precompute SCC membership of full G (for reference)
    scc_full = {n: i for i, comp in enumerate(nx.strongly_connected_components(g)) for n in comp}

    same_scc_after = []
    missing_nodes = 0
    not_an_edge = 0

    # Cache SCC index map keyed by the removed edge is unnecessary; recompute per edge.
    # Optimisation: cause & effect are in the same SCC of G' iff effect can still
    # reach cause AND cause can still reach effect in G'. After removing the single
    # cause->effect edge, mutual reachability is checked directly (cheaper than full SCC).
    for cause, effect in timeouts:
        if cause not in g or effect not in g:
            missing_nodes += 1
            continue
        had_edge = g.has_edge(cause, effect)
        if had_edge:
            g.remove_edge(cause, effect)
        else:
            not_an_edge += 1
        # same SCC iff mutually reachable in G'
        reach_ce = nx.has_path(g, cause, effect)
        reach_ec = nx.has_path(g, effect, cause)
        same = reach_ce and reach_ec
        if same:
            same_scc_after.append((cause, effect))
        if had_edge:
            g.add_edge(cause, effect)

    n = len(timeouts)
    s = len(same_scc_after)
    print(f"missing-node edges (skipped): {missing_nodes}")
    print(f"rows where cause->effect not a graph edge: {not_an_edge}")
    print(f"same SCC after removing cause->effect: {s} / {n} ({100.0 * s / n:.1f}%)")
    print(f"NOT same SCC (cycle already broken by removal): {n - s - missing_nodes}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cause", "effect"])
        w.writerows(same_scc_after)
    print(f"wrote {s} same-SCC timeout edges -> {OUT_CSV}")


if __name__ == "__main__":
    main()
