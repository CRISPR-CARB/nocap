"""List ordered treatment/outcome queries and their CyclicID status."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import networkx as nx

from nocap.cyclic_id import identify_causal_query, is_identifiable
from nocap.cyclic_single_door import same_scc
from nocap.scc_perturb import build_intervened_graph


def _demo(name: str) -> nx.DiGraph:
    """Build a demo graph using the maintained CSD graph names."""
    edges = {
        "chain": [("Z", "X"), ("X", "Y")],
        "confounded_chain": [("Z", "X"), ("X", "Y"), ("Z", "Y")],
        "cycle": [("X", "Y"), ("Y", "X"), ("Y", "Z"), ("Z", "Y")],
        "two_cycles_disconnected": [("A", "B"), ("B", "A"), ("C", "D"), ("D", "C")],
        "small_network": [
            ("TF1", "G1"),
            ("TF1", "G2"),
            ("G2", "G3"),
            ("G1", "TF2"),
            ("G3", "TF2"),
            ("TF2", "G4"),
            ("G3", "TF3"),
            ("TF3", "G2"),
            ("TF3", "TF1"),
        ],
        "frontdoor_cycle": [
            ("TF1", "TF2"),
            ("TF2", "TF3"),
            ("TF3", "TF1"),
            ("TF1", "G1"),
            ("TF2", "G1"),
            ("TF3", "G1"),
        ],
    }
    if name not in edges:
        raise ValueError(f"unknown demo graph: {name}")
    return nx.DiGraph(edges[name])


def _load(path: str | None, demo: str) -> nx.DiGraph:
    """Load and normalize a GraphML graph or built-in demo."""
    if path is None:
        return _demo(demo)
    source = nx.read_graphml(path)
    graph = nx.DiGraph()
    graph.add_nodes_from(str(node) for node in source.nodes())
    graph.add_edges_from((str(u), str(v)) for u, v in source.edges())
    return graph


def main() -> None:
    """Run the query listing command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphml")
    parser.add_argument("--demo", default="cycle")
    parser.add_argument("--prior-intervention", action="append", default=[])
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    graph = _load(args.graphml, args.demo)
    graph = build_intervened_graph(graph, sorted(set(args.prior_intervention)))
    nodes = sorted(str(node) for node in graph.nodes())
    rows = []
    for treatment in nodes:
        for outcome in nodes:
            if treatment == outcome:
                continue
            reachable = nx.has_path(graph, treatment, outcome)
            if not args.all_pairs and not reachable:
                continue
            try:
                identifiable = is_identifiable(graph, {treatment}, {outcome})
                reason = "identifiable" if identifiable else "unidentifiable"
                expression = (
                    str(identify_causal_query(graph, {treatment}, {outcome}))
                    if identifiable
                    else ""
                )
            except Exception as error:  # malformed y0 graph/query is reported per row
                identifiable, reason, expression = False, type(error).__name__, ""
            rows.append(
                {
                    "treatment": treatment,
                    "outcome": outcome,
                    "reachable": reachable,
                    "same_scc": same_scc(graph, treatment, outcome),
                    "identifiable": identifiable,
                    "status": reason,
                    "expression": expression,
                }
            )
    rows.sort(key=lambda row: (row["treatment"], row["outcome"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    else:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]) if rows else ["treatment", "outcome"]
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
