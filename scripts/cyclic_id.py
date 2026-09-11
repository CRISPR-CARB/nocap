"""Identify causal queries in directed (possibly cyclic) graphs with y0.

The graph can be loaded from GraphML or selected from a small built-in demo.
Queries use the y0 convention ``P(outcomes | do(interventions))``.  Pass sets
as comma-separated values, for example ``--intervention X,Y --outcome Z``.

Examples
--------
    uv run python scripts/cyclic_id.py --demo cycle \
        --intervention X --outcome Z

    uv run python scripts/cyclic_id.py --demo tf_gene_cycle \
        --intervention TF1 --outcome G1

    uv run python scripts/cyclic_id.py --graphml graph.graphml \
        --intervention TF1 --outcome G1,G2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
from y0.algorithm.identify.utils import Unidentifiable

from nocap.cyclic_id import identify_causal_query


def _load_graph(graphml: str | None, demo: str) -> nx.DiGraph:
    """Load a GraphML graph or construct one of the built-in demos."""
    if graphml is not None:
        graph = nx.read_graphml(graphml)
        if not isinstance(graph, nx.DiGraph):
            graph = nx.DiGraph(graph)
        result = nx.DiGraph()
        result.add_nodes_from(str(node) for node in graph.nodes())
        result.add_edges_from((str(source), str(target)) for source, target in graph.edges())
        return result

    demos = {
        "chain": [("Z", "X"), ("X", "Y")],
        "confounded_chain": [("Z", "X"), ("X", "Y"), ("Z", "Y")],
        "cycle": [("X", "Y"), ("Y", "X"), ("Y", "Z"), ("Z", "Y")],
        "circle": [("X", "Y"), ("Y", "Z"), ("Z", "X")],
        "two_cycles_disconnected": [("A", "B"), ("B", "A"), ("C", "D"), ("D", "C")],
        "tf_gene_cycle": [
            # A five-TF strongly connected component.
            ("TF1", "TF2"),
            ("TF2", "TF3"),
            ("TF3", "TF4"),
            ("TF4", "TF5"),
            ("TF5", "TF1"),
            # A two-gene cascade downstream of the cycle.
            ("TF1", "G1"),
            ("G1", "G2"),
        ],
        "frontdoor_cycle": [
            ("TF1", "TF2"),
            ("TF2", "TF3"),
            ("TF3", "TF1"),
            ("TF1", "G1"),
            ("TF2", "G1"),
            ("TF3", "G1"),
        ],
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
    }
    if demo not in demos:
        raise ValueError(f"Unknown demo {demo!r}")
    return nx.DiGraph(demos[demo])


def _parse_nodes(values: list[str], option: str) -> frozenset[str]:
    """Parse repeated and comma-separated node arguments."""
    nodes = frozenset(node.strip() for value in values for node in value.split(",") if node.strip())
    if not nodes:
        raise SystemExit(f"{option} must contain at least one node")
    return nodes


def _format_nodes(nodes: frozenset[str]) -> str:
    """Format a node set deterministically for CLI output."""
    return "{" + ", ".join(sorted(nodes)) + "}"


def main() -> None:
    """Parse arguments, run cyclic identification, and print the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphml", type=str, help="Path to the GraphML directed graph.")
    parser.add_argument(
        "--demo",
        choices=[
            "chain",
            "confounded_chain",
            "cycle",
            "circle",
            "two_cycles_disconnected",
            "tf_gene_cycle",
            "small_network",
            "frontdoor_cycle",
        ],
        default="cycle",
        help="Built-in demo used when --graphml is omitted.",
    )
    parser.add_argument(
        "--intervention",
        "--interventions",
        action="append",
        required=True,
        metavar="NODE[,NODE...]",
    )
    parser.add_argument(
        "--outcome",
        "--outcomes",
        action="append",
        required=True,
        metavar="NODE[,NODE...]",
    )
    args = parser.parse_args()

    if args.graphml is not None and not Path(args.graphml).is_file():
        raise SystemExit(f"GraphML file does not exist: {args.graphml}")

    graph = _load_graph(args.graphml, args.demo)
    interventions = _parse_nodes(args.intervention, "--intervention")
    outcomes = _parse_nodes(args.outcome, "--outcome")
    unknown = (interventions | outcomes) - set(graph.nodes())
    if unknown:
        raise SystemExit(f"Query contains nodes absent from graph: {', '.join(sorted(unknown))}")

    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Query: P({_format_nodes(outcomes)} | do({_format_nodes(interventions)}))")

    try:
        expression = identify_causal_query(graph, interventions, outcomes)
    except Unidentifiable:
        print("Identifiable: false")
        return

    print("Identifiable: true")
    print(f"Expression: {expression}")


if __name__ == "__main__":
    main()
