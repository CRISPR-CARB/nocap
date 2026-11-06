"""Tests for graph acyclification."""

from itertools import combinations

import networkx as nx
import pytest
from y0.algorithm.ioscm.utils import get_apt_order
from y0.algorithm.separation import are_d_separated, are_sigma_separated
from y0.algorithm.separation.sigma_separation import get_equivalence_classes, is_z_sigma_open
from y0.graph import NxMixedGraph

from nocap.acyclification import acyclification


def test_acyclification_replaces_cycle_with_bidirected_component():
    """A directed cycle becomes a complete bidirected SCC, without input mutation."""
    graph = NxMixedGraph.from_str_edges(
        nodes=["X", "A", "B", "C", "D"],
        directed=[("X", "A"), ("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")],
    )

    result = acyclification(graph)
    nodes = {node.name: node for node in graph.nodes()}

    assert set(result.directed.edges()) == {
        (nodes["X"], nodes["A"]),
        (nodes["C"], nodes["D"]),
        (nodes["X"], nodes["B"]),
        (nodes["X"], nodes["C"]),
    }
    assert {frozenset(edge) for edge in result.undirected.edges()} == {
        frozenset((nodes["A"], nodes["B"])),
        frozenset((nodes["A"], nodes["C"])),
        frozenset((nodes["B"], nodes["C"])),
    }
    assert set(graph.directed.edges()) == {
        (nodes["X"], nodes["A"]),
        (nodes["A"], nodes["B"]),
        (nodes["B"], nodes["C"]),
        (nodes["C"], nodes["A"]),
        (nodes["C"], nodes["D"]),
    }


def test_acyclification_preserves_acyclic_graph():
    """Acyclic directed edges remain directed and isolated nodes are retained."""
    graph = NxMixedGraph.from_str_edges(nodes=["A", "B", "C"], directed=[("A", "B")])

    result = acyclification(graph)

    assert set(result.directed.edges()) == set(graph.directed.edges())
    assert set(result.undirected.edges()) == set()
    assert set(result.nodes()) == set(graph.nodes())


def test_acyclification_expands_incoming_bidirected_edge_to_scc():
    """A bidirected neighbor connects to every member of a cyclic component."""
    graph = NxMixedGraph.from_str_edges(
        nodes=["X", "A", "B"],
        directed=[("A", "B"), ("B", "A")],
        undirected=[("X", "A")],
    )
    nodes = {node.name: node for node in graph.nodes()}

    result = acyclification(graph)

    assert {frozenset(edge) for edge in result.undirected.edges()} == {
        frozenset((nodes["X"], nodes["A"])),
        frozenset((nodes["X"], nodes["B"])),
        frozenset((nodes["A"], nodes["B"])),
    }
    assert set(graph.directed.edges()) == {
        (nodes["A"], nodes["B"]),
        (nodes["B"], nodes["A"]),
    }


def test_acyclification_rejects_unknown_type():
    """Only documented acyclification constructions are accepted."""
    graph = NxMixedGraph.from_str_edges(nodes=["A"], directed=[])

    with pytest.raises(ValueError, match="Unsupported acyclification_type"):
        acyclification(graph, acyclification_type="unknown")  # type: ignore[arg-type]


def test_dag_uses_order_and_expands_scc_adjacencies():
    """DAG mode orients each SCC by order and expands external adjacencies."""
    graph = NxMixedGraph.from_str_edges(
        nodes=["X", "A", "B", "C", "D"],
        directed=[("X", "A"), ("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")],
        undirected=[("X", "A")],
    )
    nodes = {node.name: node for node in graph.nodes()}

    result = acyclification(
        graph,
        acyclification_type="dag",
        node_order=[nodes["C"], nodes["B"], nodes["A"]],
        bidirected_edges=[(nodes["A"], nodes["C"])],
    )

    assert nx.is_directed_acyclic_graph(result.directed)
    assert set(result.directed.edges()) == {
        (nodes["C"], nodes["B"]),
        (nodes["C"], nodes["A"]),
        (nodes["B"], nodes["A"]),
        (nodes["X"], nodes["A"]),
        (nodes["X"], nodes["B"]),
        (nodes["X"], nodes["C"]),
        (nodes["C"], nodes["D"]),
    }
    assert {frozenset(edge) for edge in result.undirected.edges()} == {
        frozenset((nodes["X"], nodes["A"])),
        frozenset((nodes["X"], nodes["B"])),
        frozenset((nodes["X"], nodes["C"])),
        frozenset((nodes["A"], nodes["C"])),
    }


def test_dag_defaults_to_apt_order():
    """The default ordering within SCCs comes from y0's apt-order."""
    graph = NxMixedGraph.from_str_edges(
        nodes=["A", "B", "C"], directed=[("A", "B"), ("B", "C"), ("C", "A")]
    )

    result = acyclification(graph, acyclification_type="dag")
    apt_order = get_apt_order(graph)
    ordered_nodes = apt_order + [node for node in graph.nodes() if node not in set(apt_order)]

    assert set(result.directed.edges()) == {
        (source, target)
        for index, source in enumerate(ordered_nodes)
        for target in ordered_nodes[index + 1 :]
    }


@pytest.mark.parametrize("acyclification_type", ["bidirected", "dag"])
def test_acyclification_preserves_sigma_separation(acyclification_type):
    """Acyclification d-separation agrees with sigma-separation for all queries."""
    graph = NxMixedGraph.from_str_edges(
        nodes=["X", "A", "B", "C", "D"],
        directed=[
            ("X", "A"),
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
            ("C", "D"),
        ],
        undirected=[("X", "B"), ("D", "X")],
    )
    acyclified = acyclification(graph, acyclification_type=acyclification_type)
    nodes = list(graph.nodes())

    assert nx.is_directed_acyclic_graph(acyclified.directed)
    for left, right in combinations(nodes, 2):
        remaining = [node for node in nodes if node not in {left, right}]
        for size in range(len(remaining) + 1):
            for conditions in combinations(remaining, size):
                assert bool(are_sigma_separated(graph, left, right, conditions=conditions)) == bool(
                    are_d_separated(acyclified, left, right, conditions=conditions)
                )


@pytest.mark.parametrize("acyclification_type", ["bidirected", "dag"])
def test_sigma_separation_agrees_with_sigma_open_paths_on_mixed_graph(acyclification_type):
    """Expose path-level sigma-openness mismatches on a cyclic mixed graph."""
    graph = NxMixedGraph.from_str_edges(
        nodes=["X", "A", "B", "C", "D"],
        directed=[
            ("X", "A"),
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
            ("C", "D"),
        ],
        undirected=[("X", "B"), ("D", "X")],
    )
    acyclified = acyclification(graph, acyclification_type=acyclification_type)
    sigma = get_equivalence_classes(graph)

    for left, right in combinations(graph.nodes(), 2):
        paths = list(nx.all_simple_paths(graph.disorient(), left, right))
        remaining = [node for node in graph.nodes() if node not in {left, right}]
        for size in range(len(remaining) + 1):
            for conditions in combinations(remaining, size):
                conditions_set = set(conditions)
                has_open_path = any(
                    is_z_sigma_open(graph, path, sigma=sigma, conditions=conditions_set)
                    for path in paths
                )
                separated = bool(are_sigma_separated(graph, left, right, conditions=conditions))
                assert separated == (not has_open_path)
                assert (
                    bool(are_d_separated(acyclified, left, right, conditions=conditions))
                    == separated
                )


@pytest.mark.parametrize("acyclification_type", ["bidirected", "dag"])
def test_sigma_separation_agrees_with_sigma_open_paths(acyclification_type):
    """Sigma separation on a directed cycle matches the path openness predicate."""
    graph = NxMixedGraph.from_str_edges(
        nodes=["A", "B", "C"], directed=[("A", "B"), ("B", "C"), ("C", "A")]
    )
    acyclified = acyclification(graph, acyclification_type=acyclification_type)
    sigma = get_equivalence_classes(graph)

    for left, right in combinations(graph.nodes(), 2):
        paths = list(nx.all_simple_paths(graph.disorient(), left, right))
        remaining = [node for node in graph.nodes() if node not in {left, right}]
        for size in range(len(remaining) + 1):
            for conditions in combinations(remaining, size):
                conditions_set = set(conditions)
                has_open_path = any(
                    is_z_sigma_open(graph, path, sigma=sigma, conditions=conditions_set)
                    for path in paths
                )
                separated = bool(are_sigma_separated(graph, left, right, conditions=conditions))
                assert separated == (not has_open_path)
                assert (
                    bool(are_d_separated(acyclified, left, right, conditions=conditions))
                    == separated
                )


def test_dag_validates_additional_bidirected_edges():
    """Optional bidirected edges must connect distinct graph variables."""
    graph = NxMixedGraph.from_str_edges(nodes=["A", "B"], directed=[])
    a, b = graph.nodes()

    with pytest.raises(ValueError, match="bidirected_edges"):
        acyclification(
            graph,
            acyclification_type="dag",
            bidirected_edges=[(a, a)],
        )
    with pytest.raises(ValueError, match="bidirected_edges"):
        acyclification(
            graph,
            acyclification_type="dag",
            bidirected_edges=[(a, "outside")],  # type: ignore[list-item]
        )


def test_dag_acyclification_with_bidirected_edge():
    """DAG mode orients each SCC by node order and adds optional bidirected edges."""
    graph = NxMixedGraph.from_str_edges(
        nodes=["X", "A", "B", "C", "D"],
        directed=[("X", "A"), ("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")],
    )
    nodes = {node.name: node for node in graph.nodes()}

    result = acyclification(
        graph,
        acyclification_type="dag",
        node_order=[nodes["C"], nodes["B"], nodes["A"], nodes["X"], nodes["D"]],
        bidirected_edges=[(nodes["A"], nodes["C"])],
    )

    assert nx.is_directed_acyclic_graph(result.directed)
    assert set(result.directed.edges()) == {
        (nodes["C"], nodes["B"]),
        (nodes["C"], nodes["A"]),
        (nodes["B"], nodes["A"]),
        (nodes["X"], nodes["A"]),
        (nodes["X"], nodes["B"]),
        (nodes["X"], nodes["C"]),
        (nodes["C"], nodes["D"]),
    }
    assert set(result.undirected.edges()) == {(nodes["A"], nodes["C"])}


def test_acyclification_expands_bidirected_edges_between_sccs():
    """A bidirected adjacency across SCCs expands to all cross-component pairs."""
    graph = NxMixedGraph.from_str_edges(
        nodes=["A", "B", "X", "Y"],
        directed=[("A", "B"), ("B", "A"), ("X", "Y"), ("Y", "X")],
        undirected=[("A", "X")],
    )
    nodes = {node.name: node for node in graph.nodes()}

    result = acyclification(graph, acyclification_type="dag")

    assert {frozenset(edge) for edge in result.undirected.edges()} == {
        frozenset((nodes["A"], nodes["X"])),
        frozenset((nodes["A"], nodes["Y"])),
        frozenset((nodes["B"], nodes["X"])),
        frozenset((nodes["B"], nodes["Y"])),
    }


def test_dag_drops_original_internal_bidirected_edges_by_default():
    """DAG mode only adds internal bidirected edges when explicitly requested."""
    graph = NxMixedGraph.from_str_edges(
        nodes=["A", "B"], directed=[("A", "B"), ("B", "A")], undirected=[("A", "B")]
    )

    result = acyclification(graph, acyclification_type="dag")

    assert set(result.undirected.edges()) == set()


def test_acyclification_allows_optional_bidirected_edges_between_sccs():
    """Optional bidirected edges may be added between distinct SCCs."""
    graph = NxMixedGraph.from_str_edges(nodes=["A", "B"], directed=[("A", "B")])
    nodes = {node.name: node for node in graph.nodes()}

    result = acyclification(
        graph,
        acyclification_type="dag",
        bidirected_edges=[(nodes["A"], nodes["B"])],
    )

    assert set(result.undirected.edges()) == {(nodes["A"], nodes["B"])}


def test_dag_defaults_to_y0_apt_order():
    """Without an explicit order, the y0 assembling pseudo-topological order is used."""
    graph = NxMixedGraph.from_str_edges(
        nodes=["A", "B", "C"], directed=[("A", "B"), ("B", "C"), ("C", "A")]
    )

    result = acyclification(graph, acyclification_type="dag")
    apt_order = get_apt_order(graph)
    apt_order.extend(sorted(set(graph.nodes()) - set(apt_order)))

    assert set(result.directed.edges()) == {
        (source, target)
        for source_index, source in enumerate(apt_order)
        for target in apt_order[source_index + 1 :]
    }


def test_node_order_rejects_string_nodes():
    """Ordering follows y0's Variable node type rather than accepting names."""
    graph = NxMixedGraph.from_str_edges(nodes=["A", "B"], directed=[("A", "B"), ("B", "A")])

    with pytest.raises(ValueError, match="node_order must contain distinct graph variables"):
        acyclification(
            graph,
            acyclification_type="dag",
            node_order=["A", "B"],  # type: ignore[list-item]
        )
