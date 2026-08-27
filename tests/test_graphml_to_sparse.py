"""Tests for aligned GraphML regulatory tensors."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

from nocap.graphml_to_sparse import graphml_to_sparse


def test_gene_order_is_authoritative_and_cycles_are_retained():
    graph = nx.DiGraph()
    graph.add_edge("b", "a", polarity="activation")
    graph.add_edge("a", "b", polarity="repression")

    result = graphml_to_sparse(graph, ["b", "unused", "a"])

    assert result.gene_order == ("b", "unused", "a")
    torch.testing.assert_close(
        result.edge_index,
        torch.tensor([[2, 0], [0, 2]], dtype=torch.long),
    )
    torch.testing.assert_close(result.edge_signs, torch.tensor([-1, 1], dtype=torch.int8))
    assert result.report.isolated_measured_genes == ("unused",)


def test_graph_only_genes_and_incident_edges_are_reported_and_dropped():
    graph = nx.DiGraph()
    graph.add_edge("a", "b", polarity="+")
    graph.add_edge("external", "a", polarity="-")
    graph.add_edge("external", "also_external")

    result = graphml_to_sparse(graph, ["a", "b", "measured_only"])

    torch.testing.assert_close(result.edge_index, torch.tensor([[0], [1]], dtype=torch.long))
    assert result.report.graph_only_genes == ("also_external", "external")
    assert result.report.retained_edges == 1
    assert result.report.dropped_edges == 2
    assert result.report.isolated_measured_genes == ("measured_only",)


def test_self_loops_are_dropped_but_feedback_cycles_are_not():
    graph = nx.DiGraph()
    graph.add_edge("a", "a", polarity="+")
    graph.add_edge("a", "b", polarity="+")
    graph.add_edge("b", "a", polarity="-")

    result = graphml_to_sparse(graph, ["a", "b"])

    assert result.report.dropped_self_loops == (("a", "a"),)
    assert result.report.dropped_edges == 1
    assert result.report.retained_edges == 2


def test_graphml_path_is_supported(tmp_path):
    graph = nx.DiGraph()
    graph.add_edge("regulator", "target", polarity="negative")
    path = tmp_path / "network.graphml"
    nx.write_graphml(graph, path)

    result = graphml_to_sparse(path, ["target", "regulator"])

    torch.testing.assert_close(result.edge_index, torch.tensor([[1], [0]], dtype=torch.long))
    torch.testing.assert_close(result.edge_signs, torch.tensor([-1], dtype=torch.int8))


@pytest.mark.parametrize(
    "gene_order,match",
    [
        ([], "at least one"),
        (["a", "a"], "duplicate"),
        (["a", ""], "nonempty"),
    ],
)
def test_invalid_gene_order_is_rejected(gene_order, match):
    with pytest.raises(ValueError, match=match):
        graphml_to_sparse(nx.DiGraph(), gene_order)


def test_undirected_and_parallel_edge_graphs_are_rejected():
    with pytest.raises(ValueError, match="directed"):
        graphml_to_sparse(nx.Graph(), ["a"])
    with pytest.raises(ValueError, match="parallel"):
        graphml_to_sparse(nx.MultiDiGraph(), ["a"])


def test_ambiguous_polarity_is_unconstrained():
    graph = nx.DiGraph()
    graph.add_edge("a", "b", polarity="unknown")

    result = graphml_to_sparse(graph, ["a", "b"])

    torch.testing.assert_close(result.edge_signs, torch.tensor([0], dtype=torch.int8))
