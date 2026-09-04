"""Tests for the nx.DiGraph cyclic-ID wrapper."""

import networkx as nx
import pytest
from y0.algorithm.identify.utils import Unidentifiable
from y0.dsl import Expression

from nocap.cyclic_id import identify_causal_query, is_identifiable


def test_identifies_acyclic_query() -> None:
    graph = nx.DiGraph([("X", "Y")])

    result = identify_causal_query(graph, {"X"}, {"Y"})

    assert isinstance(result, Expression)
    assert is_identifiable(graph, {"X"}, {"Y"}) is True


def test_identifies_query_downstream_of_cycle() -> None:
    graph = nx.DiGraph(
        [
            ("TF1", "TF2"),
            ("TF2", "TF3"),
            ("TF3", "TF1"),
            ("TF1", "G1"),
            ("G1", "G2"),
        ]
    )

    assert is_identifiable(graph, {"TF1"}, {"G1"}) is True
    assert is_identifiable(graph, {"TF1"}, {"G2"}) is True


def test_reports_unidentifiable_cyclic_query() -> None:
    graph = nx.DiGraph([("X", "Y"), ("Y", "X")])

    assert is_identifiable(graph, {"X"}, {"Y"}) is False
    with pytest.raises(Unidentifiable):
        identify_causal_query(graph, {"X"}, {"Y"})


@pytest.mark.parametrize(
    ("interventions", "outcomes", "message"),
    [
        ({"missing"}, {"Y"}, "every query node"),
        (set(), {"Y"}, "interventions must not be empty"),
        ({"X"}, set(), "outcomes must not be empty"),
        ({1}, {"Y"}, "intervention nodes must be strings"),
    ],
)
def test_preconditions(
    interventions: set, outcomes: set, message: str
) -> None:
    graph = nx.DiGraph([("X", "Y")])

    with pytest.raises(AssertionError, match=message):
        identify_causal_query(graph, interventions, outcomes)


def test_requires_nx_digraph() -> None:
    with pytest.raises(AssertionError, match="graph must be an nx.DiGraph"):
        identify_causal_query(nx.Graph([("X", "Y")]), {"X"}, {"Y"})
