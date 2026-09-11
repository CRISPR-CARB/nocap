"""Contract-checked wrapper around y0's cyclic identification algorithm."""

from __future__ import annotations

from collections.abc import Collection

import networkx as nx
from y0.algorithm.identify.cyclic_id import cyclic_id as _y0_cyclic_id
from y0.algorithm.identify.utils import Unidentifiable
from y0.algorithm.ioscm.utils import get_apt_order
from y0.dsl import Expression, Variable
from y0.graph import NxMixedGraph

from nocap.cyclic_single_door import nx_digraph_to_y0

__all__ = ["identify_causal_query", "is_identifiable"]


def _validate_query(
    graph: nx.DiGraph,
    interventions: Collection[str],
    outcomes: Collection[str],
) -> None:
    """Validate the graph and query inputs shared by the public wrappers."""
    assert isinstance(graph, nx.DiGraph), "PRE: graph must be an nx.DiGraph"
    assert isinstance(interventions, (set, frozenset)), (
        "PRE: interventions must be a set or frozenset"
    )
    assert isinstance(outcomes, (set, frozenset)), "PRE: outcomes must be a set or frozenset"
    assert interventions, "PRE: interventions must not be empty"
    assert outcomes, "PRE: outcomes must not be empty"
    assert all(isinstance(node, str) for node in interventions), (
        "PRE: intervention nodes must be strings"
    )
    assert all(isinstance(node, str) for node in outcomes), "PRE: outcome nodes must be strings"
    query_nodes = interventions | outcomes
    assert query_nodes <= {str(node) for node in graph.nodes()}, (
        "PRE: every query node must be present in graph"
    )


def identify_causal_query(
    graph: nx.DiGraph,
    interventions: set[str] | frozenset[str],
    outcomes: set[str] | frozenset[str],
    unobserved: set[str] | frozenset[str] | None = None
) -> Expression:
    """Identify ``P(outcomes | do(interventions))`` in a directed graph.

    Parameters
    ----------
    graph:
        The input directed graph. Cycles are supported.
    interventions:
        Nodes in the intervention set.
    outcomes:
        Nodes whose post-intervention distribution is requested.

    Returns
    -------
    Expression
        The y0 identifying expression.

    Raises
    ------
    y0.algorithm.identify.utils.Unidentifiable
        If the query cannot be identified from the graph.

    axiomander:
        requires:
            isinstance(graph, nx.DiGraph)
            isinstance(interventions, (set, frozenset))
            isinstance(outcomes, (set, frozenset))
            interventions
            outcomes
            all(isinstance(node, str) for node in interventions)
            all(isinstance(node, str) for node in outcomes)
            set(interventions | outcomes) <= set(str(node) for node in graph.nodes())
        ensures:
            isinstance(result, Expression)
        modifies:
            none
    """
    _validate_query(graph, interventions, outcomes)

    mixed_graph: NxMixedGraph = nx_digraph_to_y0(graph, unobserved)
    ordering = get_apt_order(mixed_graph)
    result = _y0_cyclic_id(
        graph=mixed_graph,
        outcomes={Variable(node) for node in outcomes},
        interventions={Variable(node) for node in interventions},
        ordering=ordering,
    )

    # y0 currently returns an Expression on success and raises Unidentifiable
    # otherwise. Keep this boundary explicit so the wrapper's contract remains
    # stable if that implementation changes.
    assert isinstance(result, Expression), "POST: result must be a y0 Expression"
    return result


def is_identifiable(
    graph: nx.DiGraph,
    interventions: set[str] | frozenset[str],
    outcomes: set[str] | frozenset[str],
) -> bool:
    """Return whether a causal query is identifiable in ``graph``.

    The input contract is the same as :func:`identify_causal_query`. The y0
    identifying expression is intentionally discarded.

    axiomander:
        ensures:
            isinstance(result, bool)
        modifies:
            none
    """
    _validate_query(graph, interventions, outcomes)
    try:
        identify_causal_query(graph, interventions, outcomes)
    except Unidentifiable:
        result = False
    else:
        result = True

    assert isinstance(result, bool), "POST: result must be bool"
    return result
