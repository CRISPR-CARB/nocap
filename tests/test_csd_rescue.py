"""tests/test_csd_rescue.py — Unit tests for csd_rescue_worker helpers.

Tests the cause-taxonomy classifier and a tiny rescue round-trip using a
small hand-crafted graph that exercises each category.

Graph fixture
-------------
    A -> B  (with B -> A creating a 2-cycle for the A->B edge)
    A -> C  (C -> D -> A creating a long feedback for the A->C edge)
    C -> D
    D -> A
    E -> F  (no cycle; cross-SCC, but we set up a confounder via the sigma
             extension to make it unidentifiable -- or just test the classifier
             taxonomy, which is based on graph structure, not the sigma oracle)
    G -> G  (self-loop)

For the cause-taxonomy tests we only need the graph structure.
For the rescue round-trip we use a simpler 3-node 2-cycle graph.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

# ---------------------------------------------------------------------------
# Make sure scripts/ is importable (it is not a package)
# ---------------------------------------------------------------------------
_SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from csd_rescue_worker import (  # type: ignore[import]  # noqa: E402
    CAUSE_CATEGORIES,
    classify_nonident_cause,
    compute_rescue_nodes,
)

# ---------------------------------------------------------------------------
# Fixture graphs
# ---------------------------------------------------------------------------


def _make_two_cycle_graph() -> nx.DiGraph:
    """Build graph with A -> B -> A (both edges in a 2-cycle)."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "A")])
    return g


def _make_long_feedback_graph() -> nx.DiGraph:
    """Build long feedback loop A -> C -> D -> A (removing A->C leaves A in same SCC via D)."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "C"), ("C", "D"), ("D", "A")])
    return g


def _make_scc_dissolved_graph() -> nx.DiGraph:
    """Build graph where removing A->B dissolves the SCC.

    A -> B is the only link making A and B in the same SCC
    (B -> A is NOT present; instead B -> C -> A).
    """
    g = nx.DiGraph()
    # A -> B -> C -> A  (a 3-cycle; removing A->B dissolves the SCC)
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    return g


def _make_self_loop_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edge("X", "X")
    return g


def _make_cross_scc_graph() -> nx.DiGraph:
    """E -> F: no cycle, no shared SCC, so classify_nonident_cause returns cross_scc_blocked."""
    g = nx.DiGraph()
    g.add_edges_from([("E", "F"), ("F", "G")])
    return g


# ---------------------------------------------------------------------------
# Tests: classify_nonident_cause
# ---------------------------------------------------------------------------


class TestClassifyNonidentCause:
    def test_self_loop(self):
        g = _make_self_loop_graph()
        result = classify_nonident_cause(g, "X", "X")
        assert result == "self_loop"

    def test_two_cycle_forward(self):
        g = _make_two_cycle_graph()
        result = classify_nonident_cause(g, "A", "B")
        assert result == "two_cycle"

    def test_two_cycle_backward(self):
        g = _make_two_cycle_graph()
        result = classify_nonident_cause(g, "B", "A")
        assert result == "two_cycle"

    def test_same_scc_long(self):
        # Need: cause->effect is in the same SCC, no direct reverse edge,
        # and removing cause->effect still leaves them in the same SCC.
        # Graph: A->B->C->D->A  (4-cycle) plus A->C (the edge under test).
        # There is no direct C->A edge, so it's not a 2-cycle.
        # Removing A->C: A->B->C->D->A still cycles, so A and C remain in same SCC.
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")])
        result = classify_nonident_cause(g, "A", "C")
        assert result == "same_scc_long"

    def test_scc_edge_dissolved(self):
        # A->B->C->A: the 3-cycle. Removing A->B breaks it.
        g = _make_scc_dissolved_graph()
        result = classify_nonident_cause(g, "A", "B")
        assert result == "scc_edge_dissolved"

    def test_cross_scc_blocked(self):
        g = _make_cross_scc_graph()
        result = classify_nonident_cause(g, "E", "F")
        assert result == "cross_scc_blocked"

    def test_result_is_valid_category(self):
        """All returned values are in the documented category set."""
        graphs_edges = [
            (_make_self_loop_graph(), "X", "X"),
            (_make_two_cycle_graph(), "A", "B"),
            (_make_scc_dissolved_graph(), "A", "B"),
            (_make_cross_scc_graph(), "E", "F"),
        ]
        for g, c, e in graphs_edges:
            assert classify_nonident_cause(g, c, e) in CAUSE_CATEGORIES

    def test_precondition_missing_edge(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        with pytest.raises(AssertionError, match="PRE"):
            classify_nonident_cause(g, "B", "C")


# ---------------------------------------------------------------------------
# Tests: compute_rescue_nodes (tiny 2-cycle graph)
# ---------------------------------------------------------------------------


class TestComputeRescueNodes:
    def test_rescue_two_cycle(self):
        """
        Graph: A -> B -> A (2-cycle).
        do(A) removes the in-edge B->A, leaving only A->B (identifiable).
        do(B) removes the in-edge A->B — the edge itself is gone, not a rescue.
        So the rescue node for edge A->B should be A (removing B->A unblocks it).
        """
        g = _make_two_cycle_graph()
        # Pass candidates explicitly so we don't depend on min-cut pool for this tiny graph
        rescue = compute_rescue_nodes(g, "A", "B", candidates={"A", "B"})
        # do(A) removes in-edge B->A → A->B should become identifiable
        # (in the graph with only A->B, there is no confounder, so it's identifiable)
        assert "A" in rescue

    def test_rescue_preserves_edge(self):
        """Interventions that remove the target edge itself are NOT counted as rescues."""
        g = _make_two_cycle_graph()
        rescue = compute_rescue_nodes(g, "A", "B", candidates={"A", "B"})
        # do(B) removes in-edge A->B — the edge A->B is gone, so it can't be in rescue list
        assert "B" not in rescue

    def test_rescue_empty_candidates(self):
        """With an empty candidate set, no rescues are possible."""
        g = _make_two_cycle_graph()
        rescue = compute_rescue_nodes(g, "A", "B", candidates=set())
        assert rescue == []

    def test_rescue_precondition_missing_edge(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        with pytest.raises(AssertionError, match="PRE"):
            compute_rescue_nodes(g, "B", "C")
