"""tests/test_csd_break.py — Unit tests for min_scc_break_set.

Two canonical cases:

Case 1 — No intervention needed (break_size == 0):
  Graph: A→B, B→C, C→A, A→D
  Edge under study: A→D
  G' = G − {A→D}: D has no path back to A (D is a leaf), so A & D are in
  different SCCs.
  Expected: needs_intervention=False, break_set=[], break_size=0, cut_verified=True

Case 2 — Intervention needed, min-cut is exactly size 2:
  Graph: A→B, B→C, C→D, A→D, D→E, E→A, D→F, F→A
  Edge under study: A→D
  G' = G − {A→D}: forward path A→B→C→D still keeps A and D in the same SCC.
  Two node-disjoint return paths from D to A: D→E→A and D→F→A.
  Min vertex cut requires both E and F (one per path), so break_size == 2.
  Expected: needs_intervention=True, break_set=['E','F'], break_size==2,
            cause not in break_set, effect not in break_set,
            cut_verified=True
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

# Make nocap importable without an editable install
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "src"))

from nocap.scc_perturb import min_scc_break_set  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_dag() -> nx.DiGraph:
    """A→B, B→C, C→A, A→D  (A/B/C in SCC; D is a leaf)."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("A", "D")])
    return g


def _make_two_return_paths() -> nx.DiGraph:
    """A→B, B→C, C→D, A→D, D→E, E→A, D→F, F→A  (two disjoint return paths D→A).

    After removing the edge under study A→D:
      - Forward path A→B→C→D still exists, so A and D remain in the same SCC.
      - Two node-disjoint return paths: D→E→A and D→F→A.
      - Min vertex cut = {E, F} (one per return path), so break_size == 2.
    """
    g = nx.DiGraph()
    g.add_edges_from([
        ("A", "B"), ("B", "C"), ("C", "D"),  # forward path keeps A & D in same SCC
        ("A", "D"),                           # edge under study
        ("D", "E"), ("E", "A"),               # return path 1: D→E→A
        ("D", "F"), ("F", "A"),               # return path 2: D→F→A
    ])
    return g


# ---------------------------------------------------------------------------
# Case 1: no intervention needed
# ---------------------------------------------------------------------------


class TestNoInterventionNeeded:
    """Removing A→D splits cause(A) and effect(D) into different SCCs."""

    def test_needs_intervention_false(self):
        g = _make_simple_dag()
        result = min_scc_break_set("A", "D", g)
        assert result["needs_intervention"] is False

    def test_same_scc_after_removal_false(self):
        g = _make_simple_dag()
        result = min_scc_break_set("A", "D", g)
        assert result["same_scc_after_removal"] is False

    def test_break_set_empty(self):
        g = _make_simple_dag()
        result = min_scc_break_set("A", "D", g)
        assert result["break_set"] == []

    def test_break_size_zero(self):
        g = _make_simple_dag()
        result = min_scc_break_set("A", "D", g)
        assert result["break_size"] == 0

    def test_cut_verified_true(self):
        g = _make_simple_dag()
        result = min_scc_break_set("A", "D", g)
        assert result["cut_verified"] is True

    def test_cause_not_in_break_set(self):
        g = _make_simple_dag()
        result = min_scc_break_set("A", "D", g)
        assert "A" not in result["break_set"]

    def test_effect_not_in_break_set(self):
        g = _make_simple_dag()
        result = min_scc_break_set("A", "D", g)
        assert "D" not in result["break_set"]


# ---------------------------------------------------------------------------
# Case 2: intervention needed, min-cut is size 2
# ---------------------------------------------------------------------------


class TestSizeTwoCut:
    """Two return paths require a size-2 vertex cut."""

    def test_needs_intervention_true(self):
        g = _make_two_return_paths()
        result = min_scc_break_set("A", "D", g)
        assert result["needs_intervention"] is True

    def test_same_scc_after_removal_true(self):
        g = _make_two_return_paths()
        result = min_scc_break_set("A", "D", g)
        assert result["same_scc_after_removal"] is True

    def test_break_size_at_most_two(self):
        """Min cut is at most 2 (E and F cover both return paths)."""
        g = _make_two_return_paths()
        result = min_scc_break_set("A", "D", g)
        # E and F are the only intermediate nodes on the return paths;
        # min cut must be >= 1 and <= 2.
        assert 1 <= result["break_size"] <= 2

    def test_break_size_exactly_two(self):
        """Both E and F are required because the paths are node-disjoint."""
        g = _make_two_return_paths()
        result = min_scc_break_set("A", "D", g)
        assert result["break_size"] == 2

    def test_cause_not_in_break_set(self):
        g = _make_two_return_paths()
        result = min_scc_break_set("A", "D", g)
        assert "A" not in result["break_set"]

    def test_effect_not_in_break_set(self):
        g = _make_two_return_paths()
        result = min_scc_break_set("A", "D", g)
        assert "D" not in result["break_set"]

    def test_cut_verified(self):
        g = _make_two_return_paths()
        result = min_scc_break_set("A", "D", g)
        assert result["cut_verified"] is True

    def test_break_set_contains_e_and_f(self):
        """E and F are the only intermediate nodes; both must appear.

        Graph return paths:
          path 1: D→E→A  (only intermediate: E)
          path 2: D→F→A  (only intermediate: F)
        The min vertex cut must cut each path, so both E and F are in the set.
        """
        g = _make_two_return_paths()
        result = min_scc_break_set("A", "D", g)
        bs = set(result["break_set"])
        assert "E" in bs, f"E must be in break_set to cut return path 1 (D→E→A), got {bs}"
        assert "F" in bs, f"F must be in break_set to cut return path 2 (D→F→A), got {bs}"

    def test_break_set_sorted(self):
        g = _make_two_return_paths()
        result = min_scc_break_set("A", "D", g)
        assert result["break_set"] == sorted(result["break_set"])


# ---------------------------------------------------------------------------
# PRE-condition checks
# ---------------------------------------------------------------------------


class TestPreconditions:
    def test_missing_edge_raises(self):
        g = nx.DiGraph()
        g.add_edge("X", "Y")
        with pytest.raises(AssertionError, match="PRE"):
            min_scc_break_set("X", "Z", g)

    def test_non_string_cause_raises(self):
        g = nx.DiGraph()
        g.add_edge(1, 2)
        with pytest.raises(AssertionError, match="PRE"):
            min_scc_break_set(1, 2, g)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_repeated_call_same_result(self):
        g = _make_two_return_paths()
        r1 = min_scc_break_set("A", "D", g)
        r2 = min_scc_break_set("A", "D", g)
        assert r1 == r2

    def test_graph_not_mutated(self):
        g = _make_two_return_paths()
        edges_before = set(g.edges())
        min_scc_break_set("A", "D", g)
        assert set(g.edges()) == edges_before
