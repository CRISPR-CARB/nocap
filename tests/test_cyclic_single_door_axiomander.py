"""Axiomander contract tests for src/nocap/cyclic_single_door.py.

House convention: adorned copies with negative pytest.raises tests.
Production scripts stay dependency-free; contracts live as plain assert
statements with PRE:/INV:/POST: message prefixes.

Each test group:
  - Positive: valid inputs satisfy postconditions.
  - Negative: PRE violations raise AssertionError with "PRE" in the message.
"""

from __future__ import annotations

import networkx as nx
import pytest

from nocap.cyclic_single_door import (
    classify_edge,
    evaluate_all_edges,
    maximize_identifiable_edges,
    nx_digraph_to_y0,
    same_scc,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _chain() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from([("Z", "X"), ("X", "Y")])
    return g


def _cycle() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from([("X", "Y"), ("Y", "X")])
    return g


def _three_cycle() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("D", "A")])
    return g


# ---------------------------------------------------------------------------
# nx_digraph_to_y0 contracts
# ---------------------------------------------------------------------------


class TestNxDigraphToY0Contracts:
    """POST: node/edge counts preserved; PRE: must be nx.DiGraph."""

    def test_post_node_count(self):
        """POST: result.directed.number_of_nodes() == graph.number_of_nodes()."""
        g = _three_cycle()
        result = nx_digraph_to_y0(g)
        assert result.directed.number_of_nodes() == g.number_of_nodes()

    def test_post_edge_count(self):
        """POST: result.directed.number_of_edges() == graph.number_of_edges()."""
        g = _three_cycle()
        result = nx_digraph_to_y0(g)
        assert result.directed.number_of_edges() == g.number_of_edges()

    def test_pre_not_digraph(self):
        """PRE: graph must be an nx.DiGraph — raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE"):
            nx_digraph_to_y0("not_a_graph")  # type: ignore[arg-type]

    def test_pre_undirected_graph(self):
        """PRE: nx.Graph (undirected) is rejected."""
        g = nx.Graph()
        g.add_edge("X", "Y")
        with pytest.raises(AssertionError, match="PRE"):
            nx_digraph_to_y0(g)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# same_scc contracts
# ---------------------------------------------------------------------------


class TestSameSccContracts:
    """POST: result is bool; PRE: graph must be nx.DiGraph."""

    def test_post_returns_bool_true(self):
        """POST: isinstance(result, bool) — True case."""
        g = _cycle()
        result = same_scc(g, "X", "Y")
        assert isinstance(result, bool)
        assert result is True

    def test_post_returns_bool_false(self):
        """POST: isinstance(result, bool) — False case."""
        g = _chain()
        result = same_scc(g, "Z", "Y")
        assert isinstance(result, bool)
        assert result is False

    def test_pre_not_digraph(self):
        """PRE: graph must be an nx.DiGraph."""
        with pytest.raises(AssertionError, match="PRE"):
            same_scc(None, "X", "Y")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# classify_edge contracts
# ---------------------------------------------------------------------------


class TestClassifyEdgeContracts:
    """POST: status in valid set; same_scc is bool; adjustment_set iff identifiable."""

    def test_post_status_identifiable(self):
        """POST: status == 'identifiable' and adjustment_set is not None."""
        g = _chain()
        result = classify_edge(g, "X", "Y")
        assert result["status"] == "identifiable"
        assert result["adjustment_set"] is not None
        assert isinstance(result["adjustment_set"], frozenset)

    def test_post_status_unidentifiable(self):
        """POST: status == 'unidentifiable' and adjustment_set is None."""
        g = _cycle()
        result = classify_edge(g, "X", "Y")
        assert result["status"] == "unidentifiable"
        assert result["adjustment_set"] is None

    def test_post_same_scc_is_bool(self):
        """POST: isinstance(result['same_scc'], bool)."""
        g = _chain()
        result = classify_edge(g, "X", "Y")
        assert isinstance(result["same_scc"], bool)

    def test_post_implies_identifiable_adj_not_none(self):
        """POST: implies(status == 'identifiable', adjustment_set is not None)."""
        g = _three_cycle()
        result = classify_edge(g, "D", "A")
        if result["status"] == "identifiable":
            assert result["adjustment_set"] is not None
        else:
            assert result["adjustment_set"] is None

    def test_pre_not_digraph(self):
        """PRE: graph must be an nx.DiGraph."""
        with pytest.raises(AssertionError, match="PRE"):
            classify_edge("not_a_graph", "X", "Y")  # type: ignore[arg-type]

    def test_pre_cause_not_str(self):
        """PRE: cause must be a str."""
        g = _chain()
        with pytest.raises(AssertionError, match="PRE"):
            classify_edge(g, 123, "Y")  # type: ignore[arg-type]

    def test_pre_effect_not_str(self):
        """PRE: effect must be a str."""
        g = _chain()
        with pytest.raises(AssertionError, match="PRE"):
            classify_edge(g, "X", 456)  # type: ignore[arg-type]

    def test_pre_edge_missing(self):
        """PRE: edge must exist in graph."""
        g = _chain()
        with pytest.raises(AssertionError, match="PRE"):
            classify_edge(g, "Y", "Z")  # Y->Z does not exist

    def test_pre_reversed_edge_missing(self):
        """PRE: reversed edge also rejected if not in graph."""
        g = _chain()
        with pytest.raises(AssertionError, match="PRE"):
            classify_edge(g, "Y", "X")  # Y->X does not exist in chain


# ---------------------------------------------------------------------------
# evaluate_all_edges contracts
# ---------------------------------------------------------------------------


class TestEvaluateAllEdgesContracts:
    """POST: len(result) == n_edges; all statuses valid; PRE: must be DiGraph."""

    def test_post_length_all_edges(self):
        """POST: len(result) == graph.number_of_edges() when no restrict_edges."""
        g = _three_cycle()
        results = evaluate_all_edges(g)
        assert len(results) == g.number_of_edges()

    def test_post_length_restricted(self):
        """POST: len(result) == len(restrict_edges) when restrict_edges given."""
        g = _three_cycle()
        subset = [("D", "A"), ("A", "B")]
        results = evaluate_all_edges(g, restrict_edges=subset)
        assert len(results) == 2

    def test_post_all_statuses_valid(self):
        """POST: all(r['status'] in ('identifiable', 'unidentifiable') for r in result)."""
        g = _three_cycle()
        results = evaluate_all_edges(g)
        for r in results:
            assert r["status"] in ("identifiable", "unidentifiable")

    def test_post_empty_graph(self):
        """POST: empty graph returns empty list."""
        g = nx.DiGraph()
        g.add_nodes_from(["X", "Y"])
        results = evaluate_all_edges(g)
        assert results == []

    def test_pre_not_digraph(self):
        """PRE: graph must be an nx.DiGraph."""
        with pytest.raises(AssertionError, match="PRE"):
            evaluate_all_edges(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# maximize_identifiable_edges contracts
# ---------------------------------------------------------------------------


class TestMaximizeIdentifiableEdgesContracts:
    """POST: curve structure; chosen_nodes <= k; do-semantics on final_graph."""

    def test_post_curve_length(self):
        """POST: len(curve) == len(chosen_nodes) + 1."""
        g = _three_cycle()
        result = maximize_identifiable_edges(g, k=5)
        assert len(result["curve"]) == len(result["chosen_nodes"]) + 1

    def test_post_curve_starts_at_zero(self):
        """POST: curve[0][0] == 0."""
        g = _three_cycle()
        result = maximize_identifiable_edges(g, k=5)
        assert result["curve"][0][0] == 0

    def test_post_baseline_matches_curve(self):
        """POST: n_identifiable_baseline == curve[0][1]."""
        g = _three_cycle()
        result = maximize_identifiable_edges(g, k=5)
        assert result["n_identifiable_baseline"] == result["curve"][0][1]

    def test_post_final_matches_curve(self):
        """POST: n_identifiable_final == curve[-1][1]."""
        g = _three_cycle()
        result = maximize_identifiable_edges(g, k=5)
        assert result["n_identifiable_final"] == result["curve"][-1][1]

    def test_post_chosen_nodes_within_budget(self):
        """POST: len(chosen_nodes) <= k."""
        g = _three_cycle()
        for k in [0, 1, 3, 10]:
            result = maximize_identifiable_edges(g, k=k)
            assert len(result["chosen_nodes"]) <= k

    def test_post_do_semantics_in_degree_zero(self):
        """POST: intervened nodes have in_degree 0 in final_graph (do-semantics)."""
        g = _three_cycle()
        result = maximize_identifiable_edges(g, k=5)
        final_graph = result["final_graph"]
        for node in result["chosen_nodes"]:
            assert final_graph.in_degree(node) == 0, (
                f"POST: do-semantics violated for {node!r}: in_degree != 0"
            )

    def test_post_do_semantics_out_edges_preserved(self):
        """POST: intervened nodes preserve out-edges (not vertex deletion)."""
        g = _three_cycle()
        result = maximize_identifiable_edges(g, k=5)
        final_graph = result["final_graph"]
        for node in result["chosen_nodes"]:
            assert final_graph.out_degree(node) == g.out_degree(node), (
                f"POST: out-edges of {node!r} must be preserved after do()"
            )

    def test_post_node_still_present(self):
        """POST: intervened nodes are still present in final_graph."""
        g = _three_cycle()
        result = maximize_identifiable_edges(g, k=5)
        final_graph = result["final_graph"]
        for node in result["chosen_nodes"]:
            assert node in final_graph.nodes()

    def test_pre_negative_k(self):
        """PRE: k must be a non-negative int."""
        g = _chain()
        with pytest.raises(AssertionError, match="PRE"):
            maximize_identifiable_edges(g, k=-1)

    def test_pre_not_digraph(self):
        """PRE: graph must be an nx.DiGraph."""
        with pytest.raises(AssertionError, match="PRE"):
            maximize_identifiable_edges("not_a_graph", k=5)  # type: ignore[arg-type]

    def test_pre_k_not_int(self):
        """PRE: k must be an int."""
        g = _chain()
        with pytest.raises((AssertionError, TypeError)):
            maximize_identifiable_edges(g, k=1.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _split_into_n_shards contracts
# ---------------------------------------------------------------------------

import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "scripts"))
from cyclic_single_door_classify import _split_into_n_shards  # noqa: E402


class TestSplitIntoNShardsContracts:
    """POST: exact shard count; full edge coverage; PRE violations raise AssertionError."""

    def _edges(self, n: int) -> list[tuple[str, str]]:
        return [(str(i), str(i + 1)) for i in range(n)]

    # --- POST: exact shard count ---

    def test_post_exact_shard_count(self):
        """POST: len(result) == n_shards when n_edges >= n_shards."""
        edges = self._edges(100)
        chunks = _split_into_n_shards(edges, 10)
        assert len(chunks) == 10, "POST: correct shard count"

    def test_post_shard_count_capped_at_edge_count(self):
        """POST: len(result) == n_edges when n_shards > n_edges (no empty shards)."""
        edges = self._edges(5)
        chunks = _split_into_n_shards(edges, 20)
        assert len(chunks) == 5, "POST: capped at edge count"

    def test_post_single_shard(self):
        """POST: n_shards == 1 yields one chunk containing all edges."""
        edges = self._edges(50)
        chunks = _split_into_n_shards(edges, 1)
        assert len(chunks) == 1
        assert len(chunks[0]) == 50

    def test_post_n_shards_equals_n_edges(self):
        """POST: n_shards == n_edges yields one edge per shard."""
        edges = self._edges(8)
        chunks = _split_into_n_shards(edges, 8)
        assert len(chunks) == 8
        assert all(len(c) == 1 for c in chunks)

    # --- POST: full edge coverage ---

    def test_post_all_edges_covered(self):
        """POST: union of all shards == original edges (no drops)."""
        edges = self._edges(100)
        chunks = _split_into_n_shards(edges, 7)
        flat = [e for chunk in chunks for e in chunk]
        assert flat == edges, "POST: no edges dropped or reordered"

    def test_post_total_edge_count_preserved(self):
        """POST: sum(len(c) for c in result) == len(edges)."""
        for n, s in [(1, 1), (10, 3), (100, 179), (9501, 179)]:
            edges = self._edges(n)
            chunks = _split_into_n_shards(edges, s)
            assert sum(len(c) for c in chunks) == n, (
                f"POST: total edges preserved for n={n}, s={s}"
            )

    def test_post_no_empty_shards(self):
        """POST: all shards are non-empty."""
        edges = self._edges(50)
        chunks = _split_into_n_shards(edges, 179)
        assert all(len(c) > 0 for c in chunks), "POST: no empty shards emitted"

    def test_post_near_equal_sizes(self):
        """POST: max chunk size - min chunk size <= 1 (near-equal division)."""
        edges = self._edges(100)
        chunks = _split_into_n_shards(edges, 7)
        sizes = [len(c) for c in chunks]
        assert max(sizes) - min(sizes) <= 1, "POST: near-equal chunk sizes"

    def test_post_empty_input(self):
        """POST: empty edge list yields no shards."""
        chunks = _split_into_n_shards([], 10)
        assert chunks == [], "POST: empty input yields empty result"

    # --- PRE violations ---

    def test_pre_n_shards_zero(self):
        """PRE: n_shards must be >= 1."""
        with pytest.raises(AssertionError, match="PRE"):
            _split_into_n_shards(self._edges(5), 0)

    def test_pre_n_shards_negative(self):
        """PRE: n_shards must be >= 1."""
        with pytest.raises(AssertionError, match="PRE"):
            _split_into_n_shards(self._edges(5), -3)
