"""
tests/test_residual_scc_axiomander.py
======================================
Axiomander-adorned tests for ``residual_cluster_size_distribution`` and
cross-checks on ``residual_scc_analysis``.

Contract coverage
-----------------
- PRE violations → ``AssertionError`` via ``pytest.raises``.
- POST invariants verified on representative scenarios.
- Key structural properties tested on synthetic minimal graphs.
"""

from __future__ import annotations

import networkx as nx
import pytest

from nocap.scc_perturb import (
    residual_cluster_size_distribution,
    residual_scc_analysis,
)

# ---------------------------------------------------------------------------
# Helpers — tiny synthetic graphs
# ---------------------------------------------------------------------------


def _linear_dag() -> nx.DiGraph:
    """T -> a -> b -> c   (no cycles)."""
    g = nx.DiGraph()
    g.add_edges_from([("t", "a"), ("a", "b"), ("b", "c")])
    return g


def _one_residual_cluster() -> nx.DiGraph:
    """
    T -> a, t -> b, t -> c
    a <-> b  (cycle between two children; c is acyclic)
    No return path to t.
    """
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("t", "a"),
            ("t", "b"),
            ("t", "c"),
            ("a", "b"),
            ("b", "a"),  # a<->b residual cycle
        ]
    )
    return g


def _two_residual_clusters() -> nx.DiGraph:
    """
    T -> a, t -> b, t -> c, t -> d
    a <-> b  (cluster 1)
    c <-> d  (cluster 2)
    """
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("t", "a"),
            ("t", "b"),
            ("t", "c"),
            ("t", "d"),
            ("a", "b"),
            ("b", "a"),
            ("c", "d"),
            ("d", "c"),
        ]
    )
    return g


def _single_child_graph() -> nx.DiGraph:
    """T -> a only."""
    g = nx.DiGraph()
    g.add_edges_from([("t", "a")])
    return g


# ---------------------------------------------------------------------------
# residual_scc_analysis — structural invariants (POST verification)
# ---------------------------------------------------------------------------


class TestResidualSccAnalysisPostInvariants:
    def test_all_children_classified_linear(self):
        g = _linear_dag()
        children = ["a", "b", "c"]
        r = residual_scc_analysis("t", children, [], g)
        assert len(r["children_cyclic"]) + len(r["children_acyclic"]) == len(children)

    def test_no_residual_clusters_in_linear_dag(self):
        g = _linear_dag()
        children = ["a", "b", "c"]
        r = residual_scc_analysis("t", children, [], g)
        assert r["residual_clusters"] == []

    def test_one_residual_cluster_detected(self):
        g = _one_residual_cluster()
        children = ["a", "b", "c"]
        r = residual_scc_analysis("t", children, [], g)
        assert len(r["residual_clusters"]) == 1
        cluster = r["residual_clusters"][0]
        assert cluster == frozenset({"a", "b"})

    def test_acyclic_child_not_in_cluster(self):
        g = _one_residual_cluster()
        children = ["a", "b", "c"]
        r = residual_scc_analysis("t", children, [], g)
        assert "c" in r["children_acyclic"]

    def test_two_residual_clusters_detected(self):
        g = _two_residual_clusters()
        children = ["a", "b", "c", "d"]
        r = residual_scc_analysis("t", children, [], g)
        assert len(r["residual_clusters"]) == 2

    def test_cut_verified_is_bool(self):
        g = _linear_dag()
        r = residual_scc_analysis("t", ["a"], [], g)
        assert isinstance(r["cut_verified"], bool)

    def test_tf_still_cyclic_is_bool(self):
        g = _linear_dag()
        r = residual_scc_analysis("t", ["a"], [], g)
        assert isinstance(r["tf_still_cyclic"], bool)

    def test_empty_children_no_clusters(self):
        g = _linear_dag()
        r = residual_scc_analysis("t", [], [], g)
        assert r["residual_clusters"] == []
        assert r["children_cyclic"] == []
        assert r["children_acyclic"] == []


# ---------------------------------------------------------------------------
# residual_cluster_size_distribution — PRE violations
# ---------------------------------------------------------------------------


class TestResidualClusterSizeDistributionPreViolations:
    def test_non_dict_raises(self):
        with pytest.raises(AssertionError, match="PRE"):
            residual_cluster_size_distribution("not a dict")  # type: ignore[arg-type]

    def test_missing_key_raises(self):
        with pytest.raises(AssertionError, match="PRE"):
            residual_cluster_size_distribution({"other_key": []})


# ---------------------------------------------------------------------------
# residual_cluster_size_distribution — POST invariants
# ---------------------------------------------------------------------------


class TestResidualClusterSizeDistributionPostInvariants:
    def _run(self, graph: nx.DiGraph, children: list, min_cut: list | None = None) -> dict:
        if min_cut is None:
            min_cut = []
        analysis = residual_scc_analysis("t", children, min_cut, graph)
        return residual_cluster_size_distribution(analysis)

    def test_no_clusters_has_residual_false(self):
        dist = self._run(_linear_dag(), ["a", "b", "c"])
        assert dist["has_residual_cluster"] is False
        assert dist["n_clusters"] == 0
        assert dist["sizes"] == []
        assert dist["max_size"] == 0
        assert dist["total_children_in_clusters"] == 0

    def test_one_cluster_correct_size(self):
        dist = self._run(_one_residual_cluster(), ["a", "b", "c"])
        assert dist["has_residual_cluster"] is True
        assert dist["n_clusters"] == 1
        assert dist["sizes"] == [2]  # cluster {a, b} has 2 children
        assert dist["max_size"] == 2
        assert dist["total_children_in_clusters"] == 2

    def test_two_clusters_correct_sizes(self):
        dist = self._run(_two_residual_clusters(), ["a", "b", "c", "d"])
        assert dist["has_residual_cluster"] is True
        assert dist["n_clusters"] == 2
        assert dist["sizes"] == [2, 2]
        assert dist["max_size"] == 2
        assert dist["total_children_in_clusters"] == 4

    def test_n_clusters_equals_len_sizes(self):
        for graph, children in [
            (_linear_dag(), ["a", "b", "c"]),
            (_one_residual_cluster(), ["a", "b", "c"]),
            (_two_residual_clusters(), ["a", "b", "c", "d"]),
        ]:
            dist = self._run(graph, children)
            assert dist["n_clusters"] == len(dist["sizes"]), (
                f"POST violated for children={children}"
            )

    def test_has_residual_cluster_matches_n_clusters(self):
        for graph, children in [
            (_linear_dag(), ["a", "b", "c"]),
            (_one_residual_cluster(), ["a", "b", "c"]),
        ]:
            dist = self._run(graph, children)
            assert dist["has_residual_cluster"] == (dist["n_clusters"] >= 1)

    def test_max_size_zero_when_no_clusters(self):
        dist = self._run(_linear_dag(), ["a", "b", "c"])
        assert dist["max_size"] == 0

    def test_max_size_ge_2_when_clusters_present(self):
        dist = self._run(_one_residual_cluster(), ["a", "b", "c"])
        assert dist["max_size"] >= 2

    def test_total_equals_sum_of_sizes(self):
        dist = self._run(_two_residual_clusters(), ["a", "b", "c", "d"])
        assert dist["total_children_in_clusters"] == sum(dist["sizes"])

    def test_single_child_no_cluster(self):
        """Single child can never be in a residual cluster (needs >=2)."""
        dist = self._run(_single_child_graph(), ["a"])
        assert dist["has_residual_cluster"] is False
        assert dist["n_clusters"] == 0

    def test_sizes_are_sorted(self):
        """Sizes list must be non-decreasing."""
        dist = self._run(_two_residual_clusters(), ["a", "b", "c", "d"])
        assert dist["sizes"] == sorted(dist["sizes"])
