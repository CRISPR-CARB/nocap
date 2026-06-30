"""tests/test_scc_recovery_bank.py — Unit tests for scc_recovery_bank.py.

Tests cover:
  - _enumerate_tfs: correct out-degree >= 1 filter, sorted
  - _classify_tfs: singleton vs non-trivial SCC split, total count
  - _build_do_scc_info: SCC map + sizes, do() breaks cycles
  - _is_singleton_scc: singleton vs cyclic distinction
  - _build_candidate_pool: uses compute_min_cut_b, returns set
  - _greedy_bank: n/k postcondition, monotone coverage, genes in pool,
                  n=0/k=0 preconditions, early stopping
  - _score_per_tf: recovered flag, recovered_by_set, count matches bank
  - Integration: small cyclic graph, break gene recovers TF
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

# Add scripts/ to sys.path so we can import scc_recovery_bank directly
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import scc_recovery_bank as srb  # noqa: E402

# ---------------------------------------------------------------------------
# Tiny test graphs
# ---------------------------------------------------------------------------


def _chain_graph() -> nx.DiGraph:
    """Build A -> B -> C (pure DAG, no cycles). All are singleton SCCs."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C")])
    return G


def _triangle_graph() -> nx.DiGraph:
    """Build A -> B -> C -> A (one 3-cycle). Plus D -> E outside."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("D", "E")])
    return G


def _two_cycle_graph() -> nx.DiGraph:
    """Build A <-> B (2-cycle). Plus C -> D (acyclic)."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "A"), ("C", "D")])
    return G


def _larger_cyclic_graph() -> nx.DiGraph:
    """
    SCC1 = {A, B, C}: A->B->C->A.
    SCC2 = {X, Y}:    X->Y->X.
    Plus E->A (E is outside SCCs, non-trivial SCC node A has an out-degree).
    DAG TF: D->E (D is acyclic).
    """
    G = nx.DiGraph()
    # SCC1
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    # SCC2
    G.add_edges_from([("X", "Y"), ("Y", "X")])
    # extra connections
    G.add_edges_from([("D", "E"), ("E", "A")])
    return G


# ---------------------------------------------------------------------------
# Tests: _enumerate_tfs
# ---------------------------------------------------------------------------


class TestEnumerateTfs:
    def test_chain_graph(self):
        G = _chain_graph()
        tfs = srb._enumerate_tfs(G)
        # A->B, B->C; A and B have out-degree>=1; C has out-degree=0
        assert "A" in tfs
        assert "B" in tfs
        assert "C" not in tfs

    def test_sorted(self):
        G = _triangle_graph()
        tfs = srb._enumerate_tfs(G)
        assert tfs == sorted(tfs)

    def test_all_have_out_degree_ge1(self):
        G = _larger_cyclic_graph()
        tfs = srb._enumerate_tfs(G)
        for t in tfs:
            assert G.out_degree(t) >= 1, f"{t} should have out-degree >= 1"

    def test_no_sink_nodes(self):
        G = _chain_graph()
        tfs = srb._enumerate_tfs(G)
        # C is a sink; must not appear
        assert "C" not in tfs

    def test_empty_graph(self):
        G = nx.DiGraph()
        assert srb._enumerate_tfs(G) == []


# ---------------------------------------------------------------------------
# Tests: _classify_tfs
# ---------------------------------------------------------------------------


class TestClassifyTfs:
    def test_chain_all_identifiable(self):
        G = _chain_graph()
        tfs = srb._enumerate_tfs(G)
        id_, unid = srb._classify_tfs(G, tfs)
        # No cycles => all TFs identifiable
        assert len(unid) == 0
        assert set(id_) == set(tfs)

    def test_triangle_tfs_unidentifiable(self):
        G = _triangle_graph()
        tfs = srb._enumerate_tfs(G)
        id_, unid = srb._classify_tfs(G, tfs)
        # A,B,C are in the 3-cycle; D is outside (D->E, singleton SCC)
        assert "A" in unid
        assert "B" in unid
        assert "C" in unid
        assert "D" in id_

    def test_count_partition(self):
        G = _larger_cyclic_graph()
        tfs = srb._enumerate_tfs(G)
        id_, unid = srb._classify_tfs(G, tfs)
        assert len(id_) + len(unid) == len(tfs)

    def test_two_cycle_both_unidentifiable(self):
        G = _two_cycle_graph()
        tfs = srb._enumerate_tfs(G)
        id_, unid = srb._classify_tfs(G, tfs)
        # A and B are in a 2-cycle; C has out-degree>=1 (C->D) and is singleton
        assert "A" in unid
        assert "B" in unid
        assert "C" in id_


# ---------------------------------------------------------------------------
# Tests: _build_do_scc_info
# ---------------------------------------------------------------------------


class TestBuildDoSccInfo:
    def test_empty_perturbation_triangle(self):
        G = _triangle_graph()
        scc_map, scc_sizes = srb._build_do_scc_info(G, frozenset())
        # A, B, C all in same SCC
        assert scc_map["A"] == scc_map["B"] == scc_map["C"]
        assert scc_sizes[scc_map["A"]] == 3

    def test_breaking_triangle_by_perturbing_C(self):
        G = _triangle_graph()
        # Perturb C: remove B->C. Cycle A->B->C->A is broken.
        scc_map, scc_sizes = srb._build_do_scc_info(G, frozenset(["C"]))
        # Now A, B, C should each be singletons (no more cycle)
        assert scc_sizes[scc_map["A"]] == 1
        assert scc_sizes[scc_map["B"]] == 1
        assert scc_sizes[scc_map["C"]] == 1

    def test_all_nodes_covered(self):
        G = _triangle_graph()
        scc_map, scc_sizes = srb._build_do_scc_info(G, frozenset())
        for node in G.nodes():
            assert node in scc_map

    def test_requires_frozenset(self):
        G = _triangle_graph()
        with pytest.raises(AssertionError, match="PRE"):
            srb._build_do_scc_info(G, {"A"})

    def test_two_cycle_broken_by_one_gene(self):
        G = _two_cycle_graph()
        # Perturb B: remove A->B. Only B->A remains.
        scc_map, scc_sizes = srb._build_do_scc_info(G, frozenset(["B"]))
        # A and B must be in different (singleton) SCCs
        assert scc_sizes[scc_map["A"]] == 1
        assert scc_sizes[scc_map["B"]] == 1
        assert scc_map["A"] != scc_map["B"]


# ---------------------------------------------------------------------------
# Tests: _is_singleton_scc
# ---------------------------------------------------------------------------


class TestIsSingletonScc:
    def test_singleton_returns_true(self):
        # scc_id 0 has size 1
        scc_map = {"A": 0, "B": 1}
        scc_sizes = {0: 1, 1: 2}
        assert srb._is_singleton_scc("A", scc_map, scc_sizes) is True

    def test_non_trivial_returns_false(self):
        scc_map = {"A": 0, "B": 0}
        scc_sizes = {0: 2}
        assert srb._is_singleton_scc("A", scc_map, scc_sizes) is False

    def test_absent_node_returns_true(self):
        # Node not in graph is trivially isolated
        assert srb._is_singleton_scc("MISSING", {}, {}) is True


# ---------------------------------------------------------------------------
# Tests: _build_candidate_pool
# ---------------------------------------------------------------------------


class TestBuildCandidatePool:
    def test_triangle_pool_non_empty(self):
        """Triangle A->B->C->A: each TF's B(t) is the intermediate node."""
        G = _triangle_graph()
        _, unid = srb._classify_tfs(G, srb._enumerate_tfs(G))
        pool = srb._build_candidate_pool(G, unid)
        # Must be a set
        assert isinstance(pool, set)
        # For a 3-cycle, B(A) cuts return paths from B,C to A.
        # B(t) should be non-empty (C severs C->A for A, etc.)
        assert len(pool) >= 1

    def test_chain_pool_empty(self):
        """Chain has no non-trivial SCCs -> no unidentifiable TFs -> empty pool."""
        G = _chain_graph()
        _, unid = srb._classify_tfs(G, srb._enumerate_tfs(G))
        pool = srb._build_candidate_pool(G, unid)
        assert pool == set()

    def test_returns_set(self):
        G = _triangle_graph()
        _, unid = srb._classify_tfs(G, srb._enumerate_tfs(G))
        pool = srb._build_candidate_pool(G, unid)
        assert isinstance(pool, set)


# ---------------------------------------------------------------------------
# Tests: _greedy_bank
# ---------------------------------------------------------------------------


class TestGreedyBank:
    def _simple_setup(self):
        """Triangle graph with A,B,C unidentifiable. Pool = {A,B,C}."""
        G = _triangle_graph()
        _, unid = srb._classify_tfs(G, srb._enumerate_tfs(G))
        pool = srb._build_candidate_pool(G, unid)
        return G, unid, pool

    def test_returns_n_sets(self):
        G, unid, pool = self._simple_setup()
        bank = srb._greedy_bank(G, unid, pool, n=3, k=2, verbose=False)
        assert len(bank) == 3

    def test_set_index_1based(self):
        G, unid, pool = self._simple_setup()
        bank = srb._greedy_bank(G, unid, pool, n=2, k=1, verbose=False)
        assert bank[0]["set_index"] == 1
        assert bank[1]["set_index"] == 2

    def test_coverage_monotone(self):
        G, unid, pool = self._simple_setup()
        bank = srb._greedy_bank(G, unid, pool, n=5, k=2, verbose=False)
        cumulative = [item["proxy_covered_cumulative"] for item in bank]
        for i in range(len(cumulative) - 1):
            assert cumulative[i + 1] >= cumulative[i], "Coverage must be non-decreasing"

    def test_genes_in_pool(self):
        G, unid, pool = self._simple_setup()
        bank = srb._greedy_bank(G, unid, pool, n=2, k=2, verbose=False)
        for item in bank:
            for gene in item["genes"]:
                assert gene in pool, f"Chosen gene {gene!r} not in pool"

    def test_empty_pool_returns_empty_genes(self):
        G = _triangle_graph()
        _, unid = srb._classify_tfs(G, srb._enumerate_tfs(G))
        bank = srb._greedy_bank(G, unid, set(), n=2, k=2, verbose=False)
        assert len(bank) == 2
        for item in bank:
            assert item["genes"] == []

    def test_precondition_n_zero(self):
        G, unid, pool = self._simple_setup()
        with pytest.raises(AssertionError, match="PRE"):
            srb._greedy_bank(G, unid, pool, n=0, k=1, verbose=False)

    def test_precondition_k_zero(self):
        G, unid, pool = self._simple_setup()
        with pytest.raises(AssertionError, match="PRE"):
            srb._greedy_bank(G, unid, pool, n=1, k=0, verbose=False)

    def test_recovery_achieves_positive_count(self):
        """At least one TF should be recovered from the triangle."""
        G, unid, pool = self._simple_setup()
        bank = srb._greedy_bank(G, unid, pool, n=2, k=2, verbose=False)
        total = bank[-1]["proxy_covered_cumulative"]
        assert total >= 1, "At least one TF should be recoverable"

    def test_early_stop_pads_correctly(self):
        """If all targets covered before n sets, remaining sets have empty genes."""
        G, unid, pool = self._simple_setup()
        # Use large k so we can cover all in the first set
        bank = srb._greedy_bank(G, unid, pool, n=5, k=len(pool) + 2, verbose=False)
        assert len(bank) == 5
        # If all covered by set 1, sets 2-5 should have proxy_recovered_new == 0
        if bank[0]["proxy_covered_cumulative"] == len(unid):
            for item in bank[1:]:
                assert item["proxy_recovered_new"] == 0


# ---------------------------------------------------------------------------
# Tests: _score_per_tf
# ---------------------------------------------------------------------------


class TestScorePerTf:
    def test_recovered_count_matches_bank(self):
        G = _triangle_graph()
        _, unid = srb._classify_tfs(G, srb._enumerate_tfs(G))
        pool = srb._build_candidate_pool(G, unid)
        bank = srb._greedy_bank(G, unid, pool, n=3, k=2, verbose=False)
        tf_results = srb._score_per_tf(unid, bank, G)
        n_recovered = sum(1 for e in tf_results if e["recovered"])
        assert n_recovered == bank[-1]["proxy_covered_cumulative"]

    def test_recovered_by_set_valid(self):
        G = _triangle_graph()
        _, unid = srb._classify_tfs(G, srb._enumerate_tfs(G))
        pool = srb._build_candidate_pool(G, unid)
        bank = srb._greedy_bank(G, unid, pool, n=3, k=2, verbose=False)
        tf_results = srb._score_per_tf(unid, bank, G)
        valid_set_indices = {item["set_index"] for item in bank}
        for e in tf_results:
            if e["recovered"]:
                assert e["recovered_by_set"] in valid_set_indices

    def test_scc_size_reported(self):
        G = _triangle_graph()
        _, unid = srb._classify_tfs(G, srb._enumerate_tfs(G))
        pool = srb._build_candidate_pool(G, unid)
        bank = srb._greedy_bank(G, unid, pool, n=1, k=2, verbose=False)
        tf_results = srb._score_per_tf(unid, bank, G)
        for e in tf_results:
            # A,B,C are in a 3-cycle -> scc_size should be 3
            assert e["scc_size"] == 3

    def test_precondition_empty_bank(self):
        G = _triangle_graph()
        _, unid = srb._classify_tfs(G, srb._enumerate_tfs(G))
        with pytest.raises(AssertionError, match="PRE"):
            srb._score_per_tf(unid, [], G)

    def test_precondition_empty_targets(self):
        G = _triangle_graph()
        bank = [
            {
                "set_index": 1,
                "genes": ["C"],
                "proxy_recovered_new": 1,
                "proxy_covered_cumulative": 1,
            }
        ]
        with pytest.raises(AssertionError, match="PRE"):
            srb._score_per_tf([], bank, G)


# ---------------------------------------------------------------------------
# Integration test: two-SCC graph
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_larger_cyclic_bank(self):
        """
        _larger_cyclic_graph has SCC1={A,B,C} and SCC2={X,Y}.
        Unidentifiable TFs: A, B, C (all in SCC1) + X, Y (in SCC2).
        Pool = B(t) genes for each. Should achieve some recovery.
        """
        G = _larger_cyclic_graph()
        tfs = srb._enumerate_tfs(G)
        id_, unid = srb._classify_tfs(G, tfs)
        assert "D" in id_  # D->E, singleton
        assert "A" in unid
        assert "X" in unid

        pool = srb._build_candidate_pool(G, unid)
        assert isinstance(pool, set)

        bank = srb._greedy_bank(G, unid, pool, n=3, k=3, verbose=False)
        assert len(bank) == 3
        # Coverage is monotone
        cumulative = [item["proxy_covered_cumulative"] for item in bank]
        for i in range(len(cumulative) - 1):
            assert cumulative[i + 1] >= cumulative[i]

    def test_break_recovers_tf_in_cycle(self):
        """
        In the triangle A->B->C->A, perturbing C removes C's in-edges (B->C).
        After do({C}): edges remaining are A->B, C->A, D->E.
        No cycle left => A, B, C all become singletons => recovered.
        """
        G = _triangle_graph()
        scc_map, scc_sizes = srb._build_do_scc_info(G, frozenset(["C"]))
        assert srb._is_singleton_scc("A", scc_map, scc_sizes) is True
        assert srb._is_singleton_scc("B", scc_map, scc_sizes) is True
        assert srb._is_singleton_scc("C", scc_map, scc_sizes) is True

    def test_no_perturbation_leaves_tfs_cyclic(self):
        G = _triangle_graph()
        scc_map, scc_sizes = srb._build_do_scc_info(G, frozenset())
        # A, B, C are still in the same non-trivial SCC
        assert srb._is_singleton_scc("A", scc_map, scc_sizes) is False
        assert srb._is_singleton_scc("B", scc_map, scc_sizes) is False
        assert srb._is_singleton_scc("C", scc_map, scc_sizes) is False
