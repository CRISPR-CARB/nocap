"""tests/test_csd_recovery_bank.py — Unit tests for csd_recovery_bank.py.

Tests cover:
  - _proxy_recovered: same vs different SCC
  - _exact_recovered: edge-removal matters (proxy≠exact corner case)
  - _greedy_bank: n/k/len postcondition, monotone coverage
  - _build_do_scc_map: every node gets an SCC id; do() breaks cycles
  - _load_targets / _load_candidate_pool: CSV parsing edge cases
  - proxy≠exact gap: graph where removing the direct edge is the sole forward
    path in the SCC, so proxy says "different SCC" but exact confirms it too
    (proxy is sound — gap only goes the other way: proxy might miss some
    exact recoveries when the direct edge is the *sole* forward path keeping
    cause and effect in the same SCC in G, but not in G').
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

# Add scripts/ to sys.path so we can import csd_recovery_bank directly
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import csd_recovery_bank as crb  # noqa: E402

# ---------------------------------------------------------------------------
# Tiny test graphs
# ---------------------------------------------------------------------------


def _tiny_graph_triangle() -> nx.DiGraph:
    """Triangle A->B->C->A with extra D->E outside the cycle."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("D", "E")])
    return G


def _tiny_graph_two_cycles() -> nx.DiGraph:
    """Two cycles sharing node B: A->B->A and B->C->D->B."""
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("A", "B"),
            ("B", "A"),  # 2-cycle
            ("B", "C"),
            ("C", "D"),
            ("D", "B"),  # 3-cycle
        ]
    )
    return G


def _proxy_neq_exact_graph() -> nx.DiGraph:
    """
    Graph where proxy can over-count:
      A->B->C->A  (A,B,C in same SCC)
      A->D  (D outside SCC)

    Edge A->B: in G' = G - {A->B}, check if A and B still same SCC.
    G - {A->B}: B->C->A still exists, so A and B ARE still same SCC in G'.
    do({C}): removes C's in-edges (B->C removed), so B->C->A is broken.
    After do({C}), G' = G - {A->B} - {B->C}: A has no in-edges that route
    back from B. A and B end up in different SCCs. So exact_recovered=True.
    Proxy check (no edge removal): do({C}) on full G: B->C removed, C->A
    still there but B can't reach A via C. A->B still there... wait,
    proxy uses full G with do(S), not G'. Let's trace:
    Full G with do({C}): remove in-edges of C = {B->C}. Remaining:
    A->B, C->A, A->D. SCCs: {A,B} is 2-cycle (A->B and... no back-edge),
    actually A->B only, no B->A. So {A},{B},{C},{D} all singletons.
    cause=A, effect=B: different SCCs under do({C}) -> proxy_recovered=True.
    exact: G' = G-{A->B}, do({C}): remove B->C. Remaining: C->A, A->D.
    SCCs: {A},{B},{C},{D}. A and B different -> exact_recovered=True.
    So proxy=True, exact=True for this case. No gap here.

    For gap (proxy=True but exact=False):
    That would require do(S) separates cause/effect in G, but NOT in G'.
    This cannot happen because G' has one fewer edge than G, so any SCC
    in G' is a subset of an SCC in G. Removing the direct edge can only
    further split SCCs. Therefore: proxy sound means proxy_recovered=True
    implies exact_recovered=True. The gap is proxy=False, exact=True
    (proxy MISSES some recoveries where the edge itself was needed in G
    to keep A,B same-SCC, but in G' they'd be separated even without do(S)).

    So: let's build proxy-misses-exact case:
      A->B (direct edge), A->B->A=False (no cycle via A->B alone),
      but there's a cycle A->C->A, and B is reached only via A.
      In G: {A,C} SCC, B is a singleton. A and B are different SCCs in G.
      So proxy_recovered=True even without any do(S)! But that means
      the edge A->B is already in different SCCs... then it would have
      been identified by CSD already and not be in targets.

    The real gap scenario for proxy being incomplete (not wrong):
    In G: A->B->A (2-cycle), so same SCC. Edge A->B is in cycle.
    G' = G - {A->B}: only B->A remains. A and B still same SCC in G'
    (B->A is a 1-edge path from B to A, and A has no path to B in G').
    Wait: B->A only, no A->B. So SCCs in G': {A}, {B}, since B->A but
    no A->B. They are different SCCs. So exact_recovered=True with
    do({}) (empty set!). Proxy: in full G, A->B and B->A: same SCC.
    do({}) = no change. proxy_recovered = (A_scc == B_scc)? Yes, same SCC.
    proxy_recovered=False. But exact=True. This IS the gap!
    """
    # A->B->A: 2-cycle. Edge A->B is the "direct edge" being studied.
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "A")])
    return G


# ---------------------------------------------------------------------------
# Tests: _build_do_scc_map
# ---------------------------------------------------------------------------


class TestBuildDoSccMap:
    def test_every_node_assigned(self):
        G = _tiny_graph_triangle()
        scc_map = crb._build_do_scc_map(G, frozenset())
        for node in G.nodes():
            assert node in scc_map

    def test_triangle_all_same_scc(self):
        G = _tiny_graph_triangle()
        scc_map = crb._build_do_scc_map(G, frozenset())
        assert scc_map["A"] == scc_map["B"] == scc_map["C"]
        assert scc_map["D"] != scc_map["A"]

    def test_do_breaks_cycle(self):
        G = _tiny_graph_triangle()
        # Perturbing C breaks C->A, so the cycle A->B->C->A is severed
        scc_map = crb._build_do_scc_map(G, frozenset(["C"]))
        # A,B,C can no longer form a cycle since C's in-edge (B->C) is removed
        # C->A still exists. A->B exists. But B->C is removed.
        # So A->B, C->A remain. SCCs: {A,C} (C->A, no path back), actually:
        # A->B->? (B->C removed), C->A. Reachability: A->B (stop), C->A->B.
        # SCCs: A,B,C are all singletons since no cycles remain.
        assert scc_map["A"] != scc_map["B"] or scc_map["B"] != scc_map["C"]

    def test_requires_frozenset(self):
        G = _tiny_graph_triangle()
        with pytest.raises(AssertionError, match="PRE"):
            crb._build_do_scc_map(G, {"A"})  # set not frozenset


# ---------------------------------------------------------------------------
# Tests: _proxy_recovered
# ---------------------------------------------------------------------------


class TestProxyRecovered:
    def test_same_scc_not_recovered(self):
        # Both in SCC 0
        scc_map = {"A": 0, "B": 0}
        assert crb._proxy_recovered("A", "B", scc_map) is False

    def test_different_scc_recovered(self):
        scc_map = {"A": 0, "B": 1}
        assert crb._proxy_recovered("A", "B", scc_map) is True

    def test_missing_cause_recovered(self):
        # Missing node gets -1 vs -2 -> different
        scc_map = {"B": 1}
        assert crb._proxy_recovered("MISSING", "B", scc_map) is True

    def test_both_missing_recovered(self):
        # -1 != -2
        assert crb._proxy_recovered("X", "Y", {}) is True


# ---------------------------------------------------------------------------
# Tests: _exact_recovered
# ---------------------------------------------------------------------------


class TestExactRecovered:
    def test_triangle_no_intervention(self):
        """A->B->C->A: remove A->B, A and B still in same SCC (B->C->A)."""
        G = _tiny_graph_triangle()
        # After removing A->B: B->C->A still connects B to A. And A->? only A->B gone.
        # So in G' = G-{A->B}: edges B->C, C->A remain. B can reach A (via C).
        # A can reach B? No path from A to B in G'. A has no out-edges to the cycle
        # except A->B which is removed. So A and B are in DIFFERENT SCCs in G'.
        result = crb._exact_recovered("A", "B", G, frozenset())
        assert result is True  # after removing A->B, A and B are in different SCCs

    def test_two_cycle_proxy_gap(self):
        """
        A->B->A (2-cycle). Direct edge = A->B.
        G' = G - {A->B}: only B->A. B can reach A, A cannot reach B.
        SCCs in G': {A} and {B} are separate (B->A but no return).
        exact_recovered=True even with empty do(S).
        Proxy: full G, do({}): A,B same SCC -> proxy_recovered=False.
        This demonstrates the proxy-misses-exact gap.
        """
        G = _proxy_neq_exact_graph()
        # Exact: empty perturbation set
        assert crb._exact_recovered("A", "B", G, frozenset()) is True
        # Proxy on full G with empty set: A,B in same SCC
        scc_map = crb._build_do_scc_map(G, frozenset())
        assert crb._proxy_recovered("A", "B", scc_map) is False
        # This IS the proxy gap: proxy=False, exact=True
        # (proxy is sound: if proxy=True then exact=True, but not vice versa)

    def test_exact_sound_when_proxy_true(self):
        """If proxy says recovered, exact must agree."""
        G = _tiny_graph_two_cycles()
        # Try various perturbation sets; check soundness
        for genes in [frozenset(["A"]), frozenset(["C"]), frozenset(["A", "C"])]:
            scc_map = crb._build_do_scc_map(G, genes)
            for cause, effect in [("A", "B"), ("B", "C"), ("C", "D")]:
                if G.has_edge(cause, effect):
                    if crb._proxy_recovered(cause, effect, scc_map):
                        # Proxy sound: exact must also be true
                        assert crb._exact_recovered(cause, effect, G, genes), (
                            f"Soundness violated: proxy=True but exact=False "
                            f"for {cause}->{effect} do({genes})"
                        )


# ---------------------------------------------------------------------------
# Tests: _greedy_bank
# ---------------------------------------------------------------------------


class TestGreedyBank:
    def _simple_targets(self):
        return [
            {"cause": "A", "effect": "B", "same_scc": True},
            {"cause": "B", "effect": "C", "same_scc": True},
        ]

    def test_returns_n_sets(self):
        G = _tiny_graph_triangle()
        targets = self._simple_targets()
        pool = {"C", "A"}
        bank = crb._greedy_bank(G, targets, pool, n=3, k=2, verbose=False)
        assert len(bank) == 3

    def test_set_index_1based(self):
        G = _tiny_graph_triangle()
        targets = self._simple_targets()
        pool = {"C"}
        bank = crb._greedy_bank(G, targets, pool, n=2, k=1, verbose=False)
        assert bank[0]["set_index"] == 1
        assert bank[1]["set_index"] == 2

    def test_coverage_monotone(self):
        G = _tiny_graph_triangle()
        targets = self._simple_targets()
        pool = {"C", "A", "B"}
        bank = crb._greedy_bank(G, targets, pool, n=5, k=2, verbose=False)
        cumulative = [item["proxy_covered_cumulative"] for item in bank]
        for i in range(len(cumulative) - 1):
            assert cumulative[i + 1] >= cumulative[i], "Coverage must be non-decreasing"

    def test_genes_in_pool(self):
        G = _tiny_graph_triangle()
        targets = self._simple_targets()
        pool = {"C", "A", "B"}
        bank = crb._greedy_bank(G, targets, pool, n=2, k=2, verbose=False)
        for item in bank:
            for gene in item["genes"]:
                assert gene in pool, f"Chosen gene {gene!r} not in pool"

    def test_empty_pool_returns_empty_sets(self):
        G = _tiny_graph_triangle()
        targets = self._simple_targets()
        pool: set = set()
        # With empty pool, _load_candidate_pool check fails upstream,
        # but _greedy_bank itself should handle gracefully
        bank = crb._greedy_bank(G, targets, pool, n=2, k=2, verbose=False)
        assert len(bank) == 2
        for item in bank:
            assert item["genes"] == [] or len(item["genes"]) == 0

    def test_precondition_n_zero(self):
        G = _tiny_graph_triangle()
        with pytest.raises(AssertionError, match="PRE"):
            crb._greedy_bank(
                G, [{"cause": "A", "effect": "B", "same_scc": True}], {"C"}, n=0, k=1, verbose=False
            )

    def test_precondition_k_zero(self):
        G = _tiny_graph_triangle()
        with pytest.raises(AssertionError, match="PRE"):
            crb._greedy_bank(
                G, [{"cause": "A", "effect": "B", "same_scc": True}], {"C"}, n=1, k=0, verbose=False
            )


# ---------------------------------------------------------------------------
# Tests: _exact_verify_bank
# ---------------------------------------------------------------------------


class TestExactVerifyBank:
    def test_postcondition_n_recovered_matches(self):
        G = _tiny_graph_triangle()
        targets = [
            {"cause": "A", "effect": "B", "same_scc": True},
            {"cause": "B", "effect": "C", "same_scc": True},
        ]
        bank = [
            {
                "set_index": 1,
                "genes": ["C"],
                "proxy_recovered_new": 1,
                "proxy_covered_cumulative": 1,
            },
        ]
        bank_exact, edge_results = crb._exact_verify_bank(G, targets, bank)
        n_recovered_counted = sum(1 for e in edge_results if e["recovered"])
        assert n_recovered_counted == bank_exact[-1]["exact_covered_cumulative"]

    def test_empty_genes_set_contributes_zero(self):
        G = _tiny_graph_triangle()
        targets = [{"cause": "A", "effect": "B", "same_scc": True}]
        bank = [
            {"set_index": 1, "genes": [], "proxy_recovered_new": 0, "proxy_covered_cumulative": 0},
        ]
        bank_exact, edge_results = crb._exact_verify_bank(G, targets, bank)
        assert bank_exact[0]["exact_recovered_new"] == 0
        assert all(not e["recovered"] for e in edge_results)


# ---------------------------------------------------------------------------
# Tests: CSV loaders (with in-memory tmpfiles)
# ---------------------------------------------------------------------------


class TestLoadTargets:
    def test_basic(self, tmp_path):
        p = tmp_path / "clf.csv"
        p.write_text(
            "cause,effect,status,same_scc,timed_out\n"
            "A,B,unidentifiable,True,False\n"
            "C,D,identifiable,False,False\n"
            "E,F,unidentifiable,False,False\n"
        )
        targets = crb._load_targets(str(p))
        assert len(targets) == 2
        assert targets[0] == {"cause": "A", "effect": "B", "same_scc": True}
        assert targets[1] == {"cause": "E", "effect": "F", "same_scc": False}

    def test_empty_unidentifiable_raises(self, tmp_path):
        p = tmp_path / "clf.csv"
        p.write_text("cause,effect,status,same_scc\nA,B,identifiable,False\n")
        with pytest.raises(AssertionError, match="POST"):
            crb._load_targets(str(p))


class TestLoadCandidatePool:
    def test_basic(self, tmp_path):
        p = tmp_path / "break.csv"
        p.write_text(
            "cause,effect,min_break_set,min_break_size\n"
            'A,B,"[""X"", ""Y""]",2\n'
            'C,D,"[""Y"", ""Z""]",2\n'
            'E,F,"[]",0\n'
        )
        pool, edge_to_bs = crb._load_candidate_pool(str(p))
        assert "X" in pool
        assert "Y" in pool
        assert "Z" in pool
        assert len(pool) == 3
        assert edge_to_bs[("A", "B")] == frozenset(["X", "Y"])
        assert edge_to_bs[("E", "F")] == frozenset()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(AssertionError, match="PRE"):
            crb._load_candidate_pool(str(tmp_path / "nonexistent.csv"))
