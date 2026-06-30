"""
tests/test_perturbation_optimizer_axiomander.py
================================================
Axiomander-style contract tests for scripts/perturbation_optimizer.py.

Tests are organised by contract category:
  PRE  -- precondition violations raise AssertionError
  POST -- postconditions hold on valid inputs
  INV  -- loop invariants hold throughout execution

Functions under test:
  greedy_max_coverage
  greedy_min_set_cover
  build_marginal_gain_curve
  cycle_breaking_score
  rank_candidates_by_cycle_score
"""

import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from perturbation_optimizer import (
    build_marginal_gain_curve,
    cycle_breaking_score,
    greedy_max_coverage,
    greedy_min_set_cover,
    rank_candidates_by_cycle_score,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _matrix(cands, queries, coverage):
    """Build a matrix dict from a list-of-lists of bools."""
    return {c: list(row) for c, row in zip(cands, coverage, strict=False)}


# Simple 3-candidate, 4-query matrix
CANDS3 = ["A", "B", "C"]
QUERIES4 = ["q0", "q1", "q2", "q3"]
# A covers q0,q1; B covers q1,q2,q3; C covers q3
MATRIX3 = _matrix(
    CANDS3,
    QUERIES4,
    [
        [True, True, False, False],
        [False, True, True, True],
        [False, False, False, True],
    ],
)


# ---------------------------------------------------------------------------
# greedy_max_coverage — PRE
# ---------------------------------------------------------------------------


class TestGreedyMaxCoveragePreconditions:
    """Precondition violations for greedy_max_coverage raise AssertionError."""

    def test_pre_candidates_must_be_list(self):
        """Passing a tuple for candidates raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: candidates must be a list"):
            greedy_max_coverage(tuple(CANDS3), QUERIES4, MATRIX3, 2)

    def test_pre_queries_must_be_list(self):
        """Passing a tuple for queries raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: queries must be a list"):
            greedy_max_coverage(CANDS3, tuple(QUERIES4), MATRIX3, 2)

    def test_pre_budget_k_must_be_non_negative(self):
        """Negative budget_k raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: budget_k must be a non-negative int"):
            greedy_max_coverage(CANDS3, QUERIES4, MATRIX3, -1)

    def test_pre_budget_k_must_be_int(self):
        """Float budget_k raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: budget_k must be a non-negative int"):
            greedy_max_coverage(CANDS3, QUERIES4, MATRIX3, 2.0)

    def test_pre_intervenable_must_be_set_or_none(self):
        """Passing a list for intervenable raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: intervenable must be None or a set"):
            greedy_max_coverage(CANDS3, QUERIES4, MATRIX3, 2, intervenable=["A"])

    def test_pre_candidate_must_be_in_matrix(self):
        """Candidate not in matrix raises AssertionError."""
        bad_matrix = _matrix(
            ["A", "B"],
            QUERIES4,
            [
                [True, False, False, False],
                [False, True, False, False],
            ],
        )
        with pytest.raises(AssertionError, match="PRE: candidate 'C' must be a key in matrix"):
            greedy_max_coverage(["A", "B", "C"], QUERIES4, bad_matrix, 2)

    def test_pre_matrix_row_length_must_match_queries(self):
        """Matrix row with wrong length raises AssertionError."""
        bad_matrix = {"A": [True, False], "B": [False, True], "C": [False, False]}
        with pytest.raises(AssertionError, match="PRE: matrix row for"):
            greedy_max_coverage(CANDS3, QUERIES4, bad_matrix, 2)


# ---------------------------------------------------------------------------
# greedy_max_coverage — POST
# ---------------------------------------------------------------------------


class TestGreedyMaxCoveragePostconditions:
    """Postconditions hold on valid inputs for greedy_max_coverage."""

    def test_post_result_is_list(self):
        """Return value is a list."""
        result = greedy_max_coverage(CANDS3, QUERIES4, MATRIX3, 3)
        assert isinstance(result, list)

    def test_post_result_length_bounded_by_budget(self):
        """len(result) <= budget_k."""
        result = greedy_max_coverage(CANDS3, QUERIES4, MATRIX3, 1)
        assert len(result) <= 1

    def test_post_selected_tfs_are_distinct(self):
        """No TF appears twice in the selection."""
        result = greedy_max_coverage(CANDS3, QUERIES4, MATRIX3, 3)
        tfs = [tf for tf, _, _ in result]
        assert len(tfs) == len(set(tfs))

    def test_post_marginal_gains_are_positive(self):
        """Every marginal gain is >= 1."""
        result = greedy_max_coverage(CANDS3, QUERIES4, MATRIX3, 3)
        for _, gain, _ in result:
            assert gain >= 1

    def test_post_cumulative_is_non_decreasing(self):
        """Cumulative coverage is non-decreasing."""
        result = greedy_max_coverage(CANDS3, QUERIES4, MATRIX3, 3)
        cumulatives = [c for _, _, c in result]
        assert cumulatives == sorted(cumulatives)

    def test_post_cumulative_bounded_by_n_queries(self):
        """Cumulative coverage never exceeds len(queries)."""
        result = greedy_max_coverage(CANDS3, QUERIES4, MATRIX3, 3)
        for _, _, c in result:
            assert c <= len(QUERIES4)

    def test_post_zero_budget_gives_empty(self):
        """budget_k=0 always returns []."""
        result = greedy_max_coverage(CANDS3, QUERIES4, MATRIX3, 0)
        assert result == []

    def test_post_empty_candidates_gives_empty(self):
        """Empty candidates list returns []."""
        result = greedy_max_coverage([], [], {}, 5)
        assert result == []

    def test_post_all_false_matrix_gives_empty(self):
        """Matrix with no True values returns [] (no gain possible)."""
        m = _matrix(
            CANDS3,
            QUERIES4,
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ],
        )
        result = greedy_max_coverage(CANDS3, QUERIES4, m, 3)
        assert result == []

    def test_post_intervenable_restricts_pool(self):
        """Intervenable restricts which candidates can be selected."""
        result = greedy_max_coverage(CANDS3, QUERIES4, MATRIX3, 3, intervenable={"C"})
        tfs = [tf for tf, _, _ in result]
        assert all(tf in {"C"} for tf in tfs)

    def test_post_accepts_frozenset_intervenable(self):
        """Frozenset is accepted for intervenable."""
        result = greedy_max_coverage(
            CANDS3, QUERIES4, MATRIX3, 3, intervenable=frozenset({"A", "B"})
        )
        tfs = [tf for tf, _, _ in result]
        assert all(tf in {"A", "B"} for tf in tfs)


# ---------------------------------------------------------------------------
# greedy_min_set_cover — PRE
# ---------------------------------------------------------------------------


class TestGreedyMinSetCoverPreconditions:
    """Precondition violations for greedy_min_set_cover raise AssertionError."""

    def test_pre_candidates_must_be_list(self):
        """Passing a tuple for candidates raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: candidates must be a list"):
            greedy_min_set_cover(tuple(CANDS3), QUERIES4, MATRIX3)

    def test_pre_queries_must_be_list(self):
        """Passing a tuple for queries raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: queries must be a list"):
            greedy_min_set_cover(CANDS3, tuple(QUERIES4), MATRIX3)

    def test_pre_intervenable_must_be_set_or_none(self):
        """Passing a list for intervenable raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: intervenable must be None or a set"):
            greedy_min_set_cover(CANDS3, QUERIES4, MATRIX3, intervenable=["A"])


# ---------------------------------------------------------------------------
# greedy_min_set_cover — POST
# ---------------------------------------------------------------------------


class TestGreedyMinSetCoverPostconditions:
    """Postconditions hold on valid inputs for greedy_min_set_cover."""

    def test_post_result_is_list(self):
        """Return value is a list."""
        result = greedy_min_set_cover(CANDS3, QUERIES4, MATRIX3)
        assert isinstance(result, list)

    def test_post_selected_tfs_are_distinct(self):
        """No TF appears twice."""
        result = greedy_min_set_cover(CANDS3, QUERIES4, MATRIX3)
        tfs = [tf for tf, _, _ in result]
        assert len(tfs) == len(set(tfs))

    def test_post_marginal_gains_are_positive(self):
        """Every marginal gain is >= 1."""
        result = greedy_min_set_cover(CANDS3, QUERIES4, MATRIX3)
        for _, gain, _ in result:
            assert gain >= 1

    def test_post_cumulative_is_non_decreasing(self):
        """Cumulative coverage is non-decreasing."""
        result = greedy_min_set_cover(CANDS3, QUERIES4, MATRIX3)
        cumulatives = [c for _, _, c in result]
        assert cumulatives == sorted(cumulatives)

    def test_post_final_cumulative_equals_resolvable(self):
        """Final cumulative equals the number of resolvable queries."""
        result = greedy_min_set_cover(CANDS3, QUERIES4, MATRIX3)
        # All 4 queries are resolvable in MATRIX3
        assert result[-1][2] == 4

    def test_post_empty_candidates_gives_empty(self):
        """Empty candidates returns []."""
        result = greedy_min_set_cover([], [], {})
        assert result == []

    def test_post_all_false_matrix_gives_empty(self):
        """Matrix with no True values returns []."""
        m = _matrix(
            CANDS3,
            QUERIES4,
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ],
        )
        result = greedy_min_set_cover(CANDS3, QUERIES4, m)
        assert result == []


# ---------------------------------------------------------------------------
# build_marginal_gain_curve — PRE / POST
# ---------------------------------------------------------------------------


class TestBuildMarginalGainCurvePreconditions:
    """Precondition violations for build_marginal_gain_curve raise AssertionError."""

    def test_pre_max_k_must_be_non_negative(self):
        """Negative max_k raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: max_k must be a non-negative int"):
            build_marginal_gain_curve(CANDS3, QUERIES4, MATRIX3, -1)

    def test_pre_queries_must_be_non_empty(self):
        """Empty queries raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: queries must be a non-empty list"):
            build_marginal_gain_curve([], [], {}, 3)


class TestBuildMarginalGainCurvePostconditions:
    """Postconditions hold on valid inputs for build_marginal_gain_curve."""

    def test_post_starts_with_zero(self):
        """Curve always starts with (0, 0, 0.0)."""
        curve = build_marginal_gain_curve(CANDS3, QUERIES4, MATRIX3, 3)
        assert curve[0] == (0, 0, 0.0)

    def test_post_k_values_are_sequential(self):
        """K values are 0, 1, 2, ..."""
        curve = build_marginal_gain_curve(CANDS3, QUERIES4, MATRIX3, 3)
        ks = [k for k, _, _ in curve]
        assert ks == list(range(len(curve)))

    def test_post_cumulative_non_decreasing(self):
        """Cumulative values are non-decreasing."""
        curve = build_marginal_gain_curve(CANDS3, QUERIES4, MATRIX3, 3)
        cumulatives = [c for _, c, _ in curve]
        assert cumulatives == sorted(cumulatives)

    def test_post_fractions_in_unit_interval(self):
        """All fractions are in [0.0, 1.0]."""
        curve = build_marginal_gain_curve(CANDS3, QUERIES4, MATRIX3, 3)
        for _, _, frac in curve:
            assert 0.0 <= frac <= 1.0

    def test_post_zero_budget_gives_single_point(self):
        """max_k=0 returns just [(0, 0, 0.0)]."""
        curve = build_marginal_gain_curve(CANDS3, QUERIES4, MATRIX3, 0)
        assert curve == [(0, 0, 0.0)]


# ---------------------------------------------------------------------------
# cycle_breaking_score — PRE / POST
# ---------------------------------------------------------------------------


class TestCycleBreakingScorePreconditions:
    """Precondition violations for cycle_breaking_score raise AssertionError."""

    def test_pre_node_must_be_str(self):
        """Non-string node raises AssertionError."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        with pytest.raises(AssertionError, match="PRE: node must be a str"):
            cycle_breaking_score(42, G)

    def test_pre_graph_must_have_nodes(self):
        """Object without .nodes raises AssertionError."""
        with pytest.raises(AssertionError, match=r"PRE: graph must have a \.nodes"):
            cycle_breaking_score("A", object())


class TestCycleBreakingScorePostconditions:
    """Postconditions hold on valid inputs for cycle_breaking_score."""

    def test_post_result_is_non_negative_int(self):
        """Result is a non-negative int."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        result = cycle_breaking_score("A", G)
        assert isinstance(result, int) and result >= 0

    def test_post_acyclic_node_scores_zero(self):
        """Node in an acyclic graph scores 0."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C")])
        assert cycle_breaking_score("A", G) == 0
        assert cycle_breaking_score("B", G) == 0

    def test_post_node_in_cycle_scores_positive(self):
        """Node in a cycle scores >= 1."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "A")])
        assert cycle_breaking_score("A", G) >= 1

    def test_post_node_not_in_graph_scores_zero(self):
        """Node absent from graph scores 0 (not in any cycle)."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        assert cycle_breaking_score("Z", G) == 0


# ---------------------------------------------------------------------------
# rank_candidates_by_cycle_score — PRE / POST
# ---------------------------------------------------------------------------


class TestRankCandidatesByCycleScorePreconditions:
    """Precondition violations for rank_candidates_by_cycle_score raise AssertionError."""

    def test_pre_candidates_must_be_list(self):
        """Passing a tuple raises AssertionError."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        with pytest.raises(AssertionError, match="PRE: candidates must be a list"):
            rank_candidates_by_cycle_score(("A", "B"), G)

    def test_pre_graph_must_have_number_of_nodes(self):
        """Object without .number_of_nodes raises AssertionError."""
        with pytest.raises(AssertionError, match=r"PRE: graph must have a \.number_of_nodes"):
            rank_candidates_by_cycle_score(["A"], object())

    def test_pre_scc_fallback_threshold_must_be_positive(self):
        """scc_fallback_threshold=0 raises AssertionError."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        with pytest.raises(
            AssertionError, match="PRE: scc_fallback_threshold must be a positive int"
        ):
            rank_candidates_by_cycle_score(["A"], G, scc_fallback_threshold=0)


class TestRankCandidatesByCycleScorePostconditions:
    """Postconditions hold on valid inputs for rank_candidates_by_cycle_score."""

    def test_post_result_is_list(self):
        """Return value is a list."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        result = rank_candidates_by_cycle_score(["A", "B"], G)
        assert isinstance(result, list)

    def test_post_result_is_permutation(self):
        """Result contains exactly the same elements as candidates."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        cands = ["A", "B", "C"]
        result = rank_candidates_by_cycle_score(cands, G)
        assert set(result) == set(cands)
        assert len(result) == len(cands)

    def test_post_empty_candidates_gives_empty(self):
        """Empty candidates returns []."""
        G = nx.DiGraph()
        result = rank_candidates_by_cycle_score([], G)
        assert result == []

    def test_post_cycle_nodes_ranked_before_acyclic(self):
        """Nodes in cycles are ranked before acyclic nodes."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "A"), ("C", "D")])
        result = rank_candidates_by_cycle_score(["A", "B", "C", "D"], G)
        # A and B are in a cycle; C and D are not
        cycle_nodes = {"A", "B"}
        acyclic_nodes = {"C", "D"}
        cycle_positions = [i for i, n in enumerate(result) if n in cycle_nodes]
        acyclic_positions = [i for i, n in enumerate(result) if n in acyclic_nodes]
        assert max(cycle_positions) < min(acyclic_positions)

    def test_post_ties_broken_alphabetically(self):
        """Ties in score are broken alphabetically."""
        G = nx.DiGraph()
        G.add_nodes_from(["A", "B", "C"])  # no edges → all score 0
        result = rank_candidates_by_cycle_score(["C", "A", "B"], G)
        assert result == ["A", "B", "C"]

    def test_post_scc_fallback_gives_same_permutation(self):
        """SCC fallback returns a permutation of candidates."""
        G = nx.DiGraph()
        for i in range(10):
            G.add_edge(str(i), str((i + 1) % 10))
        cands = [str(i) for i in range(10)]
        result = rank_candidates_by_cycle_score(cands, G, scc_fallback_threshold=5)
        assert set(result) == set(cands)
        assert len(result) == len(cands)


# ---------------------------------------------------------------------------
# INV: loop invariants (exercised via the adorned production code)
# ---------------------------------------------------------------------------


class TestLoopInvariants:
    """
    Verify that the loop invariants in the production code fire correctly.
    These tests use inputs that would violate invariants if the logic were wrong.
    """

    def test_inv_greedy_max_coverage_unresolved_shrinks(self):
        """
        Each greedy step must strictly reduce unresolved queries.
        Verified indirectly: cumulative is strictly increasing.
        """
        result = greedy_max_coverage(CANDS3, QUERIES4, MATRIX3, 3)
        cumulatives = [c for _, _, c in result]
        # Strictly increasing (each step resolves at least 1 new query)
        for i in range(1, len(cumulatives)):
            assert cumulatives[i] > cumulatives[i - 1]

    def test_inv_greedy_min_set_cover_unresolved_shrinks(self):
        """Each min-set-cover step must strictly reduce unresolved queries."""
        result = greedy_min_set_cover(CANDS3, QUERIES4, MATRIX3)
        cumulatives = [c for _, _, c in result]
        for i in range(1, len(cumulatives)):
            assert cumulatives[i] > cumulatives[i - 1]

    def test_inv_greedy_max_coverage_gain_matches_newly_resolved(self):
        """
        The reported marginal gain must equal the actual number of newly
        resolved queries at each step.  Verified by re-simulating.
        """
        result = greedy_max_coverage(CANDS3, QUERIES4, MATRIX3, 3)
        # Re-simulate to check gain == newly resolved
        unresolved = set(range(len(QUERIES4)))
        for tf, gain, cumulative in result:
            newly = {qi for qi in unresolved if MATRIX3[tf][qi]}
            assert gain == len(newly), f"gain {gain} != newly_resolved {len(newly)}"
            unresolved -= newly
