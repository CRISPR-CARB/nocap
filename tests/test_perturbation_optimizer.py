"""
tests/test_perturbation_optimizer.py
=====================================
Unit tests for scripts/perturbation_optimizer.py

Tests cover:
  - greedy_max_coverage: correctness, budget enforcement, submodularity,
    empty/degenerate inputs
  - greedy_min_set_cover: correctness, termination on irresolvable queries
  - build_marginal_gain_curve: shape, monotonicity, boundary values
  - load_coverage_matrix: round-trip CSV serialization
  - cycle_breaking_score / rank_candidates_by_cycle_score: graph-based ranking
"""

import csv
import os
import sys

import pytest

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from perturbation_optimizer import (
    build_marginal_gain_curve,
    cycle_breaking_score,
    greedy_max_coverage,
    greedy_min_set_cover,
    load_coverage_matrix,
    rank_candidates_by_cycle_score,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_matrix():
    """
    4 candidates, 6 queries.
    g1 covers q0,q1,q2  (3 queries)
    g2 covers q2,q3,q4  (3 queries, 1 overlap with g1)
    g3 covers q4,q5     (2 queries, 1 overlap with g2)
    g4 covers nothing
    Resolvable: all 6.  Min cover: {g1, g2, g3}.
    """
    candidates = ["g1", "g2", "g3", "g4"]
    queries = ["q0", "q1", "q2", "q3", "q4", "q5"]
    matrix = {
        "g1": [True, True, True, False, False, False],
        "g2": [False, False, True, True, True, False],
        "g3": [False, False, False, False, True, True],
        "g4": [False, False, False, False, False, False],
    }
    return candidates, queries, matrix


@pytest.fixture
def disjoint_matrix():
    """
    3 candidates, 6 queries, perfectly disjoint coverage.
    g1 covers q0,q1  g2 covers q2,q3  g3 covers q4,q5
    Min cover = {g1,g2,g3}, each with gain=2.
    """
    candidates = ["g1", "g2", "g3"]
    queries = ["q0", "q1", "q2", "q3", "q4", "q5"]
    matrix = {
        "g1": [True, True, False, False, False, False],
        "g2": [False, False, True, True, False, False],
        "g3": [False, False, False, False, True, True],
    }
    return candidates, queries, matrix


@pytest.fixture
def all_zero_matrix():
    """No candidate resolves any query."""
    candidates = ["g1", "g2"]
    queries = ["q0", "q1", "q2"]
    matrix = {
        "g1": [False, False, False],
        "g2": [False, False, False],
    }
    return candidates, queries, matrix


@pytest.fixture
def single_candidate_matrix():
    """One candidate resolves all queries."""
    candidates = ["g1"]
    queries = ["q0", "q1", "q2"]
    matrix = {
        "g1": [True, True, True],
    }
    return candidates, queries, matrix


# ---------------------------------------------------------------------------
# greedy_max_coverage
# ---------------------------------------------------------------------------


class TestGreedyMaxCoverage:
    def test_greedy_order(self, simple_matrix):
        """First pick should be the candidate with highest single coverage."""
        candidates, queries, matrix = simple_matrix
        result = greedy_max_coverage(candidates, queries, matrix, budget_k=1)
        assert len(result) == 1
        # g1 and g2 both cover 3 queries; g1 comes first alphabetically in the
        # sorted candidate list, so it should win (stable tie-break)
        assert result[0][1] == 3, "First pick should resolve 3 queries"
        assert result[0][2] == 3, "Cumulative after step 1 should be 3"

    def test_marginal_gain_decreases(self, simple_matrix):
        """Marginal gains must be non-increasing (submodularity)."""
        candidates, queries, matrix = simple_matrix
        result = greedy_max_coverage(candidates, queries, matrix, budget_k=4)
        gains = [g for _, g, _ in result]
        for i in range(len(gains) - 1):
            assert gains[i] >= gains[i + 1], (
                f"Marginal gain not non-increasing at step {i}: {gains}"
            )

    def test_cumulative_monotone(self, simple_matrix):
        """Cumulative coverage must be strictly increasing at each step."""
        candidates, queries, matrix = simple_matrix
        result = greedy_max_coverage(candidates, queries, matrix, budget_k=4)
        cumulatives = [c for _, _, c in result]
        for i in range(len(cumulatives) - 1):
            assert cumulatives[i] < cumulatives[i + 1], (
                f"Cumulative not strictly increasing at step {i}: {cumulatives}"
            )

    def test_budget_respected(self, simple_matrix):
        """Result length must not exceed budget_k."""
        candidates, queries, matrix = simple_matrix
        for k in [1, 2, 3, 10]:
            result = greedy_max_coverage(candidates, queries, matrix, budget_k=k)
            assert len(result) <= k, f"Result length {len(result)} exceeds budget {k}"

    def test_full_coverage_achieved(self, simple_matrix):
        """With k=3, all 6 queries should be resolved."""
        candidates, queries, matrix = simple_matrix
        result = greedy_max_coverage(candidates, queries, matrix, budget_k=3)
        assert result[-1][2] == 6, "Expected all 6 queries resolved with k=3"

    def test_zero_coverage_matrix(self, all_zero_matrix):
        """When no candidate resolves anything, result should be empty."""
        candidates, queries, matrix = all_zero_matrix
        result = greedy_max_coverage(candidates, queries, matrix, budget_k=5)
        assert result == [], "Expected empty result for all-zero matrix"

    def test_single_candidate(self, single_candidate_matrix):
        """Single candidate covering all queries."""
        candidates, queries, matrix = single_candidate_matrix
        result = greedy_max_coverage(candidates, queries, matrix, budget_k=3)
        assert len(result) == 1
        assert result[0][0] == "g1"
        assert result[0][2] == 3

    def test_budget_zero(self, simple_matrix):
        """Budget of 0 should return empty list."""
        candidates, queries, matrix = simple_matrix
        result = greedy_max_coverage(candidates, queries, matrix, budget_k=0)
        assert result == []

    def test_intervenable_filter(self, simple_matrix):
        """Intervenable filter should restrict candidate pool."""
        candidates, queries, matrix = simple_matrix
        # Only allow g3 (covers q4,q5)
        result = greedy_max_coverage(candidates, queries, matrix, budget_k=3, intervenable={"g3"})
        assert len(result) == 1
        assert result[0][0] == "g3"
        assert result[0][2] == 2

    def test_disjoint_coverage_full(self, disjoint_matrix):
        """Disjoint coverage: k=3 should resolve all 6 queries."""
        candidates, queries, matrix = disjoint_matrix
        result = greedy_max_coverage(candidates, queries, matrix, budget_k=3)
        assert result[-1][2] == 6

    def test_no_duplicate_selections(self, simple_matrix):
        """Each TF should appear at most once in the result."""
        candidates, queries, matrix = simple_matrix
        result = greedy_max_coverage(candidates, queries, matrix, budget_k=10)
        selected_tfs = [tf for tf, _, _ in result]
        assert len(selected_tfs) == len(set(selected_tfs)), "Duplicate TF selected"

    def test_result_tuple_structure(self, simple_matrix):
        """Each result element should be a 3-tuple (str, int, int)."""
        candidates, queries, matrix = simple_matrix
        result = greedy_max_coverage(candidates, queries, matrix, budget_k=3)
        for item in result:
            assert len(item) == 3
            tf, gain, cumulative = item
            assert isinstance(tf, str)
            assert isinstance(gain, int)
            assert isinstance(cumulative, int)


# ---------------------------------------------------------------------------
# greedy_min_set_cover
# ---------------------------------------------------------------------------


class TestGreedyMinSetCover:
    def test_covers_all_resolvable(self, simple_matrix):
        """Min cover should resolve all 6 resolvable queries."""
        candidates, queries, matrix = simple_matrix
        result = greedy_min_set_cover(candidates, queries, matrix)
        assert result[-1][2] == 6

    def test_min_cover_size(self, simple_matrix):
        """Min cover for simple_matrix needs exactly 3 TFs."""
        candidates, queries, matrix = simple_matrix
        result = greedy_min_set_cover(candidates, queries, matrix)
        assert len(result) == 3

    def test_disjoint_min_cover(self, disjoint_matrix):
        """Disjoint matrix: min cover needs all 3 TFs."""
        candidates, queries, matrix = disjoint_matrix
        result = greedy_min_set_cover(candidates, queries, matrix)
        assert len(result) == 3
        assert result[-1][2] == 6

    def test_zero_coverage_terminates(self, all_zero_matrix):
        """All-zero matrix: min cover should be empty (nothing to cover)."""
        candidates, queries, matrix = all_zero_matrix
        result = greedy_min_set_cover(candidates, queries, matrix)
        assert result == []

    def test_single_candidate_covers_all(self, single_candidate_matrix):
        """Single candidate covering all: min cover size = 1."""
        candidates, queries, matrix = single_candidate_matrix
        result = greedy_min_set_cover(candidates, queries, matrix)
        assert len(result) == 1
        assert result[0][2] == 3

    def test_partial_coverage_terminates(self):
        """
        When some queries are irresolvable, min cover should stop
        after covering all resolvable ones.
        """
        candidates = ["g1", "g2"]
        queries = ["q0", "q1", "q2", "q3"]  # q2,q3 irresolvable
        matrix = {
            "g1": [True, False, False, False],
            "g2": [False, True, False, False],
        }
        result = greedy_min_set_cover(candidates, queries, matrix)
        assert len(result) == 2
        assert result[-1][2] == 2  # only 2 resolvable queries covered

    def test_cumulative_monotone(self, simple_matrix):
        """Cumulative coverage in min cover must be strictly increasing."""
        candidates, queries, matrix = simple_matrix
        result = greedy_min_set_cover(candidates, queries, matrix)
        cumulatives = [c for _, _, c in result]
        for i in range(len(cumulatives) - 1):
            assert cumulatives[i] < cumulatives[i + 1]

    def test_no_duplicate_selections(self, simple_matrix):
        """Each TF should appear at most once."""
        candidates, queries, matrix = simple_matrix
        result = greedy_min_set_cover(candidates, queries, matrix)
        selected_tfs = [tf for tf, _, _ in result]
        assert len(selected_tfs) == len(set(selected_tfs))


# ---------------------------------------------------------------------------
# build_marginal_gain_curve
# ---------------------------------------------------------------------------


class TestBuildMarginalGainCurve:
    def test_starts_at_zero(self, simple_matrix):
        """Curve must start at (0, 0, 0.0)."""
        candidates, queries, matrix = simple_matrix
        curve = build_marginal_gain_curve(candidates, queries, matrix, max_k=3)
        assert curve[0] == (0, 0, 0.0)

    def test_length(self, simple_matrix):
        """Curve length should be max_k + 1 (or fewer if coverage saturates)."""
        candidates, queries, matrix = simple_matrix
        curve = build_marginal_gain_curve(candidates, queries, matrix, max_k=3)
        assert len(curve) == 4  # k=0,1,2,3

    def test_monotone_resolved(self, simple_matrix):
        """queries_resolved must be non-decreasing."""
        candidates, queries, matrix = simple_matrix
        curve = build_marginal_gain_curve(candidates, queries, matrix, max_k=5)
        resolved = [r for _, r, _ in curve]
        for i in range(len(resolved) - 1):
            assert resolved[i] <= resolved[i + 1]

    def test_fraction_in_unit_interval(self, simple_matrix):
        """All fractions must be in [0, 1]."""
        candidates, queries, matrix = simple_matrix
        curve = build_marginal_gain_curve(candidates, queries, matrix, max_k=5)
        for _, _, frac in curve:
            assert 0.0 <= frac <= 1.0

    def test_full_coverage_at_k3(self, simple_matrix):
        """At k=3, all 6 queries should be resolved (fraction=1.0)."""
        candidates, queries, matrix = simple_matrix
        curve = build_marginal_gain_curve(candidates, queries, matrix, max_k=3)
        assert curve[3][1] == 6
        assert abs(curve[3][2] - 1.0) < 1e-9

    def test_zero_matrix_curve(self, all_zero_matrix):
        """All-zero matrix: curve stays at 0 throughout."""
        candidates, queries, matrix = all_zero_matrix
        curve = build_marginal_gain_curve(candidates, queries, matrix, max_k=3)
        assert curve[0] == (0, 0, 0.0)
        # After k=0, no progress possible
        for k, resolved, frac in curve[1:]:
            assert resolved == 0
            assert frac == 0.0


# ---------------------------------------------------------------------------
# load_coverage_matrix (round-trip)
# ---------------------------------------------------------------------------


class TestLoadCoverageMatrix:
    def test_round_trip(self, simple_matrix, tmp_path):
        """Write then read a coverage matrix CSV and verify exact round-trip."""
        candidates, queries, matrix = simple_matrix
        csv_path = tmp_path / "test_matrix.csv"

        # Write
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_tf"] + queries)
            for cand in candidates:
                row = [cand] + [int(v) for v in matrix[cand]]
                writer.writerow(row)

        # Read back
        loaded_candidates, loaded_queries, loaded_matrix = load_coverage_matrix(str(csv_path))

        assert loaded_queries == queries
        assert set(loaded_candidates) == set(candidates)
        for cand in candidates:
            assert loaded_matrix[cand] == matrix[cand], f"Matrix mismatch for {cand}"

    def test_empty_matrix(self, tmp_path):
        """CSV with header only (no candidates) should return empty structures."""
        csv_path = tmp_path / "empty.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_tf", "q0", "q1"])

        loaded_candidates, loaded_queries, loaded_matrix = load_coverage_matrix(str(csv_path))
        assert loaded_candidates == []
        assert loaded_queries == ["q0", "q1"]
        assert loaded_matrix == {}


# ---------------------------------------------------------------------------
# cycle_breaking_score / rank_candidates_by_cycle_score
# ---------------------------------------------------------------------------


class TestCycleBreakingScore:
    """
    Tests for the graph-based cycle-breaking heuristic.

    Graph topology used:
      A -> B -> C -> A   (cycle 1: A,B,C)
      B -> D -> B        (cycle 2: B,D)
      E -> F             (no cycle)

    Cycle memberships:
      A: 1 cycle  (A-B-C)
      B: 2 cycles (A-B-C and B-D)
      C: 1 cycle  (A-B-C)
      D: 1 cycle  (B-D)
      E: 0 cycles
      F: 0 cycles
    """

    @pytest.fixture
    def cycle_graph(self):
        import networkx as nx

        G = nx.DiGraph()
        G.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
                ("C", "A"),  # cycle A-B-C
                ("B", "D"),
                ("D", "B"),  # cycle B-D
                ("E", "F"),  # no cycle
            ]
        )
        return G

    def test_b_has_highest_score(self, cycle_graph):
        """B participates in 2 cycles and should have the highest score."""
        score_b = cycle_breaking_score("B", cycle_graph)
        score_a = cycle_breaking_score("A", cycle_graph)
        score_e = cycle_breaking_score("E", cycle_graph)
        assert score_b > score_a, f"B ({score_b}) should outscore A ({score_a})"
        assert score_b > score_e, f"B ({score_b}) should outscore E ({score_e})"

    def test_non_cycle_node_score_zero(self, cycle_graph):
        """E and F are not in any cycle; their score should be 0."""
        assert cycle_breaking_score("E", cycle_graph) == 0
        assert cycle_breaking_score("F", cycle_graph) == 0

    def test_cycle_node_score_positive(self, cycle_graph):
        """A, B, C, D are all in cycles; their scores should be > 0."""
        for node in ["A", "B", "C", "D"]:
            assert cycle_breaking_score(node, cycle_graph) > 0, (
                f"Expected positive score for {node}"
            )

    def test_rank_order(self, cycle_graph):
        """rank_candidates_by_cycle_score should return B first."""
        candidates = ["A", "B", "C", "D", "E", "F"]
        ranked = rank_candidates_by_cycle_score(candidates, cycle_graph)
        assert ranked[0] == "B", f"Expected B first, got {ranked[0]}"

    def test_rank_non_cycle_nodes_last(self, cycle_graph):
        """E and F (score=0) should appear at the end of the ranking."""
        candidates = ["A", "B", "C", "D", "E", "F"]
        ranked = rank_candidates_by_cycle_score(candidates, cycle_graph)
        # E and F should be in the last two positions
        assert set(ranked[-2:]) == {"E", "F"}, f"Expected E,F last, got {ranked[-2:]}"

    def test_acyclic_graph_all_zero(self):
        """In a DAG, all cycle-breaking scores should be 0."""
        import networkx as nx

        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        for node in ["A", "B", "C"]:
            assert cycle_breaking_score(node, G) == 0

    def test_single_self_loop(self):
        """A self-loop counts as a cycle of length 1."""
        import networkx as nx

        G = nx.DiGraph()
        G.add_edge("A", "A")
        G.add_edge("B", "C")
        score_a = cycle_breaking_score("A", G)
        score_b = cycle_breaking_score("B", G)
        assert score_a > 0, "Self-loop node should have positive cycle score"
        assert score_b == 0, "Non-cycle node should have score 0"

    def test_rank_candidates_empty(self):
        """Empty candidate list should return empty ranking."""
        import networkx as nx

        G = nx.DiGraph()
        G.add_edge("A", "B")
        ranked = rank_candidates_by_cycle_score([], G)
        assert ranked == []

    def test_rank_candidates_subset(self, cycle_graph):
        """Ranking should work on a subset of graph nodes."""
        ranked = rank_candidates_by_cycle_score(["A", "E"], cycle_graph)
        assert ranked[0] == "A"
        assert ranked[1] == "E"
