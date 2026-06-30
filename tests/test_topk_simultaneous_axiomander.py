"""
tests/test_topk_simultaneous_axiomander.py
==========================================
Adorned contract tests for scripts/topk_simultaneous.py.

All tests use injectable helpers — no y0, no networkx real graph required.
A tiny in-memory toy network (3-cycle A→B→C→A plus an extra node D) is built
with networkx for SCC-mass / set_cycle_break_score tests.

Test classes
------------
TestBuildQueryResolvers       -- PRE guards + resolver correctness
TestUnionLowerBound           -- PRE guards + bound vs exhaustive count
TestOptimisticUpperBound      -- PRE guards + bound >= true count (soundness)
TestEvaluatePerturbationSet   -- PRE guards + self-pair exclusion + identify_fn
TestScoreCandidateSet         -- fast-path skip + slow-path delegation + POST
TestSccMass                   -- correct SCC mass on toy graph
TestSetCycleBreakScore        -- monotonicity + correctness on toy graph
TestGreedyTopk                -- greedy returns non-decreasing gain; k=1 exact
TestExhaustiveTopk            -- exhaustive top-n + guards
TestNegative                  -- POST guard fires on bad identify_fn result

Conventions
-----------
- Production code is tested through its public interface; no axiomander import.
- identify_fn signature: (tf1: str, outcome: str, effective_set: frozenset) -> bool
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import networkx as nx
from perturbation_optimizer import scc_mass, set_cycle_break_score
from topk_simultaneous import (
    build_query_resolvers,
    evaluate_perturbation_set,
    exhaustive_topk,
    greedy_topk,
    optimistic_upper_bound,
    score_candidate_set,
    union_lower_bound,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Tiny matrix:
#   candidates: A, B, C, D
#   queries:    q0, q1, q2
#   A resolves q0, q1
#   B resolves q1, q2
#   C resolves q2
#   D resolves nothing
CANDIDATES = ["A", "B", "C", "D"]
QUERIES = ["q0", "q1", "q2"]
MATRIX = {
    "A": [True, True, False],
    "B": [False, True, True],
    "C": [False, False, True],
    "D": [False, False, False],
}
QUERY_LIST = [("tf_q0", "out_q0"), ("tf_q1", "out_q1"), ("tf_q2", "out_q2")]


def toy_graph():
    """3-cycle A→B→C→A plus singleton node D."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    g.add_node("D")
    return g


def always_false_identify(tf1, outcome, effective_set):
    """Identify function that always says unidentifiable."""
    return False


def always_true_identify(tf1, outcome, effective_set):
    """Identify function that always says identifiable."""
    return True


def identify_by_set_size(tf1, outcome, effective_set):
    """Return True iff the effective set is non-empty."""
    return len(effective_set) >= 1


# ---------------------------------------------------------------------------
# TestBuildQueryResolvers
# ---------------------------------------------------------------------------


class TestBuildQueryResolvers:
    """Tests for build_query_resolvers."""

    def test_pre_candidates_must_be_list(self):
        """PRE: candidates must be a list — str raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE"):
            build_query_resolvers("A", QUERIES, MATRIX)

    def test_pre_queries_must_be_list(self):
        """PRE: queries must be a list — str raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE"):
            build_query_resolvers(CANDIDATES, "q0", MATRIX)

    def test_pre_matrix_must_be_dict(self):
        """PRE: matrix must be a dict — list raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE"):
            build_query_resolvers(CANDIDATES, QUERIES, [])

    def test_length_equals_n_queries(self):
        """POST: len(result) == len(queries)."""
        r = build_query_resolvers(CANDIDATES, QUERIES, MATRIX)
        assert len(r) == len(QUERIES)

    def test_resolver_contents_correct(self):
        """POST: each resolver contains exactly the candidates that resolve that query."""
        r = build_query_resolvers(CANDIDATES, QUERIES, MATRIX)
        # q0 resolved only by A
        assert r[0] == frozenset({"A"})
        # q1 resolved by A and B
        assert r[1] == frozenset({"A", "B"})
        # q2 resolved by B and C
        assert r[2] == frozenset({"B", "C"})

    def test_zero_gain_candidate_not_in_resolvers(self):
        """POST: D (zero gain) is not in any resolver."""
        r = build_query_resolvers(CANDIDATES, QUERIES, MATRIX)
        for qi in range(len(QUERIES)):
            assert "D" not in r[qi]

    def test_empty_candidates(self):
        """POST: empty candidates → all resolvers are empty frozensets."""
        r = build_query_resolvers([], QUERIES, {})
        assert len(r) == len(QUERIES)
        for qi in range(len(QUERIES)):
            assert r[qi] == frozenset()


# ---------------------------------------------------------------------------
# TestUnionLowerBound
# ---------------------------------------------------------------------------


class TestUnionLowerBound:
    """Tests for union_lower_bound."""

    def setup_method(self):
        """Build resolvers from the shared toy matrix."""
        self.resolvers = build_query_resolvers(CANDIDATES, QUERIES, MATRIX)

    def test_pre_candidate_set_must_be_set(self):
        """PRE: candidate_set must be a set — list raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE"):
            union_lower_bound(["A"], QUERIES, MATRIX, self.resolvers)

    def test_empty_set_gives_zero(self):
        """POST: empty candidate set → lower bound is 0."""
        assert union_lower_bound(set(), QUERIES, MATRIX, self.resolvers) == 0

    def test_singleton_A_gives_2(self):
        """POST: {A} resolves q0 and q1 → lower bound is 2."""
        # A resolves q0, q1
        assert union_lower_bound({"A"}, QUERIES, MATRIX, self.resolvers) == 2

    def test_union_AB_gives_3(self):
        """POST: {A, B} union covers all 3 queries → lower bound is 3."""
        # A covers q0,q1; B covers q1,q2 — union covers all 3
        assert union_lower_bound({"A", "B"}, QUERIES, MATRIX, self.resolvers) == 3

    def test_D_only_gives_zero(self):
        """POST: {D} resolves nothing → lower bound is 0."""
        assert union_lower_bound({"D"}, QUERIES, MATRIX, self.resolvers) == 0

    def test_frozenset_accepted(self):
        """POST: frozenset is accepted as candidate_set."""
        # frozenset should be accepted as candidate_set
        assert union_lower_bound(frozenset({"A"}), QUERIES, MATRIX, self.resolvers) == 2

    def test_result_leq_n_queries(self):
        """POST: lower bound <= total number of queries."""
        result = union_lower_bound({"A", "B", "C"}, QUERIES, MATRIX, self.resolvers)
        assert result <= len(QUERIES)


# ---------------------------------------------------------------------------
# TestOptimisticUpperBound
# ---------------------------------------------------------------------------


class TestOptimisticUpperBound:
    """Tests for optimistic_upper_bound."""

    def setup_method(self):
        """Build resolvers from the shared toy matrix."""
        self.resolvers = build_query_resolvers(CANDIDATES, QUERIES, MATRIX)

    def test_pre_current_set_must_be_set(self):
        """PRE: current_set must be a set — list raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE"):
            optimistic_upper_bound(["A"], CANDIDATES, 2, QUERIES, self.resolvers, set())

    def test_pre_remaining_budget_non_negative(self):
        """PRE: remaining_budget must be >= 0 — negative raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE"):
            optimistic_upper_bound(set(), CANDIDATES, -1, QUERIES, self.resolvers, set())

    def test_bound_at_least_already_resolved(self):
        """POST: bound >= number of already-resolved queries."""
        already = {0}  # q0 already resolved
        ub = optimistic_upper_bound({"A"}, CANDIDATES, 2, QUERIES, self.resolvers, already)
        assert ub >= len(already)

    def test_bound_at_most_n_queries(self):
        """POST: bound <= total number of queries."""
        ub = optimistic_upper_bound(set(), CANDIDATES, 5, QUERIES, self.resolvers, set())
        assert ub <= len(QUERIES)

    def test_budget_zero_equals_already_resolved(self):
        """POST: budget=0 → bound equals already-resolved count."""
        already = {0, 1}
        ub = optimistic_upper_bound({"A"}, CANDIDATES, 0, QUERIES, self.resolvers, already)
        assert ub == len(already)

    def test_full_pool_budget_equals_all_resolvable(self):
        """POST: full budget → bound equals total resolvable queries (3)."""
        # With budget == len(candidates) and no current set, bound >= len(resolvable)
        ub = optimistic_upper_bound(
            set(), CANDIDATES, len(CANDIDATES), QUERIES, self.resolvers, set()
        )
        # Resolvable = {q0, q1, q2}; D resolves nothing; bound must be 3
        assert ub == 3


# ---------------------------------------------------------------------------
# TestEvaluatePerturbationSet
# ---------------------------------------------------------------------------


class TestEvaluatePerturbationSet:
    """Tests for evaluate_perturbation_set."""

    def test_pre_tf1_must_be_str(self):
        """PRE: tf1 must be a str — int raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE"):
            evaluate_perturbation_set(1, "out", {"A"}, None, None, always_false_identify)

    def test_pre_outcome_must_be_str(self):
        """PRE: outcome must be a str — int raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE"):
            evaluate_perturbation_set("tf", 2, {"A"}, None, None, always_false_identify)

    def test_pre_candidate_set_must_be_set(self):
        """PRE: candidate_set must be a set — list raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE"):
            evaluate_perturbation_set("tf", "out", ["A"], None, None, always_false_identify)

    def test_always_false_returns_false(self):
        """POST: result is False when identify_fn always returns False."""
        result = evaluate_perturbation_set(
            "tf1", "out1", {"A", "B"}, None, None, always_false_identify
        )
        assert result is False

    def test_always_true_returns_true(self):
        """POST: result is True when identify_fn always returns True."""
        result = evaluate_perturbation_set("tf1", "out1", {"A"}, None, None, always_true_identify)
        assert result is True

    def test_self_pair_tf1_excluded(self):
        """If the set is only {tf1}, effective_set is empty → identify_fn({}) called."""
        calls = []

        def capture_identify(tf1, outcome, effective_set):
            """Capture the effective_set passed to identify_fn."""
            calls.append(frozenset(effective_set))
            return False

        evaluate_perturbation_set("tf1", "out1", {"tf1"}, None, None, capture_identify)
        assert calls == [frozenset()]  # self-pair excluded

    def test_self_pair_outcome_excluded(self):
        """POST: outcome is excluded from effective_set passed to identify_fn."""
        calls = []

        def capture_identify(tf1, outcome, effective_set):
            """Capture the effective_set passed to identify_fn."""
            calls.append(frozenset(effective_set))
            return False

        evaluate_perturbation_set("tf1", "out1", {"out1", "A"}, None, None, capture_identify)
        # "out1" excluded; only "A" remains
        assert calls == [frozenset({"A"})]

    def test_frozenset_accepted(self):
        """POST: frozenset is accepted as candidate_set."""
        result = evaluate_perturbation_set(
            "tf1", "out1", frozenset({"A", "B"}), None, None, always_true_identify
        )
        assert result is True

    def test_empty_set_calls_identify_with_empty(self):
        """POST: empty candidate_set → identify_fn called with empty effective_set."""
        calls = []

        def capture_identify(tf1, outcome, effective_set):
            """Capture the effective_set passed to identify_fn."""
            calls.append(frozenset(effective_set))
            return True

        evaluate_perturbation_set("tf1", "out1", set(), None, None, capture_identify)
        assert calls == [frozenset()]


# ---------------------------------------------------------------------------
# TestScoreCandidateSet
# ---------------------------------------------------------------------------


class TestScoreCandidateSet:
    """Tests for score_candidate_set."""

    def setup_method(self):
        """Build resolvers from the shared toy matrix."""
        self.resolvers = build_query_resolvers(CANDIDATES, QUERIES, MATRIX)

    def test_pre_candidate_set_must_be_set(self):
        """PRE: candidate_set must be a set — list raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE"):
            score_candidate_set(
                ["A"], QUERY_LIST, None, None, self.resolvers, always_false_identify
            )

    def test_pre_query_list_must_be_list(self):
        """PRE: query_list must be a list — str raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE"):
            score_candidate_set(
                {"A"}, "not-a-list", None, None, self.resolvers, always_false_identify
            )

    def test_score_A_alone_is_2_via_fast_path(self):
        """A resolves q0,q1 in the matrix — fast path, no identify_fn called."""
        call_count = [0]

        def counting_identify(tf1, outcome, effective_set):
            """Count calls to identify_fn."""
            call_count[0] += 1
            return False

        score = score_candidate_set(
            {"A"}, QUERY_LIST, None, None, self.resolvers, counting_identify
        )
        assert score == 2
        # Fast path should have been used for both; slow path for q2 → False → 0
        # So identify_fn called exactly once (for q2, which A doesn't cover)
        assert call_count[0] == 1

    def test_score_empty_set_with_always_true(self):
        """Empty set — no fast path; all queries go to identify_fn."""
        score = score_candidate_set(
            set(), QUERY_LIST, None, None, self.resolvers, always_true_identify
        )
        assert score == len(QUERY_LIST)

    def test_score_full_set_ABC_is_3(self):
        """A+B+C via matrix alone = 3."""
        score = score_candidate_set(
            {"A", "B", "C"}, QUERY_LIST, None, None, self.resolvers, always_false_identify
        )
        assert score == 3

    def test_score_D_only_with_always_false_is_0(self):
        """POST: D resolves nothing → score is 0."""
        score = score_candidate_set(
            {"D"}, QUERY_LIST, None, None, self.resolvers, always_false_identify
        )
        assert score == 0

    def test_frozenset_accepted(self):
        """POST: frozenset is accepted as candidate_set."""
        score = score_candidate_set(
            frozenset({"A"}), QUERY_LIST, None, None, self.resolvers, always_false_identify
        )
        assert score == 2

    def test_post_result_leq_query_count(self):
        """POST: 0 <= score <= len(query_list)."""
        score = score_candidate_set(
            {"A", "B", "C"}, QUERY_LIST, None, None, self.resolvers, always_true_identify
        )
        assert 0 <= score <= len(QUERY_LIST)


# ---------------------------------------------------------------------------
# TestSccMass
# ---------------------------------------------------------------------------


class TestSccMass:
    """Tests for scc_mass."""

    def test_three_cycle_has_mass_3(self):
        """POST: 3-cycle A→B→C→A has SCC mass 3."""
        g = toy_graph()
        # A→B→C→A is one SCC of size 3; D is singleton → mass = 3
        assert scc_mass(g) == 3

    def test_dag_has_zero_mass(self):
        """POST: DAG has SCC mass 0 (no non-trivial SCCs)."""
        g = nx.DiGraph()
        g.add_edges_from([("X", "Y"), ("Y", "Z")])
        assert scc_mass(g) == 0

    def test_two_separate_cycles(self):
        """POST: two separate cycles (sizes 2 and 3) give mass 5."""
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "A")])  # 2-cycle
        g.add_edges_from([("C", "D"), ("D", "E"), ("E", "C")])  # 3-cycle
        assert scc_mass(g) == 5

    def test_pre_graph_must_have_nodes(self):
        """PRE: graph must be a DiGraph — str raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE"):
            scc_mass("not-a-graph")


# ---------------------------------------------------------------------------
# TestSetCycleBreakScore
# ---------------------------------------------------------------------------


class TestSetCycleBreakScore:
    """Tests for set_cycle_break_score."""

    def test_breaking_all_cycle_members_gives_full_mass(self):
        """POST: intervening on all cycle members gives score == scc_mass."""
        g = toy_graph()
        # Intervening on {A,B,C} breaks the only cycle → score == scc_mass
        score = set_cycle_break_score({"A", "B", "C"}, g)
        assert score == scc_mass(g)

    def test_singleton_node_in_cycle_gives_nonzero_score(self):
        """POST: intervening on a single cycle member gives nonzero score."""
        g = toy_graph()
        # A is in the 3-cycle; removing A's in-edges breaks the cycle
        score = set_cycle_break_score({"A"}, g)
        assert score > 0

    def test_node_not_in_cycle_gives_zero(self):
        """POST: intervening on a non-cycle node gives score 0."""
        g = toy_graph()
        # D is a singleton; removing its (non-existent) in-edges changes nothing
        score = set_cycle_break_score({"D"}, g)
        assert score == 0

    def test_empty_set_gives_zero(self):
        """POST: empty candidate set gives score 0."""
        g = toy_graph()
        assert set_cycle_break_score(set(), g) == 0

    def test_frozenset_accepted(self):
        """POST: frozenset is accepted as candidate_set."""
        g = toy_graph()
        score = set_cycle_break_score(frozenset({"A"}), g)
        assert score >= 0

    def test_monotonicity_adding_node_never_decreases_score(self):
        """Score({A}) <= Score({A, B}) — adding a node never un-breaks a cycle."""
        g = toy_graph()
        s1 = set_cycle_break_score({"A"}, g)
        s2 = set_cycle_break_score({"A", "B"}, g)
        assert s2 >= s1

    def test_original_graph_not_mutated(self):
        """POST: the original graph is not mutated by the score computation."""
        g = toy_graph()
        edges_before = set(g.edges())
        set_cycle_break_score({"A", "B", "C"}, g)
        assert set(g.edges()) == edges_before

    def test_pre_candidate_set_must_be_set(self):
        """PRE: candidate_set must be a set — list raises AssertionError."""
        g = toy_graph()
        with pytest.raises(AssertionError, match="PRE"):
            set_cycle_break_score(["A"], g)


# ---------------------------------------------------------------------------
# TestGreedyTopk
# ---------------------------------------------------------------------------


class TestGreedyTopk:
    """Tests for greedy_topk."""

    def setup_method(self):
        """Set up shared query list, candidates, queries, and matrix."""
        self.query_list = QUERY_LIST
        self.candidates = CANDIDATES
        self.queries = QUERIES
        self.matrix = MATRIX

    def test_k1_greedy_resolves_at_least_best_singleton(self):
        """Greedy k=1 must be at least as good as the best singleton."""
        best_singleton = max(sum(MATRIX[c]) for c in CANDIDATES if c != "D")  # A or B: 2 each
        best_set, scc_break, score, steps = greedy_topk(
            query_list=self.query_list,
            candidates=self.candidates,
            queries=self.queries,
            matrix=self.matrix,
            graph=None,
            apt_order=None,
            k=1,
            candidate_cap=10,
            beam_width=1,
            require_singleton_gain=True,
            identify_fn=always_false_identify,
        )
        assert score >= best_singleton

    def test_k2_score_at_least_k1(self):
        """Adding a gene never hurts."""
        _, _, score1, _ = greedy_topk(
            query_list=self.query_list,
            candidates=self.candidates,
            queries=self.queries,
            matrix=self.matrix,
            graph=None,
            apt_order=None,
            k=1,
            candidate_cap=10,
            beam_width=1,
            require_singleton_gain=True,
            identify_fn=always_false_identify,
        )
        _, _, score2, _ = greedy_topk(
            query_list=self.query_list,
            candidates=self.candidates,
            queries=self.queries,
            matrix=self.matrix,
            graph=None,
            apt_order=None,
            k=2,
            candidate_cap=10,
            beam_width=1,
            require_singleton_gain=True,
            identify_fn=always_false_identify,
        )
        assert score2 >= score1

    def test_result_is_four_tuple(self):
        """POST: result is a 4-tuple (best_set, scc_break, score, steps)."""
        result = greedy_topk(
            query_list=self.query_list,
            candidates=self.candidates,
            queries=self.queries,
            matrix=self.matrix,
            graph=None,
            apt_order=None,
            k=2,
            candidate_cap=10,
            beam_width=1,
            require_singleton_gain=False,
            identify_fn=always_false_identify,
        )
        assert isinstance(result, tuple) and len(result) == 4

    def test_require_singleton_gain_prunes_D(self):
        """With require_singleton_gain=True, D (zero gain) should not appear in best_set."""
        best_set, _, _, _ = greedy_topk(
            query_list=self.query_list,
            candidates=self.candidates,
            queries=self.queries,
            matrix=self.matrix,
            graph=None,
            apt_order=None,
            k=3,
            candidate_cap=10,
            beam_width=1,
            require_singleton_gain=True,
            identify_fn=always_false_identify,
        )
        assert "D" not in best_set

    def test_pre_k_must_be_positive(self):
        """PRE: k must be positive — k=0 raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE"):
            greedy_topk(
                query_list=self.query_list,
                candidates=self.candidates,
                queries=self.queries,
                matrix=self.matrix,
                graph=None,
                apt_order=None,
                k=0,
                candidate_cap=10,
                beam_width=1,
                require_singleton_gain=True,
            )


# ---------------------------------------------------------------------------
# TestExhaustiveTopk
# ---------------------------------------------------------------------------


class TestExhaustiveTopk:
    """Tests for exhaustive_topk."""

    def test_exhaustive_k1_top_result_correct(self):
        """Top-1 exhaustive for k=1 should be A or B (both score 2)."""
        results = exhaustive_topk(
            query_list=QUERY_LIST,
            candidates=CANDIDATES,
            queries=QUERIES,
            matrix=MATRIX,
            graph=None,
            apt_order=None,
            k=1,
            require_singleton_gain=True,
            top_n=5,
            identify_fn=always_false_identify,
        )
        assert len(results) >= 1
        best_set, scc_break, score = results[0]
        assert score == 2
        assert best_set <= {"A", "B"}  # frozenset subset check

    def test_exhaustive_k2_covers_all_queries(self):
        """k=2 with {A,B} or {A,C} or {B,C} should cover all 3 queries."""
        results = exhaustive_topk(
            query_list=QUERY_LIST,
            candidates=CANDIDATES,
            queries=QUERIES,
            matrix=MATRIX,
            graph=None,
            apt_order=None,
            k=2,
            require_singleton_gain=True,
            top_n=10,
            identify_fn=always_false_identify,
        )
        top_score = results[0][2]
        assert top_score == 3

    def test_top_n_respected(self):
        """POST: len(results) <= top_n."""
        results = exhaustive_topk(
            query_list=QUERY_LIST,
            candidates=CANDIDATES,
            queries=QUERIES,
            matrix=MATRIX,
            graph=None,
            apt_order=None,
            k=2,
            require_singleton_gain=False,
            top_n=2,
            identify_fn=always_false_identify,
        )
        assert len(results) <= 2

    def test_results_sorted_descending(self):
        """POST: results are sorted in descending order of score."""
        results = exhaustive_topk(
            query_list=QUERY_LIST,
            candidates=CANDIDATES,
            queries=QUERIES,
            matrix=MATRIX,
            graph=None,
            apt_order=None,
            k=2,
            require_singleton_gain=False,
            top_n=10,
            identify_fn=always_false_identify,
        )
        scores = [r[2] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_pre_k_must_be_positive(self):
        """PRE: k must be positive — k=0 raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE"):
            exhaustive_topk(
                query_list=QUERY_LIST,
                candidates=CANDIDATES,
                queries=QUERIES,
                matrix=MATRIX,
                graph=None,
                apt_order=None,
                k=0,
                require_singleton_gain=True,
            )


# ---------------------------------------------------------------------------
# TestNegative — bad identify_fn response fires POST guard
# ---------------------------------------------------------------------------


class TestNegative:
    """Negative tests: POST guard fires on bad identify_fn result."""

    def test_identify_fn_returning_non_bool_fires_post(self):
        """identify_fn that returns a non-bool must trigger the POST assertion."""

        def bad_identify(tf1, outcome, effective_set):
            """Injectable that returns a non-bool to trigger the POST guard."""
            return "yes"  # not a bool

        with pytest.raises(AssertionError, match="POST"):
            evaluate_perturbation_set("tf1", "out1", {"A"}, None, None, bad_identify)
