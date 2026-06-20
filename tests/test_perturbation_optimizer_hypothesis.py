"""
tests/test_perturbation_optimizer_hypothesis.py
================================================
Property-based tests for scripts/perturbation_optimizer.py, generated from
the axiomander contracts embedded in the production code.

Each property directly mirrors one contract line from the docstring:

  greedy_max_coverage
    POST: result is a list of (str, int, int) tuples
    POST: len(result) <= min(budget_k, len(pool))
    POST: selected TFs are distinct
    POST: marginal gains are all >= 1
    POST: cumulative_coverage is non-decreasing
    POST: cumulative_coverage[-1] <= len(queries)

  greedy_min_set_cover
    POST: result is a list of (str, int, int) tuples
    POST: selected TFs are distinct
    POST: marginal gains are all >= 1
    POST: cumulative_coverage is non-decreasing
    POST: final cumulative == len(resolvable)

  build_marginal_gain_curve
    POST: result[0] == (0, 0, 0.0)
    POST: k values are 0, 1, 2, ...
    POST: cumulative values are non-decreasing
    POST: all fractions are in [0.0, 1.0]

  rank_candidates_by_cycle_score
    POST: result is a permutation of candidates
    POST: len(result) == len(candidates)
    POST: set(result) == set(candidates)
    POST: ties broken alphabetically

Note: module-level @given functions are used throughout to avoid the
pytest-fixture-vs-@given class-method conflict.
"""

import os
import sys

import networkx as nx
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from perturbation_optimizer import (
    build_marginal_gain_curve,
    greedy_max_coverage,
    greedy_min_set_cover,
    rank_candidates_by_cycle_score,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Short uppercase identifiers for TF / query names
tf_name = st.from_regex(r"[A-Z][A-Z0-9]{0,4}", fullmatch=True)
query_name = st.from_regex(r"[a-z][a-z0-9_]{0,6}", fullmatch=True)


@st.composite
def coverage_matrix_strategy(draw, min_cands=1, max_cands=8, min_queries=1, max_queries=8):
    """
    Generate (candidates, queries, matrix, budget_k) where:
      - candidates is a list of unique TF names
      - queries is a list of unique query labels
      - matrix[cand] is a list[bool] of length len(queries)
      - budget_k is in [0, len(candidates)]
    """
    cands = draw(st.lists(tf_name, min_size=min_cands, max_size=max_cands, unique=True))
    queries = draw(st.lists(query_name, min_size=min_queries, max_size=max_queries, unique=True))
    n_q = len(queries)
    matrix = {}
    for c in cands:
        row = draw(st.lists(st.booleans(), min_size=n_q, max_size=n_q))
        matrix[c] = row
    budget_k = draw(st.integers(min_value=0, max_value=len(cands)))
    return cands, queries, matrix, budget_k


@st.composite
def coverage_matrix_no_budget(draw, min_cands=1, max_cands=8, min_queries=1, max_queries=8):
    """Same as coverage_matrix_strategy but without budget_k."""
    cands, queries, matrix, _ = draw(
        coverage_matrix_strategy(min_cands, max_cands, min_queries, max_queries)
    )
    return cands, queries, matrix


@st.composite
def digraph_with_candidates(draw, min_nodes=2, max_nodes=8):
    """Generate a DiGraph and a list of candidate node names."""
    node_names = draw(
        st.lists(tf_name, min_size=min_nodes, max_size=max_nodes, unique=True)
    )
    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    possible = [(u, v) for u in node_names for v in node_names if u != v]
    if possible:
        edges = draw(
            st.lists(
                st.sampled_from(possible),
                min_size=0,
                max_size=min(15, len(possible)),
                unique=True,
            )
        )
        G.add_edges_from(edges)
    cands = draw(
        st.lists(
            st.sampled_from(node_names),
            min_size=0,
            max_size=len(node_names),
            unique=True,
        )
    )
    return G, cands


# ---------------------------------------------------------------------------
# greedy_max_coverage properties
# ---------------------------------------------------------------------------


@given(coverage_matrix_strategy())
@settings(max_examples=300)
def test_greedy_max_coverage_result_is_list(args):
    """POST: result is a list."""
    cands, queries, matrix, k = args
    result = greedy_max_coverage(cands, queries, matrix, k)
    assert isinstance(result, list)


@given(coverage_matrix_strategy())
@settings(max_examples=300)
def test_greedy_max_coverage_length_bounded_by_budget(args):
    """POST: len(result) <= budget_k."""
    cands, queries, matrix, k = args
    result = greedy_max_coverage(cands, queries, matrix, k)
    assert len(result) <= k


@given(coverage_matrix_strategy())
@settings(max_examples=300)
def test_greedy_max_coverage_selected_tfs_distinct(args):
    """POST: selected TFs are distinct."""
    cands, queries, matrix, k = args
    result = greedy_max_coverage(cands, queries, matrix, k)
    tfs = [tf for tf, _, _ in result]
    assert len(tfs) == len(set(tfs))


@given(coverage_matrix_strategy())
@settings(max_examples=300)
def test_greedy_max_coverage_marginal_gains_positive(args):
    """POST: every marginal gain is >= 1."""
    cands, queries, matrix, k = args
    result = greedy_max_coverage(cands, queries, matrix, k)
    for _, gain, _ in result:
        assert gain >= 1


@given(coverage_matrix_strategy())
@settings(max_examples=300)
def test_greedy_max_coverage_cumulative_non_decreasing(args):
    """POST: cumulative_coverage is non-decreasing."""
    cands, queries, matrix, k = args
    result = greedy_max_coverage(cands, queries, matrix, k)
    cumulatives = [c for _, _, c in result]
    assert cumulatives == sorted(cumulatives)


@given(coverage_matrix_strategy())
@settings(max_examples=300)
def test_greedy_max_coverage_cumulative_bounded_by_n_queries(args):
    """POST: cumulative_coverage[-1] <= len(queries)."""
    cands, queries, matrix, k = args
    result = greedy_max_coverage(cands, queries, matrix, k)
    for _, _, c in result:
        assert c <= len(queries)


@given(coverage_matrix_strategy())
@settings(max_examples=300)
def test_greedy_max_coverage_result_is_list_of_3tuples(args):
    """POST: each element is a (str, int, int) tuple."""
    cands, queries, matrix, k = args
    result = greedy_max_coverage(cands, queries, matrix, k)
    for item in result:
        assert isinstance(item, tuple) and len(item) == 3
        tf, gain, cumulative = item
        assert isinstance(tf, str)
        assert isinstance(gain, int)
        assert isinstance(cumulative, int)


@given(coverage_matrix_strategy())
@settings(max_examples=300)
def test_greedy_max_coverage_idempotent(args):
    """Calling twice with same inputs gives same result."""
    cands, queries, matrix, k = args
    r1 = greedy_max_coverage(cands, queries, matrix, k)
    r2 = greedy_max_coverage(cands, queries, matrix, k)
    assert r1 == r2


@given(coverage_matrix_strategy())
@settings(max_examples=300)
def test_greedy_max_coverage_selected_tfs_in_candidates(args):
    """POST: every selected TF is in the candidates list."""
    cands, queries, matrix, k = args
    result = greedy_max_coverage(cands, queries, matrix, k)
    cand_set = set(cands)
    for tf, _, _ in result:
        assert tf in cand_set


# ---------------------------------------------------------------------------
# greedy_min_set_cover properties
# ---------------------------------------------------------------------------


@given(coverage_matrix_no_budget())
@settings(max_examples=300)
def test_greedy_min_set_cover_result_is_list(args):
    """POST: result is a list."""
    cands, queries, matrix = args
    result = greedy_min_set_cover(cands, queries, matrix)
    assert isinstance(result, list)


@given(coverage_matrix_no_budget())
@settings(max_examples=300)
def test_greedy_min_set_cover_selected_tfs_distinct(args):
    """POST: selected TFs are distinct."""
    cands, queries, matrix = args
    result = greedy_min_set_cover(cands, queries, matrix)
    tfs = [tf for tf, _, _ in result]
    assert len(tfs) == len(set(tfs))


@given(coverage_matrix_no_budget())
@settings(max_examples=300)
def test_greedy_min_set_cover_marginal_gains_positive(args):
    """POST: every marginal gain is >= 1."""
    cands, queries, matrix = args
    result = greedy_min_set_cover(cands, queries, matrix)
    for _, gain, _ in result:
        assert gain >= 1


@given(coverage_matrix_no_budget())
@settings(max_examples=300)
def test_greedy_min_set_cover_cumulative_non_decreasing(args):
    """POST: cumulative_coverage is non-decreasing."""
    cands, queries, matrix = args
    result = greedy_min_set_cover(cands, queries, matrix)
    cumulatives = [c for _, _, c in result]
    assert cumulatives == sorted(cumulatives)


@given(coverage_matrix_no_budget())
@settings(max_examples=300)
def test_greedy_min_set_cover_final_cumulative_equals_resolvable(args):
    """POST: final cumulative == number of resolvable queries."""
    cands, queries, matrix = args
    result = greedy_min_set_cover(cands, queries, matrix)
    # Compute resolvable independently
    resolvable = set()
    for c in cands:
        for qi, val in enumerate(matrix[c]):
            if val:
                resolvable.add(qi)
    if result:
        assert result[-1][2] == len(resolvable)
    else:
        # No result means no resolvable queries
        assert len(resolvable) == 0


@given(coverage_matrix_no_budget())
@settings(max_examples=300)
def test_greedy_min_set_cover_selected_tfs_in_candidates(args):
    """POST: every selected TF is in the candidates list."""
    cands, queries, matrix = args
    result = greedy_min_set_cover(cands, queries, matrix)
    cand_set = set(cands)
    for tf, _, _ in result:
        assert tf in cand_set


@given(coverage_matrix_no_budget())
@settings(max_examples=300)
def test_greedy_min_set_cover_idempotent(args):
    """Calling twice with same inputs gives same result."""
    cands, queries, matrix = args
    r1 = greedy_min_set_cover(cands, queries, matrix)
    r2 = greedy_min_set_cover(cands, queries, matrix)
    assert r1 == r2


# ---------------------------------------------------------------------------
# build_marginal_gain_curve properties
# ---------------------------------------------------------------------------


@given(coverage_matrix_strategy(min_queries=1))
@settings(max_examples=300)
def test_build_marginal_gain_curve_starts_with_zero(args):
    """POST: curve[0] == (0, 0, 0.0)."""
    cands, queries, matrix, k = args
    curve = build_marginal_gain_curve(cands, queries, matrix, k)
    assert curve[0] == (0, 0, 0.0)


@given(coverage_matrix_strategy(min_queries=1))
@settings(max_examples=300)
def test_build_marginal_gain_curve_k_values_sequential(args):
    """POST: k values are 0, 1, 2, ..."""
    cands, queries, matrix, k = args
    curve = build_marginal_gain_curve(cands, queries, matrix, k)
    ks = [ki for ki, _, _ in curve]
    assert ks == list(range(len(curve)))


@given(coverage_matrix_strategy(min_queries=1))
@settings(max_examples=300)
def test_build_marginal_gain_curve_cumulative_non_decreasing(args):
    """POST: cumulative values are non-decreasing."""
    cands, queries, matrix, k = args
    curve = build_marginal_gain_curve(cands, queries, matrix, k)
    cumulatives = [c for _, c, _ in curve]
    assert cumulatives == sorted(cumulatives)


@given(coverage_matrix_strategy(min_queries=1))
@settings(max_examples=300)
def test_build_marginal_gain_curve_fractions_in_unit_interval(args):
    """POST: all fractions are in [0.0, 1.0]."""
    cands, queries, matrix, k = args
    curve = build_marginal_gain_curve(cands, queries, matrix, k)
    for _, _, frac in curve:
        assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# rank_candidates_by_cycle_score properties
# ---------------------------------------------------------------------------


@given(digraph_with_candidates())
@settings(max_examples=300)
def test_rank_candidates_result_is_list(args):
    """POST: result is a list."""
    G, cands = args
    result = rank_candidates_by_cycle_score(cands, G)
    assert isinstance(result, list)


@given(digraph_with_candidates())
@settings(max_examples=300)
def test_rank_candidates_result_is_permutation(args):
    """POST: result is a permutation of candidates."""
    G, cands = args
    result = rank_candidates_by_cycle_score(cands, G)
    assert sorted(result) == sorted(cands)


@given(digraph_with_candidates())
@settings(max_examples=300)
def test_rank_candidates_length_preserved(args):
    """POST: len(result) == len(candidates)."""
    G, cands = args
    result = rank_candidates_by_cycle_score(cands, G)
    assert len(result) == len(cands)


@given(digraph_with_candidates())
@settings(max_examples=300)
def test_rank_candidates_set_preserved(args):
    """POST: set(result) == set(candidates)."""
    G, cands = args
    result = rank_candidates_by_cycle_score(cands, G)
    assert set(result) == set(cands)


@given(digraph_with_candidates())
@settings(max_examples=300)
def test_rank_candidates_idempotent(args):
    """Calling twice gives the same ordering."""
    G, cands = args
    r1 = rank_candidates_by_cycle_score(cands, G)
    r2 = rank_candidates_by_cycle_score(cands, G)
    assert r1 == r2


@given(digraph_with_candidates())
@settings(max_examples=300)
def test_rank_candidates_scc_fallback_is_permutation(args):
    """SCC fallback also returns a permutation."""
    G, cands = args
    # Force SCC fallback by setting threshold to 0
    result = rank_candidates_by_cycle_score(cands, G, scc_fallback_threshold=1)
    assert set(result) == set(cands)
    assert len(result) == len(cands)
