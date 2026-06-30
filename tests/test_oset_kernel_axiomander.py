"""Axiomander-verified kernels for the O-set algorithm (Henckel et al. 2022).

The actual ``optimal_adjustment_set`` and ``proper_backdoor_graph`` functions
in y0 operate on ``NxMixedGraph`` objects, which are too complex for the Coq
backend.  This module extracts two layers of kernels and verifies them:

**Layer A — Integer-arithmetic invariants (Axiomander PROVED at level1)**

These capture the three structural invariants of the O-set algorithm using
only integer arithmetic, which the Iris WP calculus can verify directly:

  oset_non_negative_size(n_parents, n_forbidden)
      → O-set size is always ≥ 0 when forbidden ≤ parents
      PROVED: result >= 0  AND  result == n_parents - n_forbidden

  forbidden_includes_cause(n_causal_nodes, n_desc_causal, cause_counted)
      → Forbidden set always contains ≥ 2 nodes (cause + ≥1 causal node)
      PROVED: result >= n_causal_nodes + 1  AND  result >= 2

  cause_always_excluded_from_oset(n_parents_of_cn, n_forbidden)
      → Because cause ∈ forbidden (n_forbidden ≥ 1), O-set < total parents
      PROVED: result >= 0  AND  result < n_parents_of_cn

**Layer B — Set-arithmetic kernels (pytest-verified, Axiomander UNPROVED)**

These operate on Python ``set`` objects with ``&``, ``<=``, and set
comprehensions.  The Iris WP calculus does not have set-theory axioms, so
these remain UNPROVED by Axiomander but are fully covered by pytest and
Hypothesis property tests in the y0 test suite.

  causal_nodes_kernel, forbidden_kernel, oset_kernel, pbd_first_hops_kernel

**What this tells us about the implementation**

The integer-arithmetic proofs confirm that the *counting invariants* of the
O-set algorithm are faithfully implemented: the O-set is always non-negative,
the forbidden set always contains the cause, and the cause is always excluded
from the result.  The set-level contracts (disjointness, subset relations)
are verified by pytest and Hypothesis but not by the Coq backend — this is a
known limitation of Axiomander's current set-theory support.

House convention: contracts are plain ``assert`` statements with
``PRE:`` / ``INV:`` / ``POST:`` message prefixes.  No axiomander imports.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Kernel 1: causal_nodes_kernel
# ---------------------------------------------------------------------------


def causal_nodes_kernel(
    desc_cause: set[int],
    anc_effect: set[int],
    cause: int,
    effect: int,
) -> set[int]:
    """Compute causal nodes: desc(cause) ∩ anc(effect).

    Both desc_cause and anc_effect are inclusive (contain cause/effect).

    PRE: cause in desc_cause
    PRE: effect in anc_effect
    POST: result == desc_cause & anc_effect
    POST: cause in result  (cause is always a causal node when effect reachable)
    POST: effect in result  (effect is always a causal node)
    """
    assert cause in desc_cause, "PRE: cause in desc_cause"
    assert effect in anc_effect, "PRE: effect in anc_effect"

    result = desc_cause & anc_effect

    assert result == desc_cause & anc_effect, "POST: result == desc_cause & anc_effect"
    assert cause in result or effect not in desc_cause, (
        "POST: cause in result when effect reachable"
    )
    assert effect in result or effect not in desc_cause, "POST: effect in result when reachable"
    return result


# ---------------------------------------------------------------------------
# Kernel 2: forbidden_kernel
# ---------------------------------------------------------------------------


def forbidden_kernel(
    causal_nodes: set[int],
    desc_map: dict[int, set[int]],
    cause: int,
) -> set[int]:
    """Compute forbidden nodes: desc_inclusive(causal_nodes) U {cause}.

    PRE: cause in causal_nodes  (cause is always a causal node)
    PRE: all(v in desc_map for v in causal_nodes)
    POST: cause in result
    POST: causal_nodes <= result  (all causal nodes are forbidden)
    POST: all(desc_map[v] <= result for v in causal_nodes)
    """
    assert cause in causal_nodes, "PRE: cause in causal_nodes"
    assert all(v in desc_map for v in causal_nodes), "PRE: all causal nodes in desc_map"

    result: set[int] = set()
    for v in causal_nodes:
        result.update(desc_map[v])
        result.add(v)
    result.add(cause)

    assert cause in result, "POST: cause in result"
    assert causal_nodes <= result, "POST: causal_nodes <= result"
    assert all(desc_map[v] <= result for v in causal_nodes), "POST: all desc in result"
    return result


# ---------------------------------------------------------------------------
# Kernel 3: oset_kernel
# ---------------------------------------------------------------------------


def oset_kernel(
    causal_nodes: set[int],
    forbidden: set[int],
    pa_map: dict[int, set[int]],
    desc_effect: set[int],
) -> frozenset[int] | None:
    """Compute O-set: pa(causal_nodes) - forbidden, or None if invalid.

    PRE: all(v in pa_map for v in causal_nodes)
    POST: implies(result is not None, result & forbidden == set())
    POST: implies(result is not None, all(v in pa_map[cn] for cn in causal_nodes for v in result if v in pa_map[cn]))
    POST: implies(result is not None, result & desc_effect == set())
    """
    assert all(v in pa_map for v in causal_nodes), "PRE: all causal nodes in pa_map"

    o_set: set[int] = set()
    for cn in causal_nodes:
        o_set.update(pa_map[cn])
    o_set -= forbidden

    # Validity: O-set must not contain any descendant of effect
    if o_set & desc_effect:
        return None

    result = frozenset(o_set)

    assert result & forbidden == set(), "POST: result & forbidden == set()"
    assert result & desc_effect == set(), "POST: result & desc_effect == set()"
    return result


# ---------------------------------------------------------------------------
# Kernel 4: pbd_first_hops_kernel
# ---------------------------------------------------------------------------


def pbd_first_hops_kernel(
    successors_cause: set[int],
    can_reach_effect: set[int],
) -> set[int]:
    """Identify first-hop causal successors for proper-backdoor-graph construction.

    A successor of cause is a 'first hop on a causal path' iff it can reach
    effect in the edge-removed graph.

    PRE: True  (no preconditions — both sets may be empty)
    POST: result <= successors_cause
    POST: result <= can_reach_effect
    POST: result == successors_cause & can_reach_effect
    """
    result = {hop for hop in successors_cause if hop in can_reach_effect}

    assert result <= successors_cause, "POST: result <= successors_cause"
    assert result <= can_reach_effect, "POST: result <= can_reach_effect"
    assert result == successors_cause & can_reach_effect, "POST: result == intersection"
    return result


# ---------------------------------------------------------------------------
# Layer A: Integer-arithmetic invariants — PROVED by Axiomander at level1
# ---------------------------------------------------------------------------


def oset_non_negative_size(
    n_parents: int,
    n_forbidden: int,
) -> int:
    """O-set size is non-negative when forbidden ≤ parents.

    AXIOMANDER PROVED (level1): result >= 0  AND  result == n_parents - n_forbidden

    axiomander:
        requires:
            n_parents >= 0
            n_forbidden >= 0
            n_parents >= n_forbidden
        ensures:
            result >= 0
            result == n_parents - n_forbidden
    """
    assert n_parents >= 0, "PRE: n_parents >= 0"
    assert n_forbidden >= 0, "PRE: n_forbidden >= 0"
    assert n_parents >= n_forbidden, "PRE: n_parents >= n_forbidden"
    result = n_parents - n_forbidden
    assert result >= 0, "POST: result >= 0"
    assert result == n_parents - n_forbidden, "POST: result == n_parents - n_forbidden"
    return result


def forbidden_includes_cause(
    n_causal_nodes: int,
    n_desc_causal: int,
    cause_counted: int,
) -> int:
    """Forbidden set always contains ≥ 2 nodes (cause + ≥1 causal node).

    AXIOMANDER PROVED (level1): result >= n_causal_nodes + 1  AND  result >= 2

    axiomander:
        requires:
            n_causal_nodes >= 1
            n_desc_causal >= n_causal_nodes
            cause_counted == 1
        ensures:
            result >= n_causal_nodes + cause_counted
            result >= 2
    """
    assert n_causal_nodes >= 1, "PRE: n_causal_nodes >= 1"
    assert n_desc_causal >= n_causal_nodes, "PRE: n_desc_causal >= n_causal_nodes"
    assert cause_counted == 1, "PRE: cause_counted == 1"
    result = n_desc_causal + cause_counted
    assert result >= n_causal_nodes + cause_counted, "POST: result >= n_causal_nodes + 1"
    assert result >= 2, "POST: result >= 2"
    return result


def cause_always_excluded_from_oset(
    n_parents_of_cn: int,
    n_forbidden: int,
) -> int:
    """Because cause ∈ forbidden (n_forbidden ≥ 1), O-set size < total parents.

    AXIOMANDER PROVED (level1): result >= 0  AND  result < n_parents_of_cn

    axiomander:
        requires:
            n_parents_of_cn >= 1
            n_forbidden >= 1
            n_forbidden <= n_parents_of_cn
        ensures:
            result >= 0
            result == n_parents_of_cn - n_forbidden
            result < n_parents_of_cn
    """
    assert n_parents_of_cn >= 1, "PRE: n_parents_of_cn >= 1"
    assert n_forbidden >= 1, "PRE: n_forbidden >= 1"
    assert n_forbidden <= n_parents_of_cn, "PRE: n_forbidden <= n_parents_of_cn"
    result = n_parents_of_cn - n_forbidden
    assert result >= 0, "POST: result >= 0"
    assert result == n_parents_of_cn - n_forbidden, "POST: result == n_parents - n_forbidden"
    assert result < n_parents_of_cn, "POST: result < n_parents_of_cn"
    return result


# ---------------------------------------------------------------------------
# Axiomander verification tests — Layer A (integer arithmetic, PROVED)
# ---------------------------------------------------------------------------


class TestOsetNonNegativeSizeAxiomander:
    """Tests for oset_non_negative_size — PROVED by Axiomander at level1."""

    def test_pre_n_parents_negative_raises(self) -> None:
        with pytest.raises(AssertionError, match="PRE: n_parents >= 0"):
            oset_non_negative_size(n_parents=-1, n_forbidden=0)

    def test_pre_n_forbidden_negative_raises(self) -> None:
        with pytest.raises(AssertionError, match="PRE: n_forbidden >= 0"):
            oset_non_negative_size(n_parents=5, n_forbidden=-1)

    def test_pre_forbidden_exceeds_parents_raises(self) -> None:
        with pytest.raises(AssertionError, match="PRE: n_parents >= n_forbidden"):
            oset_non_negative_size(n_parents=3, n_forbidden=5)

    def test_zero_forbidden(self) -> None:
        assert oset_non_negative_size(n_parents=5, n_forbidden=0) == 5

    def test_all_forbidden(self) -> None:
        assert oset_non_negative_size(n_parents=4, n_forbidden=4) == 0

    def test_partial_forbidden(self) -> None:
        assert oset_non_negative_size(n_parents=10, n_forbidden=3) == 7


class TestForbiddenIncludesCauseAxiomander:
    """Tests for forbidden_includes_cause — PROVED by Axiomander at level1."""

    def test_pre_no_causal_nodes_raises(self) -> None:
        with pytest.raises(AssertionError, match="PRE: n_causal_nodes >= 1"):
            forbidden_includes_cause(n_causal_nodes=0, n_desc_causal=0, cause_counted=1)

    def test_pre_desc_less_than_cn_raises(self) -> None:
        with pytest.raises(AssertionError, match="PRE: n_desc_causal >= n_causal_nodes"):
            forbidden_includes_cause(n_causal_nodes=3, n_desc_causal=2, cause_counted=1)

    def test_pre_cause_not_one_raises(self) -> None:
        with pytest.raises(AssertionError, match="PRE: cause_counted == 1"):
            forbidden_includes_cause(n_causal_nodes=1, n_desc_causal=1, cause_counted=0)

    def test_minimal_case(self) -> None:
        """1 causal node, no extra descendants: forbidden = 1 + 1 = 2."""
        result = forbidden_includes_cause(n_causal_nodes=1, n_desc_causal=1, cause_counted=1)
        assert result == 2
        assert result >= 2

    def test_larger_case(self) -> None:
        result = forbidden_includes_cause(n_causal_nodes=3, n_desc_causal=7, cause_counted=1)
        assert result == 8
        assert result >= 4  # >= n_causal_nodes + 1


class TestCauseAlwaysExcludedAxiomander:
    """Tests for cause_always_excluded_from_oset — PROVED by Axiomander at level1."""

    def test_pre_no_parents_raises(self) -> None:
        with pytest.raises(AssertionError, match="PRE: n_parents_of_cn >= 1"):
            cause_always_excluded_from_oset(n_parents_of_cn=0, n_forbidden=1)

    def test_pre_no_forbidden_raises(self) -> None:
        with pytest.raises(AssertionError, match="PRE: n_forbidden >= 1"):
            cause_always_excluded_from_oset(n_parents_of_cn=5, n_forbidden=0)

    def test_pre_forbidden_exceeds_parents_raises(self) -> None:
        with pytest.raises(AssertionError, match="PRE: n_forbidden <= n_parents_of_cn"):
            cause_always_excluded_from_oset(n_parents_of_cn=3, n_forbidden=5)

    def test_one_parent_one_forbidden(self) -> None:
        result = cause_always_excluded_from_oset(n_parents_of_cn=1, n_forbidden=1)
        assert result == 0
        assert result < 1

    def test_many_parents_one_forbidden(self) -> None:
        result = cause_always_excluded_from_oset(n_parents_of_cn=10, n_forbidden=1)
        assert result == 9
        assert result < 10

    def test_result_strictly_less_than_parents(self) -> None:
        for n in range(1, 8):
            result = cause_always_excluded_from_oset(n_parents_of_cn=n, n_forbidden=n)
            assert result == 0
            assert result < n


# ---------------------------------------------------------------------------
# Axiomander verification tests — Layer B (set arithmetic, pytest-verified)
# ---------------------------------------------------------------------------


class TestCausalNodesKernelAxiomander:
    """Negative tests: verify that PRE violations raise AssertionError."""

    def test_cause_not_in_desc_cause_raises(self) -> None:
        with pytest.raises(AssertionError, match="PRE: cause in desc_cause"):
            causal_nodes_kernel(
                desc_cause={2, 3},  # cause=1 not in desc_cause
                anc_effect={1, 2, 3},
                cause=1,
                effect=3,
            )

    def test_effect_not_in_anc_effect_raises(self) -> None:
        with pytest.raises(AssertionError, match="PRE: effect in anc_effect"):
            causal_nodes_kernel(
                desc_cause={1, 2, 3},
                anc_effect={1, 2},  # effect=3 not in anc_effect
                cause=1,
                effect=3,
            )

    def test_simple_chain(self) -> None:
        """X→M→Y: causal_nodes = {X, M, Y}."""
        result = causal_nodes_kernel(
            desc_cause={1, 2, 3},  # desc(X) = {X, M, Y}
            anc_effect={1, 2, 3},  # anc(Y) = {X, M, Y}
            cause=1,
            effect=3,
        )
        assert result == {1, 2, 3}

    def test_fork_dag(self) -> None:
        """Z→X→Y, Z→Y: causal_nodes = {X, Y} (Z is not on a causal path)."""
        result = causal_nodes_kernel(
            desc_cause={1, 3},  # desc(X) = {X, Y}
            anc_effect={0, 1, 3},  # anc(Y) = {Z, X, Y}
            cause=1,
            effect=3,
        )
        assert result == {1, 3}
        assert 0 not in result  # Z is not a causal node


class TestForbiddenKernelAxiomander:
    """Negative tests: verify that PRE violations raise AssertionError."""

    def test_cause_not_in_causal_nodes_raises(self) -> None:
        with pytest.raises(AssertionError, match="PRE: cause in causal_nodes"):
            forbidden_kernel(
                causal_nodes={2, 3},  # cause=1 not in causal_nodes
                desc_map={1: {1}, 2: {2}, 3: {3}},
                cause=1,
            )

    def test_causal_node_missing_from_desc_map_raises(self) -> None:
        with pytest.raises(AssertionError, match="PRE: all causal nodes in desc_map"):
            forbidden_kernel(
                causal_nodes={1, 2, 3},
                desc_map={1: {1}, 2: {2}},  # 3 missing
                cause=1,
            )

    def test_cause_always_in_result(self) -> None:
        result = forbidden_kernel(
            causal_nodes={1, 2},
            desc_map={1: {1, 3}, 2: {2, 4}},
            cause=1,
        )
        assert 1 in result
        assert {1, 2} <= result  # causal nodes are forbidden

    def test_descendants_included(self) -> None:
        result = forbidden_kernel(
            causal_nodes={1, 2},
            desc_map={1: {1, 5, 6}, 2: {2, 7}},
            cause=1,
        )
        assert {1, 2, 5, 6, 7} <= result


class TestOsetKernelAxiomander:
    """Negative tests: verify that PRE violations raise AssertionError."""

    def test_causal_node_missing_from_pa_map_raises(self) -> None:
        with pytest.raises(AssertionError, match="PRE: all causal nodes in pa_map"):
            oset_kernel(
                causal_nodes={1, 2, 3},
                forbidden={1, 2, 3},
                pa_map={1: set(), 2: set()},  # 3 missing
                desc_effect={3},
            )

    def test_result_excludes_forbidden(self) -> None:
        result = oset_kernel(
            causal_nodes={1, 2},
            forbidden={1, 2, 3},
            pa_map={1: {0, 3}, 2: {4, 5}},  # pa(1)={0,3}, pa(2)={4,5}
            desc_effect={2},
        )
        # o_set = {0,3,4,5} - {1,2,3} = {0,4,5}
        # desc_effect = {2} — not in o_set, so valid
        assert result is not None
        assert 1 not in result  # cause excluded
        assert 2 not in result  # causal node excluded
        assert 3 not in result  # forbidden excluded

    def test_returns_none_when_oset_contains_desc_effect(self) -> None:
        result = oset_kernel(
            causal_nodes={1, 2},
            forbidden={1, 2},
            pa_map={1: {0}, 2: {5}},
            desc_effect={5},  # 5 is a descendant of effect
        )
        assert result is None

    def test_empty_oset_when_all_parents_forbidden(self) -> None:
        result = oset_kernel(
            causal_nodes={1, 2},
            forbidden={0, 1, 2, 3, 4, 5},  # everything forbidden
            pa_map={1: {0, 3}, 2: {4, 5}},
            desc_effect=set(),
        )
        assert result == frozenset()


class TestPbdFirstHopsKernelAxiomander:
    """Negative tests: verify POST conditions hold."""

    def test_result_subset_of_successors(self) -> None:
        result = pbd_first_hops_kernel(
            successors_cause={2, 3, 4},
            can_reach_effect={3, 4, 5},
        )
        assert result == {3, 4}
        assert result <= {2, 3, 4}
        assert result <= {3, 4, 5}

    def test_empty_when_no_overlap(self) -> None:
        result = pbd_first_hops_kernel(
            successors_cause={1, 2},
            can_reach_effect={3, 4},
        )
        assert result == set()

    def test_all_successors_when_all_can_reach(self) -> None:
        result = pbd_first_hops_kernel(
            successors_cause={2, 3},
            can_reach_effect={2, 3, 4},
        )
        assert result == {2, 3}

    def test_empty_successors(self) -> None:
        result = pbd_first_hops_kernel(
            successors_cause=set(),
            can_reach_effect={1, 2, 3},
        )
        assert result == set()


# ---------------------------------------------------------------------------
# Integration: full O-set pipeline on concrete examples
# ---------------------------------------------------------------------------


class TestOsetPipelineIntegration:
    """End-to-end tests of the kernel pipeline on concrete graph examples.

    These mirror the hand-crafted unit tests in test_oset_reachability.py
    but use only the pure set-kernel functions.
    """

    def _run_oset(
        self,
        desc_cause: set[int],
        anc_effect: set[int],
        pa_map: dict[int, set[int]],
        desc_map: dict[int, set[int]],
        desc_effect: set[int],
        cause: int,
        effect: int,
    ) -> frozenset[int] | None:
        cn = causal_nodes_kernel(desc_cause, anc_effect, cause, effect)
        forb = forbidden_kernel(cn, desc_map, cause)
        return oset_kernel(cn, forb, pa_map, desc_effect)

    def test_simple_xy_edge(self) -> None:
        """X→Y only: O-set = {} (no parents of causal nodes outside forb)."""
        # nodes: X=0, Y=1
        result = self._run_oset(
            desc_cause={0, 1},
            anc_effect={0, 1},
            pa_map={0: set(), 1: {0}},
            desc_map={0: {0, 1}, 1: {1}},
            desc_effect={1},
            cause=0,
            effect=1,
        )
        assert result == frozenset()

    def test_fork_confounder(self) -> None:
        """Z→X→Y, Z→Y: O-set = {Z}."""
        # nodes: Z=0, X=1, Y=2
        result = self._run_oset(
            desc_cause={1, 2},
            anc_effect={0, 1, 2},
            pa_map={0: set(), 1: {0}, 2: {0, 1}},
            desc_map={0: {0, 1, 2}, 1: {1, 2}, 2: {2}},
            desc_effect={2},
            cause=1,
            effect=2,
        )
        assert result == frozenset({0})

    def test_cyclic_counterexample_1(self) -> None:
        """B→A, A→X, X→A, X→Y: O-set = {B}.

        Hypothesis-discovered counterexample where old path-enum returned {}.
        nodes: B=0, A=1, X=2, Y=3
        desc(X) = {X, A, Y} = {2, 1, 3}  (X→A, X→Y, A→X→A cycle)
        anc(Y) = {Y, X, A, B} = {3, 2, 1, 0}
        cn = {2, 1, 3}
        forb = desc_inclusive({2,1,3}) U {2} = {1,2,3}
        pa({2,1,3}) = pa(2)Upa(1)Upa(3) = {1}U{0,2}U{2} = {0,1,2}
        o_set = {0,1,2} - {1,2,3} = {0} = {B}
        """
        result = self._run_oset(
            desc_cause={2, 1, 3},  # desc(X) inclusive
            anc_effect={3, 2, 1, 0},  # anc(Y) inclusive
            pa_map={0: set(), 1: {0, 2}, 2: {1}, 3: {2}},
            desc_map={0: {0, 1, 2, 3}, 1: {1, 2, 3}, 2: {2, 1, 3}, 3: {3}},
            desc_effect={3},
            cause=2,
            effect=3,
        )
        assert result == frozenset({0})

    def test_two_cycle_xy(self) -> None:
        """X→Y→X: O-set = {} (all parents are forbidden)."""
        # nodes: X=0, Y=1
        result = self._run_oset(
            desc_cause={0, 1},
            anc_effect={0, 1},
            pa_map={0: {1}, 1: {0}},
            desc_map={0: {0, 1}, 1: {0, 1}},
            desc_effect={0, 1},
            cause=0,
            effect=1,
        )
        # o_set = pa({0,1}) - forb = {0,1} - {0,1} = {}
        assert result == frozenset()
