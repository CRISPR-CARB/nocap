"""
test_scc_perturb_axiomander.py
==============================
Axiomander-style PRE/POST contract tests for the SCC-perturbation pipeline.

All tests are injectable — no real graph file, no y0, no SLURM.

Toy network used throughout:

    A → B → C → A          (non-trivial SCC: {A, B, C})
    A → D                   (D is outside the SCC, a descendant of A)
    B → E                   (E is outside the SCC, a descendant of B)

For TF = A:
  - in_scc_children = [B]   (A→B is the only direct edge A→SCC-member)
  - Return paths: B→C→A, so we need to cut all B⇝A paths inside the SCC.
  - Minimum cut (excluding A and B): {C}  (removing do(C) severs B→C→A)
  - Post-intervention descendants of A: D, E  (C is removed from the return
    path but is still a descendant; B→E so E is reachable)

For TF = B (not tested in all fixtures, but verified in basic tests):
  - in_scc_children = [C]
  - Minimum cut (excluding B and C): {A} (B→C→A→B, cutting A severs C→A→B)
"""

import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from scc_perturb_prepare import compute_min_cut_b, find_in_scc_children
from scc_perturb_worker import (
    build_intervened_graph,
    get_descendants,
    run_joint_cyclic_id,
    run_per_gene_cyclic_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def toy_graph():
    """
    Build the toy DiGraph:
      A→B, B→C, C→A  (SCC = {A,B,C})
      A→D, B→E        (DAG leaves)
    """
    g = nx.DiGraph()
    g.add_edges_from([
        ("A", "B"), ("B", "C"), ("C", "A"),  # cycle
        ("A", "D"), ("B", "E"),               # out-of-SCC descendants
    ])
    return g


@pytest.fixture()
def scc_abc():
    return frozenset({"A", "B", "C"})


@pytest.fixture()
def in_scc_children_a(toy_graph, scc_abc):
    return find_in_scc_children("A", scc_abc, toy_graph)


@pytest.fixture()
def min_cut_a(toy_graph, scc_abc, in_scc_children_a):
    return compute_min_cut_b("A", scc_abc, in_scc_children_a, toy_graph)


@pytest.fixture()
def intervened_graph_a(toy_graph, min_cut_a):
    return build_intervened_graph(toy_graph, min_cut_a)


# ---------------------------------------------------------------------------
# TestFindInSccChildren
# ---------------------------------------------------------------------------


class TestFindInSccChildren:
    def test_a_has_b_as_in_scc_child(self, toy_graph, scc_abc):
        children = find_in_scc_children("A", scc_abc, toy_graph)
        assert "B" in children

    def test_a_does_not_have_d_as_in_scc_child(self, toy_graph, scc_abc):
        children = find_in_scc_children("A", scc_abc, toy_graph)
        assert "D" not in children

    def test_no_self_loops_in_result(self, toy_graph, scc_abc):
        children = find_in_scc_children("A", scc_abc, toy_graph)
        assert "A" not in children

    def test_result_is_list(self, toy_graph, scc_abc):
        children = find_in_scc_children("A", scc_abc, toy_graph)
        assert isinstance(children, list)

    def test_all_children_in_scc(self, toy_graph, scc_abc):
        children = find_in_scc_children("A", scc_abc, toy_graph)
        assert all(c in scc_abc for c in children)

    def test_node_with_no_scc_children(self, toy_graph):
        """D has no outgoing edges into any SCC."""
        children = find_in_scc_children("D", frozenset({"D"}), toy_graph)
        assert children == []

    def test_pre_tf_must_be_str(self, toy_graph, scc_abc):
        with pytest.raises(AssertionError, match="PRE: tf must be a str"):
            find_in_scc_children(123, scc_abc, toy_graph)

    def test_pre_scc_nodes_must_be_set(self, toy_graph):
        with pytest.raises(AssertionError, match="PRE: scc_nodes must be a set or frozenset"):
            find_in_scc_children("A", ["A", "B", "C"], toy_graph)


# ---------------------------------------------------------------------------
# TestComputeMinCutB
# ---------------------------------------------------------------------------


class TestComputeMinCutB:
    def test_cut_excludes_tf(self, min_cut_a):
        assert "A" not in min_cut_a

    def test_cut_excludes_direct_children(self, min_cut_a, in_scc_children_a):
        for c in in_scc_children_a:
            assert c not in min_cut_a

    def test_cut_nodes_are_in_scc(self, min_cut_a, scc_abc):
        assert all(n in scc_abc for n in min_cut_a)

    def test_cut_is_list(self, min_cut_a):
        assert isinstance(min_cut_a, list)

    def test_cut_is_nonempty_for_a(self, min_cut_a):
        """A is in a 3-cycle so B(A) should be non-empty (C severs B→C→A)."""
        assert len(min_cut_a) >= 1

    def test_cut_for_b(self, toy_graph, scc_abc):
        """For TF=B: in-SCC child is C; min cut severs C→A→B, should contain A."""
        children_b = find_in_scc_children("B", scc_abc, toy_graph)
        assert "C" in children_b
        cut_b = compute_min_cut_b("B", scc_abc, children_b, toy_graph)
        assert "B" not in cut_b
        assert "C" not in cut_b
        assert all(n in scc_abc for n in cut_b)

    def test_empty_in_scc_children_gives_empty_cut(self, toy_graph, scc_abc):
        cut = compute_min_cut_b("A", scc_abc, [], toy_graph)
        assert cut == []

    def test_trivial_scc_gives_empty_cut(self, toy_graph):
        cut = compute_min_cut_b("D", frozenset({"D"}), ["E"], toy_graph)
        assert cut == []

    def test_cut_severs_all_return_paths(self, toy_graph, scc_abc, min_cut_a, in_scc_children_a):
        """After removing B(A) from the graph, no child of A should reach A."""
        intervened = build_intervened_graph(toy_graph, min_cut_a)
        for c in in_scc_children_a:
            reachable_from_c = nx.descendants(intervened, c) if c in intervened else set()
            assert "A" not in reachable_from_c, (
                f"After do(B(A)), child {c} can still reach A via {reachable_from_c}"
            )

    def test_pre_tf_must_be_str(self, toy_graph, scc_abc):
        with pytest.raises(AssertionError, match="PRE: tf must be a str"):
            compute_min_cut_b(42, scc_abc, [], toy_graph)

    def test_pre_scc_nodes_must_be_set(self, toy_graph):
        with pytest.raises(AssertionError, match="PRE: scc_nodes must be a set or frozenset"):
            compute_min_cut_b("A", ["A", "B", "C"], [], toy_graph)

    def test_pre_in_scc_children_must_be_list(self, toy_graph, scc_abc):
        with pytest.raises(AssertionError, match="PRE: in_scc_children must be a list"):
            compute_min_cut_b("A", scc_abc, {"B"}, toy_graph)


# ---------------------------------------------------------------------------
# TestBuildIntervenedGraph
# ---------------------------------------------------------------------------


class TestBuildIntervenedGraph:
    def test_cut_nodes_have_zero_in_degree(self, toy_graph, min_cut_a, intervened_graph_a):
        for n in min_cut_a:
            assert intervened_graph_a.in_degree(n) == 0, (
                f"Node {n} should have in-degree 0 after intervention"
            )

    def test_original_graph_not_mutated(self, toy_graph, min_cut_a):
        original_edges = set(toy_graph.edges())
        _ = build_intervened_graph(toy_graph, min_cut_a)
        assert set(toy_graph.edges()) == original_edges

    def test_empty_cut_leaves_graph_unchanged(self, toy_graph):
        intervened = build_intervened_graph(toy_graph, [])
        assert set(intervened.edges()) == set(toy_graph.edges())

    def test_intervened_is_different_object(self, toy_graph, min_cut_a):
        intervened = build_intervened_graph(toy_graph, min_cut_a)
        assert intervened is not toy_graph

    def test_pre_min_cut_must_be_list(self, toy_graph):
        with pytest.raises(AssertionError, match="PRE: min_cut must be a list"):
            build_intervened_graph(toy_graph, {"C"})


# ---------------------------------------------------------------------------
# TestGetDescendants
# ---------------------------------------------------------------------------


class TestGetDescendants:
    def test_a_descendants_include_d_and_e(self, intervened_graph_a):
        desc = get_descendants("A", intervened_graph_a)
        # D is A→D, E is A→B→E (B is still reachable from A)
        assert "D" in desc
        assert "E" in desc

    def test_a_not_in_own_descendants(self, intervened_graph_a):
        desc = get_descendants("A", intervened_graph_a)
        assert "A" not in desc

    def test_result_is_sorted_list(self, intervened_graph_a):
        desc = get_descendants("A", intervened_graph_a)
        assert isinstance(desc, list)
        assert desc == sorted(desc)

    def test_unknown_node_returns_empty(self, intervened_graph_a):
        desc = get_descendants("Z", intervened_graph_a)
        assert desc == []

    def test_leaf_node_has_no_descendants(self, toy_graph):
        intervened = build_intervened_graph(toy_graph, [])
        desc = get_descendants("D", intervened)
        assert desc == []

    def test_pre_tf_must_be_str(self, toy_graph):
        with pytest.raises(AssertionError, match="PRE: tf must be a str"):
            get_descendants(123, toy_graph)


# ---------------------------------------------------------------------------
# TestRunJointCyclicId  (injectable identify_fn)
# ---------------------------------------------------------------------------


class TestRunJointCyclicId:
    def _always_true(self, tf, outcome_set, min_cut):
        return True

    def _always_false(self, tf, outcome_set, min_cut):
        return False

    def _check_inputs(self, tf, outcome_set, min_cut):
        """Injectable that validates the inputs it receives."""
        assert isinstance(tf, str)
        assert isinstance(outcome_set, (set, frozenset))
        assert isinstance(min_cut, list)
        return True

    def test_injectable_true(self):
        result = run_joint_cyclic_id(
            tf="A",
            outcome_set=frozenset({"D", "E"}),
            min_cut=["C"],
            ecoli_mixed=None,
            apt_order=None,
            all_network_vars=set(),
            identify_fn=self._always_true,
        )
        assert result is True

    def test_injectable_false(self):
        result = run_joint_cyclic_id(
            tf="A",
            outcome_set=frozenset({"D", "E"}),
            min_cut=["C"],
            ecoli_mixed=None,
            apt_order=None,
            all_network_vars=set(),
            identify_fn=self._always_false,
        )
        assert result is False

    def test_injectable_receives_correct_types(self):
        run_joint_cyclic_id(
            tf="A",
            outcome_set=frozenset({"D"}),
            min_cut=["C"],
            ecoli_mixed=None,
            apt_order=None,
            all_network_vars=set(),
            identify_fn=self._check_inputs,
        )

    def test_result_is_bool(self):
        result = run_joint_cyclic_id(
            tf="A",
            outcome_set=frozenset(),
            min_cut=[],
            ecoli_mixed=None,
            apt_order=None,
            all_network_vars=set(),
            identify_fn=self._always_true,
        )
        assert isinstance(result, bool)

    def test_pre_tf_must_be_str(self):
        with pytest.raises(AssertionError, match="PRE: tf must be a str"):
            run_joint_cyclic_id(
                tf=42,
                outcome_set=frozenset(),
                min_cut=[],
                ecoli_mixed=None,
                apt_order=None,
                all_network_vars=set(),
                identify_fn=self._always_true,
            )

    def test_pre_outcome_set_must_be_set(self):
        with pytest.raises(AssertionError, match="PRE: outcome_set must be a set or frozenset"):
            run_joint_cyclic_id(
                tf="A",
                outcome_set=["D"],
                min_cut=[],
                ecoli_mixed=None,
                apt_order=None,
                all_network_vars=set(),
                identify_fn=self._always_true,
            )

    def test_pre_min_cut_must_be_list(self):
        with pytest.raises(AssertionError, match="PRE: min_cut must be a list"):
            run_joint_cyclic_id(
                tf="A",
                outcome_set=frozenset({"D"}),
                min_cut={"C"},
                ecoli_mixed=None,
                apt_order=None,
                all_network_vars=set(),
                identify_fn=self._always_true,
            )

    def test_injectable_non_bool_fires_post(self):
        def bad_fn(tf, outcome_set, min_cut):
            return "yes"  # not bool

        with pytest.raises(AssertionError, match="POST: identify_fn must return bool"):
            run_joint_cyclic_id(
                tf="A",
                outcome_set=frozenset({"D"}),
                min_cut=[],
                ecoli_mixed=None,
                apt_order=None,
                all_network_vars=set(),
                identify_fn=bad_fn,
            )


# ---------------------------------------------------------------------------
# TestRunPerGeneCyclicId  (injectable identify_fn)
# ---------------------------------------------------------------------------


class TestRunPerGeneCyclicId:
    """
    Tests for the per-gene fallback path.

    We use a tiny mock Variable-like object so we don't import y0.
    """

    class FakeVar:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return self.name

    def _always_true(self, tf, outcome_set, min_cut):
        return True

    def _always_false(self, tf, outcome_set, min_cut):
        return False

    def test_returns_dict_keyed_by_gene(self):
        vars_ = [self.FakeVar("D"), self.FakeVar("E")]
        result = run_per_gene_cyclic_id(
            tf="A",
            outcome_vars=vars_,
            min_cut=["C"],
            ecoli_mixed=None,
            apt_order=None,
            all_network_vars=set(),
            identify_fn=self._always_true,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"D", "E"}

    def test_all_true_when_injectable_returns_true(self):
        vars_ = [self.FakeVar("D"), self.FakeVar("E")]
        result = run_per_gene_cyclic_id(
            tf="A",
            outcome_vars=vars_,
            min_cut=[],
            ecoli_mixed=None,
            apt_order=None,
            all_network_vars=set(),
            identify_fn=self._always_true,
        )
        assert all(v is True for v in result.values())

    def test_all_false_when_injectable_returns_false(self):
        vars_ = [self.FakeVar("D"), self.FakeVar("E")]
        result = run_per_gene_cyclic_id(
            tf="A",
            outcome_vars=vars_,
            min_cut=[],
            ecoli_mixed=None,
            apt_order=None,
            all_network_vars=set(),
            identify_fn=self._always_false,
        )
        assert all(v is False for v in result.values())

    def test_length_matches_outcome_vars(self):
        vars_ = [self.FakeVar(f"gene_{i}") for i in range(5)]
        result = run_per_gene_cyclic_id(
            tf="A",
            outcome_vars=vars_,
            min_cut=[],
            ecoli_mixed=None,
            apt_order=None,
            all_network_vars=set(),
            identify_fn=self._always_true,
        )
        assert len(result) == 5

    def test_empty_outcome_vars_gives_empty_dict(self):
        result = run_per_gene_cyclic_id(
            tf="A",
            outcome_vars=[],
            min_cut=[],
            ecoli_mixed=None,
            apt_order=None,
            all_network_vars=set(),
            identify_fn=self._always_true,
        )
        assert result == {}

    def test_pre_tf_must_be_str(self):
        with pytest.raises(AssertionError, match="PRE: tf must be a str"):
            run_per_gene_cyclic_id(
                tf=99,
                outcome_vars=[],
                min_cut=[],
                ecoli_mixed=None,
                apt_order=None,
                all_network_vars=set(),
                identify_fn=self._always_true,
            )

    def test_pre_outcome_vars_must_be_list(self):
        with pytest.raises(AssertionError, match="PRE: outcome_vars must be a list"):
            run_per_gene_cyclic_id(
                tf="A",
                outcome_vars={self.FakeVar("D")},
                min_cut=[],
                ecoli_mixed=None,
                apt_order=None,
                all_network_vars=set(),
                identify_fn=self._always_true,
            )

    def test_pre_min_cut_must_be_list(self):
        with pytest.raises(AssertionError, match="PRE: min_cut must be a list"):
            run_per_gene_cyclic_id(
                tf="A",
                outcome_vars=[],
                min_cut={"C"},
                ecoli_mixed=None,
                apt_order=None,
                all_network_vars=set(),
                identify_fn=self._always_true,
            )


# ---------------------------------------------------------------------------
# TestEndToEndToyGraph  (integration, no y0)
# ---------------------------------------------------------------------------


class TestEndToEndToyGraph:
    """
    Verify the full per-TF pipeline on the toy graph:
      A→B→C→A (SCC), A→D, B→E

    Pipeline: find_in_scc_children → compute_min_cut_b → build_intervened_graph
              → get_descendants → run_joint_cyclic_id (injectable)
    """

    def test_pipeline_a_joint_identifiable(self, toy_graph, scc_abc):
        children = find_in_scc_children("A", scc_abc, toy_graph)
        cut = compute_min_cut_b("A", scc_abc, children, toy_graph)
        intervened = build_intervened_graph(toy_graph, cut)
        desc = get_descendants("A", intervened)
        assert len(desc) >= 1  # D and E must be reachable

        outcome_set = frozenset(desc)  # use plain str set for inject test
        result = run_joint_cyclic_id(
            tf="A",
            outcome_set=outcome_set,
            min_cut=cut,
            ecoli_mixed=None,
            apt_order=None,
            all_network_vars=set(),
            identify_fn=lambda tf, o, b: True,
        )
        assert result is True

    def test_cut_severs_in_scc_children_from_tf(self, toy_graph, scc_abc):
        """Core correctness: after do(B(t)), no in-SCC child can reach t."""
        children = find_in_scc_children("A", scc_abc, toy_graph)
        cut = compute_min_cut_b("A", scc_abc, children, toy_graph)
        intervened = build_intervened_graph(toy_graph, cut)
        for c in children:
            if c in intervened:
                reachable = nx.descendants(intervened, c)
                assert "A" not in reachable, (
                    f"Child {c} can still reach A after do({cut})"
                )

    def test_dag_tf_has_empty_in_scc_children(self, toy_graph):
        """D is a leaf — it has no in-SCC children (singleton SCC)."""
        singleton = frozenset({"D"})
        children = find_in_scc_children("D", singleton, toy_graph)
        assert children == []

    def test_b_pipeline(self, toy_graph, scc_abc):
        """Run the same pipeline for TF=B."""
        children = find_in_scc_children("B", scc_abc, toy_graph)
        cut = compute_min_cut_b("B", scc_abc, children, toy_graph)
        assert "B" not in cut
        for c in children:
            assert c not in cut
        intervened = build_intervened_graph(toy_graph, cut)
        for c in children:
            if c in intervened:
                reachable = nx.descendants(intervened, c)
                assert "B" not in reachable
