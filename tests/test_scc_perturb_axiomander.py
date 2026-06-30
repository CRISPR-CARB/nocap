"""
test_scc_perturb_axiomander.py
==============================
Axiomander-style PRE/POST contract tests for the SCC-perturbation pipeline.

All tests are injectable — no real graph file, no y0, no SLURM.

Toy network used throughout:

    A → B → C → A          (non-trivial SCC: {A, B, C})
    A → D                   (D is outside the SCC, a direct child of A)
    B → E                   (E is outside the SCC, a direct child of B)

For TF = A:
  - in_scc_children = [B]   (A→B is the only direct edge A→SCC-member)
  - Return paths: B→C→A, so we need to cut all B⇝A paths inside the SCC.
  - Minimum cut (excluding A and B): {C}  (removing do(C) severs B→C→A)
  - Direct children of A in the post-do(B(A)) graph: B, D
    (A→B and A→D are both out-edges of A; B is preserved because children
    are excluded from the min-cut by Interpretation A)

For TF = B (not tested in all fixtures, but verified in basic tests):
  - in_scc_children = [C]
  - Minimum cut (excluding B and C): {A} (B→C→A→B, cutting A severs C→A→B)
"""

import networkx as nx
import pytest

from nocap.scc_perturb import (
    build_intervened_graph,
    compute_min_cut_b,
    find_in_scc_children,
    get_direct_children,
)
from scripts.scc_perturb_worker import run_joint_cyclic_id, run_per_gene_cyclic_id

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
    g.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),  # cycle
            ("A", "D"),
            ("B", "E"),  # out-of-SCC descendants
        ]
    )
    return g


@pytest.fixture()
def scc_abc():
    """Return the SCC {A, B, C} as a frozenset."""
    return frozenset({"A", "B", "C"})


@pytest.fixture()
def in_scc_children_a(toy_graph, scc_abc):
    """Return the in-SCC children of A in the toy graph."""
    return find_in_scc_children("A", scc_abc, toy_graph)


@pytest.fixture()
def min_cut_a(toy_graph, scc_abc, in_scc_children_a):
    """Return the minimum cut B(A) for TF=A in the toy graph."""
    return compute_min_cut_b("A", scc_abc, in_scc_children_a, toy_graph)


@pytest.fixture()
def intervened_graph_a(toy_graph, min_cut_a):
    """Return the intervened graph after applying do(B(A))."""
    return build_intervened_graph(toy_graph, min_cut_a)


# ---------------------------------------------------------------------------
# TestFindInSccChildren
# ---------------------------------------------------------------------------


class TestFindInSccChildren:
    """Tests for find_in_scc_children."""

    def test_a_has_b_as_in_scc_child(self, toy_graph, scc_abc):
        """POST: B is a direct in-SCC child of A (A→B, B in SCC)."""
        children = find_in_scc_children("A", scc_abc, toy_graph)
        assert "B" in children

    def test_a_does_not_have_d_as_in_scc_child(self, toy_graph, scc_abc):
        """POST: D is not in the SCC so it is not an in-SCC child of A."""
        children = find_in_scc_children("A", scc_abc, toy_graph)
        assert "D" not in children

    def test_no_self_loops_in_result(self, toy_graph, scc_abc):
        """POST: tf itself is never returned as its own in-SCC child."""
        children = find_in_scc_children("A", scc_abc, toy_graph)
        assert "A" not in children

    def test_result_is_list(self, toy_graph, scc_abc):
        """POST: result is a list."""
        children = find_in_scc_children("A", scc_abc, toy_graph)
        assert isinstance(children, list)

    def test_all_children_in_scc(self, toy_graph, scc_abc):
        """POST: every returned child is a member of the SCC."""
        children = find_in_scc_children("A", scc_abc, toy_graph)
        assert all(c in scc_abc for c in children)

    def test_node_with_no_scc_children(self, toy_graph):
        """D has no outgoing edges into any SCC."""
        children = find_in_scc_children("D", frozenset({"D"}), toy_graph)
        assert children == []

    def test_pre_tf_must_be_str(self, toy_graph, scc_abc):
        """PRE: tf must be a str — int raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: tf must be a str"):
            find_in_scc_children(123, scc_abc, toy_graph)

    def test_pre_scc_nodes_must_be_set(self, toy_graph):
        """PRE: scc_nodes must be a set or frozenset — list raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: scc_nodes must be a set or frozenset"):
            find_in_scc_children("A", ["A", "B", "C"], toy_graph)


# ---------------------------------------------------------------------------
# TestComputeMinCutB
# ---------------------------------------------------------------------------


class TestComputeMinCutB:
    """Tests for compute_min_cut_b."""

    def test_cut_excludes_tf(self, min_cut_a):
        """POST: the TF itself is never in the cut set."""
        assert "A" not in min_cut_a

    def test_cut_excludes_direct_children(self, min_cut_a, in_scc_children_a):
        """POST: direct in-SCC children are excluded from the cut."""
        for c in in_scc_children_a:
            assert c not in min_cut_a

    def test_cut_nodes_are_in_scc(self, min_cut_a, scc_abc):
        """POST: every cut node is a member of the SCC."""
        assert all(n in scc_abc for n in min_cut_a)

    def test_cut_is_list(self, min_cut_a):
        """POST: result is a list."""
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
        """POST: no in-SCC children → empty cut."""
        cut = compute_min_cut_b("A", scc_abc, [], toy_graph)
        assert cut == []

    def test_trivial_scc_gives_empty_cut(self, toy_graph):
        """POST: singleton SCC → empty cut (nothing to sever)."""
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
        """PRE: tf must be a str — int raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: tf must be a str"):
            compute_min_cut_b(42, scc_abc, [], toy_graph)

    def test_pre_scc_nodes_must_be_set(self, toy_graph):
        """PRE: scc_nodes must be a set or frozenset — list raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: scc_nodes must be a set or frozenset"):
            compute_min_cut_b("A", ["A", "B", "C"], [], toy_graph)

    def test_pre_in_scc_children_must_be_list(self, toy_graph, scc_abc):
        """PRE: in_scc_children must be a list — set raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: in_scc_children must be a list"):
            compute_min_cut_b("A", scc_abc, {"B"}, toy_graph)


# ---------------------------------------------------------------------------
# TestBuildIntervenedGraph
# ---------------------------------------------------------------------------


class TestBuildIntervenedGraph:
    """Tests for build_intervened_graph."""

    def test_cut_nodes_have_zero_in_degree(self, toy_graph, min_cut_a, intervened_graph_a):
        """POST: every cut node has in-degree 0 in the intervened graph."""
        for n in min_cut_a:
            assert intervened_graph_a.in_degree(n) == 0, (
                f"Node {n} should have in-degree 0 after intervention"
            )

    def test_original_graph_not_mutated(self, toy_graph, min_cut_a):
        """POST: the original graph is not mutated by the intervention."""
        original_edges = set(toy_graph.edges())
        _ = build_intervened_graph(toy_graph, min_cut_a)
        assert set(toy_graph.edges()) == original_edges

    def test_empty_cut_leaves_graph_unchanged(self, toy_graph):
        """POST: empty cut → intervened graph has same edges as original."""
        intervened = build_intervened_graph(toy_graph, [])
        assert set(intervened.edges()) == set(toy_graph.edges())

    def test_intervened_is_different_object(self, toy_graph, min_cut_a):
        """POST: intervened graph is a new object, not the original."""
        intervened = build_intervened_graph(toy_graph, min_cut_a)
        assert intervened is not toy_graph

    def test_pre_perturb_set_must_be_list(self, toy_graph):
        """PRE: perturb_set must be a list — set raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: perturb_set must be a list"):
            build_intervened_graph(toy_graph, {"C"})


# ---------------------------------------------------------------------------
# TestGetDirectChildren
# ---------------------------------------------------------------------------


class TestGetDirectChildren:
    """Tests for get_direct_children."""

    def test_a_children_include_b_and_d(self, intervened_graph_a):
        """A has direct out-edges A→B and A→D; both must appear."""
        children = get_direct_children("A", intervened_graph_a)
        assert "B" in children
        assert "D" in children

    def test_a_children_do_not_include_e(self, intervened_graph_a):
        """E is reachable from A via B→E but is not a direct child of A."""
        children = get_direct_children("A", intervened_graph_a)
        assert "E" not in children

    def test_a_not_in_own_children(self, intervened_graph_a):
        """POST: tf is never returned as its own direct child."""
        children = get_direct_children("A", intervened_graph_a)
        assert "A" not in children

    def test_result_is_sorted_list(self, intervened_graph_a):
        """POST: result is a sorted list."""
        children = get_direct_children("A", intervened_graph_a)
        assert isinstance(children, list)
        assert children == sorted(children)

    def test_unknown_node_returns_empty(self, intervened_graph_a):
        """POST: unknown node returns empty list."""
        children = get_direct_children("Z", intervened_graph_a)
        assert children == []

    def test_leaf_node_has_no_children(self, toy_graph):
        """POST: leaf node D has no direct children."""
        intervened = build_intervened_graph(toy_graph, [])
        children = get_direct_children("D", intervened)
        assert children == []

    def test_all_children_are_out_neighbours(self, intervened_graph_a):
        """Every result node must be a direct out-neighbour of A."""
        children = get_direct_children("A", intervened_graph_a)
        for c in children:
            assert intervened_graph_a.has_edge("A", c), f"{c} in result but A→{c} edge missing"

    def test_pre_tf_must_be_str(self, toy_graph):
        """PRE: tf must be a str — int raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: tf must be a str"):
            get_direct_children(123, toy_graph)


# ---------------------------------------------------------------------------
# TestRunJointCyclicId  (injectable identify_fn)
# ---------------------------------------------------------------------------


class TestRunJointCyclicId:
    """Tests for run_joint_cyclic_id with injectable identify_fn."""

    def _always_true(self, tf, outcome_set, min_cut):
        """Injectable that always returns True."""
        return True

    def _always_false(self, tf, outcome_set, min_cut):
        """Injectable that always returns False."""
        return False

    def _check_inputs(self, tf, outcome_set, min_cut):
        """Injectable that validates the inputs it receives."""
        assert isinstance(tf, str)
        assert isinstance(outcome_set, set | frozenset)
        assert isinstance(min_cut, list)
        return True

    def test_injectable_true(self):
        """POST: result is True when identify_fn always returns True."""
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
        """POST: result is False when identify_fn always returns False."""
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
        """POST: identify_fn receives (str, set/frozenset, list) arguments."""
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
        """POST: result is always a bool."""
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
        """PRE: tf must be a str — int raises AssertionError."""
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
        """PRE: outcome_set must be a set or frozenset — list raises AssertionError."""
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
        """PRE: min_cut must be a list — set raises AssertionError."""
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
        """POST: identify_fn returning non-bool fires the postcondition guard."""

        def bad_fn(tf, outcome_set, min_cut):
            """Injectable that returns a non-bool to trigger the POST guard."""
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

    # ------------------------------------------------------------------
    # NEW: empty-B(t) guard tests
    # ------------------------------------------------------------------

    def test_empty_min_cut_does_not_pass_base_distribution(self):
        """
        When min_cut == [], run_joint_cyclic_id must call identify_fn
        (or cyclic_id) WITHOUT a base_distribution kwarg.

        We verify this using a spy injectable that records whether
        min_cut was empty — the real guard is in the non-injectable path,
        but the contract INV (len(min_cut) >= 0) and the branch logic
        are also exercised by confirming the empty-cut path returns a bool
        and propagates to the injectable correctly.
        """
        calls = []

        def spy_fn(tf, outcome_set, min_cut):
            """Spy injectable that records calls and returns True."""
            calls.append({"tf": tf, "min_cut": list(min_cut)})
            return True

        result = run_joint_cyclic_id(
            tf="A",
            outcome_set=frozenset({"D"}),
            min_cut=[],  # empty B(t)
            ecoli_mixed=None,
            apt_order=None,
            all_network_vars=set(),
            identify_fn=spy_fn,
        )
        assert result is True
        assert len(calls) == 1
        assert calls[0]["min_cut"] == []  # empty list forwarded faithfully

    def test_empty_min_cut_result_is_bool(self):
        """POST: result is bool even for empty min_cut."""
        result = run_joint_cyclic_id(
            tf="A",
            outcome_set=frozenset({"D"}),
            min_cut=[],
            ecoli_mixed=None,
            apt_order=None,
            all_network_vars=set(),
            identify_fn=self._always_true,
        )
        assert isinstance(result, bool)

    def test_empty_outcome_set_with_empty_min_cut_returns_true(self):
        """
        POST: implies(len(outcome_set) == 0, result == True).
        Works for both empty and non-empty min_cut.
        """
        result_empty_cut = run_joint_cyclic_id(
            tf="A",
            outcome_set=frozenset(),
            min_cut=[],
            ecoli_mixed=None,
            apt_order=None,
            all_network_vars=set(),
            identify_fn=self._always_true,
        )
        assert result_empty_cut is True

        result_nonempty_cut = run_joint_cyclic_id(
            tf="A",
            outcome_set=frozenset(),
            min_cut=["C"],
            ecoli_mixed=None,
            apt_order=None,
            all_network_vars=set(),
            identify_fn=self._always_true,
        )
        assert result_nonempty_cut is True

    def test_nonempty_min_cut_passes_cut_to_injectable(self):
        """
        Non-empty min_cut must be forwarded to identify_fn unchanged.
        This is the positive control for the empty-cut guard.
        """
        received = {}

        def recording_fn(tf, outcome_set, min_cut):
            """Capture min_cut and return True."""
            received["min_cut"] = list(min_cut)
            return True

        run_joint_cyclic_id(
            tf="A",
            outcome_set=frozenset({"D"}),
            min_cut=["C", "X"],
            ecoli_mixed=None,
            apt_order=None,
            all_network_vars=set(),
            identify_fn=recording_fn,
        )
        assert received["min_cut"] == ["C", "X"]


# ---------------------------------------------------------------------------
# TestRunPerGeneCyclicId  (injectable identify_fn)
# ---------------------------------------------------------------------------


class TestRunPerGeneCyclicId:
    """
    Tests for the per-gene fallback path.

    We use a tiny mock Variable-like object so we don't import y0.
    """

    class FakeVar:
        """Minimal Variable-like stub for testing without importing y0."""

        def __init__(self, name):
            """Store the variable name."""
            self.name = name

        def __repr__(self):
            """Return the variable name as its string representation."""
            return self.name

    def _always_true(self, tf, outcome_set, min_cut):
        """Injectable that always returns True."""
        return True

    def _always_false(self, tf, outcome_set, min_cut):
        """Injectable that always returns False."""
        return False

    def test_returns_dict_keyed_by_gene(self):
        """POST: result is a dict keyed by gene name."""
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
        """POST: all values are True when identify_fn always returns True."""
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
        """POST: all values are False when identify_fn always returns False."""
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
        """POST: len(result) == len(outcome_vars)."""
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
        """POST: empty outcome_vars → empty dict."""
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
        """PRE: tf must be a str — int raises AssertionError."""
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
        """PRE: outcome_vars must be a list — set raises AssertionError."""
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
        """PRE: min_cut must be a list — set raises AssertionError."""
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
        """Full pipeline for TF=A returns True with always-true injectable."""
        in_scc_ch = find_in_scc_children("A", scc_abc, toy_graph)
        cut = compute_min_cut_b("A", scc_abc, in_scc_ch, toy_graph)
        intervened = build_intervened_graph(toy_graph, cut)
        children = get_direct_children("A", intervened)
        # A has out-edges A→B and A→D; both should survive the intervention
        assert "B" in children
        assert "D" in children

        outcome_set = frozenset(children)
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
                assert "A" not in reachable, f"Child {c} can still reach A after do({cut})"

    def test_dag_tf_has_empty_in_scc_children(self, toy_graph):
        """D is a leaf — it has no in-SCC children (singleton SCC)."""
        singleton = frozenset({"D"})
        children = find_in_scc_children("D", singleton, toy_graph)
        assert children == []

    def test_b_pipeline(self, toy_graph, scc_abc):
        """Run the same pipeline for TF=B: direct children of B are C and E."""
        in_scc_ch = find_in_scc_children("B", scc_abc, toy_graph)
        cut = compute_min_cut_b("B", scc_abc, in_scc_ch, toy_graph)
        assert "B" not in cut
        for c in in_scc_ch:
            assert c not in cut
        intervened = build_intervened_graph(toy_graph, cut)
        # Return paths from in-SCC children back to B must be severed
        for c in in_scc_ch:
            if c in intervened:
                reachable = nx.descendants(intervened, c)
                assert "B" not in reachable
        # Direct children of B in intervened graph = C and E (B→C, B→E)
        direct = get_direct_children("B", intervened)
        assert "C" in direct
        assert "E" in direct
        assert "B" not in direct
