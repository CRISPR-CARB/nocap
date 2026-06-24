"""
test_cut_complete_axiomander.py
================================
Axiomander-style tests for ``verify_cut_complete`` in
``nocap.scc_perturb``.

House convention: production code uses plain ``assert`` with PRE/INV/POST
prefixes; test modules use ``pytest.raises`` for negative cases and direct
assertion for positive cases.

Two structural patterns are covered:

1. **2-cycle (Interpretation A cut failure)**
   tf → child → tf.  Under Interpretation A, the child is forbidden from
   B(t).  No intermediate node exists, so B(t) = [].  After do([]) the
   return path child→tf survives.
   Expected: ``complete=False``, ``tf_still_cyclic=True``,
   ``surviving_children=[child]``.

2. **Benign residual cluster (complete cut)**
   tf → child1 → child2 → child1 (children form a 2-cycle among
   themselves), but the return path to tf is severed by a valid
   intermediate node.
   Expected: ``complete=True``, ``tf_still_cyclic=False``,
   ``surviving_children=[]``.

3. **Linear DAG (trivially complete)**
   tf → child (no return path at all, B(t) = []).
   Expected: ``complete=True``, ``tf_still_cyclic=False``.

4. **Negative: type contract violations**
   ``verify_cut_complete`` asserts PRE conditions; passing wrong types
   raises ``AssertionError``.
"""

from __future__ import annotations

import networkx as nx
import pytest

from nocap.scc_perturb import verify_cut_complete

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _digraph(*edges) -> nx.DiGraph:
    """Build a DiGraph from a sequence of (src, dst) tuples."""
    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g


# ---------------------------------------------------------------------------
# 1. 2-cycle: Interpretation A cut failure
# ---------------------------------------------------------------------------

class TestTwoCycleCutFailure:
    """
    Graph: tf ↔ child (a 2-cycle).
    in_scc_children = [child]  (direct child in same SCC)
    min_cut = []               (Interp A: child is forbidden, no intermediate)

    do([]) removes no edges → child can still reach tf.
    """

    def setup_method(self):
        self.tf = "marR"
        self.child = "marA"
        self.graph = _digraph((self.tf, self.child), (self.child, self.tf))
        self.in_scc_children = [self.child]
        self.min_cut = []

    def test_complete_is_false(self):
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert result["complete"] is False, (
            "2-cycle with empty B(t) must yield complete=False"
        )

    def test_tf_still_cyclic(self):
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert result["tf_still_cyclic"] is True, (
            "TF must still be cyclic when 2-cycle survives do([])"
        )

    def test_surviving_children_contains_child(self):
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert self.child in result["surviving_children"], (
            "The in-SCC child that forms the return path must appear in surviving_children"
        )

    def test_postcondition_complete_eq_no_surviving(self):
        """Postcondition: complete == (len(surviving_children) == 0)."""
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert result["complete"] == (len(result["surviving_children"]) == 0)

    def test_surviving_children_is_sorted(self):
        """surviving_children must be in sorted order."""
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert result["surviving_children"] == sorted(result["surviving_children"])


# ---------------------------------------------------------------------------
# 2. Benign residual cluster: cut severs tf return path; children cyclic
# ---------------------------------------------------------------------------

class TestBenignResidualCluster:
    """
    Graph:
        tf → c1 → inter → tf    (c1 is in-SCC child; returns to tf via inter)
        tf → c2 → inter          (c2 also routes through inter)
        c1 ↔ c2                  (residual 2-cycle between children after cut)
        min_cut = [inter]        (inter is the intermediate node; cut severs both)

    After do([inter]):
        inter has no in-edges → c1 and c2 cannot reach tf via inter
        c1 ↔ c2 form a benign residual cluster (cyclic among themselves)
        but neither can reach tf

    in_scc_children = [c1, c2]  (both are direct children of tf in the SCC)
    Expected: complete=True, tf_still_cyclic=False, surviving=[].
    """

    def setup_method(self):
        self.tf = "tf"
        self.inter = "inter"
        self.c1 = "c1"
        self.c2 = "c2"
        self.graph = _digraph(
            (self.tf, self.c1),
            (self.tf, self.c2),
            (self.c1, self.inter),
            (self.c2, self.inter),
            (self.inter, self.tf),      # return path via inter
            (self.c1, self.c2),
            (self.c2, self.c1),         # residual cycle among children (benign)
        )
        self.in_scc_children = [self.c1, self.c2]
        self.min_cut = [self.inter]

    def test_complete_is_true(self):
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert result["complete"] is True, (
            "After do([inter]), no in-SCC child can reach tf — cut must be complete"
        )

    def test_tf_not_cyclic(self):
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert result["tf_still_cyclic"] is False, (
            "After do([inter]), tf has no in-edges → tf is a singleton SCC"
        )

    def test_surviving_children_empty(self):
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert result["surviving_children"] == [], (
            "No in-SCC children should survive after the cut is complete"
        )

    def test_postcondition_complete_eq_no_surviving(self):
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert result["complete"] == (len(result["surviving_children"]) == 0)


# ---------------------------------------------------------------------------
# 3. Linear DAG: no return path, trivially complete
# ---------------------------------------------------------------------------

class TestLinearDag:
    """
    Graph: tf → child (no cycle at all).
    in_scc_children = []  (child is not in same SCC as tf)
    min_cut = []

    Expected: complete=True, tf_still_cyclic=False, surviving=[].
    """

    def setup_method(self):
        self.tf = "tf"
        self.child = "child"
        self.graph = _digraph((self.tf, self.child))
        self.in_scc_children = []
        self.min_cut = []

    def test_complete_is_true(self):
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert result["complete"] is True

    def test_tf_not_cyclic(self):
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert result["tf_still_cyclic"] is False

    def test_surviving_empty(self):
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert result["surviving_children"] == []


# ---------------------------------------------------------------------------
# 4. Negative: PRE contract violations raise AssertionError
# ---------------------------------------------------------------------------

class TestPreContractViolations:
    """
    verify_cut_complete asserts that:
    - tf is a str
    - in_scc_children is a list
    - min_cut is a list
    Passing wrong types must raise AssertionError.
    """

    def setup_method(self):
        self.graph = _digraph(("tf", "child"))

    def test_tf_not_str_raises(self):
        with pytest.raises(AssertionError, match="PRE"):
            verify_cut_complete(123, [], [], self.graph)

    def test_in_scc_children_not_list_raises(self):
        with pytest.raises(AssertionError, match="PRE"):
            verify_cut_complete("tf", ("child",), [], self.graph)  # tuple, not list

    def test_min_cut_not_list_raises(self):
        with pytest.raises(AssertionError, match="PRE"):
            verify_cut_complete("tf", [], {"child"}, self.graph)  # set, not list


# ---------------------------------------------------------------------------
# 5. Multiple in-SCC children — only some survive
# ---------------------------------------------------------------------------

class TestPartialSurvival:
    """
    Graph:
        tf → c1 → tf   (2-cycle: c1 is in-SCC child, cannot be cut)
        tf → c2 → mid → tf  (c2 returns via intermediate mid)
        min_cut = [mid]      (cuts c2's return path, but not c1's)

    After do([mid]):
        c1 can still reach tf (direct edge c1→tf survives)
        c2 cannot reach tf (path c2→mid→tf: mid's in-edges removed)

    Expected: complete=False, surviving_children=[c1].
    """

    def setup_method(self):
        self.tf = "tf"
        self.c1 = "c1"
        self.c2 = "c2"
        self.mid = "mid"
        self.graph = _digraph(
            (self.tf, self.c1),
            (self.c1, self.tf),          # direct 2-cycle
            (self.tf, self.c2),
            (self.c2, self.mid),
            (self.mid, self.tf),         # c2 returns via mid
        )
        self.in_scc_children = [self.c1, self.c2]
        self.min_cut = [self.mid]

    def test_complete_false(self):
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert result["complete"] is False

    def test_c1_survives(self):
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert self.c1 in result["surviving_children"]

    def test_c2_does_not_survive(self):
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert self.c2 not in result["surviving_children"]

    def test_tf_still_cyclic(self):
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert result["tf_still_cyclic"] is True

    def test_postcondition(self):
        result = verify_cut_complete(
            self.tf, self.in_scc_children, self.min_cut, self.graph
        )
        assert result["complete"] == (len(result["surviving_children"]) == 0)
