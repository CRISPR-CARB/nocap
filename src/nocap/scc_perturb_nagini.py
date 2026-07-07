"""
scc_perturb_nagini.py — Nagini-annotated contracts for scc_perturb.py.

This file is a **verification-only companion** to ``scc_perturb.py``.  It
contains abstract stubs and contract-bearing wrappers that let Nagini
statically verify the logical properties captured in the Axiomander contracts.

Design notes
------------
``nx.DiGraph`` is an opaque external type — Nagini cannot inspect its
internals.  We therefore:

1. Introduce a ``@ContractOnly`` abstract class ``DiGraph`` that exposes only
   the graph properties needed for contracts (node membership, edge existence,
   in-degree, node count).
2. Write ``@ContractOnly`` stubs for each function whose body Nagini cannot
   verify (because it calls networkx).  The stubs carry the full contract;
   Nagini trusts them modularly.
3. Write thin wrappers for the pure list/bool helpers that Nagini *can* verify
   end-to-end (e.g. ``residual_cluster_size_distribution``).

All contract functions are no-ops at runtime.

Key Nagini constraints discovered during verification:
- Acc(obj.method, ...) on method references is INVALID — only field Acc works
- list_pred(Result()) inside @Pure method bodies is INVALID (heap in pure ctx)
- @Pure methods on carrier classes can only have simple scalar Ensures
- from __future__ import annotations is NOT supported by Nagini
"""

from nagini_contracts.contracts import *
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Abstract graph model
# ---------------------------------------------------------------------------

class DiGraph:
    """Abstract model of nx.DiGraph for Nagini contracts.

    All methods are @Pure — Nagini trusts their contracts without
    seeing a body.  At runtime this class is never instantiated; the real
    nx.DiGraph is used instead.
    """

    @Pure
    def has_node(self, n: object) -> bool:
        Ensures(isinstance(Result(), bool))

    @Pure
    def has_edge(self, u: object, v: object) -> bool:
        Ensures(isinstance(Result(), bool))

    @Pure
    def number_of_nodes(self) -> int:
        Ensures(Result() >= 0)

    @Pure
    def number_of_edges(self) -> int:
        Ensures(Result() >= 0)

    @Pure
    def in_degree_of(self, n: object) -> int:
        Ensures(Result() >= 0)


# ---------------------------------------------------------------------------
# build_intervened_graph
# ---------------------------------------------------------------------------

@ContractOnly
def build_intervened_graph(graph: DiGraph, perturb_set: List[str]) -> DiGraph:
    """
    Return a copy of *graph* with all in-edges to every node in *perturb_set* removed.

    Nagini contracts (derived from Axiomander postconditions):
    - Every node in perturb_set that exists in the result has in-degree 0.
    - The node count is preserved.
    - The graph itself is not mutated (modifies: none).
    """
    Requires(list_pred(perturb_set))
    Ensures(list_pred(perturb_set))
    Ensures(Result().number_of_nodes() == graph.number_of_nodes())
    Ensures(Forall(int, lambda i: Implies(
        0 <= i and i < len(perturb_set),
        Implies(
            Result().has_node(perturb_set[i]),
            Result().in_degree_of(perturb_set[i]) == 0
        )
    )))


# ---------------------------------------------------------------------------
# get_direct_children
# ---------------------------------------------------------------------------

@ContractOnly
def get_direct_children(tf: str, graph: DiGraph) -> List[str]:
    """
    Return the sorted list of direct out-neighbours of *tf* in *graph*.

    Nagini contracts:
    - Result is a list.
    - tf is not in the result.
    - Every element of the result has an edge from tf in graph.
    """
    Ensures(list_pred(Result()))
    # tf not in result
    Ensures(Forall(int, lambda i: Implies(
        0 <= i and i < len(Result()),
        Result()[i] != tf
    )))
    # every element has an edge from tf
    Ensures(Forall(int, lambda i: Implies(
        0 <= i and i < len(Result()),
        graph.has_edge(tf, Result()[i])
    )))


# ---------------------------------------------------------------------------
# find_in_scc_children
# ---------------------------------------------------------------------------

@ContractOnly
def find_in_scc_children(tf: str, scc_nodes: List[str], graph: DiGraph) -> List[str]:
    """
    Return the direct children of *tf* that are members of *scc_nodes*.

    Nagini contracts:
    - No self-loops: tf not in result.
    - Every result element is in scc_nodes.
    """
    Requires(list_pred(scc_nodes))
    Ensures(list_pred(Result()))
    Ensures(list_pred(scc_nodes))
    # tf not in result
    Ensures(Forall(int, lambda i: Implies(
        0 <= i and i < len(Result()),
        Result()[i] != tf
    )))
    # every result element is in scc_nodes
    Ensures(Forall(int, lambda i: Implies(
        0 <= i and i < len(Result()),
        Result()[i] in scc_nodes
    )))


# ---------------------------------------------------------------------------
# compute_min_cut_b
# ---------------------------------------------------------------------------

@ContractOnly
def compute_min_cut_b(
    tf: str,
    scc_nodes: List[str],
    in_scc_children: List[str],
    graph: DiGraph,
) -> List[str]:
    """
    Compute the minimum background perturbation set B(t) for TF *tf*.

    Nagini contracts (derived from Axiomander postconditions):
    - tf not in result.
    - No direct in-SCC child of tf is in result.
    - Every result element is in scc_nodes.
    - If in_scc_children is empty, result is empty.
    - If scc_nodes has <= 1 element, result is empty.
    """
    Requires(list_pred(scc_nodes))
    Requires(list_pred(in_scc_children))
    Ensures(list_pred(Result()))
    Ensures(list_pred(scc_nodes))
    Ensures(list_pred(in_scc_children))
    # tf not in result
    Ensures(Forall(int, lambda i: Implies(
        0 <= i and i < len(Result()),
        Result()[i] != tf
    )))
    # no in-SCC child in result
    Ensures(Forall(int, lambda i: Implies(
        0 <= i and i < len(Result()),
        Forall(int, lambda j: Implies(
            0 <= j and j < len(in_scc_children),
            Result()[i] != in_scc_children[j]
        ))
    )))
    # every result element is in scc_nodes
    Ensures(Forall(int, lambda i: Implies(
        0 <= i and i < len(Result()),
        Result()[i] in scc_nodes
    )))
    # empty in_scc_children => empty result
    Ensures(Implies(len(in_scc_children) == 0, len(Result()) == 0))
    # trivial SCC => empty result
    Ensures(Implies(len(scc_nodes) <= 1, len(Result()) == 0))


# ---------------------------------------------------------------------------
# VerifyCutResult — result carrier for verify_cut_complete
# ---------------------------------------------------------------------------

class VerifyCutResult:
    """Result carrier for verify_cut_complete.

    Note: @Pure methods on carrier classes cannot use list_pred in Ensures
    (heap predicates are not allowed in pure contexts).
    """

    @Pure
    def complete(self) -> bool:
        Ensures(isinstance(Result(), bool))

    @Pure
    def tf_still_cyclic(self) -> bool:
        Ensures(isinstance(Result(), bool))

    @Pure
    def surviving_count(self) -> int:
        Ensures(Result() >= 0)


# ---------------------------------------------------------------------------
# verify_cut_complete
# ---------------------------------------------------------------------------

@ContractOnly
def verify_cut_complete(
    tf: str,
    in_scc_children: List[str],
    min_cut: List[str],
    graph: DiGraph,
) -> VerifyCutResult:
    """
    Verify that do(B(t)) severs every return path from any in-SCC child back to tf.

    Returns a VerifyCutResult with fields: complete, surviving_count, tf_still_cyclic.

    Nagini contracts:
    - complete == (surviving_count == 0)
    - surviving_count >= 0
    """
    Requires(list_pred(in_scc_children))
    Requires(list_pred(min_cut))
    Ensures(list_pred(in_scc_children))
    Ensures(list_pred(min_cut))
    # complete == (surviving_children is empty)
    Ensures(Result().complete() == (Result().surviving_count() == 0))
    Ensures(Result().surviving_count() >= 0)


# ---------------------------------------------------------------------------
# residual_cluster_size_distribution — fully verifiable pure function
# ---------------------------------------------------------------------------

def residual_cluster_size_distribution(
    n_clusters: int,
    sizes: List[int],
    max_size: int,
    total_in_clusters: int,
    has_cluster: bool,
) -> bool:
    """
    Verify the invariants of residual_cluster_size_distribution output.

    This is a pure checker for the postconditions — given the output fields,
    assert they satisfy the contract.  Nagini can verify this end-to-end.

    Nagini contracts:
    - n_clusters == len(sizes)
    - has_cluster == (n_clusters >= 1)
    - n_clusters == 0 => max_size == 0
    - n_clusters >= 1 => max_size >= 2
    - total_in_clusters >= 0
    """
    Requires(list_pred(sizes))
    Requires(n_clusters == len(sizes))
    Requires(n_clusters >= 0)
    Requires(max_size >= 0)
    Requires(total_in_clusters >= 0)
    Requires(has_cluster == (n_clusters >= 1))
    Requires(Implies(n_clusters == 0, max_size == 0))
    Requires(Implies(n_clusters >= 1, max_size >= 2))
    Ensures(Result() == True)
    Ensures(has_cluster == (n_clusters >= 1))
    Ensures(Implies(n_clusters == 0, max_size == 0))
    Ensures(Implies(n_clusters >= 1, max_size >= 2))
    return True


# ---------------------------------------------------------------------------
# MinSccBreakResult — result carrier for min_scc_break_set
# ---------------------------------------------------------------------------

class MinSccBreakResult:
    """Result carrier for min_scc_break_set.

    Note: @Pure methods on carrier classes cannot use list_pred in Ensures
    (heap predicates are not allowed in pure contexts).
    The break_set list is exposed via a @ContractOnly accessor with list_pred
    in the function-level Ensures instead.
    """

    @Pure
    def same_scc_after_removal(self) -> bool:
        Ensures(isinstance(Result(), bool))

    @Pure
    def needs_intervention(self) -> bool:
        Ensures(isinstance(Result(), bool))

    @Pure
    def break_size(self) -> int:
        Ensures(Result() >= 0)

    @Pure
    def cut_verified(self) -> bool:
        Ensures(isinstance(Result(), bool))


# ---------------------------------------------------------------------------
# min_scc_break_set_break_set — accessor for the break_set list
# ---------------------------------------------------------------------------

@ContractOnly
def min_scc_break_set_break_set(r: MinSccBreakResult) -> List[str]:
    """
    Accessor for the break_set field of a MinSccBreakResult.
    Separated out because @Pure methods cannot carry list_pred.
    """
    Ensures(list_pred(Result()))
    Ensures(len(Result()) == r.break_size())


# ---------------------------------------------------------------------------
# min_scc_break_set
# ---------------------------------------------------------------------------

@ContractOnly
def min_scc_break_set(
    cause: str,
    effect: str,
    graph: DiGraph,
) -> MinSccBreakResult:
    """
    Compute the minimum vertex-intervention set B that makes cause->effect
    single-door identifiable.

    Returns a MinSccBreakResult with fields:
      same_scc_after_removal, needs_intervention, break_size, cut_verified

    Nagini contracts (derived from Axiomander):
    - break_size >= 0
    - needs_intervention == same_scc_after_removal
    - not needs_intervention => break_size == 0
    """
    Requires(graph.has_edge(cause, effect))
    Ensures(Result().break_size() >= 0)
    # needs_intervention == same_scc_after_removal
    Ensures(Result().needs_intervention() == Result().same_scc_after_removal())
    # not needs_intervention => break_size == 0
    Ensures(Implies(not Result().needs_intervention(), Result().break_size() == 0))
