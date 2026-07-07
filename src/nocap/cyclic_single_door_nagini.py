"""
cyclic_single_door_nagini.py — Nagini-annotated contracts for cyclic_single_door.py.

This file is a **verification-only companion** to ``cyclic_single_door.py``.
It contains abstract stubs and contract-bearing wrappers that let Nagini
statically verify the logical properties captured in the Axiomander contracts.

Design notes
------------
All external types (``nx.DiGraph``, ``NxMixedGraph``) are opaque to Nagini.
We reuse the ``DiGraph`` abstract model from ``scc_perturb_nagini.py`` and
introduce a ``MixedGraph`` abstract model for y0's ``NxMixedGraph``.

Functions whose bodies call networkx/y0 are modelled as ``@ContractOnly``
stubs.  The one fully-verifiable function is ``same_scc_checker``, a pure
boolean checker for the ``same_scc`` postcondition.

Key Nagini constraints (learned from scc_perturb_nagini.py):
- ``from __future__ import annotations`` is NOT supported
- ``Acc(obj.method, ...)`` on method references is INVALID
- ``list_pred(Result())`` inside ``@Pure`` method bodies is INVALID
- ``@Pure`` methods on carrier classes can only have simple scalar Ensures
"""

from nagini_contracts.contracts import *
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Abstract graph models
# ---------------------------------------------------------------------------

class DiGraph:
    """Abstract model of nx.DiGraph (reused from scc_perturb_nagini.py)."""

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


class MixedGraph:
    """Abstract model of y0's NxMixedGraph for Nagini contracts."""

    @Pure
    def directed_number_of_nodes(self) -> int:
        Ensures(Result() >= 0)

    @Pure
    def directed_number_of_edges(self) -> int:
        Ensures(Result() >= 0)


# ---------------------------------------------------------------------------
# nx_digraph_to_y0
# ---------------------------------------------------------------------------

@ContractOnly
def nx_digraph_to_y0(graph: DiGraph) -> MixedGraph:
    """
    Convert a plain nx.DiGraph to a y0 NxMixedGraph.

    Nagini contracts (from Axiomander postconditions):
    - directed node count is preserved
    - directed edge count is preserved
    """
    Ensures(Result().directed_number_of_nodes() == graph.number_of_nodes())
    Ensures(Result().directed_number_of_edges() == graph.number_of_edges())


# ---------------------------------------------------------------------------
# same_scc_checker — fully verifiable pure function
# ---------------------------------------------------------------------------

def same_scc_checker(result: bool) -> bool:
    """
    Verify the postcondition of same_scc: result must be a bool.

    This is a pure checker — given the output, assert it satisfies the contract.
    Nagini can verify this end-to-end.

    Nagini contracts:
    - result is bool => checker returns True
    """
    Requires(isinstance(result, bool))
    Ensures(Result() == True)
    Ensures(isinstance(result, bool))
    return True


# ---------------------------------------------------------------------------
# ClassifyEdgeResult — result carrier for classify_edge
# ---------------------------------------------------------------------------

class ClassifyEdgeResult:
    """Result carrier for classify_edge.

    Note: @Pure methods on carrier classes cannot use list_pred in Ensures.
    """

    @Pure
    def is_identifiable(self) -> bool:
        Ensures(isinstance(Result(), bool))

    @Pure
    def is_same_scc(self) -> bool:
        Ensures(isinstance(Result(), bool))

    @Pure
    def has_adjustment_set(self) -> bool:
        Ensures(isinstance(Result(), bool))


# ---------------------------------------------------------------------------
# classify_edge
# ---------------------------------------------------------------------------

@ContractOnly
def classify_edge(
    graph: DiGraph,
    cause: str,
    effect: str,
) -> ClassifyEdgeResult:
    """
    Classify a single directed edge as identifiable or unidentifiable.

    Nagini contracts (from Axiomander postconditions):
    - is_identifiable is bool
    - is_same_scc is bool
    - is_identifiable == has_adjustment_set  (adjustment_set non-None iff identifiable)
    """
    Requires(graph.has_edge(cause, effect))
    Ensures(Result().is_identifiable() == Result().has_adjustment_set())
    Ensures(isinstance(Result().is_same_scc(), bool))


# ---------------------------------------------------------------------------
# EvaluateAllEdgesResult — result carrier for evaluate_all_edges
# ---------------------------------------------------------------------------

class EvaluateAllEdgesResult:
    """Result carrier for evaluate_all_edges."""

    @Pure
    def result_count(self) -> int:
        Ensures(Result() >= 0)

    @Pure
    def all_statuses_valid(self) -> bool:
        Ensures(isinstance(Result(), bool))


# ---------------------------------------------------------------------------
# evaluate_all_edges
# ---------------------------------------------------------------------------

@ContractOnly
def evaluate_all_edges(graph: DiGraph) -> EvaluateAllEdgesResult:
    """
    Classify every directed edge in graph under the σ-single-door criterion.

    Nagini contracts (from Axiomander postconditions):
    - result_count == graph.number_of_edges()  (when no restrict_edges)
    - all_statuses_valid is True
    """
    Ensures(Result().result_count() == graph.number_of_edges())
    Ensures(Result().all_statuses_valid() == True)


# ---------------------------------------------------------------------------
# MaximizeResult — result carrier for maximize_identifiable_edges
# ---------------------------------------------------------------------------

class MaximizeResult:
    """Result carrier for maximize_identifiable_edges."""

    @Pure
    def curve_length(self) -> int:
        Ensures(Result() >= 1)

    @Pure
    def chosen_count(self) -> int:
        Ensures(Result() >= 0)

    @Pure
    def curve_first_step(self) -> int:
        Ensures(Result() == 0)

    @Pure
    def n_identifiable_baseline(self) -> int:
        Ensures(Result() >= 0)

    @Pure
    def n_identifiable_final(self) -> int:
        Ensures(Result() >= 0)

    @Pure
    def curve_first_value(self) -> int:
        Ensures(Result() >= 0)

    @Pure
    def curve_last_value(self) -> int:
        Ensures(Result() >= 0)


# ---------------------------------------------------------------------------
# maximize_identifiable_edges
# ---------------------------------------------------------------------------

@ContractOnly
def maximize_identifiable_edges(graph: DiGraph, k: int) -> MaximizeResult:
    """
    Greedily maximise identifiable edges via up to k hard interventions.

    Nagini contracts (from Axiomander postconditions):
    - curve_length == chosen_count + 1
    - curve_first_step == 0
    - n_identifiable_baseline == curve_first_value
    - n_identifiable_final == curve_last_value
    - chosen_count <= k
    """
    Requires(k >= 0)
    # curve_length == chosen_count + 1
    Ensures(Result().curve_length() == Result().chosen_count() + 1)
    # curve starts at step 0
    Ensures(Result().curve_first_step() == 0)
    # baseline matches curve[0]
    Ensures(Result().n_identifiable_baseline() == Result().curve_first_value())
    # final matches curve[-1]
    Ensures(Result().n_identifiable_final() == Result().curve_last_value())
    # chosen_count <= k
    Ensures(Result().chosen_count() <= k)
    # non-negative counts
    Ensures(Result().n_identifiable_baseline() >= 0)
    Ensures(Result().n_identifiable_final() >= 0)
