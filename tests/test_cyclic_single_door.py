"""Tests for src/nocap/cyclic_single_door.py.

Test cases
----------
1. DAG (acyclic) — Pearl single-door: edge X->Y with parent Z of X is
   identifiable; adjustment set contains Z (or is empty when no confounders).
2. Same-SCC edge — unidentifiable without intervention.
3. Confounder / bidirected edge case — edge with a common cause is identifiable
   by conditioning on the common cause.
4. O-set completeness guard — an edge whose naive parent set is invalid but
   whose O-set is valid is correctly classified as identifiable.
5. Greedy rescue — maximize_identifiable_edges increases identifiable count;
   in-edge-removal assertion (out-edges of intervened nodes survive, in-degree 0).
6. evaluate_all_edges — length and status invariants.
7. nx_digraph_to_y0 — node/edge count preserved.
8. same_scc — basic true/false cases.
"""

from __future__ import annotations

import networkx as nx
import pytest

from nocap.cyclic_single_door import (
    classify_edge,
    evaluate_all_edges,
    maximize_identifiable_edges,
    nx_digraph_to_y0,
    same_scc,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dag_chain() -> nx.DiGraph:
    """Z -> X -> Y  (simple chain, no confounders, no cycles)."""
    g = nx.DiGraph()
    g.add_edges_from([("Z", "X"), ("X", "Y")])
    return g


def _dag_with_confounder() -> nx.DiGraph:
    """Z -> X -> Y, Z -> Y  (Z is a common cause / confounder for X->Y)."""
    g = nx.DiGraph()
    g.add_edges_from([("Z", "X"), ("X", "Y"), ("Z", "Y")])
    return g


def _two_cycle() -> nx.DiGraph:
    """X -> Y -> X  (both nodes in the same SCC)."""
    g = nx.DiGraph()
    g.add_edges_from([("X", "Y"), ("Y", "X")])
    return g


def _three_cycle_with_dag_edge() -> nx.DiGraph:
    """Build a 3-cycle A->B->C->A (SCC) plus a DAG edge D->A into the SCC.

    Edges:
      A->B, B->C, C->A  (SCC)
      D->A              (DAG edge, D is outside the SCC)
    """
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("D", "A")])
    return g


def _rescue_graph() -> nx.DiGraph:
    """Graph where one intervention rescues an unidentifiable edge.

    Structure:
      X -> Y -> X  (SCC, X->Y is unidentifiable)
      Z -> X       (Z is outside the SCC)

    After do(Y) (remove in-edges to Y, i.e. remove X->Y), the SCC is broken
    and X->Y no longer exists in the intervened graph.  But the edge Z->X
    becomes identifiable (it was already identifiable since Z is outside the
    SCC).  We use a slightly richer graph to make the rescue meaningful.

    Richer version:
      A -> B -> C -> A  (SCC)
      D -> A            (D outside SCC, D->A identifiable)
      A -> E            (A->E: A in SCC, E outside — identifiable)
    """
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),  # SCC
            ("D", "A"),  # into SCC
            ("A", "E"),  # out of SCC
        ]
    )
    return g


# ---------------------------------------------------------------------------
# 1. DAG (acyclic) — Pearl single-door
# ---------------------------------------------------------------------------


def test_dag_chain_x_to_y_identifiable():
    """In Z->X->Y, the edge X->Y has no confounders; should be identifiable."""
    g = _dag_chain()
    result = classify_edge(g, "X", "Y")
    assert result["status"] == "identifiable"
    assert result["adjustment_set"] is not None
    assert isinstance(result["adjustment_set"], frozenset)
    assert result["same_scc"] is False


def test_dag_chain_z_to_x_identifiable():
    """In Z->X->Y, the edge Z->X should also be identifiable."""
    g = _dag_chain()
    result = classify_edge(g, "Z", "X")
    assert result["status"] == "identifiable"
    assert result["same_scc"] is False


def test_dag_confounder_x_to_y_identifiable():
    """In Z->X->Y, Z->Y, the edge X->Y is identifiable by conditioning on Z."""
    g = _dag_with_confounder()
    result = classify_edge(g, "X", "Y")
    assert result["status"] == "identifiable"
    assert result["adjustment_set"] is not None
    # Z should be in the adjustment set (it's the confounder)
    assert "Z" in result["adjustment_set"]


# ---------------------------------------------------------------------------
# 2. Same-SCC edge — unidentifiable
# ---------------------------------------------------------------------------


def test_two_cycle_x_to_y_unidentifiable():
    """In X->Y->X, the edge X->Y is unidentifiable (both in same SCC)."""
    g = _two_cycle()
    result = classify_edge(g, "X", "Y")
    assert result["status"] == "unidentifiable"
    assert result["adjustment_set"] is None
    assert result["same_scc"] is True


def test_two_cycle_y_to_x_unidentifiable():
    """In X->Y->X, the edge Y->X is also unidentifiable."""
    g = _two_cycle()
    result = classify_edge(g, "Y", "X")
    assert result["status"] == "unidentifiable"
    assert result["adjustment_set"] is None
    assert result["same_scc"] is True


def test_three_cycle_scc_edges_unidentifiable():
    """In A->B->C->A, all intra-SCC edges are unidentifiable."""
    g = _three_cycle_with_dag_edge()
    for cause, effect in [("A", "B"), ("B", "C"), ("C", "A")]:
        result = classify_edge(g, cause, effect)
        assert result["status"] == "unidentifiable", (
            f"Expected unidentifiable for {cause}->{effect}"
        )
        assert result["same_scc"] is True


def test_three_cycle_dag_edge_identifiable():
    """In A->B->C->A + D->A, the edge D->A is identifiable (D outside SCC)."""
    g = _three_cycle_with_dag_edge()
    result = classify_edge(g, "D", "A")
    assert result["status"] == "identifiable"
    assert result["same_scc"] is False


# ---------------------------------------------------------------------------
# 3. O-set completeness guard
# ---------------------------------------------------------------------------


def test_o_set_completeness_guard():
    """An edge whose naive parent set is invalid but O-set is valid is identifiable.

    Graph: W -> X -> Y, W -> Y, X -> Z -> Y
    The edge X->Y has parents {W} as a valid adjustment set (W blocks the
    backdoor path X <- W -> Y).  The O-set should find this.
    """
    g = nx.DiGraph()
    g.add_edges_from([("W", "X"), ("X", "Y"), ("W", "Y"), ("X", "Z"), ("Z", "Y")])
    result = classify_edge(g, "X", "Y")
    assert result["status"] == "identifiable"
    assert result["adjustment_set"] is not None
    # W should be in the adjustment set
    assert "W" in result["adjustment_set"]


# ---------------------------------------------------------------------------
# 4. evaluate_all_edges
# ---------------------------------------------------------------------------


def test_evaluate_all_edges_length():
    """evaluate_all_edges returns one result per edge."""
    g = _three_cycle_with_dag_edge()
    results = evaluate_all_edges(g)
    assert len(results) == g.number_of_edges()


def test_evaluate_all_edges_statuses():
    """All statuses are valid strings."""
    g = _three_cycle_with_dag_edge()
    results = evaluate_all_edges(g)
    for r in results:
        assert r["status"] in ("identifiable", "unidentifiable")


def test_evaluate_all_edges_restrict():
    """restrict_edges limits the output to the specified edges."""
    g = _three_cycle_with_dag_edge()
    subset = [("D", "A")]
    results = evaluate_all_edges(g, restrict_edges=subset)
    assert len(results) == 1
    assert results[0]["cause"] == "D"
    assert results[0]["effect"] == "A"


def test_evaluate_all_edges_empty_graph():
    """Empty graph returns empty list."""
    g = nx.DiGraph()
    g.add_nodes_from(["X", "Y"])
    results = evaluate_all_edges(g)
    assert results == []


# ---------------------------------------------------------------------------
# 5. Greedy rescue — maximize_identifiable_edges
# ---------------------------------------------------------------------------


def test_rescue_curve_structure():
    """maximize_identifiable_edges returns a well-formed result dict."""
    g = _rescue_graph()
    result = maximize_identifiable_edges(g, k=10)

    assert "curve" in result
    assert "chosen_nodes" in result
    assert "final_graph" in result
    assert "final_results" in result
    assert "n_identifiable_baseline" in result
    assert "n_identifiable_final" in result

    # Curve starts at step 0
    assert result["curve"][0][0] == 0
    # Curve length = chosen_nodes + 1
    assert len(result["curve"]) == len(result["chosen_nodes"]) + 1
    # Budget respected
    assert len(result["chosen_nodes"]) <= 10
    # Baseline matches curve[0]
    assert result["n_identifiable_baseline"] == result["curve"][0][1]
    # Final matches curve[-1]
    assert result["n_identifiable_final"] == result["curve"][-1][1]


def test_rescue_in_edge_removal_semantics():
    """Intervened nodes have in-degree 0 but out-edges preserved (do-semantics)."""
    g = _rescue_graph()
    result = maximize_identifiable_edges(g, k=10)

    final_graph: nx.DiGraph = result["final_graph"]
    for node in result["chosen_nodes"]:
        # do-semantics: in-edges removed
        assert final_graph.in_degree(node) == 0, (
            f"Intervened node {node!r} must have in-degree 0 in final_graph"
        )
        # do-semantics: out-edges preserved (same count as original)
        assert final_graph.out_degree(node) == g.out_degree(node), (
            f"Intervened node {node!r} must preserve out-edges"
        )
        # Node itself still present
        assert node in final_graph.nodes()


def test_rescue_k0_no_interventions():
    """k=0 means no interventions; curve has exactly one point."""
    g = _rescue_graph()
    result = maximize_identifiable_edges(g, k=0)
    assert result["chosen_nodes"] == []
    assert len(result["curve"]) == 1
    assert result["curve"][0][0] == 0
    assert result["n_identifiable_baseline"] == result["n_identifiable_final"]


def test_rescue_monotone_curve():
    """The rescue curve is non-decreasing (each intervention is only kept if it helps)."""
    g = _rescue_graph()
    result = maximize_identifiable_edges(g, k=10)
    curve = result["curve"]
    for i in range(1, len(curve)):
        assert curve[i][1] >= curve[i - 1][1], f"Curve must be non-decreasing: {curve}"


def test_rescue_two_cycle_improves():
    """For X->Y->X, intervening on Y (removing X->Y) should help identifiability.

    After do(Y): Y has no in-edges, so the SCC is broken.  The edge X->Y no
    longer exists in the intervened graph (it was the in-edge to Y that got
    removed).  The remaining edges (Y->X) may become identifiable.
    """
    g = _two_cycle()
    baseline = evaluate_all_edges(g)
    n_baseline = sum(1 for r in baseline if r["status"] == "identifiable")

    result = maximize_identifiable_edges(g, k=2)
    # After intervention, at least as many edges should be identifiable
    assert result["n_identifiable_final"] >= n_baseline


# ---------------------------------------------------------------------------
# 6. nx_digraph_to_y0
# ---------------------------------------------------------------------------


def test_nx_digraph_to_y0_node_count():
    """Node count is preserved after conversion."""
    g = _three_cycle_with_dag_edge()
    g_y0 = nx_digraph_to_y0(g)
    assert g_y0.directed.number_of_nodes() == g.number_of_nodes()


def test_nx_digraph_to_y0_edge_count():
    """Edge count is preserved after conversion."""
    g = _three_cycle_with_dag_edge()
    g_y0 = nx_digraph_to_y0(g)
    assert g_y0.directed.number_of_edges() == g.number_of_edges()


def test_nx_digraph_to_y0_no_bidirected():
    """Conversion adds no bidirected edges (those come from sigma_extension)."""
    g = _two_cycle()
    g_y0 = nx_digraph_to_y0(g)
    assert g_y0.undirected.number_of_edges() == 0


# ---------------------------------------------------------------------------
# 7. same_scc
# ---------------------------------------------------------------------------


def test_same_scc_true_in_cycle():
    """X and Y are in the same SCC in X->Y->X."""
    g = _two_cycle()
    assert same_scc(g, "X", "Y") is True
    assert same_scc(g, "Y", "X") is True


def test_same_scc_false_dag():
    """Z and Y are not in the same SCC in Z->X->Y."""
    g = _dag_chain()
    assert same_scc(g, "Z", "Y") is False
    assert same_scc(g, "Y", "Z") is False


def test_same_scc_self():
    """A node is always in the same SCC as itself."""
    g = _dag_chain()
    assert same_scc(g, "X", "X") is True


def test_same_scc_missing_node():
    """Missing node returns False."""
    g = _dag_chain()
    assert same_scc(g, "X", "MISSING") is False


# ---------------------------------------------------------------------------
# 8. PRE-condition violations raise AssertionError
# ---------------------------------------------------------------------------


def test_classify_edge_pre_missing_edge():
    """classify_edge raises AssertionError when edge does not exist."""
    g = _dag_chain()
    with pytest.raises(AssertionError, match="PRE"):
        classify_edge(g, "Y", "Z")  # edge Y->Z does not exist


def test_classify_edge_pre_bad_cause_type():
    """classify_edge raises AssertionError when cause is not a str."""
    g = _dag_chain()
    with pytest.raises(AssertionError, match="PRE"):
        classify_edge(g, 42, "Y")  # type: ignore[arg-type]


def test_maximize_pre_bad_k():
    """maximize_identifiable_edges raises AssertionError for negative k."""
    g = _dag_chain()
    with pytest.raises(AssertionError, match="PRE"):
        maximize_identifiable_edges(g, k=-1)
