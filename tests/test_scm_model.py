"""Tests for the neutral synthetic SCM model and builder."""

import networkx as nx
import numpy as np
import pytest

from nocap.scm_model import DirectedScm, build_synthetic_scm


def test_directed_scm_builds_beta_matrix_in_stable_node_order():
    """Build the coefficient matrix according to the requested node order."""
    graph = nx.DiGraph([("B", "A"), ("A", "C")])
    scm = DirectedScm(
        nodes=("C", "A", "B"),
        graph=graph,
        betas={("B", "A"): 2.0, ("A", "C"): -0.5},
    )

    assert scm.nodes == ("C", "A", "B")
    np.testing.assert_array_equal(
        scm.beta_matrix,
        np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 2.0, 0.0]]),
    )


def test_directed_scm_rejects_coefficients_for_non_edges():
    """Reject coefficients whose directed pair is not present in the graph."""
    graph = nx.DiGraph([("A", "B")])

    with pytest.raises(ValueError, match="exactly to graph edges"):
        DirectedScm(
            nodes=("A", "B"),
            graph=graph,
            betas={("B", "A"): 1.0},
        )


def test_directed_scm_defensively_copies_graph_and_stringifies_nodes():
    """Copy the graph and normalize node labels without exposing mutable state."""
    graph = nx.DiGraph([(1, 2)])
    scm = DirectedScm(nodes=(1, 2), graph=graph, betas={("1", "2"): 0.25})

    graph.add_edge(2, 1)

    assert scm.nodes == ("1", "2")
    assert set(scm.graph.edges()) == {("1", "2")}
    with pytest.raises(nx.NetworkXError):
        scm.graph.add_edge("2", "1")


def test_build_synthetic_scm_is_reproducible():
    """Produce identical synthetic SCM results from identical random seeds."""
    graph = nx.DiGraph([("A", "B"), ("B", "A")])
    kwargs = {
        "missing_edge_rate": 0.0,
        "beta_med": 2.0,
        "beta_log_sd": 0.5,
        "beta_p": 0.5,
        "beta_abs_max": 5.0,
    }

    first = build_synthetic_scm(graph, ["A", "B"], rng=np.random.default_rng(11), **kwargs)
    second = build_synthetic_scm(graph, ["A", "B"], rng=np.random.default_rng(11), **kwargs)

    assert first.added_edges == second.added_edges
    assert first.condition_number == second.condition_number
    assert first.scm.betas == second.scm.betas


def test_build_synthetic_scm_uses_explicit_edge_signs():
    """Use caller-provided signs when sampling synthetic edge coefficients."""
    graph = nx.DiGraph([("A", "B"), ("B", "A")])
    result = build_synthetic_scm(
        graph,
        ["A", "B"],
        missing_edge_rate=0.0,
        beta_med=2.0,
        beta_log_sd=0.5,
        beta_p=0.5,
        beta_abs_max=5.0,
        rng=np.random.default_rng(11),
        signs={("A", "B"): 1.0, ("B", "A"): -1.0},
    )

    assert result.scm.betas[("A", "B")] > 0
    assert result.scm.betas[("B", "A")] < 0


def test_build_synthetic_scm_only_adds_true_edges():
    """Restrict synthetic edge additions to pairs allowed by the true graph."""
    graph = nx.DiGraph([("A", "B")])
    result = build_synthetic_scm(
        graph,
        ["A", "B", "C"],
        missing_edge_rate=1.0,
        beta_med=2.0,
        beta_log_sd=0.5,
        beta_p=0.5,
        beta_abs_max=5.0,
        rng=np.random.default_rng(4),
    )

    assert set(graph.edges()).issubset(result.scm.graph.edges())
    assert set(result.added_edges).issubset(result.scm.graph.edges())
    assert len(result.added_edges) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beta_med": 0.0, "beta_log_sd": 0.5, "beta_abs_max": 5.0},
        {"beta_med": 2.0, "beta_log_sd": 0.0, "beta_abs_max": 5.0},
        {"beta_med": 2.0, "beta_log_sd": 0.5, "beta_abs_max": 0.5},
        {"beta_med": 2.0, "beta_log_sd": 0.5, "beta_abs_max": 5.0, "beta_p": 1.5},
    ],
)
def test_build_synthetic_scm_validates_beta_parameters(kwargs):
    """Reject invalid beta-distribution parameters before building the model."""
    parameters = {
        "missing_edge_rate": 0.0,
        "beta_med": 2.0,
        "beta_log_sd": 0.5,
        "beta_p": 0.5,
        "beta_abs_max": 5.0,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError):
        build_synthetic_scm(
            nx.DiGraph([("A", "B")]),
            ["A", "B"],
            rng=np.random.default_rng(1),
            **parameters,
        )
