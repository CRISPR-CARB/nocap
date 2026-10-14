"""Tests for reusable causal simulation helpers."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from nocap.estimation import estimated_scm_ate
from nocap.scm_model import DirectedScm, build_intervened_scm
from nocap.simulation import SimulationConfig, simulate_intervention_levels, true_scm_ate


def test_hard_intervention_preserves_outgoing_edges_and_original_scm() -> None:
    """Preserve outgoing intervention edges without mutating the original SCM."""
    graph = nx.DiGraph([("T", "Y"), ("Y", "T")])
    scm = DirectedScm(("T", "Y"), graph, {("T", "Y"): 0.2, ("Y", "T"): 0.1})
    intervened = build_intervened_scm(scm, ["T"])
    assert set(intervened.graph.edges()) == {("T", "Y")}
    assert set(scm.graph.edges()) == {("T", "Y"), ("Y", "T")}


def test_intervention_data_has_fixed_treatment_values() -> None:
    """Generate intervention states whose treatment columns equal each level."""
    graph = nx.DiGraph([("T", "Y")])
    scm = DirectedScm(("T", "Y"), graph, {("T", "Y"): 0.5})
    config = SimulationConfig(n_samples=20, size_factor_log_sd=0.0, use_latent_expression_hat=False)
    states = simulate_intervention_levels(scm, config, "T", (-1.0, 1.0), seed=4)
    assert np.all(states[-1.0].latent_log_expression[:, 0] == -1.0)
    assert np.all(states[1.0].latent_log_expression[:, 0] == 1.0)


def test_true_scm_ate_matches_linear_effect() -> None:
    """Match the known linear effect when estimating the true SCM ATE."""
    graph = nx.DiGraph([("T", "Y")])
    scm = DirectedScm(("T", "Y"), graph, {("T", "Y"): 0.5})
    assert true_scm_ate(scm, "T", "Y", (-1.0, 1.0), n_samples=1000) == 1.0


def test_estimated_scm_ate_uses_csd_coefficients() -> None:
    """Use estimated cyclic single-door coefficients to calculate the ATE."""
    graph = nx.DiGraph([("T", "Y")])
    scm = DirectedScm(("T", "Y"), graph, {("T", "Y"): 0.5})
    noise = np.random.default_rng(3).normal(size=(500, 2))
    from nocap.simulation import numpy_linear_solver

    data_values = numpy_linear_solver(scm, noise)
    data = pd.DataFrame(data_values, columns=("T", "Y"))
    ate, estimates, estimated = estimated_scm_ate(
        graph, data, "T", "Y", (-1.0, 1.0), n_samples=1000, min_rows=30
    )
    assert estimates[0].status == "estimated"
    assert estimated.betas[("T", "Y")] == pytest.approx(0.5, abs=0.1)
    assert ate == pytest.approx(1.0, abs=0.2)
