"""Tests for the composable synthetic simulation pipeline."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from nocap.scm_model import DirectedScm
from nocap.simulation import (
    SimulationConfig,
    SimulationState,
    default_simulation_stages,
    generate_from_scm,
    numpy_linear_solver,
    sample_baseline_abundances,
    sample_library_sizes,
    sample_umi_counts,
)


def _one_edge_scm() -> DirectedScm:
    return DirectedScm(
        nodes=("A", "B"),
        graph=nx.DiGraph([("A", "B")]),
        betas={("A", "B"): 2.0},
    )


def test_numpy_linear_solver_solves_one_edge_model():
    """Solve a one-edge structural model using the NumPy linear solver."""
    scm = _one_edge_scm()
    exogenous = np.array([[3.0, 5.0]])

    latent = numpy_linear_solver(scm, exogenous)

    np.testing.assert_allclose(latent, [[3.0, 11.0]])


def test_numpy_linear_solver_handles_cycles():
    """Solve cyclic structural equations by inverting their coefficient system."""
    scm = DirectedScm(
        nodes=("A", "B"),
        graph=nx.DiGraph([("A", "B"), ("B", "A")]),
        betas={("A", "B"): 0.2, ("B", "A"): 0.3},
    )
    exogenous = np.array([[1.0, 2.0]])

    latent = numpy_linear_solver(scm, exogenous)

    np.testing.assert_allclose(latent, [[1.70212766, 2.34042553]])


def test_numpy_linear_solver_supports_fixed_intervention_values():
    """Hold an intervened variable fixed while solving downstream equations."""
    scm = _one_edge_scm()
    exogenous = np.array([[3.0, 5.0]])

    latent = numpy_linear_solver(scm, exogenous, {"A": 7.0})

    np.testing.assert_allclose(latent, [[7.0, 19.0]])


def test_fixed_intervention_changes_generated_latent_values_but_not_noise():
    """Fixed interventions affect the solve while preserving generated noise."""
    config = SimulationConfig(n_samples=8, fixed_intervention_values={"A": 0.0})
    result = generate_from_scm(_one_edge_scm(), config, seed=11)

    np.testing.assert_array_equal(result.latent_log_expression[:, 0], 0.0)
    np.testing.assert_allclose(result.latent_log_expression[:, 1], result.exogenous_noise[:, 1])


def test_default_pipeline_is_reproducible_and_retains_state_fields():
    """Keep generated state fields and reproduce them for a fixed seed."""
    config = SimulationConfig(n_samples=12)
    first = generate_from_scm(_one_edge_scm(), config, seed=7)
    second = generate_from_scm(_one_edge_scm(), config, seed=7)

    np.testing.assert_array_equal(first.exogenous_noise, second.exogenous_noise)
    np.testing.assert_array_equal(first.latent_log_expression, second.latent_log_expression)
    np.testing.assert_array_equal(first.umi_counts, second.umi_counts)
    pd.testing.assert_frame_equal(first.observed_data, second.observed_data)
    assert first.library_sizes.shape == (12,)
    assert first.baseline_abundances.shape == (2,)


def test_default_stages_can_be_extended():
    """Allow callers to insert custom stages into the default pipeline."""
    stages = list(default_simulation_stages())
    observed_metadata = {}

    def record_stage(state: SimulationState) -> SimulationState:
        """Record the latent expression shape while passing state through."""
        observed_metadata["latent_shape"] = state.latent_log_expression.shape
        return state

    stages.insert(2, record_stage)
    result = generate_from_scm(_one_edge_scm(), SimulationConfig(n_samples=4), stages=stages)

    assert observed_metadata["latent_shape"] == (4, 2)
    assert result.observed_data.shape == (4, 2)


def test_pipeline_rejects_observation_before_solving():
    """Reject a pipeline that attempts observation before latent expression exists."""
    with pytest.raises(ValueError, match="Latent expression must be generated"):
        generate_from_scm(
            _one_edge_scm(),
            SimulationConfig(n_samples=3),
            stages=(default_simulation_stages()[2],),
        )


def test_biological_missingness_writes_zero_values():
    """Represent complete biological missingness as zero observed values."""
    result = generate_from_scm(
        _one_edge_scm(),
        SimulationConfig(n_samples=100, missing_data_rate=1.0),
        seed=3,
    )

    assert (result.observed_data == 0.0).all().all()


def test_instrument_missingness_validates_direction():
    """Validate the self-masking direction for instrument-error missingness."""
    with pytest.raises(ValueError, match="self_mask_direction"):
        generate_from_scm(
            _one_edge_scm(),
            SimulationConfig(
                n_samples=4,
                missing_data_rate=0.5,
                missing_data_mechanism="instrument_error",
                self_mask_direction="invalid",
            ),
        )


def test_combined_missingness_applies_both_mechanisms():
    """Allow biological and instrument missingness in one observation run."""
    result = generate_from_scm(
        _one_edge_scm(),
        SimulationConfig(
            n_samples=100,
            missing_data_rate=0.5,
            missing_data_mechanism="biological_error+instrument_error",
        ),
        seed=3,
    )

    assert (result.observed_data == 0.0).any().all()
    assert (result.observed_data == 0.0).to_numpy().sum() > 100


def test_observation_helpers_validate_shapes_and_parameters():
    """Validate observation helper parameters and compatible array shapes."""
    with pytest.raises(ValueError, match="library_size_log_sd"):
        sample_library_sizes(2, log_mean=1.0, log_sd=-1.0, rng=np.random.default_rng(1))

    with pytest.raises(ValueError, match="n_genes"):
        sample_baseline_abundances(0, log_sd=1.0, rng=np.random.default_rng(1))

    with pytest.raises(ValueError, match="shape"):
        sample_umi_counts(
            np.zeros(2),
            np.ones(2),
            np.ones(1),
            dispersion=0.1,
            rng=np.random.default_rng(1),
        )
