"""Tests for the composable synthetic simulation pipeline."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from nocap.scm_model import DirectedScm
from nocap.simulation import (
    SimulationConfig,
    SimulationState,
    counts_to_log_expression,
    counts_to_normalized_expression,
    default_simulation_stages,
    generate_from_scm,
    normalize_dispersions,
    numpy_linear_solver,
    paired_view_stages,
    sample_baseline_expression,
    sample_size_factors,
    sample_umi_counts,
)


def _one_edge_scm() -> DirectedScm:
    """Build the small two-node SCM shared by simulation tests."""
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

    assert result.latent_log_expression is not None
    assert result.exogenous_noise is not None
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
    assert first.observed_data is not None
    assert second.observed_data is not None
    pd.testing.assert_frame_equal(first.observed_data, second.observed_data)
    assert first.size_factors is not None
    assert first.baseline_expression is not None
    assert first.dispersions is not None
    assert first.normalized_expression is not None
    assert first.size_factors.shape == (12,)
    assert first.baseline_expression.shape == (2,)
    assert first.dispersions.shape == (2,)
    assert first.normalized_expression.shape == (12, 2)


def test_pipeline_can_expose_raw_umi_counts_as_observed_data():
    """Use integer UMI counts as the estimator-facing observed data when requested."""
    result = generate_from_scm(
        _one_edge_scm(),
        SimulationConfig(n_samples=12, use_umi_counts_as_observed_data=True),
        seed=7,
    )

    assert result.observed_data is not None
    assert result.umi_counts is not None
    pd.testing.assert_frame_equal(
        result.observed_data,
        pd.DataFrame(result.umi_counts, columns=("A", "B")),
    )
    assert np.issubdtype(result.observed_data.to_numpy().dtype, np.integer)


def test_binary_expression_state_is_truth_separate_from_biological_missingness():
    """Keep active/off truth distinct from sampling zeros and missing values."""
    config = SimulationConfig(
        n_samples=500,
        use_binary_expression_state=True,
        expression_state_intercept=0.0,
        expression_state_slope=2.0,
        missing_data_rate=0.2,
    )
    result = generate_from_scm(_one_edge_scm(), config, seed=19)

    assert result.latent_expression_state is not None
    assert result.umi_counts is not None
    assert result.observed_data is not None
    assert result.latent_expression_state.dtype == bool
    assert np.any(result.latent_expression_state)
    assert np.any(~result.latent_expression_state)
    assert np.all(result.umi_counts[~result.latent_expression_state] == 0)
    assert np.any(result.umi_counts[result.latent_expression_state] > 0)
    assert np.any(result.umi_counts[result.latent_expression_state] == 0)
    assert np.any(result.observed_data.to_numpy() == 0)


def test_expression_state_can_be_read_from_paired_artifact_in_sample_order():
    """Align paired binary truth with the artifact's paired count view."""
    from nocap.simulation import generate_paired_data_artifact

    config = SimulationConfig(
        n_samples=4,
        use_binary_expression_state=True,
        expression_state_intercept=0.0,
        expression_state_slope=1.0,
    )
    artifact = generate_paired_data_artifact(
        _one_edge_scm(),
        config,
        max_samples=8,
        scm_seed=8,
        data_seed=9,
    )

    truth = artifact.latent_expression_state_view(config)
    expected = artifact.latent_expression_state[artifact.sample_order[:4]]
    np.testing.assert_array_equal(truth.to_numpy(), expected)
    np.testing.assert_array_equal(artifact.umi_counts[artifact.sample_order[:4]][~expected], 0)


def test_binary_expression_state_is_opt_in_for_paired_artifacts():
    """Leave the default count model unchanged unless binary states are enabled."""
    from nocap.simulation import generate_paired_data_artifact

    artifact = generate_paired_data_artifact(
        _one_edge_scm(),
        SimulationConfig(n_samples=3),
        max_samples=3,
        scm_seed=3,
        data_seed=4,
    )

    assert artifact.latent_expression_state is None
    with pytest.raises(ValueError, match="use_binary_expression_state=True"):
        artifact.latent_expression_state_view(SimulationConfig(n_samples=3))


def test_count_observation_flag_uses_counts_in_paired_views():
    """Apply the count-observation flag to paired artifact materialization."""
    from nocap.simulation import generate_paired_data_artifact

    artifact = generate_paired_data_artifact(
        _one_edge_scm(),
        SimulationConfig(n_samples=4),
        max_samples=4,
        scm_seed=1,
        data_seed=2,
    )
    observed = artifact.view(SimulationConfig(n_samples=4, use_umi_counts_as_observed_data=True))

    pd.testing.assert_frame_equal(
        observed,
        pd.DataFrame(
            artifact.umi_counts[artifact.sample_order],
            columns=("A", "B"),
        ),
    )


def test_paired_binarization_is_generated_once_and_aligned(monkeypatch):
    """Normalize and log counts before binarizing, then align nested views."""
    from nocap import binarization
    from nocap.simulation import generate_paired_data_artifact

    fitted = []

    class FakeBinarizer:
        """Capture fit input and return reproducible binary expression."""

        def __init__(self, **kwargs):
            """Store fake constructor options for assertions."""
            self.kwargs = kwargs

        def fit(self, frame, *, simulation):
            """Record the fitting input and preserve its gene labels."""
            assert simulation is False
            fitted.append(frame.copy())
            self.columns = frame.columns
            return self

        def binarize(self, frame):
            """Return deterministic alternating binary expression values."""
            return pd.DataFrame(
                np.arange(frame.size, dtype=float).reshape(frame.shape) % 2,
                columns=self.columns,
            )

    monkeypatch.setattr(binarization, "scBoolSeq", FakeBinarizer)
    artifact = generate_paired_data_artifact(
        _one_edge_scm(),
        SimulationConfig(n_samples=3),
        max_samples=6,
        scm_seed=11,
        data_seed=12,
        binarize=True,
        binarizer_kwargs={"confidence": 0.9},
    )
    counts_before = artifact.umi_counts.copy()
    expected_expression = np.log1p(counts_before / artifact.size_factors[:, None])
    small_config = SimulationConfig(n_samples=3)
    large_config = SimulationConfig(n_samples=6)

    small = artifact.binarized_view(small_config)
    large = artifact.binarized_view(large_config)

    assert len(fitted) == 1
    assert artifact.binarized_expression is not None
    np.testing.assert_allclose(fitted[0].to_numpy(), expected_expression)
    assert fitted[0].columns.tolist() == ["A", "B"]
    expected_rows = artifact.sample_order[:3]
    np.testing.assert_array_equal(small.to_numpy(), artifact.binarized_expression[expected_rows])
    np.testing.assert_array_equal(
        large.to_numpy(), artifact.binarized_expression[artifact.sample_order]
    )
    pd.testing.assert_frame_equal(
        small,
        large.iloc[:3].set_axis(small.index),
    )
    np.testing.assert_array_equal(artifact.umi_counts, counts_before)


def test_paired_binary_unknown_policy_and_missingness_are_separate(monkeypatch):
    """Apply the generation unknown policy, then mask binary entries with NaN."""
    from nocap import binarization
    from nocap.simulation import generate_paired_data_artifact

    class FakeBinarizer:
        """Produce a deterministic binary matrix containing an unknown."""

        def __init__(self, **kwargs):
            """Accept the scBoolSeq-compatible constructor options."""

        def fit(self, frame, *, simulation):
            """Complete the fake fit without additional state."""
            return self

        def binarize(self, frame):
            """Return zero expression with one undetermined entry."""
            values = np.zeros(frame.shape, dtype=float)
            values[0, 0] = np.nan
            return pd.DataFrame(values, columns=frame.columns)

    monkeypatch.setattr(binarization, "scBoolSeq", FakeBinarizer)
    artifact = generate_paired_data_artifact(
        _one_edge_scm(),
        SimulationConfig(n_samples=4),
        max_samples=4,
        scm_seed=2,
        data_seed=3,
        binarize=True,
        unknown="off",
    )
    config = SimulationConfig(n_samples=4, missing_data_rate=1.0)
    original_counts = artifact.umi_counts.copy()
    original_uniforms = artifact.missingness_uniforms.copy()
    binary_expression = artifact.binarized_view(config)

    assert binary_expression.isna().all().all()
    np.testing.assert_array_equal(artifact.umi_counts, original_counts)
    np.testing.assert_array_equal(artifact.missingness_uniforms, original_uniforms)

    with pytest.raises(ValueError, match="unknown states"):
        generate_paired_data_artifact(
            _one_edge_scm(),
            SimulationConfig(n_samples=4),
            max_samples=4,
            scm_seed=4,
            data_seed=5,
            binarize=True,
            unknown="error",
        )


def test_paired_views_keep_legacy_frame_output_and_default_stages():
    """Keep default paired views continuous and require opt-in binary generation."""
    from nocap.simulation import generate_paired_data_artifact

    artifact = generate_paired_data_artifact(
        _one_edge_scm(),
        SimulationConfig(n_samples=4),
        max_samples=4,
        scm_seed=2,
        data_seed=3,
    )
    viewed = artifact.view(SimulationConfig(n_samples=4))

    assert callable(paired_view_stages()[0])
    assert len(paired_view_stages()) == 2
    assert isinstance(viewed, pd.DataFrame)
    assert viewed.columns.tolist() == ["A", "B"]
    with pytest.raises(ValueError, match="binarize=True"):
        artifact.binarized_view(SimulationConfig(n_samples=4))


def test_paired_custom_stage_can_transform_binary_output():
    """Allow custom stages to transform generated binary expression."""
    from nocap.simulation import generate_paired_data_artifact

    artifact = generate_paired_data_artifact(
        _one_edge_scm(),
        SimulationConfig(n_samples=3),
        max_samples=3,
        scm_seed=2,
        data_seed=3,
        binarize=True,
    )

    def custom_binary(state):
        """Populate binary expression from positive raw UMI counts."""
        state.binarized_data = pd.DataFrame(
            (state.umi_counts > 0).astype(float), columns=state.scm.nodes
        )
        return state

    binary = artifact.binarized_view(SimulationConfig(n_samples=3), stages=(custom_binary,))

    assert binary.shape == (3, 2)
    assert binary.columns.tolist() == ["A", "B"]


def test_size_factors_are_geometrically_centered():
    """Sample positive size factors whose geometric mean is one."""
    values = sample_size_factors(100, 0.4, np.random.default_rng(1))

    assert np.all(values > 0)
    assert np.isclose(np.exp(np.mean(np.log(values))), 1.0)


def test_baseline_expression_is_not_compositionally_normalized():
    """Sample positive q0 baselines without forcing a unit sum."""
    values = sample_baseline_expression(5, mean=1.0, dispersion=2.25, rng=np.random.default_rng(1))

    assert np.all(values > 0)
    assert not np.isclose(values.sum(), 1.0)


def test_baseline_expression_uses_negative_binomial_sampling():
    """Use the seeded negative-binomial baseline realization."""
    values = sample_baseline_expression(5, mean=1.0, dispersion=2.25, rng=np.random.default_rng(1))

    expected = (
        np.random.default_rng(1)
        .negative_binomial(n=1 / 2.25, p=(1 / 2.25) / ((1 / 2.25) + 1.0), size=5)
        .astype(float)
        + 1.0
    )
    np.testing.assert_array_equal(values, expected)


def test_dispersions_broadcast_and_copy_gene_vectors():
    """Normalize scalar and gene-specific dispersion inputs."""
    scalar = normalize_dispersions(0.1, 3)
    vector = normalize_dispersions([0.1, 0.2, 0.3], 3)

    np.testing.assert_array_equal(scalar, [0.1, 0.1, 0.1])
    np.testing.assert_array_equal(vector, [0.1, 0.2, 0.3])


def test_count_transforms_use_canonical_qhat_and_xhat_formulas():
    """Keep q-hat and X-hat conversions distinct and explicit."""
    counts = np.array([[9.0, 19.0]])
    size_factors = np.array([2.0])
    baseline = np.array([2.0, 5.0])

    np.testing.assert_allclose(
        counts_to_normalized_expression(counts, size_factors, 1.0), [[5.0, 10.0]]
    )
    np.testing.assert_allclose(
        counts_to_log_expression(counts, size_factors, baseline, 1.0),
        np.log2([[10.0 / 4.0, 20.0 / 10.0]]),
    )


def test_default_stages_can_be_extended():
    """Allow callers to insert custom stages into the default pipeline."""
    stages = list(default_simulation_stages())
    observed_metadata = {}

    def record_stage(state: SimulationState) -> SimulationState:
        """Record the latent expression shape while passing state through."""
        assert state.latent_log_expression is not None
        observed_metadata["latent_shape"] = state.latent_log_expression.shape
        return state

    stages.insert(2, record_stage)
    result = generate_from_scm(_one_edge_scm(), SimulationConfig(n_samples=4), stages=stages)

    assert observed_metadata["latent_shape"] == (4, 2)
    assert result.observed_data is not None
    assert result.observed_data.shape == (4, 2)


def test_pipeline_rejects_observation_before_solving():
    """Reject a pipeline that attempts observation before latent expression exists."""
    with pytest.raises(ValueError, match="Latent expression must be generated"):
        generate_from_scm(
            _one_edge_scm(),
            SimulationConfig(n_samples=3),
            stages=(default_simulation_stages()[3],),
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
    with pytest.raises(ValueError, match="size_factor_log_sd"):
        sample_size_factors(2, -1.0, np.random.default_rng(1))

    with pytest.raises(ValueError, match="n_genes"):
        sample_baseline_expression(0, dispersion=1.0, rng=np.random.default_rng(1))

    with pytest.raises(ValueError, match="shape"):
        sample_umi_counts(
            np.zeros(2),
            np.ones(2),
            np.ones(1),
            0.1,
            np.random.default_rng(1),
        )


def test_sample_umi_counts_accepts_gene_specific_dispersion():
    """Use one negative-binomial dispersion value per gene."""
    counts = sample_umi_counts(
        np.zeros((2, 2)),
        np.ones(2),
        np.ones(2),
        [0.1, 0.2],
        np.random.default_rng(1),
    )

    assert counts.shape == (2, 2)
    assert np.issubdtype(counts.dtype, np.integer)
