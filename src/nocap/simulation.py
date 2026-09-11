"""Composable synthetic SCM generation and UMI observation pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import networkx as nx
import numpy as np
import pandas as pd

from .experiment import StageSeeds
from .scm_model import DirectedScm


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for generation and observation."""

    n_samples: int
    dispersion: float | Sequence[float] = 0.1
    size_factor_log_sd: float = 0.4
    umi_pseudocount: float = 1.0
    baseline_expression_mean: float = 1.0
    baseline_expression_dispersion: float = 2.25
    missing_data_rate: float = 0.0
    missing_data_mechanism: str = "biological_error"
    self_mask_quantile: float = 0.25
    self_mask_k: float = 8.0
    self_mask_direction: str = "low"
    scc_confounding_strength: float = 0.0
    estimation_graph: nx.DiGraph | None = None
    fixed_intervention_values: dict[str, float] = field(default_factory=dict)
    use_latent_expression_hat: bool = True
    use_umi_counts_as_observed_data: bool = False


@dataclass
class SimulationState:
    """Mutable state shared by ordered generation stages."""

    scm: DirectedScm
    config: SimulationConfig
    rng: np.random.Generator
    exogenous_noise: np.ndarray | None = None
    latent_log_expression: np.ndarray | None = None
    umi_counts: np.ndarray | None = None
    size_factors: np.ndarray | None = None
    baseline_expression: np.ndarray | None = None
    dispersions: np.ndarray | None = None
    normalized_expression: np.ndarray | None = None
    observed_data: pd.DataFrame | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PairedDataArtifact:
    """Immutable maximum-size data realization used by paired conditions."""

    scm: DirectedScm
    exogenous_noise: np.ndarray
    latent_log_expression: np.ndarray
    umi_counts: np.ndarray
    size_factors: np.ndarray
    baseline_expression: np.ndarray
    dispersions: np.ndarray
    normalized_expression: np.ndarray
    sample_order: np.ndarray
    missingness_uniforms: np.ndarray
    stage_seeds: StageSeeds

    def view(
        self,
        config: SimulationConfig,
        *,
        stages: Sequence[SimulationStage] | None = None,
    ) -> pd.DataFrame:
        """Materialize a deterministic prefix view through configurable stages.

        The default stages only materialize the immutable observation artifact and
        apply condition-specific missingness. Callers may provide a replacement
        stage sequence to add transformations or alternative observation models.
        Stages receive a normal :class:`SimulationState`; the artifact fields are
        also available in ``state.metadata`` for custom stages.
        """
        if config.n_samples > len(self.sample_order):
            raise ValueError("Condition n_samples exceeds artifact maximum size")
        indices = self.sample_order[: config.n_samples]
        state = SimulationState(
            scm=self.scm,
            config=config,
            rng=np.random.default_rng(0),
            exogenous_noise=self.exogenous_noise[indices],
            latent_log_expression=self.latent_log_expression[indices],
            umi_counts=self.umi_counts[indices],
            size_factors=self.size_factors[indices],
            baseline_expression=self.baseline_expression,
            dispersions=self.dispersions,
            normalized_expression=self.normalized_expression[indices],
            metadata={
                "missingness_uniforms": self.missingness_uniforms[indices],
                "sample_parent_id": "maximum",
                "sample_selection_rule": "prefix",
                "pairing_scope": "shared_complete_observation_shared_scores",
            },
        )
        ordered = paired_view_stages() if stages is None else stages
        for stage in ordered:
            state = stage(state)
        if state.observed_data is None:
            raise ValueError("Paired view stages must produce observed_data")
        state.observed_data.attrs.update(state.metadata)
        return state.observed_data


def paired_view_stages() -> tuple[SimulationStage, ...]:
    """Return the default stages used to materialize a paired artifact view."""
    return (_view_observe, _view_missing)


def _view_observe(state: SimulationState) -> SimulationState:
    """Convert immutable artifact counts into condition-view expression data."""
    if (
        state.umi_counts is None
        or state.size_factors is None
        or state.baseline_expression is None
        or state.dispersions is None
    ):
        raise ValueError("Paired view requires immutable observation arrays")
    if state.normalized_expression is None:
        state.normalized_expression = counts_to_normalized_expression(
            state.umi_counts,
            state.size_factors,
            pseudocount=state.config.umi_pseudocount,
        )
    if state.config.use_umi_counts_as_observed_data:
        observed_expression = state.umi_counts.copy()
    elif state.config.use_latent_expression_hat:
        observed_expression = counts_to_log_expression(
            state.umi_counts,
            state.size_factors,
            state.baseline_expression,
            pseudocount=state.config.umi_pseudocount,
        )
    else:
        if state.latent_log_expression is None:
            raise ValueError("Latent expression must be generated before observation.")
        observed_expression = state.latent_log_expression
    state.observed_data = pd.DataFrame(observed_expression, columns=state.scm.nodes)
    return state


def _view_missing(state: SimulationState) -> SimulationState:
    """Apply condition-specific missingness using persisted shared uniforms."""
    if state.observed_data is None:
        raise ValueError("Paired observation must be materialized before missingness")
    uniforms = state.metadata.get("missingness_uniforms")
    if uniforms is None:
        raise ValueError("Paired view requires persisted missingness uniforms")
    if state.config.missing_data_rate > 0:
        state.observed_data = _apply_missingness_arrays(
            state.observed_data,
            state.latent_log_expression,
            uniforms,
            state.config,
        )
    return state


def _apply_missingness_arrays(data, latent, uniforms, config):
    mechanisms = _missing_data_mechanisms(config.missing_data_mechanism)
    for j, col in enumerate(data.columns):
        rate = np.clip(float(config.missing_data_rate), 0, 1)
        probabilities = []
        if "biological_error" in mechanisms:
            probabilities.append(np.full(len(data), rate))
        if "instrument_error" in mechanisms:
            x = latent[:, j]
            threshold = np.quantile(x, config.self_mask_quantile)
            direction = config.self_mask_direction.lower()
            raw = (
                1 / (1 + np.exp(-config.self_mask_k * (threshold - x)))
                if direction == "low"
                else 1 / (1 + np.exp(-config.self_mask_k * (x - threshold)))
            )
            probabilities.append(np.clip(rate * raw / raw.mean(), 0, 1))
        # The mechanisms are independent causes of missingness.
        probability = 1.0 - np.prod([1.0 - p for p in probabilities], axis=0)
        missing_value = 0.0
        data.loc[uniforms[:, j] < probability, col] = missing_value
    return data


def _missing_data_mechanisms(value: str) -> set[str]:
    """Normalize one or more missingness mechanism names."""
    aliases = {
        "biological_error": "biological_error",
        "biological": "biological_error",
        "bio": "biological_error",
        "instrument_error": "instrument_error",
        "instrument": "instrument_error",
        "instrument-self-masking": "instrument_error",
    }
    names = {part.strip().lower() for part in value.replace("+", ",").split(",") if part.strip()}
    mechanisms = {aliases.get(name) for name in names}
    if not names or None in mechanisms:
        raise ValueError(f"Unknown missing-data mechanism: {value!r}")
    return {mechanism for mechanism in mechanisms if mechanism is not None}


def generate_paired_data_artifact(
    scm: DirectedScm,
    config: SimulationConfig,
    *,
    max_samples: int,
    scm_seed: int,
    data_seed: int,
    fixed_intervention_values: dict[str, float] | None = None,
) -> PairedDataArtifact:
    """Generate one immutable maximum-size realization for paired conditions."""
    if max_samples < config.n_samples:
        raise ValueError("max_samples must be at least config.n_samples")
    seeds = StageSeeds.from_roots(scm_seed, data_seed)
    latent_rng = np.random.default_rng(seeds.data_latent_seed)
    observation_rng = np.random.default_rng(seeds.observation_seed)
    missing_rng = np.random.default_rng(seeds.missingness_score_seed)
    order_rng = np.random.default_rng(seeds.sample_order_seed)

    # Generate exogenous noise and solve the linear SCM according to this noise
    noise = latent_rng.normal(size=(max_samples, len(scm.nodes)))
    latent = numpy_linear_solver(
        scm, noise, fixed_intervention_values or config.fixed_intervention_values
    )

    # Generate observational UMI counts
    size_factors = sample_size_factors(
        max_samples,
        log_sd=config.size_factor_log_sd,
        rng=observation_rng,
    )
    baseline = sample_baseline_expression(
        len(scm.nodes),
        mean=config.baseline_expression_mean,
        dispersion=config.baseline_expression_dispersion,
        rng=observation_rng,
    )
    dispersions = normalize_dispersions(config.dispersion, len(scm.nodes))
    counts = sample_umi_counts(latent, size_factors, baseline, dispersions, observation_rng)
    normalized_expression = counts_to_normalized_expression(
        counts,
        size_factors,
        pseudocount=config.umi_pseudocount,
    )

    return PairedDataArtifact(
        scm=scm,
        exogenous_noise=noise,
        latent_log_expression=latent,
        umi_counts=counts,
        size_factors=size_factors,
        baseline_expression=baseline,
        dispersions=dispersions,
        normalized_expression=normalized_expression,
        sample_order=order_rng.permutation(max_samples),
        missingness_uniforms=missing_rng.random((max_samples, len(scm.nodes))),
        stage_seeds=seeds,
    )


class SimulationStage(Protocol):
    """Protocol for an ordered stage that transforms simulation state."""

    def __call__(self, state: SimulationState) -> SimulationState:
        """Transform and return the shared simulation state."""
        ...


class NumpyLinearSolver(Protocol):
    """Protocol for solvers consuming a neutral SCM and exogenous values."""

    def __call__(
        self,
        scm: DirectedScm,
        exogenous_noise: np.ndarray,
        fixed_intervention_values: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Solve the SCM and return sample-row latent values."""
        ...


def default_simulation_stages() -> tuple[SimulationStage, ...]:
    """Return the standard ordered synthetic-data generation stages.

    The returned tuple can be converted to a list and extended with custom
    stages while retaining the default behavior:

    ``noise -> solve -> UMI observation -> missingness``.
    """
    return (_noise, _solve, _observe, _missing)


def _noise(state):
    """Generate exogenous noise with optional SCC-level latent confounding."""
    n, p = state.config.n_samples, len(state.scm.nodes)
    eps = state.rng.normal(size=(n, p))
    graph = state.config.estimation_graph or state.scm.graph

    # Add one shared latent factor to every node in each non-trivial SCC.
    if state.config.scc_confounding_strength > 0:
        for scc in nx.strongly_connected_components(graph):
            if len(scc) > 1:
                factor = state.rng.normal(size=n)
                for node in scc:
                    eps[:, state.scm.nodes.index(str(node))] += (
                        state.config.scc_confounding_strength * factor
                    )
    state.exogenous_noise = eps
    return state


def sample_size_factors(
    n_samples: int,
    log_sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample positive size factors with geometric mean one."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if not np.isfinite(log_sd) or log_sd < 0:
        raise ValueError("size_factor_log_sd must be finite and non-negative.")

    raw = rng.lognormal(mean=0.0, sigma=log_sd, size=n_samples)
    return raw / np.exp(np.mean(np.log(raw)))


def sample_baseline_expression(
    n_genes: int,
    *,
    mean: float = 1.0,
    dispersion: float = 2.25,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample positive gene-specific q0 baselines from a zero-inflated NB.

    The negative-binomial mean is ``mean`` and its dispersion is ``dispersion``
    in the variance parameterization ``var = mean + dispersion * mean**2``.
    A unit pseudocount keeps the baseline strictly positive, which is required
    by the count-to-expression transforms.
    """
    if n_genes <= 0:
        raise ValueError("n_genes must be positive.")
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("baseline_expression_mean must be finite and positive.")
    if not np.isfinite(dispersion) or dispersion <= 0:
        raise ValueError("baseline_expression_dispersion must be finite and positive.")

    n = 1.0 / dispersion
    p = n / (n + mean)
    return rng.negative_binomial(n=n, p=p, size=n_genes).astype(float) + 1.0


def sample_zero_inflated_negative_binomial(
    means: np.ndarray,
    dispersions: np.ndarray,
    rng: np.random.Generator,
    zero_inflation: float = 0.1,
) -> np.ndarray:
    """Sample a zero-inflated negative-binomial array.

    The NB uses ``var = mean + dispersion * mean**2``.  A single inflation
    probability is intentionally kept here so this helper can be shared by
    baseline and UMI generation without adding configuration parameters.
    """
    means = np.asarray(means, dtype=float)
    dispersions = np.asarray(dispersions, dtype=float)
    if means.shape != dispersions.shape or not np.all(np.isfinite(means)):
        raise ValueError("means and dispersions must have matching finite shapes.")
    if np.any(means <= 0) or not np.all(np.isfinite(dispersions)) or np.any(dispersions <= 0):
        raise ValueError("means must be positive and dispersions must be finite and positive.")
    if not 0 <= zero_inflation < 1:
        raise ValueError("zero_inflation must be in [0, 1).")
    n = 1.0 / dispersions
    p = n / (n + means)
    counts = rng.negative_binomial(n=n, p=p)
    return np.where(rng.random(means.shape) < zero_inflation, 0, counts)


def normalize_dispersions(
    dispersions: float | Sequence[float] | np.ndarray, n_genes: int
) -> np.ndarray:
    """Broadcast a positive scalar dispersion or validate a gene vector."""
    if n_genes <= 0:
        raise ValueError("n_genes must be positive.")
    values = np.asarray(dispersions, dtype=float)
    if values.ndim == 0:
        values = np.full(n_genes, float(values))
    elif values.ndim == 1 and len(values) == n_genes:
        values = values.copy()
    else:
        raise ValueError("dispersions must be a scalar or have shape (n_genes,).")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("dispersions must be finite and positive.")
    return values


def _validate_observation_arrays(
    latent_log_expression: np.ndarray,
    size_factors: np.ndarray,
    baseline_expression: np.ndarray,
    dispersions: np.ndarray,
) -> tuple[int, int]:
    """Validate observation dimensions and return sample and gene counts."""
    if latent_log_expression.ndim != 2:
        raise ValueError("latent_log_expression must have shape (n_samples, n_genes).")
    n_samples, n_genes = latent_log_expression.shape
    if not np.all(np.isfinite(latent_log_expression)):
        raise ValueError("latent_log_expression must be finite.")
    if size_factors.shape != (n_samples,):
        raise ValueError("size_factors must have shape (n_samples,).")
    if not np.all(np.isfinite(size_factors)) or np.any(size_factors <= 0):
        raise ValueError("size_factors must be finite and positive.")
    if baseline_expression.shape != (n_genes,):
        raise ValueError("baseline_expression must have shape (n_genes,).")
    if not np.all(np.isfinite(baseline_expression)) or np.any(baseline_expression <= 0):
        raise ValueError("baseline_expression must be finite and positive.")
    if dispersions.shape != (n_genes,):
        raise ValueError("dispersions must have shape (n_genes,).")
    if not np.all(np.isfinite(dispersions)) or np.any(dispersions <= 0):
        raise ValueError("dispersions must be finite and positive.")
    return n_samples, n_genes


def sample_umi_counts(
    latent_log_expression: np.ndarray,
    size_factors: np.ndarray,
    baseline_expression: np.ndarray,
    dispersions: float | Sequence[float] | np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate counts from the latent expression and NB observation model.

    The model is ``mu_ig = s_i * q0_g * 2**X_ig`` and
    ``Y_ig ~ NB(mu_ig, alpha_g)`` with variance ``mu + alpha_g * mu**2``.
    """
    if latent_log_expression.ndim != 2:
        raise ValueError("latent_log_expression must have shape (n_samples, n_genes).")
    _, n_genes = latent_log_expression.shape
    normalized_dispersions = normalize_dispersions(dispersions, n_genes)
    _validate_observation_arrays(
        latent_log_expression,
        size_factors,
        baseline_expression,
        normalized_dispersions,
    )

    mu = size_factors[:, None] * baseline_expression[None, :] * np.exp2(latent_log_expression)
    n = 1.0 / normalized_dispersions[None, :]
    p = n / (n + mu)
    return rng.negative_binomial(n=n, p=p)


def counts_to_log_expression(
    counts: np.ndarray,
    size_factors: np.ndarray,
    baseline_expression: np.ndarray,
    pseudocount: float,
) -> np.ndarray:
    """Estimate latent X as ``log2((Y + c) / (s * q0))``."""
    counts, size_factors, baseline_expression = _validate_count_inputs(
        counts, size_factors, baseline_expression, pseudocount
    )
    return np.log2((counts + pseudocount) / (size_factors[:, None] * baseline_expression[None, :]))


def counts_to_normalized_expression(
    counts: np.ndarray,
    size_factors: np.ndarray,
    pseudocount: float,
) -> np.ndarray:
    """Convert counts to q-hat on the normalized q scale.

    The pseudocount is added to counts before dividing by the dimensionless
    size factor: ``q_hat = (Y + c) / s``. This is intentionally separate from
    the X-hat conversion, which also divides by the gene-specific q0 baseline.
    """
    counts = np.asarray(counts)
    if counts.ndim != 2:
        raise ValueError("counts must have shape (n_samples, n_genes).")
    if size_factors.shape != (counts.shape[0],):
        raise ValueError("size_factors must have shape (n_samples,).")
    if not np.all(np.isfinite(size_factors)) or np.any(size_factors <= 0):
        raise ValueError("size_factors must be finite and positive.")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be finite and non-negative.")
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("umi_pseudocount must be finite and positive.")
    return (counts + pseudocount) / size_factors[:, None]


def _validate_count_inputs(
    counts: np.ndarray,
    size_factors: np.ndarray,
    baseline_expression: np.ndarray,
    pseudocount: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate arrays used by count-to-expression conversions."""
    counts = np.asarray(counts)
    size_factors = np.asarray(size_factors)
    baseline_expression = np.asarray(baseline_expression)
    if counts.ndim != 2:
        raise ValueError("counts must have shape (n_samples, n_genes).")
    if size_factors.shape != (counts.shape[0],):
        raise ValueError("size_factors must have shape (n_samples,).")
    if baseline_expression.shape != (counts.shape[1],):
        raise ValueError("baseline_expression must have shape (n_genes,).")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be finite and non-negative.")
    if not np.all(np.isfinite(size_factors)) or np.any(size_factors <= 0):
        raise ValueError("size_factors must be finite and positive.")
    if not np.all(np.isfinite(baseline_expression)) or np.any(baseline_expression <= 0):
        raise ValueError("baseline_expression must be finite and positive.")
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("umi_pseudocount must be finite and positive.")
    return counts, size_factors, baseline_expression


def numpy_linear_solver(
    scm: DirectedScm,
    exogenous_noise: np.ndarray,
    fixed_intervention_values: dict[str, float] | None = None,
) -> np.ndarray:
    """Solve ``X = B @ X + eps`` for sample-row data.

    The public matrix is edge-oriented, with ``beta_matrix[source, target]``.
    Its transpose is the mathematical target-parent matrix, so the system for
    each sample is ``(I - beta_matrix.T) X = eps``.

    If fixed_intervention_values is not None, then explicit hard interventions
    are applied.
    In the case of solving this for some SCM that represents the mutilated SCM
    after applying an intervention, but no fixed_values are set, then solving
    is equivalent to a stochastic hard intervention where the values of the
    intervened nodes are equal to their exogenous noise terms.
    """
    coefficient_matrix = np.eye(len(scm.nodes)) - scm.beta_matrix.T

    fixed_values = fixed_intervention_values or {}
    try:
        if fixed_values:
            indices = {node: i for i, node in enumerate(scm.nodes)}
            fixed = {indices[node]: value for node, value in fixed_values.items()}
            free = [i for i in range(len(scm.nodes)) if i not in fixed]
            latent_transposed = np.zeros((len(scm.nodes), exogenous_noise.shape[0]))
            for index, value in fixed.items():
                latent_transposed[index] = value
            if free:
                fixed_indices = list(fixed)
                rhs = (
                    exogenous_noise[:, free].T
                    - coefficient_matrix[np.ix_(free, fixed_indices)]
                    @ latent_transposed[fixed_indices]
                )
                latent_transposed[free] = np.linalg.solve(
                    coefficient_matrix[np.ix_(free, free)], rhs
                )
        else:
            latent_transposed = np.linalg.solve(coefficient_matrix, exogenous_noise.T)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "Synthetic SCM linear system is singular / ill-conditioned. "
            "Try smaller beta magnitudes."
        ) from error

    return latent_transposed.T


def _solve(
    state: SimulationState,
    solver: NumpyLinearSolver | None = None,
) -> SimulationState:
    """Solve the neutral SCM using the configured solver."""
    if state.exogenous_noise is None:
        raise ValueError("Exogenous noise must be generated before solving the SCM.")

    state.latent_log_expression = (solver or numpy_linear_solver)(
        state.scm, state.exogenous_noise, state.config.fixed_intervention_values
    )
    return state


def _observe(state: SimulationState) -> SimulationState:
    """Generate integer counts, q-hat, and estimator-facing X-hat."""
    config = state.config

    if state.latent_log_expression is None:
        raise ValueError("Latent expression must be generated before observation.")

    state.size_factors = sample_size_factors(
        config.n_samples,
        log_sd=config.size_factor_log_sd,
        rng=state.rng,
    )
    state.baseline_expression = sample_baseline_expression(
        len(state.scm.nodes),
        mean=config.baseline_expression_mean,
        dispersion=config.baseline_expression_dispersion,
        rng=state.rng,
    )
    state.dispersions = normalize_dispersions(config.dispersion, len(state.scm.nodes))

    state.umi_counts = sample_umi_counts(
        state.latent_log_expression,
        state.size_factors,
        state.baseline_expression,
        state.dispersions,
        rng=state.rng,
    )

    state.normalized_expression = counts_to_normalized_expression(
        state.umi_counts,
        state.size_factors,
        pseudocount=config.umi_pseudocount,
    )
    if config.use_umi_counts_as_observed_data:
        observed_expression = state.umi_counts.copy()
    elif config.use_latent_expression_hat:
        observed_expression = counts_to_log_expression(
            state.umi_counts,
            state.size_factors,
            state.baseline_expression,
            pseudocount=config.umi_pseudocount,
        )
    else:
        if state.latent_log_expression is None:
            raise ValueError("Latent expression must be generated before observation.")
        observed_expression = state.latent_log_expression
    state.observed_data = pd.DataFrame(
        observed_expression,
        columns=state.scm.nodes,
    )
    return state


def _missing(state: SimulationState) -> SimulationState:
    """Apply biological or expression-dependent instrument missingness."""
    config = state.config
    if state.observed_data is None:
        raise ValueError("Observed data must be generated before missingness.")
    rate = np.clip(float(config.missing_data_rate), 0, 1)
    if rate <= 0:
        return state
    mechanisms = _missing_data_mechanisms(config.missing_data_mechanism)
    for j, col in enumerate(state.scm.nodes):
        probabilities = []
        if "biological_error" in mechanisms:
            probabilities.append(np.full(config.n_samples, rate))
        if "instrument_error" in mechanisms:
            latent_log_expression = state.latent_log_expression
            if latent_log_expression is None:
                raise ValueError("Latent expression must be generated before missingness.")
            x = latent_log_expression[:, j]
            threshold = np.quantile(x, config.self_mask_quantile)
            direction = config.self_mask_direction.lower()
            if direction == "low":
                raw = 1 / (1 + np.exp(-config.self_mask_k * (threshold - x)))
            elif direction == "high":
                raw = 1 / (1 + np.exp(-config.self_mask_k * (x - threshold)))
            else:
                raise ValueError("self_mask_direction must be one of {low,high}.")
            probabilities.append(np.clip(rate * raw / raw.mean(), 0, 1))
        probability = 1.0 - np.prod([1.0 - p for p in probabilities], axis=0)
        missing_value = 0.0
        state.observed_data.loc[state.rng.random(config.n_samples) < probability, col] = (
            missing_value
        )
    return state


def generate_from_scm(
    scm: DirectedScm,
    config: SimulationConfig,
    *,
    seed: int = 0,
    rng: np.random.Generator | None = None,
    stages: Sequence[SimulationStage] | None = None,
) -> SimulationState:
    """Generate data from a preconstructed SCM using ordered stages."""
    state = SimulationState(scm, config, np.random.default_rng(seed) if rng is None else rng)
    ordered = default_simulation_stages() if stages is None else stages
    for stage in ordered:
        state = stage(state)
    return state
