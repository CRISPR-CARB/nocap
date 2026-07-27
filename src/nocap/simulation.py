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
    umi_dispersion: float = 0.1
    library_size_log_mean: float = float(np.log(10_000))
    library_size_log_sd: float = 0.4
    umi_pseudocount: float = 1.0
    baseline_log_sd: float = 2.0
    missing_data_rate: float = 0.0
    missing_data_mechanism: str = "biological_error"
    self_mask_quantile: float = 0.25
    self_mask_k: float = 8.0
    self_mask_direction: str = "low"
    scc_confounding_strength: float = 0.0
    estimation_graph: nx.DiGraph | None = None


@dataclass
class SimulationState:
    """Mutable state shared by ordered generation stages."""

    scm: DirectedScm
    config: SimulationConfig
    rng: np.random.Generator
    exogenous_noise: np.ndarray | None = None
    latent_log_expression: np.ndarray | None = None
    umi_counts: np.ndarray | None = None
    library_sizes: np.ndarray | None = None
    baseline_abundances: np.ndarray | None = None
    observed_data: pd.DataFrame | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PairedDataArtifact:
    """Immutable maximum-size data realization used by paired conditions."""

    scm: DirectedScm
    exogenous_noise: np.ndarray
    latent_log_expression: np.ndarray
    umi_counts: np.ndarray
    library_sizes: np.ndarray
    baseline_abundances: np.ndarray
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
            library_sizes=self.library_sizes[indices],
            baseline_abundances=self.baseline_abundances,
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
    if state.umi_counts is None or state.library_sizes is None:
        raise ValueError("Paired view requires immutable observation arrays")
    normalized = counts_to_log_expression(
        state.umi_counts,
        state.library_sizes,
        pseudocount=state.config.umi_pseudocount,
        target_library_size=np.exp(state.config.library_size_log_mean),
    )
    state.observed_data = pd.DataFrame(normalized, columns=state.scm.nodes)
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
    for j, col in enumerate(data.columns):
        rate = np.clip(float(config.missing_data_rate), 0, 1)
        if config.missing_data_mechanism.lower() in {"biological_error", "biological", "bio"}:
            probability = np.full(len(data), rate)
        elif config.missing_data_mechanism.lower() in {
            "instrument_error",
            "instrument",
            "instrument-self-masking",
        }:
            x = latent[:, j]
            threshold = np.quantile(x, config.self_mask_quantile)
            direction = config.self_mask_direction.lower()
            raw = (
                1 / (1 + np.exp(-config.self_mask_k * (threshold - x)))
                if direction == "low"
                else 1 / (1 + np.exp(-config.self_mask_k * (x - threshold)))
            )
            probability = np.clip(rate * raw / raw.mean(), 0, 1)
        else:
            raise ValueError(f"Unknown missing-data mechanism: {config.missing_data_mechanism!r}")
        data.loc[uniforms[:, j] < probability, col] = 0.0
    return data


def generate_paired_data_artifact(
    scm: DirectedScm,
    config: SimulationConfig,
    *,
    max_samples: int,
    scm_seed: int,
    data_seed: int,
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
    latent = numpy_linear_solver(scm, noise)

    # Generate observational UMI counts
    libraries = sample_library_sizes(
        max_samples,
        log_mean=config.library_size_log_mean,
        log_sd=config.library_size_log_sd,
        rng=observation_rng,
    )
    baseline = sample_baseline_abundances(
        len(scm.nodes), log_sd=config.baseline_log_sd, rng=observation_rng
    )
    counts = sample_umi_counts(
        latent, libraries, baseline, dispersion=config.umi_dispersion, rng=observation_rng
    )

    return PairedDataArtifact(
        scm,
        noise,
        latent,
        counts,
        libraries,
        baseline,
        order_rng.permutation(max_samples),
        missing_rng.random((max_samples, len(scm.nodes))),
        seeds,
    )


class SimulationStage(Protocol):
    """Protocol for an ordered stage that transforms simulation state."""

    def __call__(self, state: SimulationState) -> SimulationState:
        """Transform and return the shared simulation state."""
        ...


class NumpyLinearSolver(Protocol):
    """Protocol for solvers consuming a neutral SCM and exogenous values."""

    def __call__(self, scm: DirectedScm, exogenous_noise: np.ndarray) -> np.ndarray:
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


def sample_library_sizes(
    n_samples: int,
    *,
    log_mean: float,
    log_sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate positive per-sample library sizes.

    The library sizes represent total sequencing depth: the approximate
    number of UMIs available across all genes in a sample or cell.
    """
    if log_sd < 0:
        raise ValueError("library_size_log_sd must be non-negative.")

    return rng.lognormal(mean=log_mean, sigma=log_sd, size=n_samples)


def sample_baseline_abundances(
    n_genes: int,
    *,
    log_sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample gene-specific baseline abundance proportions.

    Returns ``q`` such that ``q[h] >= 0`` and ``sum(q) = 1``.
    """
    if n_genes <= 0:
        raise ValueError("n_genes must be positive.")
    if log_sd < 0:
        raise ValueError("baseline_log_sd must be non-negative.")

    raw = rng.lognormal(mean=0.0, sigma=log_sd, size=n_genes)
    return raw / raw.sum()


def sample_umi_counts(
    latent_log_expression: np.ndarray,
    library_sizes: np.ndarray,
    baseline_abundances: np.ndarray,
    *,
    dispersion: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate UMI counts from latent expression.

    The model is ``mu_hs = L_s * q_h * 2**X_hs`` and
    ``Y_hs ~ NB(mu_hs, alpha)`` with variance ``mu + alpha * mu**2``.
    """
    if dispersion <= 0:
        raise ValueError("umi_dispersion must be positive.")
    if latent_log_expression.ndim != 2:
        raise ValueError("latent_log_expression must have shape (n_samples, n_genes).")

    n_samples, n_genes = latent_log_expression.shape
    if library_sizes.shape != (n_samples,):
        raise ValueError("library_sizes must have shape (n_samples,).")
    if baseline_abundances.shape != (n_genes,):
        raise ValueError("baseline_abundances must have shape (n_genes,).")
    if not np.isclose(baseline_abundances.sum(), 1.0):
        raise ValueError("baseline_abundances must sum to one.")

    latent_log_expression = np.clip(latent_log_expression, -30.0, 30.0)
    mu = library_sizes[:, None] * baseline_abundances[None, :] * np.exp2(latent_log_expression)
    n = 1.0 / dispersion
    p = n / (n + mu)
    return rng.negative_binomial(n=n, p=p)


def counts_to_log_expression(
    counts: np.ndarray,
    library_sizes: np.ndarray,
    *,
    pseudocount: float,
    target_library_size: float,
) -> np.ndarray:
    """Convert UMI counts to normalized log2-expression.

    Counts are rescaled to target_library_size before applying log1p.
    """
    if pseudocount <= 0:
        raise ValueError("umi_pseudocount must be positive.")
    if target_library_size <= 0:
        raise ValueError("target_library_size must be positive.")

    normalized_counts = counts / library_sizes[:, None] * target_library_size
    return np.log2(normalized_counts + pseudocount)


def numpy_linear_solver(
    scm: DirectedScm,
    exogenous_noise: np.ndarray,
) -> np.ndarray:
    """Solve ``X = B.T @ X + eps`` for sample-row data.

    With ``B[u, v]`` representing ``u -> v``, the system for each sample is
    ``(I - B.T) X = eps``.
    """
    coefficient_matrix = np.eye(len(scm.nodes)) - scm.beta_matrix.T

    try:
        latent_transposed = np.linalg.solve(coefficient_matrix, exogenous_noise.T)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "Synthetic SCM linear system is singular / ill-conditioned. "
            "Try smaller beta magnitudes."
        ) from error

    return latent_transposed.T


def _solve(
    state: SimulationState,
    solver: NumpyLinearSolver = numpy_linear_solver,
) -> SimulationState:
    """Solve the neutral SCM using the configured solver."""
    if state.exogenous_noise is None:
        raise ValueError("Exogenous noise must be generated before solving the SCM.")

    state.latent_log_expression = solver(state.scm, state.exogenous_noise)
    return state


def _observe(state: SimulationState) -> SimulationState:
    """Generate integer UMI counts and normalized log-expression."""
    config = state.config

    if state.latent_log_expression is None:
        raise ValueError("Latent expression must be generated before observation.")

    # Generate sample-specific sequencing depths and gene-specific baseline
    # abundance proportions before drawing observed UMI counts.
    state.library_sizes = sample_library_sizes(
        config.n_samples,
        log_mean=config.library_size_log_mean,
        log_sd=config.library_size_log_sd,
        rng=state.rng,
    )
    state.baseline_abundances = sample_baseline_abundances(
        len(state.scm.nodes),
        log_sd=config.baseline_log_sd,
        rng=state.rng,
    )

    # The observation model is mu_hs = L_s * q_h * 2**X_hs with
    # Var(Y_hs) = mu_hs + dispersion * mu_hs**2.
    state.umi_counts = sample_umi_counts(
        state.latent_log_expression,
        state.library_sizes,
        state.baseline_abundances,
        dispersion=config.umi_dispersion,
        rng=state.rng,
    )

    # Normalize counts to the target library size before applying log2.
    normalized = counts_to_log_expression(
        state.umi_counts,
        state.library_sizes,
        pseudocount=config.umi_pseudocount,
        target_library_size=np.exp(config.library_size_log_mean),
    )
    state.observed_data = pd.DataFrame(
        normalized,
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
    for j, col in enumerate(state.scm.nodes):
        if config.missing_data_mechanism.lower() in {"biological_error", "biological", "bio"}:
            probability = np.full(config.n_samples, rate)
        elif config.missing_data_mechanism.lower() in {
            "instrument_error",
            "instrument",
            "instrument-self-masking",
        }:
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
            probability = np.clip(rate * raw / raw.mean(), 0, 1)
        else:
            raise ValueError(f"Unknown missing-data mechanism: {config.missing_data_mechanism!r}")
        state.observed_data.loc[state.rng.random(config.n_samples) < probability, col] = 0.0
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
