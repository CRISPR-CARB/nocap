"""Composable synthetic SCM generation and UMI observation pipeline."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import networkx as nx
import numpy as np
import pandas as pd

from .experiment import StageSeeds
from .scm_model import DirectedScm, build_intervened_scm

PairedUnknownState = Literal["keep", "off", "on", "error"]


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
    fixed_expression_state_values: dict[str, bool] = field(default_factory=dict)
    activation_fold_change: float = 2.0
    interference_fold_change: float = 0.1
    knockout_log_expression: float = -20.0
    use_latent_expression_hat: bool = True
    use_umi_counts_as_observed_data: bool = False
    use_binary_expression_state: bool = False
    # Active-state probability is sigmoid(intercept + slope * latent X).
    expression_state_intercept: float = 0.0
    expression_state_slope: float = 1.0


def simulate_intervention_levels(
    scm: DirectedScm,
    config: SimulationConfig,
    treatment: str,
    treatment_levels: Iterable[float],
    *,
    seed: int = 0,
) -> dict[float, SimulationState]:
    """Generate numeric hard-intervention states for explicit levels.

    Deprecated: use :func:`simulate_expression_treatment` for binary-state
    biological treatments. This generic numeric helper does not clamp the
    binary expression state and is retained temporarily for compatibility.
    """
    warnings.warn(
        "simulate_intervention_levels is deprecated; use simulate_expression_treatment "
        "for binary-state biological treatments",
        DeprecationWarning,
        stacklevel=2,
    )
    levels = tuple(float(level) for level in treatment_levels)
    if treatment not in scm.nodes:
        raise ValueError(f"unknown treatment node: {treatment}")
    invalid = set(config.fixed_intervention_values) - {treatment}
    if invalid:
        raise ValueError(f"intervention config contains unlisted nodes: {sorted(invalid)}")
    results: dict[float, SimulationState] = {}
    for offset, level in enumerate(levels):
        intervened = build_intervened_scm(scm, [treatment])
        intervention_config = SimulationConfig(
            **{
                **config.__dict__,
                "fixed_intervention_values": {treatment: level},
                "use_latent_expression_hat": False,
                "use_umi_counts_as_observed_data": False,
            }
        )
        state = generate_from_scm(intervened, intervention_config, seed=seed + offset)
        if state.observed_data is None or not np.all(state.observed_data[treatment] == level):
            raise RuntimeError("intervention treatment column is not fixed at its requested level")
        results[level] = state
    return results


def simulate_expression_treatment(
    scm: DirectedScm,
    config: SimulationConfig,
    target: str,
    treatment: Literal["A", "I", "K"],
    *,
    seed: int = 0,
) -> SimulationState:
    """Simulate one activation, interference, or knockout intervention.

    Activation and interference fix latent log2-expression to the configured
    fold change and clamp the target ON. Knockout fixes the configured latent
    expression floor and clamps the target OFF.
    """
    if target not in scm.nodes:
        raise ValueError(f"unknown treatment node: {target}")
    if treatment not in ("A", "I", "K"):
        raise ValueError("treatment must be one of {'A', 'I', 'K'}")
    if not config.use_binary_expression_state:
        raise ValueError("expression treatments require use_binary_expression_state=True")

    latent_value, state_value, fold_change = _expression_treatment_values(config, treatment)
    _validate_expression_treatment_config(config, target, latent_value, state_value)

    intervention_config = SimulationConfig(
        **{
            **config.__dict__,
            "fixed_intervention_values": {target: latent_value},
            "fixed_expression_state_values": {target: state_value},
        }
    )
    state = generate_from_scm(build_intervened_scm(scm, [target]), intervention_config, seed=seed)
    target_index = scm.nodes.index(target)
    if state.latent_log_expression is None or not np.all(
        state.latent_log_expression[:, target_index] == latent_value
    ):
        raise RuntimeError("expression treatment latent value is not fixed as requested")
    if state.latent_expression_state is None or not np.all(
        state.latent_expression_state[:, target_index] == state_value
    ):
        raise RuntimeError("expression treatment state is not fixed as requested")

    if (
        state.umi_counts is not None
        and not state_value
        and np.any(state.umi_counts[:, target_index] != 0)
    ):
        raise RuntimeError("expression treatment OFF counts are not structurally zero")

    state.metadata.update(
        {
            "expression_treatment": treatment,
            "expression_treatment_target": target,
            "expression_treatment_latent_value": latent_value,
            "expression_treatment_state": state_value,
        }
    )
    if fold_change is not None:
        state.metadata["expression_treatment_fold_change"] = fold_change
    return state


def _expression_treatment_values(
    config: SimulationConfig, treatment: Literal["A", "I", "K"]
) -> tuple[float, bool, float | None]:
    """Resolve the configured latent value, state, and optional fold change."""
    if treatment == "A":
        fold_change = config.activation_fold_change
        if not np.isfinite(fold_change) or fold_change <= 0:
            raise ValueError("activation_fold_change must be finite and positive")
        return float(np.log2(fold_change)), True, float(fold_change)
    if treatment == "I":
        fold_change = config.interference_fold_change
        if not np.isfinite(fold_change) or not 0 < fold_change < 1:
            raise ValueError("interference_fold_change must be finite and between zero and one")
        return float(np.log2(fold_change)), True, float(fold_change)
    latent_value = config.knockout_log_expression
    if not np.isfinite(latent_value):
        raise ValueError("knockout_log_expression must be finite")
    return float(latent_value), False, None


def _validate_expression_treatment_config(
    config: SimulationConfig,
    target: str,
    latent_value: float,
    state_value: bool,
) -> None:
    """Reject configured interventions incompatible with a treatment request."""
    fixed_latent = config.fixed_intervention_values
    fixed_state = config.fixed_expression_state_values
    other_nodes = (set(fixed_latent) | set(fixed_state)) - {target}
    if other_nodes:
        raise ValueError(
            f"expression treatment config intervenes on other nodes: {sorted(other_nodes)}"
        )
    if target in fixed_latent and fixed_latent[target] != latent_value:
        raise ValueError(
            "expression treatment conflicts with the configured latent intervention value"
        )
    if target in fixed_state and fixed_state[target] != state_value:
        raise ValueError(
            "expression treatment conflicts with the configured expression-state value"
        )


def true_scm_ate(
    scm: DirectedScm,
    treatment: str,
    outcome: str,
    treatment_levels: tuple[float, float],
    *,
    n_samples: int = 100_000,
    seed: int = 0,
) -> float:
    """Estimate a true linear-SCM ATE using common exogenous noise samples."""
    if treatment not in scm.nodes or outcome not in scm.nodes:
        raise ValueError("treatment and outcome must be SCM nodes")
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    noise = np.random.default_rng(seed).normal(size=(n_samples, len(scm.nodes)))
    means = []
    for level in treatment_levels:
        intervened = build_intervened_scm(scm, [treatment])
        values = numpy_linear_solver(intervened, noise, {treatment: float(level)})
        means.append(float(values[:, intervened.nodes.index(outcome)].mean()))
    return means[1] - means[0]


@dataclass
class SimulationState:
    """Mutable state shared by ordered generation stages."""

    scm: DirectedScm
    config: SimulationConfig
    rng: np.random.Generator
    exogenous_noise: np.ndarray | None = None
    latent_log_expression: np.ndarray | None = None
    latent_expression_state: np.ndarray | None = None
    umi_counts: np.ndarray | None = None
    size_factors: np.ndarray | None = None
    baseline_expression: np.ndarray | None = None
    dispersions: np.ndarray | None = None
    normalized_expression: np.ndarray | None = None
    observed_data: pd.DataFrame | None = None
    binarized_data: pd.DataFrame | None = None
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
    binarized_expression: np.ndarray | None = field(default=None, repr=False, compare=False)
    latent_expression_state: np.ndarray | None = None

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
        ordered = paired_view_stages() if stages is None else stages
        state = self._materialize(config, ordered)
        if state.observed_data is None:
            raise ValueError("Paired view stages must produce observed_data")
        state.observed_data.attrs.update(state.metadata)
        return state.observed_data

    def binarized_view(
        self,
        config: SimulationConfig,
        *,
        stages: Sequence[SimulationStage] | None = None,
    ) -> pd.DataFrame:
        """Return the artifact's binary expression matrix for a paired sample view.

        Generate the artifact with ``binarize=True`` to populate its binary
        expression matrix. Custom stages may replace or further transform it.
        """
        ordered = paired_view_stages() if stages is None else stages
        state = self._materialize(config, ordered)
        if state.binarized_data is None:
            raise ValueError("Generate the paired artifact with binarize=True first")
        state.binarized_data.attrs.update(state.metadata)
        return state.binarized_data

    def latent_expression_state_view(self, config: SimulationConfig) -> pd.DataFrame:
        """Return ground-truth active/off labels aligned to a paired sample view."""
        if self.latent_expression_state is None:
            raise ValueError("Generate the artifact with use_binary_expression_state=True first")
        if config.n_samples > len(self.sample_order):
            raise ValueError("Condition n_samples exceeds artifact maximum size")
        indices = self.sample_order[: config.n_samples]
        return pd.DataFrame(self.latent_expression_state[indices].copy(), columns=self.scm.nodes)

    def _materialize(
        self, config: SimulationConfig, stages: Sequence[SimulationStage]
    ) -> SimulationState:
        """Build a view state and run the requested paired stages."""
        if config.n_samples > len(self.sample_order):
            raise ValueError("Condition n_samples exceeds artifact maximum size")
        indices = self.sample_order[: config.n_samples]
        state = SimulationState(
            scm=self.scm,
            config=config,
            rng=np.random.default_rng(0),
            exogenous_noise=self.exogenous_noise[indices],
            latent_log_expression=self.latent_log_expression[indices],
            latent_expression_state=(
                self.latent_expression_state[indices]
                if self.latent_expression_state is not None
                else None
            ),
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
            binarized_data=(
                pd.DataFrame(self.binarized_expression[indices].copy(), columns=self.scm.nodes)
                if self.binarized_expression is not None
                else None
            ),
        )
        for stage in stages:
            state = stage(state)
        return state


def paired_view_stages() -> tuple[SimulationStage, ...]:
    """Return the default paired-view observation and missingness stages."""
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
        mask = _missingness_mask(
            state.latent_log_expression, uniforms, state.config, len(state.observed_data)
        )
        state.observed_data = state.observed_data.mask(mask, 0.0)
        if state.binarized_data is not None:
            state.binarized_data = state.binarized_data.mask(mask, np.nan)
    return state


def _missingness_mask(latent, uniforms, config, n_samples: int) -> np.ndarray:
    """Compute the shared condition-specific missingness mask."""
    if uniforms is None:
        raise ValueError("Paired view requires persisted missingness uniforms")
    if latent is None and "instrument_error" in _missing_data_mechanisms(
        config.missing_data_mechanism
    ):
        raise ValueError("Latent expression must exist before instrument missingness")
    mechanisms = _missing_data_mechanisms(config.missing_data_mechanism)
    mask = np.zeros(uniforms.shape, dtype=bool)
    for j in range(uniforms.shape[1]):
        rate = np.clip(float(config.missing_data_rate), 0, 1)
        probabilities = []
        if "biological_error" in mechanisms:
            probabilities.append(np.full(n_samples, rate))
        if "instrument_error" in mechanisms:
            if latent is None:
                raise ValueError("Latent expression must exist before instrument missingness")
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
        mask[:, j] = uniforms[:, j] < probability
    return mask


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
    binarize: bool = False,
    unknown: PairedUnknownState = "keep",
    binarizer_kwargs: dict[str, object] | None = None,
) -> PairedDataArtifact:
    """Generate one maximum-size paired realization, optionally binarizing counts.

    When ``binarize`` is true, scBoolSeq is fitted once on the complete
    size-factor-normalized, log-transformed UMI-count matrix and the resulting
    binary expression is stored directly on the artifact. Its optional
    dependency is imported only for this path.
    """
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
    expression_state = (
        sample_expression_state(latent, config, observation_rng)
        if config.use_binary_expression_state
        else None
    )
    counts = sample_umi_counts(
        latent, size_factors, baseline, dispersions, observation_rng, expression_state
    )
    normalized_expression = counts_to_normalized_expression(
        counts,
        size_factors,
        pseudocount=config.umi_pseudocount,
    )

    binary_expression: np.ndarray | None = None
    if binarize:
        from .binarization import binarize_expression

        expression = pd.DataFrame(np.log1p(counts / size_factors[:, None]), columns=scm.nodes)
        binary_result = binarize_expression(
            expression,
            unknown=unknown,
            binarizer_kwargs=binarizer_kwargs,
        )
        if not isinstance(binary_result, pd.DataFrame):
            raise TypeError("DataFrame binarization must return a labeled DataFrame")
        binary_expression = binary_result.to_numpy(copy=True)

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
        binarized_expression=binary_expression,
        latent_expression_state=expression_state,
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

    ``noise -> solve -> expression state -> UMI observation -> missingness``.
    """
    return (_noise, _solve, _expression_state, _observe, _missing)


def _expression_state(state: SimulationState) -> SimulationState:
    """Sample known active/off states when the hurdle model is enabled."""
    if not state.config.use_binary_expression_state:
        if state.config.fixed_expression_state_values:
            raise ValueError(
                "expression-state interventions require use_binary_expression_state=True"
            )
        return state
    if state.latent_log_expression is None:
        raise ValueError("Latent expression must be generated before expression states.")
    state.latent_expression_state = sample_expression_state(
        state.latent_log_expression, state.config, state.rng
    )
    if state.config.fixed_expression_state_values:
        indices = {node: i for i, node in enumerate(state.scm.nodes)}
        invalid = set(state.config.fixed_expression_state_values) - set(indices)
        if invalid:
            raise ValueError(f"unknown expression-state intervention nodes: {sorted(invalid)}")
        for node, value in state.config.fixed_expression_state_values.items():
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError("fixed expression-state intervention values must be boolean")
            state.latent_expression_state[:, indices[node]] = value
    return state


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


def sample_expression_state(
    latent_log_expression: np.ndarray,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample biological on/off state using a logistic latent-expression link."""
    latent_log_expression = np.asarray(latent_log_expression, dtype=float)
    if latent_log_expression.ndim != 2 or not np.all(np.isfinite(latent_log_expression)):
        raise ValueError("latent_log_expression must be a finite 2D array.")
    if not np.isfinite(config.expression_state_intercept) or not np.isfinite(
        config.expression_state_slope
    ):
        raise ValueError("expression-state intercept and slope must be finite.")
    logits = (
        config.expression_state_intercept + config.expression_state_slope * latent_log_expression
    )
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -709, 709)))
    return (rng.random(latent_log_expression.shape) < probabilities).astype(bool)


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
    expression_state: np.ndarray | None = None,
) -> np.ndarray:
    """Generate counts from the latent expression and NB observation model.

    The model is ``mu_ig = s_i * q0_g * 2**X_ig`` and
    ``Y_ig ~ NB(mu_ig, alpha_g)`` with variance ``mu + alpha_g * mu**2``.
    If ``expression_state`` is provided, inactive genes have structural zero
    counts while active genes retain ordinary NB sampling zeros.
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
    counts = rng.negative_binomial(n=n, p=p)
    if expression_state is not None:
        expression_state = np.asarray(expression_state)
        if expression_state.shape != latent_log_expression.shape:
            raise ValueError("expression_state must have the same shape as latent expression.")
        if not np.all(np.isin(expression_state, [False, True, 0, 1])):
            raise ValueError("expression_state must contain only boolean values.")
        counts = np.where(expression_state, counts, 0)
    return counts


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

    if config.use_binary_expression_state and state.latent_expression_state is None:
        state.latent_expression_state = sample_expression_state(
            state.latent_log_expression, config, state.rng
        )

    state.umi_counts = sample_umi_counts(
        state.latent_log_expression,
        state.size_factors,
        state.baseline_expression,
        state.dispersions,
        rng=state.rng,
        expression_state=state.latent_expression_state,
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
