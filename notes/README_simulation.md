# Synthetic SCM Simulation

This document describes the maintained synthetic-data pipeline in
`src/nocap/scm_model.py` and `src/nocap/simulation.py`, including the latent
structural causal model (SCM), the negative-binomial observation model, missing
data, paired artifacts, and the `csd_estimate.py` command-line workflow.

## Overview

The simulation has two layers:

1. A latent linear SCM generates log2-expression values `X` from exogenous
   noise and a directed graph.
2. An observation model converts latent expression into sample-specific counts,
   then estimates normalized expression for downstream causal estimation.

The standard generation pipeline is:

```text
exogenous noise
      |
      v
latent SCM solve
      |
      v
negative-binomial counts
      |
      v
q-hat and X-hat transformations
      |
      v
condition-specific missingness
```

The estimator continues to receive a pandas DataFrame. That DataFrame is
`X-hat`, the estimated latent log2-expression. The count matrix and `q-hat`
remain available on `SimulationState` and `PairedDataArtifact` for inspection
and custom stages.

## Latent SCM

### Structural equations

For sample `i` and gene `g`, the latent variable is log2-expression:

```text
X_i = A X_i + epsilon_i
```

where:

- `X_i` is the vector of latent gene expression values for sample `i`.
- `A[target, parent]` is the mathematical target-parent coefficient matrix.
- `epsilon_i` is a vector of exogenous noise values.

The solver computes the equilibrium solution by solving:

```text
(I - A) X_i = epsilon_i
```

or, equivalently:

```text
X_i = (I - A)^(-1) epsilon_i
```

### Matrix convention

The public `DirectedScm.beta_matrix` retains the repository's existing
edge-oriented convention:

```text
beta_matrix[source, target] = beta(source -> target)
```

Therefore:

```text
A = beta_matrix.T
```

The NumPy solver explicitly uses:

```python
coefficient_matrix = np.eye(n_genes) - scm.beta_matrix.T
latent = np.linalg.solve(coefficient_matrix, exogenous_noise.T).T
```

Do not transpose `beta_matrix` when looking up a structural edge coefficient.
For an edge `(parent, child)`, the stored coefficient is
`scm.beta_matrix[parent_index, child_index]`, while the solver uses its
transpose to place that coefficient in the target equation.

### Graphs, cycles, and stability

`DirectedScm` stores:

- An ordered tuple of node names.
- A directed graph whose edges must match the coefficient dictionary.
- A coefficient dictionary keyed by `(source, target)` edge tuples.

`build_synthetic_scm()` can add true edges that are absent from the estimation
graph. The `missing_edge_rate` controls the expected number of added hidden
edges; existing estimation edges are retained.

Structural coefficients are sampled as signed continuous log2 effects. For an
edge coefficient `beta`:

- `beta > 0` is activation.
- `beta < 0` is inhibition.
- A one-unit increase in parent log2-expression changes target log2-expression
  by `beta`.
- `2**beta` is the corresponding multiplicative fold change on the expression
  scale.

For cyclic graphs, the coefficient matrix is rescaled when necessary so its
spectral radius is below the target stability threshold. The builder also
rejects poorly conditioned linear systems. This keeps the equilibrium solve
numerically usable while preserving the graph and signed-effect semantics.

## Observation Model

The latent expression is converted to positive, unconstrained expression using
gene-specific control baselines:

```text
q_ig = q0_g * 2**X_ig
```

Here `q0_g` is not a proportion and is not normalized across genes. This is
intentional: the baseline expression scale is part of the observation model.

### Size factors

Each sample receives a positive dimensionless size factor `s_i`. Raw factors
are sampled from a zero-location lognormal distribution:

```text
raw_s_i ~ LogNormal(0, size_factor_log_sd**2)
```

They are then divided by their geometric mean:

```text
s_i = raw_s_i / exp(mean(log(raw_s)))
```

Consequently:

```text
geomean(s) = 1
```

Size factors represent relative sample-specific offsets. The simulation does
not introduce an effective library-size relationship such as
`M_i = M_ref * s_i`.

### Baseline expression

Gene-specific baselines are sampled directly:

```text
q0_g ~ LogNormal(baseline_expression_log_mean,
                 baseline_expression_log_sd**2)
```

The lognormal parameters are in natural-log units. The defaults are:

```text
baseline_expression_log_mean = 0.0
baseline_expression_log_sd   = 2.0
```

The geometric baseline is therefore one by default, but individual `q0_g`
values can be much larger or smaller. Their sum is not constrained.

### Negative-binomial counts

Counts are generated with sample and gene offsets:

```text
mu_ig = s_i * q0_g * 2**X_ig
Y_ig ~ NB(mu_ig, alpha_g)
```

The dispersion parameterization is:

```text
E[Y_ig]   = mu_ig
Var[Y_ig] = mu_ig + alpha_g * mu_ig**2
```

`alpha_g` may be supplied as one positive scalar, which is broadcast to every
gene, or as a positive vector with exactly one value per gene. NumPy's
negative-binomial parameters are derived as:

```text
n_g = 1 / alpha_g
p_ig = n_g / (n_g + mu_ig)
```

The default scalar dispersion is `0.1`.

## q-hat and X-hat

The pipeline exposes two different count transformations.

### Normalized expression q-hat

The normalized expression array is:

```text
q_hat_ig = (Y_ig + pseudocount) / s_i
```

The pseudocount is added before division by the size factor. This output is
stored in `state.normalized_expression` or
`artifact.normalized_expression`.

### Estimated latent expression X-hat

The estimator-facing DataFrame uses the gene-specific baseline as well:

```text
X_hat_ig = log2((Y_ig + pseudocount) / (s_i * q0_g))
```

This is implemented by `counts_to_log_expression()` and stored in
`state.observed_data` as a DataFrame with SCM node names as columns.

The two transformations must not be conflated. `q-hat` is on the normalized
expression scale, while `X-hat` estimates the latent log2-expression scale
used by the causal regression estimator.

## Simulation Configuration

`SimulationConfig` controls both latent generation options and observation
settings:

| Field | Meaning | Default |
| --- | --- | --- |
| `n_samples` | Number of generated samples | required |
| `dispersion` | Positive scalar or one value per gene | `0.1` |
| `size_factor_log_sd` | Natural-log SD for raw size factors | `0.4` |
| `umi_pseudocount` | Positive count pseudocount | `1.0` |
| `baseline_expression_log_mean` | Natural-log mean of `q0` | `0.0` |
| `baseline_expression_log_sd` | Natural-log SD of `q0` | `2.0` |
| `missing_data_rate` | Base missingness probability | `0.0` |
| `missing_data_mechanism` | Biological, instrument, or both | `biological_error` |
| `self_mask_quantile` | Instrument masking threshold quantile | `0.25` |
| `self_mask_k` | Instrument masking logistic slope | `8.0` |
| `self_mask_direction` | Mask low or high latent expression | `low` |
| `scc_confounding_strength` | Shared noise-factor strength in nontrivial SCCs | `0.0` |
| `estimation_graph` | Graph used for SCC confounding structure | `None` |
| `fixed_intervention_values` | Hard-intervention values by node | `{}` |

Example:

```python
from nocap.simulation import SimulationConfig

config = SimulationConfig(
    n_samples=500,
    dispersion=[0.05, 0.1, 0.2],
    size_factor_log_sd=0.4,
    baseline_expression_log_mean=0.0,
    baseline_expression_log_sd=2.0,
    missing_data_rate=0.1,
    missing_data_mechanism="biological_error+instrument_error",
)
```

The dispersion vector must have the same length as the SCM node list.

## Generation Stages

The default stages are returned by `default_simulation_stages()`:

```python
(
    _noise,
    _solve,
    _observe,
    _missing,
)
```

### 1. Noise

`_noise` samples an `(n_samples, n_genes)` standard normal array. If
`scc_confounding_strength > 0`, each nontrivial strongly connected component
receives an additional shared latent factor. This introduces correlated
exogenous noise within that SCC.

### 2. Solve

`_solve` applies the configured hard interventions and solves the linear SCM.
The result is stored as `latent_log_expression`.

### 3. Observe

`_observe` samples size factors and `q0`, normalizes the dispersion input,
generates integer counts, computes q-hat, computes X-hat, and builds the
estimator-facing DataFrame.

The main state fields after observation are:

```python
state.exogenous_noise
state.latent_log_expression
state.size_factors
state.baseline_expression
state.dispersions
state.umi_counts
state.normalized_expression
state.observed_data
```

### 4. Missingness

Missingness is applied after X-hat is materialized. Missing entries are written
as `0.0` in `observed_data`, preserving the existing estimator workflow.

Supported mechanisms are:

- `biological_error`: each gene/sample entry has the configured base missingness
  probability.
- `instrument_error`: missingness depends on latent `X`, not X-hat or q-hat.
  The logistic mechanism can preferentially mask low or high expression using
  `self_mask_direction`.
- `biological_error+instrument_error`: combines both mechanisms as independent
  causes.

Using latent `X` for instrument error is important because missingness should
not be driven by a representation that already contains observation noise or
missing values.

## Paired Artifacts

`generate_paired_data_artifact()` creates one immutable maximum-size realization
for paired conditions. It stores:

- The `DirectedScm`.
- Exogenous noise.
- Latent expression.
- Counts.
- Size factors.
- Baseline expression `q0`.
- Per-gene dispersions.
- Normalized expression q-hat.
- A deterministic sample order.
- Missingness uniforms.
- Stage-specific seed metadata.

For a condition with fewer samples than the artifact maximum, `view()` selects
a deterministic prefix of the shared sample order. It then reconstructs X-hat
from the shared counts, size factors, and q0, followed by condition-specific
missingness.

This provides common random numbers across conditions:

```python
from nocap.simulation import SimulationConfig, generate_paired_data_artifact

artifact = generate_paired_data_artifact(
    scm,
    SimulationConfig(n_samples=1000),
    max_samples=1000,
    scm_seed=101,
    data_seed=202,
)

small = artifact.view(SimulationConfig(n_samples=250))
large = artifact.view(SimulationConfig(n_samples=1000))
```

The observation quantities are drawn once per artifact rather than regenerated
for each view. Paired views therefore share counts, q0, size factors, and
dispersions, while missingness settings can vary by condition.

## Reproducibility

For ordinary generation, pass a seed to `generate_from_scm()`:

```python
state = generate_from_scm(scm, SimulationConfig(n_samples=100), seed=7)
```

For paired experiments, the seed hierarchy is managed by `StageSeeds`:

- The SCM seed controls structural graph additions and coefficient draws.
- The data seed controls latent noise, observation quantities, counts, sample
  order, and missingness scores.
- Stage-specific seeds separate latent noise, observation, missingness, and
  ordering streams.

Changing the data seed does not change the SCM. Changing the SCM seed does not
reuse the same structural realization.

## Command-Line Usage

The maintained CLI is `scripts/csd_estimate.py`.

### Scalar dispersion

```bash
uv run python scripts/csd_estimate.py \
    --demo cycle \
    --output-csv results/csd.csv \
    --seed 7 \
    --dispersion 0.1
```

When `--dispersion` is omitted, the CLI uses scalar `alpha=0.1`.

### Gene-specific dispersion

The comma-separated vector must contain one value per graph node:

```bash
uv run python scripts/csd_estimate.py \
    --demo cycle \
    --output-csv results/csd.csv \
    --seed 7 \
    --dispersions 0.05,0.1,0.2
```

`--dispersion` and `--dispersions` cannot be used together.

Other observation options include:

```text
--size-factor-log-sd
--baseline-expression-log-mean
--baseline-expression-log-sd
--umi-pseudocount
```

The CLI also supports `--design-mode paired_hierarchical`, in which the
maximum-size artifact is shared across the configured sample-size conditions.
See `README_paired_experiments.md` for task manifests, SLURM execution, and
replicate/bootstrap workflows.

## Public Helpers

The main simulation helpers are exported from `nocap` and
`nocap.simulation`:

```python
from nocap.simulation import (
    counts_to_log_expression,
    counts_to_normalized_expression,
    generate_from_scm,
    normalize_dispersions,
    sample_baseline_expression,
    sample_size_factors,
    sample_umi_counts,
)
```

Validation is intentionally strict:

- Counts must be finite and nonnegative.
- Size factors must be finite and positive.
- Baseline expression must be finite and positive.
- Dispersions must be finite and positive.
- Array dimensions must match the sample and gene counts.
- Pseudocounts must be finite and positive.

The old library-size and compositional-baseline helper names are not part of
the maintained public API. The current canonical names are
`sample_size_factors()` and `sample_baseline_expression()`.

## Relationship to Causal Estimation

The simulation generates the data used by the causal path-coefficient estimator;
it does not replace that estimator with a count-aware differential-expression
fitting procedure. The estimator still operates on the pandas `X-hat`
DataFrame, performs its existing regression and missing-data handling, and
compares estimated path coefficients with the structural `ground_truth_beta`
values from the SCM.

The observation model is therefore DESeq2-style in its size-factor and
negative-binomial mean/variance semantics, while the downstream causal
estimation remains the existing regression-based workflow.
