# Paired CSD Experiments

This document describes the reproducible experiment workflow for synthetic CSD
estimation. It covers experiment setup, seed derivation, paired simulation
artifacts, SLURM execution, replicate design, and downstream bootstrap
confidence intervals.

## Workflow

The workflow is split into four stages:

```text
setup_csd_experiment.py
        |
        +-- experiment.json
        +-- tasks/*.json
                    |
                    v
          submit_csd_estimate.sh
                    |
                    v
             csd_estimate.py
                    |
                    v
              result CSVs
                    |
                    v
          simulation_analysis.py
```

Experiment setup creates stable task records. The SLURM submitter schedules
those records, and `csd_estimate.py` generates data and estimation results for
each task. Statistical bootstrap confidence intervals are computed later from
completed result units; bootstrap iterations do not generate additional
simulations.

## Experiment Setup

Create a manifest and one JSON record per task with:

```bash
uv run python scripts/setup_csd_experiment.py \
    --output results/experiment.json \
    --tasks-dir results/tasks \
    --output-dir results/csv \
    --experiment-id ecoli-csd-v1 \
    --design-mode paired_hierarchical \
    --base-seed 123 \
    --scm-replicates 2 \
    --data-replicates 3 \
    --n-samples-list 100,500,1000 \
    --missing-edge-rates 0.0,0.2,0.4 \
    --missing-data-rates 0.0,0.3 \
    --mechanisms biological_error,instrument_error \
    --graphml notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml \
    --intervention-csv notebooks/Ecoli_Analysis_Notebooks/csd_recovery_n5_k6.csv \
    --include-observational
```

The command writes:

- One experiment manifest at `--output`.
- One task JSON record per job under `--tasks-dir`.
- An `output_csv` path in each task when `--output-dir` is supplied.

A task can be run directly:

```bash
uv run python scripts/csd_estimate.py \
    --task-json results/tasks/<task>.json \
    --graphml path/to/network.graphml \
    --adjustments-csv path/to/adjustments.csv
```

The SLURM wrapper invokes the setup script automatically.

## Replicate Hierarchy

Simulation units have this hierarchy:

```text
SCM replicate
└── data replicate
    └── parameter condition
```

An SCM replicate represents a different underlying causal system. It can vary
the true graph, added hidden edges, structural beta coefficients, stability,
and conditioning.

A data replicate represents a new observational realization conditional on an
SCM. It can vary exogenous noise, SCM solutions, size factors, q0 baselines,
counts, missingness, and sample order.

A parameter condition changes settings such as sample size, missing-edge rate,
missing-data rate, and missing-data mechanism. A `trial` is only a local grid
index and is not a replicate identity.

## Stable Identities and Seeds

`src/nocap/experiment.py` provides stable SHA-256-based seed derivation through:

```python
derive_seed(base_seed, *tokens)
```

The same base seed and tokens always produce the same uint32 seed. The seed
schema is versioned, so changing the schema version intentionally changes the
derived streams.

Task identities include:

```text
experiment_id
design_mode
scm_replicate_id
parameter_condition_id
```

The complete job identity, rather than the seed pair alone, is used for task
deduplication and resume behavior. Repeated seed pairs are valid when they are
intentional common-random-number pairings across conditions.

## Design Modes

### Paired Hierarchical

This is the primary paired design. Seeds are stable at the replicate level:

```python
scm_seed = derive_seed(base_seed, "scm_replicate", scm_replicate_id)
data_seed = derive_seed(
    base_seed,
    "data_replicate",
    scm_replicate_id,
    data_replicate_id,
)
```

The same replicate-level seed roots are reused across parameter conditions.
This shares latent and observational random streams, allowing condition
effects to be estimated with paired comparisons:

```text
SCM 0 / data 0
├── condition A
├── condition B
└── condition C
```

Use this mode for paired condition contrasts and hierarchical inference.

### Full

Full mode derives condition-specific SCM and data seeds. It measures
end-to-end variation across:

- SCM construction.
- Structural beta generation.
- True hidden-edge generation.
- SCM solving.
- Negative-binomial count observation.
- Library and count generation.
- Missingness.

It is appropriate when each parameter cell should represent an independent
realization of the complete synthetic pipeline. It is not a common-random-
number paired design.

### Fixed-SCM

Fixed-SCM mode derives one SCM seed per SCM replicate and missing-edge-rate
block, then varies data seeds across observational conditions. It estimates
observational-generation variation conditional on fixed SCM blocks.

The current setup implementation includes condition tokens in the data seed
for this mode. Therefore it is suitable for conditional observational
variation, but it does not provide the strongest common-random-number pairing
across conditions. Use `paired_hierarchical` when paired comparisons are the
primary objective.

## Paired Data Artifacts

`PairedDataArtifact` in `src/nocap/simulation.py` stores one immutable,
maximum-size realization containing:

- SCM and beta structure.
- Exogenous noise.
- Latent expression.
- Counts from the negative-binomial observation model.
- Geometrically centered size factors.
- Positive, unconstrained q0 baseline expression.
- Gene-specific negative-binomial dispersions.
- Normalized expression q-hat and estimator-facing X-hat.
- Sample ordering.
- Missingness uniforms.
- Stage seeds.

If the experiment includes sample sizes `100,500,1000`, one artifact is
generated at `max_samples=1000`. Smaller conditions use deterministic prefixes
of the shared sample order, making sample-size views nested rather than
independently regenerated.

`PairedDataArtifact.view()` uses the simulation stage architecture. The default
view stages are:

```python
paired_view_stages()  # (_view_observe, _view_missing)
```

Custom stage sequences can be supplied to add or replace view transformations
without mutating the shared artifact.

## Missing-Edge Rates

Paired execution creates one artifact per distinct `missing_edge_rate` and
selects the matching artifact for each condition. This ensures that the true
SCM graph, ground-truth beta map, and true missing-edge count correspond to the
condition being evaluated.

The current implementation builds these edge-rate artifacts as separate SCM
constructions. They correctly use their requested rates, but they are not yet
fully nested coefficient-preserving SCM variants. In particular, the stronger
policy of sharing candidate edge order and preserving base coefficients across
variants still requires a dedicated nested SCM builder.

## SLURM Execution

`scripts/slurm/submit_csd_estimate.sh` now:

1. Calls `setup_csd_experiment.py`.
2. Writes the experiment manifest and task JSON files.
3. Creates a pending list from task output paths.
4. Splits pending JSON records into batches.
5. Runs `csd_estimate.py --task-json <task>` on SLURM workers.
6. Skips nonempty output files during resume.

The active scheduler no longer reconstructs the grid or seeds in shell, uses
shell `cksum`, rejects intentional repeated seed pairs, or relies on filenames
for task identity.

## Bootstrap Confidence Intervals

Simulation replication and statistical bootstrap are separate operations.

First generate actual SCM/data replicate units. Then compute metrics or paired
deltas. Finally resample those completed units downstream to construct
confidence intervals. Bootstrap iterations do not regenerate SCMs or datasets.

### Full Bootstrap

Full-mode bootstrap resamples complete `(scm_replicate_id, data_replicate_id)`
units. It includes both between-SCM and within-SCM observational variation and
therefore estimates end-to-end uncertainty.

### Fixed-SCM Bootstrap

Fixed-SCM bootstrap resamples data replicates independently within each SCM.
SCM identities remain fixed. Each bootstrap iteration:

1. Resamples data replicates within every SCM.
2. Computes the statistic within each SCM.
3. Averages the SCM-level statistics.

This estimates observational-generation variation conditional on the available
SCM blocks.

The helpers are in `src/nocap/simulation_analysis.py`:

- `pair_deltas()` joins reference and comparison rows and computes signed
  deltas.
- `bootstrap_replicates()` creates design-aware resampled units.
- `bootstrap_confidence_interval()` computes percentile confidence intervals.
- `percentile_interval()` computes generic percentile bounds.

Bootstrap multiplicity is preserved. If a replicate is selected twice, it
appears twice in the draw. Draw metadata includes:

```text
bootstrap_iteration
bootstrap_draw_position
bootstrap_scm_draw_position
bootstrap_data_draw_position
```

For paired comparisons, compute the comparison-minus-reference delta within
each replicate and target unit before bootstrapping. Resampling the two
conditions independently would destroy the pairing.

Example:

```python
from nocap.simulation_analysis import (
    bootstrap_confidence_interval,
    pair_deltas,
)

deltas = pair_deltas(
    results,
    reference={"missing_data_rate": 0.0},
    comparison={"missing_data_rate": 0.3},
    value_column="metric_value",
)

ci = bootstrap_confidence_interval(
    deltas,
    value_column="delta",
    design_mode="fixed_scm",
    iterations=2000,
    seed=123,
)
```

## Result Schema

Result CSVs retain legacy fields such as:

```text
trial
seed
scm_seed
cause
```

They also include hierarchical metadata such as:

```text
experiment_id
seed_schema_version
scm_replicate_id
parameter_condition_id
pairing_scope
sample_parent_id
sample_selection_rule
run_status
```

`csd_estimate_gather.py` normalizes legacy and current files. Legacy outputs
are labeled `legacy_independent` with pairing scope `none`; they are not
reclassified as paired results.

## Task Counts

The number of tasks grows multiplicatively:

```text
mechanisms
× missing-edge rates
× missing-data rates
× sample sizes
× SCM replicates
× data replicates
```

For example:

```text
2 mechanisms
× 3 edge rates
× 9 data rates
× 7 sample sizes
× 2 SCM replicates
× 3 data replicates
= 2,268 tasks
```

Grid selection, replicate counts, batch size, and available SLURM capacity
should be considered before launching a large experiment.

## Important Caveats

- `paired_hierarchical` is the primary paired design.
- `fixed_scm` provides conditional observational variation, but its current
  data seeds include condition tokens and therefore do not maximize paired
  common-random-number behavior.
- Edge-rate artifacts currently use separate SCM builds rather than fully
  nested coefficient-preserving variants.
- Bootstrap resamples completed results and does not generate additional
  simulation tasks.
- Paired analysis requires explicit experiment, SCM replicate, data replicate,
  target effect, and numeric metric columns.
- Legacy files can be gathered, but they do not support hierarchical inference
  unless their replicate identity is independently established.
