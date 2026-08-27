# Regulatory VAE Formalization Plan

**Date:** 2026-08-26  
**Status:** Proposed execution plan  
**Scope:** E. coli regulatory VAE, sparse cyclic equilibrium solver, and
ChiRho counterfactual execution in nocap

## 1. Purpose

Create a versioned, normative specification for the graph-aware regulatory VAE
and connect every implemented requirement to an appropriate executable
verification artifact.

The formalization covers:

- strict YAML experiment configuration;
- H5AD count and perturbation ingestion;
- GraphML alignment to authoritative expression-column order;
- sparse cyclic equilibrium solving with `message` and `linalg` backends;
- an unlabeled Pyro regulatory VAE;
- ChiRho soft interventions and batched counterfactual worlds;
- provenance, diagnostics, and failure behavior.

This document is an implementation plan, not the normative specification. The
authoritative behavior will live in
`docs/specifications/regulatory_vae.md`. That specification will supersede
conflicting statements in
`docs/plans/nocap-regulatory-scanvi-implementation-plan.md`, which remains as
historical design input.

## 2. Fixed Decisions

The formalization MUST preserve these decisions:

1. All production code belongs in nocap. Pyro and ChiRho remain upstream
   dependencies and are not modified for this feature.
2. The initial model is an unlabeled regulatory scVI-style VAE, named
   `RegulatoryVAE`, rather than label-specific SCANVI.
3. YAML configuration selects an H5AD expression source and a GraphML network.
4. AnnData feature order is authoritative. GraphML order, traversal order, and
   alphabetical order MUST NOT determine expression tensor columns.
5. GraphML cycles are retained. Graph polarity is a hard sign constraint on
   learned edge coefficients.
6. The adjacency convention is `A[target, regulator]`.
7. `message` is the scalable default solver. `linalg` is a guarded exact
   backend for small systems and reference testing.
8. Perturbation drive is `u = p * eta`, where `p` is signed dose and `eta` is
   one learned nonnegative efficacy per gene.
9. Zero graph weights recover the vanilla decoder boundary. Zero perturbation
   alone does not remove regulatory feedback.
10. Fixed-step locality applies to pre-softmax equilibrium logits, not to final
    normalized proportions.
11. ChiRho intervenes on an exogenous `perturbation` site. It does not soften
    observations or clamp downstream values.
12. Counterfactual worlds replay factual latent state and library size. The
    observed likelihood contributes only in the factual world.

## 3. Assurance Strategy

No single tool should be stretched beyond the claims it can support.

| Layer | Artifact | Responsibility |
| --- | --- | --- |
| Normative | Markdown with RFC 2119 terms | Semantic source of truth and stable requirement IDs |
| Runtime | Nocap PRE/POST assertions | Cheap boundary, shape, range, and topology checks |
| Examples | Pytest unit tests | Hand-computed regressions and integration behavior |
| Properties | Hypothesis | Algebraic, shape, permutation, and data-generation properties |
| Proof | Axiomander | Universal contracts over its supported Python subset |
| Acceptance | Specsaver-linked Gherkin | Stakeholder-visible workflows and domain errors |
| Lifecycle | FizzBee | Finite solver mode transitions, guards, and terminal states |

Gherkin MUST NOT be used to express floating-point convergence theorems,
gradient correctness, asymptotic complexity, or universal shape
quantification. FizzBee MUST NOT be presented as proving real-valued numerical
convergence. These claims belong in mathematics, property tests, and reference
comparisons.

## 4. Normative Specification Structure

Create `docs/specifications/regulatory_vae.md` as specification version `0.1`.
Use `MUST`, `MUST NOT`, `SHOULD`, and `MAY` consistently. Every normative
statement receives exactly one stable ID from these families:

| Prefix | Subject |
| --- | --- |
| `CFG-*` | YAML syntax, defaults, path resolution, and validation |
| `DATA-*` | H5AD counts, genes, perturbations, and batches |
| `GRAPH-*` | GraphML projection, polarity, topology, and reporting |
| `SOLVER-*` | Sparse operator, stability, convergence, and backends |
| `MODEL-*` | Pyro model, guide, likelihood, training, and identifiability |
| `CF-*` | ChiRho intervention and counterfactual semantics |
| `PROV-*` | Checkpoint, configuration, graph, and gene-order provenance |
| `ERR-*` | Stable validation and runtime failure taxonomy |

The specification begins with terminology and notation:

- `G`: number of measured genes;
- `E`: number of retained candidate edges;
- `S`: arbitrary leading batch/world shape;
- `A[target, regulator]`: signed direct regulatory coefficient;
- `b(z)`: basal decoder logits;
- `p`: signed perturbation dose;
- `eta`: nonnegative perturbation efficacy;
- `u = p * eta`: perturbation drive;
- `d = b(z) + u`: total exogenous drive;
- `g`: equilibrium logits satisfying `g = d + A g`.

## 5. Configuration and Data Contracts

### 5.1 YAML configuration

Specify a typed `ExperimentConfig` with these behaviors:

- reject unknown keys;
- resolve relative paths from the YAML file directory;
- require an explicit count source, either `X` or a named layer;
- configure the perturbation `obs` column, control values, target delimiter,
  CRISPR type-to-sign mapping, and default dose;
- validate cross-field constraints deterministically;
- serialize the fully resolved configuration into checkpoints.

### 5.2 H5AD ingestion

Specify a typed `RegulatoryDataset`:

- counts are two-dimensional, finite, nonnegative, and integer-valued;
- gene identifiers are unique, normalized strings;
- gene order is preserved exactly from configured AnnData metadata;
- multi-target perturbations are parsed without reordering targets;
- controls produce zero perturbation drive;
- unknown perturbation targets follow an explicit configured error policy;
- sparse perturbation records remain sparse until a model boundary requires a
  tensor of shape `S + (G,)`.

### 5.3 GraphML projection

Specify `RegulatoryGraph` and `GraphAlignmentReport` records:

- input must be directed and have string node identifiers;
- cycles are retained;
- self-loops and duplicate ordered edges are rejected or dropped according to
  one documented policy;
- graph-only genes and their incident edges are dropped and reported;
- measured genes absent from GraphML remain represented as isolated genes;
- positive edges are nonnegative, negative edges are nonpositive, and retained
  ambiguous edges are unconstrained;
- confidence metadata MUST NOT silently scale biological effect coefficients;
- all node and edge accounting appears in the alignment report.

The implementation already has initial solver and graph projection slices in
`src/nocap/sparse_regulatory.py` and `src/nocap/graphml_to_sparse.py`. Their
public behavior remains provisional until the normative specification and
traceability rows are accepted.

## 6. Solver Formalization

### 6.1 Sparse operator

For each target gene `i`, define:

$$
(A g)[..., i] = \sum_{e:\,target(e)=i} w_e g[..., regulator(e)].
$$

Production message mode MUST implement this with edge gather and out-of-place
accumulation. It MUST NOT allocate a dense `G x G` matrix.

Hard sign parameterization is applied before target-row normalization. For row
absolute sum `r_i` and configured `0 <= alpha < 1`, define:

$$
s_i =
\begin{cases}
\min(1, \alpha/r_i), & r_i > 0, \\
1, & r_i = 0.
\end{cases}
$$

Then each edge entering target `i` receives scale `s_i`. Normalized mode MUST
preserve topology and signs and satisfy `||A||_\infty <= alpha`.

### 6.2 Fixed message mode

Define iteration indexing without ambiguity:

$$
g_0=d, \qquad g_{k+1}=d+A g_k.
$$

After `K` propagation steps:

$$
g_K=\sum_{j=0}^{K}A^j d.
$$

Therefore `K=0` returns `d` exactly.

### 6.3 Tolerance message mode

After forming each candidate `g_next`, compute the per-instance residual:

$$
r_{next}=||d+A g_{next}-g_{next}||_\infty.
$$

Stop globally when every leading instance satisfies:

$$
r_{next} \le atol + rtol\,||g_{next}||_\infty,
$$

or when `max_steps` is reached. Raw mode is permitted only after dynamically
checking `||A||_\infty < 1`; otherwise it raises the specified stability
error.

### 6.4 Dense linalg mode

The reference backend constructs `M = I - A`, flattens leading dimensions into
right-hand-side columns, calls differentiable `torch.linalg.solve(M, D)`, and
restores shape `S + (G,)`.

It MUST enforce `max_dense_genes` before dense allocation. Singular or failed
factorization maps to a stable nocap domain error. Algebraic solvability does
not imply dynamic stability, so diagnostics distinguish `solved` from
`stable`.

### 6.5 Mathematical claims

Normalized mode gives `||A||_infinity <= alpha < 1`, hence a unique equilibrium
`g* = (I-A)^-1 d` and convergence of message iteration. The testable truncation
bound is:

$$
||g^*-g_K||_\infty
\le
\frac{\alpha^{K+1}}{1-\alpha}||d||_\infty.
$$

The specification MUST NOT require componentwise or strictly monotone residual
decrease for arbitrary signed systems.

### 6.6 Common result contract

Define an `EquilibriumResult` shared by both backends:

- state shape `S + (G,)`;
- residual shape `S`, scalar when `S` is empty;
- method and stopping mode;
- shared executed step count;
- maximum incoming row norm;
- distinct convergence, algebraic-solve, and stability status;
- preserved state dtype and device.

Specify all invalid topology, parameter, drive, option, dense-limit,
instability, and singular-system errors. Inputs and fixed topology MUST NOT be
mutated.

## 7. Probabilistic Model Formalization

Define the initial generative model and guide using a site dependency table.
At minimum include:

- latent cell state `z`;
- library size `l`;
- observed/intervenable factual `perturbation`;
- observed counts `x`;
- per-gene dispersion;
- basal and dropout logits;
- deterministic equilibrium logits and expected counts.

The structural path is:

$$
b(z) \rightarrow d=b(z)+p*\eta \rightarrow g=(I-A)^{-1}d
\rightarrow \mu=softmax(g).
$$

Expected counts are library-scaled and `x` uses the selected ZINB
parameterization. The specification must fix all plate and event dimensions.

Training requirements include point-estimated edge and efficacy parameters,
an L1 penalty over candidate coefficients, minibatch scaling, finite ELBO and
gradients, solver diagnostics, and reproducible checkpoint provenance.

The efficacy constraint and initialization must be explicit. The specification
must discuss dose-efficacy scale ambiguity and MUST NOT describe learned direct
edge coefficients as causal effects without the required intervention and
identification assumptions.

## 8. ChiRho Counterfactual Formalization

Represent factual perturbation as an exogenous Pyro site and apply ChiRho
`do(actions={"perturbation": ...})`. `AutoSoftConditioning` is excluded because
it softens observations rather than defining a causal intervention.

Specify abduction-action-prediction:

1. Infer factual posterior `z` and `l` once from factual observations.
2. Replay those latent values in every world.
3. Replace only the perturbation action.
4. Recompute equilibrium and deterministic expected counts.

For `N` scenarios, actions have shape `S + (N, G)`, `event_dim=1`, and masks
have shape `S + (N)`. World index `0` is factual and indices `1..N` are named
counterfactuals. Observed count likelihood is restricted to the factual world
through `SelectFactual` or equivalent conditioning semantics.

Default comparisons use deterministic means. Sampled counts MUST NOT claim
shared observation noise unless a future model introduces explicit shared
exogenous noise for the ZINB mechanism.

## 9. Executable Artifacts

### 9.1 Specsaver and Gherkin

Create feature files under `features/regulatory_vae/`:

- `experiment_config.feature`;
- `anndata_ingestion.feature`;
- `graphml_alignment.feature`;
- `equilibrium_solver.feature`;
- `regulatory_vae.feature`;
- `counterfactuals.feature`.

Tag each scenario with requirement IDs. Use precise outcomes such as
`preserved gene order`, `rejected with <domain error>`, `reported stable`,
`reported algebraically solved`, and `converged within tolerance`.

Add public-boundary Specsaver contracts and fixtures under
`tests/specifications/`. Verify links with:

```bash
specsaver trace <contract-module> --verify
```

Do not add `pytest-bdd` unless Specsaver cannot express a required execution
behavior.

### 9.2 Axiomander

Use nocap's established production convention:

```python
# --- PRE ---
assert condition, "PRE: ..."

# --- POST ---
assert property_of_result, "POST: ..."
```

Contracts should cover inexpensive public-boundary facts only. Add positive
and negative adorned tests for configuration, graph projection, solver, and
model boundaries. Expensive spectral checks and tensor-wide equalities remain
outside production paths.

### 9.3 Hypothesis

Generate valid and invalid edge lists, stable signed weights, arbitrary leading
drive shapes, gene permutations, small H5AD schemas, and GraphML/data overlaps.
Required properties include:

- sparse/dense matrix-vector equivalence;
- message/linalg equilibrium agreement;
- the Neumann truncation bound;
- linearity and disconnected superposition;
- gene-permutation equivariance;
- sign and support preservation;
- contraction in normalized mode;
- gradients against the dense reference;
- pre-softmax `K`-hop locality;
- configuration round trips;
- authoritative feature-order preservation;
- complete graph alignment accounting.

### 9.4 FizzBee

Create `specs/regulatory_solver.fizz` with finite states:

`UNCONFIGURED`, `READY`, `ITERATING`, `SOLVED`, `CAPPED`, and `ERROR`.

Model method selection, validation, stable/unstable row-norm classes,
dense-limit class, singularity class, iteration count, and terminal
diagnostics. Check these invariants:

- message mode never allocates dense state;
- linalg never starts above the dense limit;
- tolerance message mode never iterates with unstable weights;
- successful terminal states contain a result and diagnostics;
- errors never claim convergence;
- steps never exceed the cap;
- fixed `K=0` terminates with identity status;
- linalg success does not imply stability.

Add reachability checks for message convergence, message cap, linalg success,
singular failure, and validation failure.

## 10. Traceability

Create `docs/specifications/regulatory_vae_traceability.md` with one row per
normative requirement:

| Requirement | Implementation | Runtime contract | BDD scenario | Unit/property/proof/model check | Status |
| --- | --- | --- | --- | --- | --- |

Every `MUST` must have a unique requirement ID and a traceability row. An
implemented requirement cannot be marked complete without at least one mapped
executable artifact. Behavior changes require a specification version update
and corresponding traceability change.

## 11. Verification Dependencies

Add a Python 3.12+ `verification` optional dependency group:

```toml
verification = [
    "specsaver @ git+https://github.com/scidonia/specsaver.git@4e317b0291875a015de76cdfea41640202acf089",
    "axiomander[testgen] @ git+https://github.com/scidonia/axiomander.git@7362ec77e853502ca1b5b4a0699bc6723bcd49ba",
    "hypothesis>=6.155.6",
]
```

Keep Hypothesis in the ordinary `tests` extra. Do not raise nocap's base Python
requirement above 3.11; document that the complete verification environment
requires Python 3.12 because Specsaver does.

Regenerate `uv.lock` with uv. Replace the machine-specific editable Axiomander
dependency in `tox.ini` with `extras = verification` and use Python 3.12 for
Axiomander/Specsaver environments. Rocq and SMT configuration remains separate
from Python package installation.

## 12. Delivery Sequence

### Phase 1: Specification skeleton

1. Create the normative specification and terminology.
2. Assign requirement IDs to every mandatory behavior.
3. Create the traceability matrix with status `planned`.
4. Mark the old SCANVI plan as historical and link to the specification.

**Gate:** all normative statements have IDs; orientation, shapes, iteration
indexing, and residual timing are internally consistent.

### Phase 2: Data and graph boundaries

1. Specify YAML and H5AD contracts.
2. Specify GraphML projection and reporting.
3. Add Gherkin scenarios, runtime contracts, and Hypothesis tests.
4. Reconcile the initial `graphml_to_sparse` implementation with accepted
   requirements.

**Gate:** arbitrary GraphML ordering cannot change expression-column meaning,
and all dropped graph content is accounted for.

### Phase 3: Solver

1. Finalize common result and domain-error types.
2. Formalize message, tolerance, and linalg behavior.
3. Add hand-computed, property, gradient, and Axiomander tests.
4. Add and model-check the FizzBee lifecycle.
5. Reconcile the initial `sparse_regulatory` implementation with the accepted
   contract.

**Gate:** both backends satisfy common result semantics; normalized message
mode satisfies the stated bound; dense allocation is guarded.

### Phase 4: Regulatory VAE

1. Specify model and guide sites, plates, and events.
2. Implement config/data/model modules in requirement order.
3. Verify zero-edge equivalence, finite training behavior, and checkpoint
   provenance.

**Gate:** model and guide traces agree, gradients are finite, and a checkpoint
reload reproduces gene order, graph, configuration, and solver method.

### Phase 5: ChiRho worlds

1. Specify factual and scenario indexing.
2. Implement intervention and multi-world execution.
3. Verify factual-world equivalence and latent replay.

**Gate:** ordinary factual execution equals world `0` under identical replayed
latents, and scenario likelihood cannot condition on counterfactual counts.

### Phase 6: CI closure

1. Install and lock the verification extra under Python 3.12.
2. Run Specsaver trace verification.
3. Run Axiomander positive and negative contract tests.
4. Run Hypothesis and focused Pytest suites.
5. Run the FizzBee model checker.
6. Mark traceability rows complete only after their checks pass.

## 13. Validation Commands

Commands may be refined when artifact names land, but the intended gates are:

```bash
uv sync --extra verification
uv run pytest -q tests/test_graphml_to_sparse.py tests/test_sparse_regulatory.py
uv run pytest -q tests/test_*_hypothesis.py tests/test_*_axiomander.py
uv run ruff check src tests
uv run specsaver trace tests/specifications/<module>.py --verify
fizz specs/regulatory_solver.fizz
```

Run full model and ChiRho tests only after graph and solver conformance passes.
Optional CUDA checks must not replace CPU acceptance coverage.

## 14. Completion Criteria

Formalization is complete when:

- the normative specification is versioned and internally consistent;
- every `MUST` has a stable ID and traceability row;
- every implemented requirement maps to an executable verification artifact;
- graph ordering, edge orientation, solver indexing, residual timing, and
  counterfactual world indexing are unambiguous;
- Specsaver reports no unlinked contracts or scenarios;
- Axiomander negative tests demonstrate invalid behavior is rejected;
- Hypothesis checks the numerical and permutation properties;
- FizzBee safety and reachability checks pass;
- model/guide and ChiRho factual-world tests pass;
- the verification environment contains no machine-specific dependency path;
- the historical SCANVI plan points readers to the normative specification.

## 15. Explicit Non-Goals

- Modifying Pyro or ChiRho upstream.
- Claiming FizzBee proves floating-point convergence.
- Treating graph confidence as an effect-size multiplier.
- Defining GraphML order as expression order.
- Inferring causal effects without stated identification assumptions.
- Requiring monotonic biological response through signed cyclic paths.
- Sharing sampled observation noise across worlds without an explicit
  exogenous-noise model.
- Scaling dense linalg solving to production-size gene sets.
