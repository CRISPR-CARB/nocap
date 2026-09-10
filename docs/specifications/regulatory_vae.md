# Regulatory VAE Specification

**Version:** 0.1  
**Status:** Draft normative specification  
**Date:** 2026-08-26

## 1. Authority and Scope

This document is the semantic source of truth for nocap's graph-aware,
unlabeled regulatory VAE for E. coli perturbation data. The terms MUST, MUST
NOT, SHOULD, SHOULD NOT, and MAY are normative.

This specification supersedes conflicting statements in
`docs/plans/nocap-regulatory-scanvi-implementation-plan.md`. In particular,
the model is not label-specific SCANVI, GraphML never defines expression-column
order, and both sparse message and guarded dense solvers are supported.

## 2. Terminology and Notation

- `G` is the number of measured genes and `E` the number of retained edges.
- `S` is any leading batch or counterfactual-world shape.
- `edge_index[0, e]` is the regulator and `edge_index[1, e]` is the target.
- `A[target, regulator]` is the signed direct regulatory coefficient.
- `b(z)` is the basal decoder logit vector.
- `p` is signed perturbation dose and `eta` is nonnegative efficacy.
- `u = p * eta` is perturbation drive and `d = b(z) + u` is total drive.
- `g` is the equilibrium logit vector satisfying `g = d + A g`.

## 3. Configuration

- **CFG-001:** Configuration MUST be loaded from YAML into a typed,
  serializable `ExperimentConfig`.
- **CFG-002:** Unknown configuration keys MUST be rejected.
- **CFG-003:** Relative paths MUST resolve relative to the containing YAML
  file, not the process working directory.
- **CFG-004:** Count source MUST explicitly select AnnData `X` or one named
  layer.
- **CFG-005:** Configuration MUST identify the perturbation `obs` field,
  control values, multi-target delimiter, CRISPR type-to-sign mapping, default
  dose, and unknown-target policy.
- **CFG-006:** Cross-field validation MUST be deterministic and report the
  invalid field or relationship.

## 4. AnnData Ingestion

- **DATA-001:** Counts MUST be two-dimensional, finite, nonnegative, and
  integer-valued before model conversion.
- **DATA-002:** Gene identifiers MUST be nonempty unique strings after the
  configured normalization.
- **DATA-003:** The configured AnnData feature order MUST be preserved exactly
  as model gene order.
- **DATA-004:** GraphML order, alphabetical order, and graph traversal order
  MUST NOT reorder expression columns.
- **DATA-005:** Control observations MUST produce zero perturbation dose.
- **DATA-006:** CRISPRi and CRISPRa direction MUST be represented by configured
  signed dose; efficacy MUST NOT independently reverse that direction.
- **DATA-007:** Multi-target perturbations MUST preserve every parsed target
  and combine duplicate target doses under one documented rule.
- **DATA-008:** Unknown perturbation targets MUST follow the configured reject
  or report-and-drop policy.
- **DATA-009:** Sparse perturbation records SHOULD remain sparse until the
  model boundary requires a tensor of shape `S + (G,)`.

## 5. Graph Projection

- **GRAPH-001:** `graphml_to_sparse` MUST require a directed graph with string
  node identifiers and MUST reject parallel edges.
- **GRAPH-002:** The caller-supplied gene order MUST be authoritative for all
  tensor indices.
- **GRAPH-003:** Each retained edge MUST use regulator-to-target orientation,
  with adjacency entry `A[target, regulator]`.
- **GRAPH-004:** Directed cycles involving distinct genes MUST be retained.
- **GRAPH-005:** GraphML self-loops MUST be dropped and listed in the alignment
  report; direct construction of a solver topology containing self-loops MUST
  be rejected.
- **GRAPH-006:** Duplicate ordered edges MUST NOT reach the solver topology.
- **GRAPH-007:** Graph-only genes and their incident edges MUST be dropped and
  reported.
- **GRAPH-008:** Measured genes absent from the retained graph MUST remain in
  gene order as isolated dimensions and be reported.
- **GRAPH-009:** Positive polarity MUST constrain effective weight to be
  nonnegative, negative polarity MUST constrain it to be nonpositive, and
  unknown polarity MUST remain unconstrained.
- **GRAPH-010:** Confidence metadata MUST NOT silently scale learned effect
  coefficients.
- **GRAPH-011:** Alignment accounting MUST report measured genes, graph genes,
  retained edges, dropped edges, self-loops, graph-only genes, and isolated
  measured genes.

## 6. Sparse Regulatory Solver

### 6.1 Construction and effective weights

- **SOLVER-001:** `num_genes` MUST be a positive integer and `edge_index` MUST
  have integer shape `(2, E)` with valid endpoints, no self-loops, and no
  duplicate ordered edges.
- **SOLVER-002:** Initial weights MUST be finite floating values of shape
  `(E,)`; edge signs MUST be `int8` values in `{-1, 0, 1}` of shape `(E,)`.
- **SOLVER-003:** Topology, weights, signs, drive, dtype, and device mismatches
  MUST fail before solving with a stable domain-level error.
- **SOLVER-004:** Hard sign parameterization MUST be applied before stability
  normalization.
- **SOLVER-005:** For incoming absolute row sum `r_i`, normalized mode MUST use
  scale `min(1, alpha / r_i)` when `r_i > 0` and `1` otherwise, where
  `0 <= alpha < 1`.
- **SOLVER-006:** Normalized effective weights MUST preserve topology and hard
  signs and satisfy `||A||_infinity <= alpha`.

### 6.2 Sparse multiplication

- **SOLVER-007:** Sparse multiplication MUST compute

  $$
  (A g)[...,i] = \sum_{e:\,target(e)=i}w_e g[...,regulator(e)].
  $$

- **SOLVER-008:** Message mode MUST use edge gather and out-of-place indexed
  accumulation and MUST NOT allocate a dense `G x G` matrix.
- **SOLVER-009:** Solver inputs and registered topology MUST NOT be mutated.

### 6.3 Fixed and tolerance message solving

- **SOLVER-010:** Fixed message mode MUST define `g_0 = d` and
  `g_(k+1) = d + A g_k`; exactly `K` steps MUST return
  `g_K = sum_(j=0)^K A^j d`.
- **SOLVER-011:** Fixed message mode with `K = 0`, an empty graph, or zero
  effective weights MUST return `d` exactly.
- **SOLVER-012:** Tolerance mode MUST evaluate the residual of the returned
  candidate state as `||d + A g_next - g_next||_infinity` per leading instance.
- **SOLVER-013:** Tolerance mode MUST stop only when every leading instance
  satisfies `residual <= atol + rtol * ||g_next||_infinity`, or when
  `max_steps` is reached.
- **SOLVER-014:** Raw tolerance mode MUST reject `||A||_infinity >= 1` before
  iteration. Normalized mode is contractive because `alpha < 1`.
- **SOLVER-015:** A fixed-step result MAY report `converged = false`; its step
  count and state remain valid fixed-truncation output.

### 6.4 Dense solving and common results

- **SOLVER-016:** Linalg mode MUST enforce `max_dense_genes` before allocating
  a dense adjacency matrix.
- **SOLVER-017:** Linalg mode MUST solve `(I-A)g=d` with a differentiable
  `torch.linalg.solve`, batching leading instances as right-hand-side columns.
- **SOLVER-018:** Singular or failed factorization MUST raise a stable domain
  error rather than return a false equilibrium.
- **SOLVER-019:** Algebraic solvability MUST NOT imply dynamic stability;
  diagnostics MUST distinguish solved/converged status from `stable`.
- **SOLVER-020:** Both methods MUST return state shape `S + (G,)`, residual
  shape `S` (scalar when `S` is empty), method, convergence, steps, maximum row
  norm, and stability status while preserving state dtype and device.

### 6.5 Mathematical guarantee

- **SOLVER-021:** In normalized mode, the implementation and documentation MAY
  claim a unique equilibrium and convergence only from
  `||A||_infinity <= alpha < 1`.
- **SOLVER-022:** Fixed-step validation MUST use the bound

  $$
  ||g^*-g_K||_infinity
  \le \frac{\alpha^{K+1}}{1-\alpha}||d||_infinity.
  $$

- **SOLVER-023:** The system MUST NOT claim componentwise or strictly monotone
  residual decrease for arbitrary signed systems.

## 7. Probabilistic Model

- **MODEL-001:** The initial model MUST be an unlabeled `RegulatoryVAE`; it
  MUST NOT introduce a label latent solely to retain SCANVI naming.
- **MODEL-002:** The decoder MUST produce basal logits `b(z)` and dropout
  logits, with one learned dispersion parameter per gene.
- **MODEL-003:** Perturbation efficacy MUST be one learned nonnegative scalar
  per gene and drive MUST be `u = p * eta`.
- **MODEL-004:** Equilibrium logits MUST be computed from `d = b(z) + u`, and
  proportions MUST be `softmax(g)` before library scaling.
- **MODEL-005:** Observed counts MUST use the selected zero-inflated negative
  binomial parameterization with documented plate and event dimensions.
- **MODEL-006:** If `A = 0`, equilibrium logits MUST equal `d`; if `p = 0` but
  `A != 0`, regulatory feedback remains active and vanilla-decoder equivalence
  MUST NOT be claimed.
- **MODEL-007:** In fixed `K`-step mode, perturbation influence on pre-softmax
  logits MUST NOT propagate beyond `K` directed hops. No equivalent locality
  claim MAY be made after softmax normalization.
- **MODEL-008:** Training MUST support point-estimated edge and efficacy
  parameters, L1 regularization over candidate coefficients, minibatch
  scaling, finite ELBO values, and finite gradients.
- **MODEL-009:** Learned edge coefficients MUST NOT be described as causal
  effects without the stated intervention, model, and identification
  assumptions.

## 8. ChiRho Interventions and Counterfactuals

- **CF-001:** Factual perturbation MUST be represented by an exogenous Pyro
  `perturbation` sample site with gene dimension as event dimension.
- **CF-002:** ChiRho `do` MUST replace only `perturbation`; it MUST NOT clamp
  counts, expected means, or intermediate regulatory values.
- **CF-003:** Counterfactual prediction MUST infer factual posterior `z` and
  library size once and replay those same values in every world.
- **CF-004:** Scenario actions MUST have shape `S + (N, G)`, masks shape
  `S + (N)`, and `event_dim = 1`.
- **CF-005:** World index `0` MUST be factual and indices `1..N` MUST map to
  named scenarios in stable order.
- **CF-006:** Observed count likelihood MUST contribute only in the factual
  world through `SelectFactual` or equivalent conditioning.
- **CF-007:** Factual world execution MUST equal ordinary execution under the
  same replayed latent values.
- **CF-008:** Default comparisons MUST use deterministic means. Sampled counts
  MUST NOT claim shared observation noise without an explicit shared
  exogenous-noise model.

## 9. Provenance

- **PROV-001:** Checkpoints MUST contain the resolved configuration, exact gene
  order, projected graph topology and signs, solver settings, and learned model
  parameters.
- **PROV-002:** Reload MUST reject incompatible gene order or graph identity
  rather than silently permute or partially load parameters.
- **PROV-003:** Alignment reports and source perturbation metadata MUST be
  retained with trained-model provenance.
- **PROV-004:** A reproducible run SHOULD record package versions and immutable
  VCS revisions for unreleased dependencies.

## 10. Error Taxonomy

- **ERR-001:** Invalid configuration and data schema MUST raise validation
  errors identifying the responsible field.
- **ERR-002:** Invalid graph topology or alignment MUST raise graph-domain
  errors before model construction.
- **ERR-003:** Invalid solver options, dtype, device, shape, or stability MUST
  raise solver-domain errors before returning a result.
- **ERR-004:** Dense-limit and singular-system failures MUST be distinguishable.
- **ERR-005:** Errors MUST NOT report convergence or a valid equilibrium.

## 11. Verification Boundaries

- Runtime contracts verify inexpensive boundary facts.
- Unit tests pin hand-computed examples and error behavior.
- Hypothesis checks algebraic, permutation, shape, and gradient properties.
- Axiomander proves contracts only over its supported Python subset.
- Specsaver-linked Gherkin specifies externally observable workflows.
- FizzBee models finite solver lifecycle transitions, not real arithmetic.

Every normative requirement MUST have a row in
`docs/specifications/regulatory_vae_traceability.md`. A requirement is complete
only when its implementation and mapped executable evidence both exist.
