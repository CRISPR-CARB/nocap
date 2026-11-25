# ChiRho + regulatory-network prior: CRISPR knockdown effects from synthetic scRNA-seq

Date: 2026-11-24 · Branch: `chirho-grn-prior-poc`

## Intent

A proof-of-concept Jupyter notebook, shared with colleagues for review, that shows **how to put
prior knowledge of a gene regulatory network — its edges and their polarity (+ / − / unknown) —
into a ChiRho causal model** and use it to estimate CRISPR knockdown effects from single-cell
count data.

The starting point is the ChiRho SciPlex tutorial (https://basisresearch.github.io/chirho/sciplex.html),
which fits an independent Poisson GLM per gene with latent confounders and a drug dose treatment.
It uses no network structure: genes are linked only through shared latent factors. This notebook
distills that idea to one cell type and synthetic data, and replaces "drug → each gene" with a
network-structured model where a CRISPR guide perturbs one gene and the effect propagates along
the prior network.

The primary message is pedagogical: *"here is the pattern for encoding a regulatory network in
ChiRho."* It is **not** a benchmark of network-aware vs. network-free models.

## Goals

- A clear, readable walkthrough a colleague can follow top to bottom and critique.
- Prior network structure encoded as a mask (absent edges fixed at 0).
- Polarity encoded in the weight priors: `+` → weight > 0, `−` → weight < 0, `?` → unconstrained.
- Supports feedback loops via a **steady-state** assumption (scRNA-seq snapshots treated as cells
  at equilibrium).
- CRISPR knockdown modeled as a **soft intervention** (guide shifts the target's own structural
  equation; its regulation stays intact), with **learned per-guide knockdown strength**.
- Knockdown expressed as a genuine ChiRho `do()` (on a guide variable), and usable with ChiRho's
  counterfactual handlers (`MultiWorldCounterfactual`) to compare the same cells with and without
  a knockdown.
- Validation against known ground truth: recovered edge weights, recovered knockdown strengths,
  and predicted effects of a **held-out knockdown** vs. the simulated truth.

## Non-goals

- Comparing network-aware vs. network-free models as a headline result.
- Multiple cell types, batch/replicate confounders, real data.
- Time dynamics / ODEs (`chirho.dynamical`).
- Learning network structure; the structure is given.
- Library code in `src/nocap` or unit tests — this is a notebook.

## Anti-goals

- ChiRho usage hidden behind hand-rolled intervention logic so the "how do I do this in ChiRho"
  story is lost.
- A notebook too long or too clever to review; prefer clarity over generality.
- Claims that overreach the synthetic, well-specified setting.

## Approaches considered

1. **Per-gene structural equations in topological order** (one named site per gene, `do()` on the
   gene directly). Most readable, but cannot represent feedback loops. Rejected once the user
   chose the steady-state framing, which handles DAGs and cycles uniformly.
2. **Steady-state matrix model with guide variables (chosen).** All genes solved jointly at
   equilibrium; knockdowns enter as guide sites feeding the solve, so `do(guide=1)` is a real
   ChiRho intervention whose effects propagate correctly around loops.
3. **SciPlex-style GLM with knockdown as treatment**, network only choosing responder genes.
   Rejected: no gene-to-gene propagation, which is the point.

Hard vs. soft interventions: soft is the main model (matches CRISPRi: transcription reduced,
regulation intact, feedback can partially buffer the knockdown). Hard (clamp target, cut incoming
edges) is shown as a short contrast section.

## Model

Genes `g = 1..G`, cells `c = 1..N`, guides targeting genes in a set `K` (one guide per targeted gene).

- Prior network: adjacency `A` (`A[g, p] = 1` if `p → g`) and sign `S[g, p] ∈ {+1, −1, 0(unknown)}`.
- Edge weights `W[g, p]` for edges in `A`:
  - `+`: `W = HalfNormal(σ_w)` (or LogNormal)
  - `−`: `W = −HalfNormal(σ_w)`
  - `?`: `W ~ Normal(0, σ_w)`
  - not in `A`: `W = 0`
- Baseline log-expression `b_g ~ Normal(μ_b, 1)`.
- Guide indicators `guide_k[c] ∈ {0,1}` (observed per cell; at most one active per cell).
- Knockdown strength per guide `s_k > 0` in log space, drawn from a **shared population of guide
  strengths** (hierarchical): `μ_s ~ Normal(log 1.2, 0.5)`, `τ_s ~ HalfNormal(0.3)`,
  `s_k ~ LogNormal(μ_s, τ_s)`. The hyperprior implies roughly 50–90% reduction. A held-out (new)
  guide's strength is drawn from this learned population (user decision).
- Cell-level noise `e[c, g] ~ Normal(0, σ_e)`.
- **Steady state (soft):** `z[c] = (I − W)^{-1} (b + e[c] − Σ_k guide_k[c] · s_k · 1_{target(k)})`.
  Requires spectral radius of `W` < 1 for a stable equilibrium (see Stability).
- **Steady state (hard, contrast section):** for a guided cell, zero row `target(k)` of `W` and set
  that gene's equation to a fixed low value before solving.
- Observation: `y[c, g] ~ Poisson(L[c] · exp(z[c, g]))`, `L[c]` = known per-cell size factor
  (Poisson-lognormal; overdispersion comes from `e`). No zero-inflation.

ChiRho intervention: `do(actions={"guide_X": 1})` (and vehicle `0`) on the guide sites. Average
effects computed as the contrast of expected expression under `do(guide_X=1)` vs `do(guide_X=0)`,
mirroring the tutorial's ATE construction, optionally under `MultiWorldCounterfactual` for
same-cell counterfactuals. Note: ChiRho also accepts callable interventions (`do` with
`lambda x: x - δ`), but a shift applied to a gene's value *after* the steady-state solve would not
propagate around loops — hence the guide-in-the-equation design. The notebook should explain this.

### Stability

Synthetic ground truth must satisfy spectral radius(W) < 1 (borrow the idea of
`nocap.scm_model.stabilize_cyclic_beta`). In inference, at minimum check posterior samples for
stability and report the fraction stable; optionally penalize/reject unstable draws.

### Identifiability notes (to state in the notebook)

- Knockdown cells for each guide shift mean expression by `s_k · (I − W)^{-1}[:, target]`.
  If the target has no regulators, `s_k` is just the target's own observed shift. In a loop,
  `s_k` trades off against the target's incoming weights; control-cell covariation, other
  knockdowns, and the prior resolve it.
- Learning `s_k` removes the bias from assuming a wrong fixed strength; the cost is wider
  posteriors. Demonstrate by also fitting with `s_k` wrongly fixed at "full knockout" and showing
  downstream edges get shrunk.
- Cyclic linear models are weakly identified from control cells alone, which is why knockdown
  cells are included in training.

## Synthetic data

- Toy network of ~6 genes with **one feedback loop**, mixed polarities, and **one edge whose sign
  is marked unknown** in the prior. Draw the network figure in the notebook (nodes, signed edges,
  dashed for unknown).
- Ground-truth weights drawn consistent with signs, then stabilized.
- Conditions: control cells + knockdowns of 2–3 genes used for training; **one additional
  knockdown held out** for prediction. A few hundred to ~1000 cells per condition.
- Size factors `L[c]` lognormal around a modest depth so some low-count genes appear.
- Data are generated from the same model family (well-specified) — state this as a limitation.

## Notebook outline

1. Motivation & link to the SciPlex tutorial; what's new (network prior, polarity, loops, CRISPR).
2. The prior network: adjacency + polarity table, figure.
3. Simulate ground truth and synthetic counts.
4. The ChiRho/Pyro model, explained cell by cell (mask, sign-constrained priors, steady-state solve,
   guide sites, observation model).
5. Inference: SVI, `AutoNormal`, `Trace_ELBO`, Adam; loss curve.
6. Checks: posterior edge weights vs truth (incl. sign recovery for the unknown edge), knockdown
   strengths vs truth, stability fraction.
7. Causal queries with `do()`: predicted held-out knockdown effects vs simulated truth, with
   uncertainty; same-cell counterfactual example. Two views side by side (user decision):
   (main) end-to-end prediction with the new guide's strength drawn from the learned guide
   population; (secondary) effect per unit of knockdown, isolating how well the network
   propagation `(I − W)^{-1}` was learned.
8. Soft vs hard contrast: feedback buffering of the knockdown.
9. Sensitivity: knockdown strength wrongly fixed at full knockout → biased edges.
10. Limitations & next steps (misspecified noise e.g. NegBin, larger networks, real Perturb-seq,
    multiple cell types, structure uncertainty).

## Constraints

- Python ≥3.11, existing repo tooling (uv, ruff with notebook ignores).
- New dependencies: `pyro-ppl`, `chirho`.
- Runs end-to-end on a laptop CPU in a few minutes.
- Fixed seeds for reproducibility.

## Builder's calls (decided by Claude, user may override)

- Notebook path: `notebooks/chirho_grn_prior_poc.ipynb`; all code lives in the notebook.
- Dependencies added as a new optional extra (`[project.optional-dependencies] chirho = [...]`)
  rather than core dependencies.
- One guide per targeted gene; one strength per guide (not per cell).
- Poisson-lognormal observation model; size factors treated as known.
- Hard-intervention contrast (section 8) is **query-time only**: a different intervention applied
  to the fitted soft-model posterior (and to ground truth), with a fixed clamp value. No second fit.
- `nocap.scm_model.stabilize_cyclic_beta` uses the transposed convention (`B[p, g]`); convert
  explicitly to this spec's `W[g, p]` if reusing it.

## Left to the builder

- Exact toy network topology and parameter values (within the constraints above).
- Prior hyperparameters (`σ_w`, `σ_e`, strength prior), SVI steps/learning rate.
- Whether to sample `e` and solve, or sample `z` directly as an MVN — whichever reads cleaner and
  fits well.
- How to enforce/check stability during inference.
- Plot styling.

## Visualization requirements (user, at approval)

- **Regulatory network figure** with gene labels, edge polarity (+ / − / unknown, e.g. color +
  dashed for unknown), the feedback loop visible, and guide targets marked.
- **Probabilistic model figure**: a rendered graphical model of the Pyro/ChiRho model (e.g.
  `pyro.render_model` with plates and distributions shown), plus a short plain-language legend.
- **Results plots**: ELBO loss curve; posterior edge weights vs truth (with intervals, sign of the
  unknown edge highlighted); knockdown strengths vs truth incl. learned guide population; held-out
  knockdown predicted vs true effects (both views); soft vs hard knockdown contrast; sensitivity
  (strength fixed at full knockout) edge-bias plot; a posterior predictive check of counts.
