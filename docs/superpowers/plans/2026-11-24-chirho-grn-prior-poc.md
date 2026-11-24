# ChiRho GRN-Prior Knockdown POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A self-contained, reviewable Jupyter notebook showing how to encode a prior regulatory network (edges + polarity) in a ChiRho model of steady-state scRNA-seq counts, and use `do()` on CRISPR guide variables to estimate soft-knockdown effects on synthetic data.

**Architecture:** One notebook. Ground truth and model share a steady-state linear-in-log solver `z = (I − W)^{-1}(b + e − guide·s)`. The Pyro model masks `W` by the prior adjacency and sign-constrains its priors; guides are observed Bernoulli sample sites so `chirho.interventional.handlers.do` intervenes on them; counts are Poisson(size_factor · exp(z)). SVI with `AutoNormal`; causal queries via `do` + `MultiWorldCounterfactual`, cross-checked against the analytic effect formula.

**Tech Stack:** Python ≥3.11, torch, pyro-ppl, chirho, networkx, matplotlib, graphviz (system `dot` present at `/opt/homebrew/bin/dot`), uv, ruff, jupyter nbconvert.

**Spec:** `docs/superpowers/specs/2026-11-24-chirho-grn-prior-design.md`

## Global Constraints

- All code lives in `notebooks/chirho_grn_prior_poc.ipynb`; no changes to `src/nocap`, no unit-test files (spec non-goal). "Tests" are `assert` check cells inside the notebook.
- New deps go in a new optional extra `chirho = ["pyro-ppl", "chirho", "graphviz"]` in `pyproject.toml` `[project.optional-dependencies]`; not core deps.
- Notebook must pass "restart & run all" on a laptop CPU in a few minutes: `uv run --extra chirho jupyter nbconvert --to notebook --execute --inplace notebooks/chirho_grn_prior_poc.ipynb --ExecutePreprocessor.timeout=1200` exits 0.
- `uv run ruff check notebooks/chirho_grn_prior_poc.ipynb` and `uv run ruff format --check notebooks/chirho_grn_prior_poc.ipynb` pass.
- Fixed seeds: `pyro.set_rng_seed(0)`, `torch.manual_seed(0)`, `numpy.random.default_rng(0)`.
- Matrix convention everywhere: `W[g, p]` = weight of edge `p → g` (child row, parent column). If borrowing from `nocap.scm_model.stabilize_cyclic_beta` (uses `B[p, g]`), transpose explicitly.
- Polarity prior: `+` → `HalfNormal(σ_w)`, `−` → `−HalfNormal(σ_w)`, `?` → `Normal(0, σ_w)`, absent edge → exactly 0.
- Soft knockdown is the main model; hard knockdown is query-time only on the fitted soft posterior (no second fit for hard).
- Guide strength is hierarchical: `μ_s ~ Normal(log 1.2, 0.5)`, `τ_s ~ HalfNormal(0.3)`, `s_k ~ LogNormal(μ_s, τ_s)`; the held-out guide's strength comes from this population.
- Data are generated from the model family being fit; the notebook states this limitation explicitly.
- Each section starts with a plain-language markdown cell aimed at colleagues reviewing it.

## Review Focus

1. **Unstable posterior draws** (spectral radius of `W` ≥ 1): the steady state is meaningless for those draws; the notebook must report the stable fraction and compute effects only from stable draws. → check cell in Task 4.
2. **NaN/exploding loss from near-singular `I − W` during SVI**: a reader re-running with other seeds should see finite loss, not silent NaNs. → assert all losses finite in Task 4.
3. **Low-count genes / weakly identified strengths**: intervals should widen, not mislead — true values should be covered by 90% intervals. → coverage asserts in Tasks 4 and 5.
4. **Cells with more than one guide or a guide on an untargeted gene**: spec assumes ≤1 guide per cell. → assert in Task 2 data simulation.
5. **ChiRho multi-world indexing**: reading the factual vs. intervened worlds wrong silently swaps signs of effects. → Task 5 cross-checks the ChiRho effect against the analytic `s_k·(I−W)^{-1}[:, t]` formula.

---

### Task 1: Dependencies, notebook skeleton, prior network + figure (spec sections 1–2)

**Files:**
- Modify: `pyproject.toml` (`[project.optional-dependencies]`, add `chirho` extra)
- Create: `notebooks/chirho_grn_prior_poc.ipynb`

**Interfaces:**
- Produces (notebook globals):
  - `GENES: list[str] = ["A", "B", "C", "D", "E", "F"]`
  - `EDGES: list[tuple[str, str, str]]` as `(parent, child, sign)`:
    `("A","B","+"), ("B","C","+"), ("C","A","-"), ("A","D","+"), ("D","E","-"), ("B","E","?"), ("E","F","+")` — loop A→B→C⊣A (negative feedback), `B→E` is the unknown-sign edge.
  - `TRAIN_TARGETS: list[str] = ["A", "D", "E"]`, `HELDOUT_TARGET: str = "B"`, `GUIDE_TARGETS = TRAIN_TARGETS + [HELDOUT_TARGET]`
  - `prior_matrices(genes, edges) -> tuple[torch.Tensor, torch.Tensor]` returning `mask[G,G]` (float 0/1) and `sign[G,G]` (+1, −1, 0 for unknown/absent), indexed `[child, parent]`.
  - `plot_network(genes, edges, targets, heldout, weights=None, ax=None) -> matplotlib.axes.Axes`

- [ ] **Step 1:** Add the `chirho` extra to `pyproject.toml`; run `uv lock && uv run --extra chirho python -c "import pyro, chirho, graphviz; from chirho.interventional.handlers import do; from chirho.counterfactual.handlers import MultiWorldCounterfactual; print(pyro.__version__)"`. Expected: prints a version, no ImportError. If chirho pins an incompatible torch/pyro, record the resolved versions in the notebook's intro cell.
- [ ] **Step 2:** Create the notebook with: title + motivation markdown (link SciPlex tutorial; what's new: network prior, polarity, loops, CRISPR soft knockdown, steady-state assumption); imports + seeds cell; section 2 markdown explaining mask/sign encoding.
- [ ] **Step 3: Write the check cell first** (end of section 2):
  ```python
  mask, sign = prior_matrices(GENES, EDGES)
  i = {g: k for k, g in enumerate(GENES)}
  assert mask.sum() == len(EDGES)
  assert mask[i["B"], i["A"]] == 1 and sign[i["B"], i["A"]] == 1
  assert sign[i["A"], i["C"]] == -1
  assert mask[i["E"], i["B"]] == 1 and sign[i["E"], i["B"]] == 0
  assert mask[i["A"], i["B"]] == 0
  ```
- [ ] **Step 4:** Run the nbconvert command; expected FAIL with `NameError: prior_matrices`.
- [ ] **Step 5:** Implement `prior_matrices`, `plot_network` (networkx layout, gene labels, green arrows `+`, red flat-head `−`, grey dashed `?`, sign labels on edges, guide targets ringed, held-out target ringed in a distinct style, legend; optional `weights` annotates edge values). Add a cell drawing the prior network and a markdown polarity table.
- [ ] **Step 6:** Run nbconvert + ruff check/format; expected PASS, figure rendered.
- [ ] **Step 7:** Commit: `git add pyproject.toml uv.lock notebooks/chirho_grn_prior_poc.ipynb && git commit -m "Add chirho extra and notebook skeleton with prior network"`

### Task 2: Steady-state solver, ground truth, synthetic counts (spec section 3)

**Interfaces:**
- Consumes: `GENES`, `EDGES`, `prior_matrices`, target lists.
- Produces:
  - `steady_state(W: Tensor[...,G,G], inputs: Tensor[...,N,G]) -> Tensor[...,N,G]` solving `(I − W) z = inputs` with `torch.linalg.solve` (batch over leading dims; inputs broadcast).
  - `knockdown_inputs(b, e, guides, s, target_idx) -> Tensor[N,G]` = `b + e − guides @ (s[:,None] * onehot(target_idx))`, where `guides: Tensor[N,K]` 0/1, `s: Tensor[K]`, `target_idx: LongTensor[K]`.
  - `spectral_radius(W) -> Tensor` (batched, max |eigval|).
  - `TRUE: dict` with keys `W [G,G]`, `b [G]`, `s [K]` (K = len(GUIDE_TARGETS)), `sigma_e: float`.
  - `data: dict` with `guides [N,K]` float (held-out column all zero), `size_factors [N]`, `counts [N,G]` (training cells: control + TRAIN_TARGETS), and `heldout_data` with the same keys for HELDOUT_TARGET cells (used only for truth/plots).
  - `true_effect(W, s_k, t) -> Tensor[G]` = `s_k * inv(I − W)[:, t]` with a negative sign (log-expression change under knockdown).

Values: true weights magnitude in [0.4, 0.9] with correct signs (pick fixed values; the unknown edge `B→E` true sign negative so recovery is non-trivial); `b` in [0.5, 3.5]; true `s` ≈ log of 60–85% reduction (e.g. `[1.2, 1.6, 1.0, 1.4]`); `sigma_e = 0.3`; size factors `LogNormal(0, 0.3)`; 800 cells per condition.

- [ ] **Step 1: Write check cells first:**
  ```python
  assert spectral_radius(TRUE["W"]) < 1
  # solver equals fixed-point iteration
  x = torch.randn(5, len(GENES)); z = torch.zeros_like(x)
  for _ in range(500): z = x + z @ TRUE["W"].T
  assert torch.allclose(steady_state(TRUE["W"], x), z, atol=1e-5)
  assert (data["guides"].sum(-1) <= 1).all()
  assert (data["guides"][:, GUIDE_TARGETS.index(HELDOUT_TARGET)] == 0).all()
  assert ((TRUE["W"] != 0).float() == mask).all()
  # empirical knockdown shift on target matches analytic effect (log of mean-normalized counts)
  ```
  plus one assert per training target that the empirical mean `log((counts+0.5)/size_factor)` shift vs control has the sign of `true_effect(...)` for the target gene itself.
- [ ] **Step 2:** Run nbconvert; expected FAIL (NameError).
- [ ] **Step 3:** Implement the functions, `TRUE`, and simulation (with `numpy.random.default_rng(0)` / torch seed). Add markdown explaining the steady-state equation and the soft knockdown; plot the true-weighted network via `plot_network(..., weights=TRUE["W"])` and a small panel of count histograms per condition.
- [ ] **Step 4:** Run nbconvert + ruff; expected PASS.
- [ ] **Step 5:** Commit: `"Add steady-state simulator and synthetic CRISPR counts"`

### Task 3: ChiRho/Pyro model + rendered model figure (spec section 4)

**Interfaces:**
- Consumes: `mask`, `sign`, `steady_state`, `knockdown_inputs`, `data`, `GUIDE_TARGETS`.
- Produces:
  - `grn_model(guides: Tensor[N,K], size_factors: Tensor[N], counts: Tensor[N,G] | None = None, hard: bool = False, fixed_strength: float | None = None) -> Tensor[N,G]` (returns `z`).
  - Sample-site names (later tasks rely on them): `"w_pos"`, `"w_neg"`, `"w_unk"` (vectors over edges of each sign), `"b"`, `"mu_s"`, `"tau_s"`, `"s"` (shape [K], replaced by a `pyro.deterministic` of `fixed_strength` when given), `"sigma_e"` (`HalfNormal(0.5)`), per-cell `"e"` inside `pyro.plate("cells", N)`, per-guide observed sites `f"guide_{t}"` (`Bernoulli(0.25)`, `obs=guides[:, k]`), deterministic `"W"`, `"z"`, observed `"counts"` (`Poisson(size_factors[:,None] * exp(z))`).
  - `hard=True`: for cells with an active guide on target t, zero row t of `W` for that cell and set the input to `b_t + e_t − s_k`, i.e. same shift, regulation cut (feedback removed). Implement via per-cell batched W.
  - `SIGMA_W = 1.0`.

- [ ] **Step 1: Write check cells first:**
  ```python
  prior = Predictive(grn_model, num_samples=200, return_sites=["W"])(data["guides"][:10], data["size_factors"][:10])
  Wp = prior["W"]
  assert (Wp[:, mask == 0] == 0).all()
  assert (Wp[:, sign == 1] > 0).all() and (Wp[:, sign == -1] < 0).all()
  unk = (mask == 1) & (sign == 0)
  assert (Wp[:, unk] > 0).any() and (Wp[:, unk] < 0).any()
  # do() on a guide lowers the target's z, with everything else fixed
  t = GUIDE_TARGETS.index("A")
  with pyro.poutine.seed(rng_seed=1): z0 = do(grn_model, {f"guide_A": torch.zeros(10)})(data["guides"][:10]*0, data["size_factors"][:10])
  with pyro.poutine.seed(rng_seed=1): z1 = do(grn_model, {f"guide_A": torch.ones(10)})(data["guides"][:10]*0, data["size_factors"][:10])
  assert (z1[:, GENES.index("A")] < z0[:, GENES.index("A")]).all()
  ```
- [ ] **Step 2:** Run nbconvert; expected FAIL.
- [ ] **Step 3:** Implement `grn_model` with markdown cells walking through mask, sign-constrained priors, guide sites, steady-state solve, observation model. Add a cell `pyro.render_model(grn_model, model_args=(...small slice...), render_distributions=True, render_params=True)` and a plain-language legend table mapping each site to its biological meaning.
- [ ] **Step 4:** Run nbconvert + ruff; expected PASS, model graph rendered.
- [ ] **Step 5:** Commit: `"Add network-structured ChiRho model with guide sites"`

### Task 4: Inference + diagnostic plots (spec sections 5–6)

**Interfaces:**
- Consumes: `grn_model`, `data`, `TRUE`.
- Produces:
  - `fit(model_kwargs: dict, num_steps: int = 3000, lr: float = 0.02, seed: int = 0) -> tuple[AutoNormal, list[float]]`
  - `guide_soft`, `losses_soft`
  - `posterior: dict[str, Tensor]` from `Predictive(grn_model, guide=guide_soft, num_samples=500, return_sites=["W","s","mu_s","tau_s","b","sigma_e"])` on training data.
  - `stable: BoolTensor[500]` = `spectral_radius(posterior["W"]) < 1`.

- [ ] **Step 1: Write check cells first:**
  ```python
  assert all(math.isfinite(l) for l in losses_soft)
  assert np.mean(losses_soft[-100:]) < np.mean(losses_soft[:100])
  assert stable.float().mean() > 0.95
  lo, hi = posterior["W"][stable].quantile(0.05, 0), posterior["W"][stable].quantile(0.95, 0)
  covered = ((TRUE["W"] >= lo) & (TRUE["W"] <= hi))[mask == 1]
  assert covered.float().mean() >= 0.8
  eB = (GENES.index("E"), GENES.index("B"))
  assert (posterior["W"][stable][:, eB[0], eB[1]] < 0).float().mean() > 0.9  # unknown sign recovered
  ks = [GUIDE_TARGETS.index(t) for t in TRAIN_TARGETS]
  slo, shi = posterior["s"].quantile(0.05, 0), posterior["s"].quantile(0.95, 0)
  assert ((TRUE["s"][ks] >= slo[ks]) & (TRUE["s"][ks] <= shi[ks])).all()
  ```
- [ ] **Step 2:** Run nbconvert; expected FAIL.
- [ ] **Step 3:** Implement `fit` (`pyro.clear_param_store`, `AutoNormal`, `Trace_ELBO`, `ClippedAdam`/`Adam`), run it, compute `posterior` and `stable`. Plots: ELBO curve; edge weights posterior vs truth (interval plot per edge, coloured by prior sign, unknown edge highlighted, with prior sign constraint noted); strengths per guide vs truth with the learned population density `LogNormal(μ_s, τ_s)` overlaid and the held-out guide shown as "prior from population"; posterior predictive check of per-gene mean and variance of counts by condition (observed vs replicated). Markdown on identifiability (from spec) next to the strength plot.
- [ ] **Step 4:** Run nbconvert + ruff; expected PASS; total runtime noted in a markdown cell. If coverage asserts fail, tune `num_steps`/`lr`/`SIGMA_W` first — do not loosen thresholds without a markdown note explaining why.
- [ ] **Step 5:** Commit: `"Fit model with SVI and add posterior diagnostics"`

### Task 5: Causal queries with do() — held-out knockdown + counterfactual (spec section 7)

**Interfaces:**
- Consumes: `grn_model`, `guide_soft`, `posterior`, `stable`, `true_effect`, `data`, `heldout_data`.
- Produces:
  - `knockdown_effect(model, guide, target: str, guides, size_factors, num_samples=500, **model_kwargs) -> Tensor[S,G]`: runs `do(actions={f"guide_{target}": (torch.tensor(0.), torch.tensor(1.))})` under `MultiWorldCounterfactual()` with `Predictive(..., guide=guide)` on control training cells (per-cell `e` from the guide → same-cell counterfactuals); returns per-sample mean over cells of `z[do=1] − z[do=0]` using `chirho.indexed.ops.gather`/`IndexSet` to select worlds.
  - `effect_heldout_soft: Tensor[S,G]`.

- [ ] **Step 1: Write check cells first:**
  ```python
  t = GENES.index(HELDOUT_TARGET); k = GUIDE_TARGETS.index(HELDOUT_TARGET)
  analytic = torch.stack([true_effect(W, s[k], t) for W, s in zip(posterior["W"], posterior["s"])])
  assert torch.allclose(effect_heldout_soft.mean(0), analytic[stable].mean(0), atol=0.05)  # Review Focus 5
  truth = true_effect(TRUE["W"], TRUE["s"][k], t)
  lo, hi = effect_heldout_soft.quantile(0.05, 0), effect_heldout_soft.quantile(0.95, 0)
  assert ((truth >= lo) & (truth <= hi)).all()
  per_unit = analytic[stable] / posterior["s"][stable][:, k:k+1]
  pu_true = truth / TRUE["s"][k]
  assert ((pu_true >= per_unit.quantile(0.05, 0)) & (pu_true <= per_unit.quantile(0.95, 0))).float().mean() >= 0.8
  ```
  (If the ChiRho and posterior samples are drawn separately, compare means with the given tolerance rather than sample-by-sample.)
- [ ] **Step 2:** Run nbconvert; expected FAIL.
- [ ] **Step 3:** Implement `knockdown_effect`, compute `effect_heldout_soft`. Plots, two panels side by side: (main) end-to-end predicted effect per gene with 90% intervals vs simulated truth (and vs empirical held-out cell shift as a third marker); (secondary) per-unit-knockdown effect vs truth. Then a same-cell counterfactual example: for ~5 control cells, factual vs counterfactual `z` under `do(guide_B=1)`, plotted as paired points. Markdown explaining why the held-out guide's interval is wider (strength from population).
- [ ] **Step 4:** Run nbconvert + ruff; expected PASS.
- [ ] **Step 5:** Commit: `"Add do()-based held-out knockdown prediction and counterfactuals"`

### Task 6: Soft vs hard contrast + strength sensitivity (spec sections 8–9)

**Interfaces:**
- Consumes: `knockdown_effect`, `grn_model` (`hard=True`, `fixed_strength`), `fit`, `guide_soft`, `TRUE`.
- Produces: `effect_A_soft`, `effect_A_hard` (`Tensor[S,G]`, target "A", in the loop), `guide_fixed`, `posterior_fixed`.

Full-knockout strength: `FULL_KO = math.log(100.0)` (99% reduction).

- [ ] **Step 1: Write check cells first:**
  ```python
  a = GENES.index("A")
  # negative feedback A→B→C⊣A buffers soft knockdown: soft drop in A is smaller than hard
  assert effect_A_soft[:, a].mean().abs() < effect_A_hard[:, a].mean().abs()
  out_A = [(GENES.index(c), a) for p, c, _ in EDGES if p == "A"]
  shrunk = [posterior_fixed["W"][:, i, j].abs().mean() < posterior["W"][:, i, j].abs().mean() for i, j in out_A]
  assert all(shrunk)
  ```
  Also assert the same truth-side inequality on `TRUE` (soft vs hard via analytic formulas) so the claim is a property of the system, not the fit.
- [ ] **Step 2:** Run nbconvert; expected FAIL.
- [ ] **Step 3:** Implement: compute soft and hard effects of knocking down A with the fitted soft guide (`hard=True` passed as model kwarg; no refit). Bar plot soft vs hard per gene with truth markers, markdown explaining feedback buffering. Fit `guide_fixed` with `fixed_strength=FULL_KO` (same `fit`), plot outgoing edge weights of guided genes: learned-strength vs fixed-strength vs truth. Markdown on bias from mis-specified strength.
- [ ] **Step 4:** Run nbconvert + ruff; expected PASS; check total runtime still "a few minutes" (target < 10 min); if not, reduce cells per condition to 500 and note it.
- [ ] **Step 5:** Commit: `"Add soft-vs-hard contrast and knockdown-strength sensitivity"`

### Task 7: Limitations, final polish, clean full run (spec section 10)

- [ ] **Step 1:** Add closing markdown: summary of the pattern (mask + sign priors + guide sites + steady-state solve + `do`); limitations (well-specified synthetic data, linear-in-log, steady-state assumption, fixed structure, one cell type, stability handled by filtering); next steps (NegBin/misspecified generator, larger networks, real Perturb-seq, multiple cell types, structure uncertainty, `chirho.dynamical`).
- [ ] **Step 2:** Re-read every markdown cell top to bottom for a colleague audience; make sure each figure has a title, axis labels, and a one-line caption cell.
- [ ] **Step 3:** Run the full nbconvert command from a fresh kernel and both ruff commands; expected exit 0 for all, outputs saved in the notebook. Record wall time.
- [ ] **Step 4:** Commit: `"Finalize ChiRho GRN-prior POC notebook"`
