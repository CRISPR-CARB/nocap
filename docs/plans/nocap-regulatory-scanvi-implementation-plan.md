# Regulatory SCANVI Extension Implementation Plan for nocap

## Executive Summary

Move the sparse regulatory SCANVI extension from pyro/examples into the nocap package, integrating causal cyclic coefficient estimation (σ-separation single-door criterion), sparse graph equilibrium solvers, and ChiRho multi-world counterfactual execution.

**Timeline segments:**
1. **Phase 1:** Graph → Sparse Operator abstractions
2. **Phase 2:** Standalone `SparseRegulatoryLayer` operator in nocap
3. **Phase 3:** SCANVI model integration
4. **Phase 4:** ChiRho multi-world orchestration
5. **Phase 5:** Test coverage and documentation

---

## Architecture Overview

### Layers (Bottom → Top)

```
┌─────────────────────────────────────────────────────────────────┐
│ Application: nocap regulatory SCANVI + ChiRho counterfactuals   │
│ (downstream project or optional nocap.extensions.scanvi_chirho) │
├─────────────────────────────────────────────────────────────────┤
│ nocap.scanvi.model (extended SCANVI)                            │
│   - factual_perturbation batch tensor (sparse gene/dose)        │
│   - pyro.sample("perturbation", dist.Delta(...))               │
│   - learned efficacy: u = perturbation * efficacy               │
├─────────────────────────────────────────────────────────────────┤
│ nocap.sparse_regulatory (operator + normalization)              │
│   - SparseRegulatoryLayer: fixed topology, learned weights      │
│   - forward(drive) → equilibrium via fixed-point iteration      │
│   - normalization: ||A||_∞ < alpha < 1 stability guarantee      │
├─────────────────────────────────────────────────────────────────┤
│ nocap.graphml (utilities)                                       │
│   - graphml_to_edge_index(path) → edge_index, weights           │
│   - edge_index_to_nx_digraph(edge_index) → validation/analysis  │
│   - Alignment to gene indices (via GeneOntology or extern. map) │
├─────────────────────────────────────────────────────────────────┤
│ nocap.cyclic (existing)                                         │
│   - classify_edge → identifiability + adjustment_set            │
│   - estimate_path_coefficient_for_edge → (β, stderr, σ², t)     │
│   - σ-extension, y0 integration                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Abstractions

1. **Gene Indices** (0-based, contiguous):
   - `num_genes`: Positive integer, e.g., 2000 or 20000
   - All gene references in sparse graphs use these indices
   - External mapping (e.g., gene symbol → index) handled upstream

2. **Sparse Graph Representation**:
   - `edge_index: torch.Tensor` (2, E) int64: `[regulators, targets]`
   - `edge_weights: torch.Tensor` (E,) float: signed coefficients ± per edge
   - Invariants: no self-loops, no duplicate (u, v) pairs, all endpoints ∈ [0, num_genes)

3. **Dense Adjacency** (A):
   - `A[target_idx, regulator_idx] = weight` (num_genes × num_genes)
   - Only in tests/reference code; never in forward path for >2000 genes

4. **Equilibrium State** (g):
   - Shape: `(..., num_genes)` with arbitrary leading batch/world dimensions
   - Satisfies: `g ≈ drive + A @ g` (fixed-point equation)
   - Computed via: `g = Σ_{k=0}^{K} A^k @ drive` or tolerance iteration

---

## Package Structure

### New/Modified Files

```
src/nocap/
├── scanvi/                          [NEW]
│   ├── __init__.py
│   ├── model.py                     [NEW] Extended SCANVI model
│   ├── data.py                      [NEW] Data loaders, gene mappings
│   └── integration_chirho.py        [NEW] ChiRho orchestration (optional)
├── sparse_regulatory.py             [NEW] SparseRegulatoryLayer operator
├── graphml_to_sparse.py             [REFACTOR from graphml_to_semopy.py]
│                                        Extract gene-aligned sparse extraction
├── graphml_to_semopy.py             [KEEP] Existing semopy conversion
├── cyclic_single_door.py            [KEEP] σ-separation + coeff estimation
├── scc_perturb.py                   [KEEP] Graph manipulation primitives
└── scm.py                           [KEEP] Causal model utilities

tests/
├── test_sparse_regulatory.py        [NEW] Operator preconditions/postconditions
├── test_graphml_to_sparse.py        [NEW] GraphML → edge_index validation
├── test_scanvi_model.py             [NEW] Extended SCANVI model tests
├── test_scanvi_integration.py       [NEW] Model + operator integration
└── test_examples.py                 [KEEP] CLI smoke tests (add regulatory)

docs/source/
├── scanvi.rst                       [NEW] SCANVI extension tutorial
├── regulatory_graphs.rst            [NEW] GraphML → sparse conversion guide
└── api/
    ├── sparse_regulatory.rst        [NEW] Layer API reference
    └── scanvi_model.rst             [NEW] Model API reference

examples/
├── scanvi_regulatory.py             [NEW] Standalone CLI example
├── scanvi_regulatory_chirho.py      [NEW] ChiRho counterfactual example (optional)
└── notebooks/
    └── regulatory_scanvi_tutorial.ipynb [NEW]
```

---

## Public API Boundaries

### 1. `nocap.sparse_regulatory.SparseRegulatoryLayer`

**Purpose:** PyTorch module for sparse equilibrium computation.

```python
class SparseRegulatoryLayer(nn.Module):
    def __init__(
        self,
        num_genes: int,                      # Total genes (0 to num_genes-1)
        edge_index: torch.Tensor,            # (2, E) int64
        init_weights: torch.Tensor | None,   # (E,) float, default: uniform [0, 0.1]
        stability_bound: float = 0.95,       # max row norm alpha < 1
        weight_mode: str = "normalized",     # "normalized" | "raw"
    ) -> None: ...
    
    def forward(
        self,
        drive: torch.Tensor,                # (..., num_genes) float
        num_steps: int | None = None,       # None → use tolerance mode
        tolerance_mode: Literal["fixed", "tolerance"] = "fixed",
        atol: float = 1e-5,
        rtol: float = 1e-4,
        max_steps: int | None = None,       # cap for tolerance mode
    ) -> dict[str, Any]: ...
    
    # Postcondition returns:
    # {
    #     "state": torch.Tensor,           # (..., num_genes), finite
    #     "residual": torch.Tensor | None, # scalar, ||d + A*g - g||_∞
    #     "converged": bool,               # tolerance mode only
    #     "steps": int,                    # iterations executed
    #     "max_row_norm": float,           # stability diagnostic
    # }
    
    def effective_weights(self) -> torch.Tensor:
        """Return edge weights after normalization."""
        # (E,) float
    
    def incoming_row_norms(self, weight_mode: Literal["effective", "raw"] = "effective") -> torch.Tensor:
        """Return absolute-sum per target."""
        # (num_genes,) float
    
    def dense_adjacency_for_test(self) -> torch.Tensor:
        """Debug utility: materialize dense (num_genes, num_genes) adjacency."""
        # Not part of production forward path
```

**Constructor Preconditions** (type/value checks, raises `TypeError`/`ValueError`):
- `num_genes: int > 0`
- `edge_index: torch.Tensor` dtype int64, shape (2, E)
  - All endpoints ∈ [0, num_genes)
  - No self-loops (edge_index[0] != edge_index[1])
  - No duplicate directed edges (unique (u,v) pairs)
- `init_weights: torch.Tensor | None` if provided: dtype float, shape (E,), finite values
- `0 ≤ stability_bound < 1`
- `weight_mode ∈ {"normalized", "raw"}`

**Call Preconditions**:
- `drive: torch.Tensor` finite float, shape (..., num_genes)
- `num_steps ≥ 0` (int)
- If `tolerance_mode == "tolerance"` and `weight_mode == "raw"`: raise `ValueError` (unstable raw weights)
- Device/dtype compatible with self (PyTorch `.to(...)` standard semantics)

**Postconditions**:
- Output state shape/dtype/device matches drive
- Output finite for stable finite drive
- Input and topology not mutated
- num_steps=0, zero weights, empty graph → output ≈ drive exactly
- Genes with no incoming edges equal their drive at every iteration

---

### 2. `nocap.graphml_to_sparse.GraphMLConverter`

**Purpose:** Convert GraphML regulatory networks to sparse edge-indexed format.

```python
class GraphMLConverter:
    def __init__(
        self,
        graphml_path: str | Path,
        gene_symbol_to_index: dict[str, int] | None = None,
        polarity_attr: str = "polarity",
        pos_values: Iterable[str] = {"+", "activation", "positive", "1"},
        neg_values: Iterable[str] = {"-", "repression", "negative", "-1"},
    ) -> None:
        # Load and parse GraphML file
        # gene_symbol_to_index: external or inferred from node list (alphabetical)
    
    def to_edge_index(
        self,
        signed: bool = True,
        exclude_ambiguous: bool = True,  # Skip '+/-' or None polarities
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            edge_index: (2, E) int64 [regulator_idx, target_idx]
            weights:    (E,) float signed coefficients
        """
    
    def to_nx_digraph(self) -> nx.DiGraph:
        """Return networkx.DiGraph for validation/analysis."""
        # Nodes: gene indices; edges: weighted
    
    def gene_index_map(self) -> dict[str, int]:
        """Export symbol → index mapping used."""
    
    @property
    def num_genes(self) -> int:
        """Total genes (max index + 1)."""
```

**Preconditions**:
- `graphml_path` exists and is valid GraphML
- Node attributes/IDs parse to gene symbols (non-empty strings)
- If `gene_symbol_to_index` provided: all GraphML nodes map to indices ∈ [0, num_genes)

**Postconditions**:
- `edge_index[0, i]` (regulator) ≠ `edge_index[1, i]` (target) for all i
- No duplicate (regulator, target) pairs
- `weights[i]` ≠ 0 (zero edges dropped)
- `|weights[i]| > 0` and finite

---

### 3. `nocap.scanvi.model.RegulatoryVAE` (or `RegulatoryCANVI`)

**Purpose:** Extended SCANVI with sparse regulatory layer in latent space.

```python
class RegulatoryVAE(nn.Module):
    def __init__(
        self,
        num_genes: int,
        num_labels: int,
        edge_index: torch.Tensor,              # Regulatory graph
        latent_dim: int = 10,
        hidden_dims: dict[str, list] = None,   # encoder/decoder dimensions
        l_loc: float = 0.0,
        l_scale: float = 0.5,
        alpha: float = 0.01,
        scale_factor: float = 1.0,
        regulatory_weight_mode: str = "normalized",
        regulatory_num_steps: int = 5,
    ) -> None: ...
    
    def model(
        self,
        x: torch.Tensor,          # (batch, num_genes) counts
        y: torch.Tensor | None,   # (batch,) cell type labels or None
        perturbation: torch.Tensor | None = None,  # (batch, num_genes) or sparse
    ) -> None:
        """Generative model with regulatory equilibrium."""
        # Pyro sites:
        # - "z1": latent cell type embedding
        # - "y": cell type label (obs=y if provided)
        # - "z2": latent transcriptomic state
        # - "l": library size
        # - "perturbation": factual perturbation (Delta, obs=perturbation)
        # - "x": count observation (obs=x, ZeroInflatedNegativeBinomial)
        
        # Structure:
        # 1. Sample z1, y, z2, l
        # 2. basal_logits = decoder_basal(z2) → (batch, num_genes)
        # 3. if perturbation is not None:
        #      efficacy = learned_efficacy_parameter → (num_genes,)
        #      perturbation_drive = perturbation * efficacy
        # 4. regulatory_state = sparse_layer(basal_logits + perturbation_drive)
        # 5. mu = softmax(regulatory_state)
        # 6. Observe x ~ ZeroInflatedNegativeBinomial(mu, theta, gate_logits)
    
    def guide(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        perturbation: torch.Tensor | None = None,
    ) -> None:
        """Approximate posterior (may initialize to factual observations)."""
    
    @property
    def sparse_layer(self) -> SparseRegulatoryLayer:
        """Access to regulatory equilibrium solver."""
    
    @property
    def efficacy_matrix(self) -> torch.Tensor | None:
        """Learned efficacy (num_genes,) if perturbations are active."""
```

**Preconditions**:
- `edge_index` consistent with `num_genes` (all endpoints ∈ [0, num_genes))
- `x` dtype int64 or float (counts)
- `y` None or shape (batch,) int64 (label indices ∈ [0, num_labels))
- `perturbation` None or shape (batch, num_genes) float

**Postconditions** (model + guide):
- Factual (zero perturbation) path reproduces baseline SCANVI behavior
- Perturbation affects only downstream targets of regulatory edges
- Gradients flow to edge weights and efficacy via backprop through solver

---

### 4. `nocap.scanvi.integration_chirho.predict_counterfactuals` (Optional)

**Purpose:** Execute multi-world counterfactual predictions with ChiRho.

```python
def predict_counterfactuals(
    model: RegulatoryVAE,
    guide: pyro.infer.autoguide.AutoGuide,
    x_obs: torch.Tensor,            # (batch, num_genes) factual counts
    y_obs: torch.Tensor | None,     # (batch,) factual labels
    factual_perturbation: torch.Tensor,  # (batch, num_genes)
    intervention_scenarios: dict[str, torch.Tensor],
    # intervention_scenarios = {
    #     "scenario_A": torch.randn(batch, num_genes),  # alternative dose/target
    #     "scenario_B": torch.randn(batch, num_genes),
    # }
    num_samples: int = 1,            # Posterior samples per cell
) -> dict[str, torch.Tensor]:
    """
    Returns:
    {
        "factual_mean": (batch, num_genes),
        "factual_count": (batch, num_genes) if num_samples > 0,
        "scenario_A_mean": (batch, num_genes),
        "scenario_A_count": (batch, num_genes, num_samples) if > 0,
        ...
    }
    
    Under hood:
    1. Infer z1, z2, l from factual (x_obs, y_obs)
    2. BatchedWorldCounterfactual + do(actions={"perturbation": batched_scenarios})
    3. Replay inferred z1, z2, l into factual world
    4. Evaluate model under each scenario
    5. gather(..., IndexSet(batched_interventions={0, 1, 2, ...}), event_dim=1)
    """
```

**Requires** ChiRho optional dependency; raises `ImportError` if unavailable.

---

## Implementation Details by Phase

### Phase 1: Refactor GraphML ↔ Sparse Graph Utilities

**Files modified:**
- [graphml_to_sparse.py](graphml_to_sparse.py) (new)
- [test_graphml_to_sparse.py](test_graphml_to_sparse.py) (new)

**Tasks:**
1. Extract from `graphml_to_semopy.py` the core logic:
   - Parse GraphML (networkx + pydot)
   - Resolve gene symbols → indices (alphabetical or external map)
   - Build edge_index, weights from polarity attributes
   
2. Implement `GraphMLConverter` class:
   - Constructor: load GraphML, store symbol→index map
   - `to_edge_index(signed, exclude_ambiguous)` method
   - `to_nx_digraph()` for validation
   - Docstrings + type hints
   
3. Tests (`test_graphml_to_sparse.py`):
   - Roundtrip: GraphML → edge_index → nx.DiGraph
   - Invariants: no self-loops, no duplicate edges, all indices ∈ [0, num_genes)
   - Polarity parsing (pos/neg/ambiguous)
   - External gene_symbol_to_index map application
   - Empty graph, disconnected components, cycles

**Blockers:** None; orthogonal to existing code

---

### Phase 2: Implement SparseRegulatoryLayer

**Files:**
- [sparse_regulatory.py](sparse_regulatory.py) (new, ~400–500 LOC)
- [test_sparse_regulatory.py](test_sparse_regulatory.py) (new, ~1500 LOC)

**Steps:**

1. **Precondition checking** (constructor):
   ```python
   def __init__(self, num_genes, edge_index, init_weights=None, stability_bound=0.95, weight_mode="normalized"):
       _check_num_genes(num_genes)
       _check_edge_index(edge_index, num_genes)
       _check_weights(init_weights, edge_index.shape[1] if init_weights is not None else 0)
       _check_stability_bound(stability_bound)
       # Register as buffers/parameters
   ```

2. **Sparse matvec** (no dense A):
   ```python
   def _sparse_matvec(self, state: Tensor, weights: Tensor) -> Tensor:
       # state: (..., num_genes)
       # weights: (E,) [effective or raw]
       # output: (..., num_genes) = sum_{i: edge_index[0,i]=reg, edge_index[1,i]=tgt} 
       #                            weights[i] * state[..., reg] → output[..., tgt]
       # Use PyTorch's segment sum or index_add
   ```

3. **Normalization** (incoming row norms → stability):
   ```python
   def _normalize_weights(self, raw_weights: Tensor, stability_bound: float) -> Tensor:
       # Compute per-target absolute incoming sum
       # Apply row scale: min(1, stability_bound / max_abs_sum)
       # Return normalized weights (preserve sign, guard zeros)
   ```

4. **Fixed-step iteration**:
   ```python
   def _forward_fixed_steps(self, drive: Tensor, weights: Tensor, num_steps: int) -> dict:
       state = drive
       for k in range(num_steps):
           state = drive + self._sparse_matvec(state, weights)
       residual = self._compute_residual(drive, state, weights)
       return {
           "state": state,
           "residual": residual,
           "converged": None,
           "steps": num_steps,
           "max_row_norm": self.incoming_row_norms().abs().max().item(),
       }
   ```

5. **Tolerance iteration**:
   ```python
   def _forward_tolerance(self, drive: Tensor, weights: Tensor, atol: float, rtol: float, max_steps: int) -> dict:
       state = drive
       for k in range(max_steps):
           new_state = drive + self._sparse_matvec(state, weights)
           residual = ||drive + matvec(state, weights) - state||_∞
           if residual <= atol + rtol * ||state||_∞:
               return {"state": new_state, "residual": residual, "converged": True, "steps": k+1, ...}
           state = new_state
       # Did not converge
       return {"state": state, "residual": residual, "converged": False, "steps": max_steps, ...}
   ```

6. **Stability checking**:
   - Constructor: accept `stability_bound ∈ [0, 1)`
   - In normalized mode: guarantee `||A||_∞ ≤ stability_bound` via row scaling
   - In raw mode: log warning if unstable; reject tolerance mode with `ValueError`

7. **Tests** (`test_sparse_regulatory.py`):

   **a) Preconditions** (~200 LOC):
   - Malformed `num_genes` (not positive int)
   - Malformed `edge_index` (wrong dtype, shape, endpoints out of range)
   - Self-loops, duplicate edges
   - Malformed weights (wrong shape, dtype, non-finite, empty)
   - Invalid `stability_bound`, `num_steps`, tolerances
   - Tolerance mode with raw unstable weights → ValueError

   **b) Postconditions** (~200 LOC):
   - Output shape/dtype/device matches drive
   - Input and topology not mutated
   - num_steps=0 → output = drive exactly
   - Zero weights → output = drive
   - Empty graph → output = drive
   - No incoming edges → gene value = drive at all steps
   - Outputs and diagnostics finite for stable finite inputs

   **c) Algebraic properties** (~400 LOC):
   - Dense A construction (test-only)
   - Sparse matvec = state @ A.T for multiple leading dims, float32/64
   - Fixed-step = explicit dense recurrence = Σ A^k d for K ∈ {0, 1, 2, 5}
   - Linearity: T(a*x + b*y) = a*T(x) + b*T(y)
   - Permutation equivariance under consistent relabeling
   - Locality: K-step perturbation reaches K hops max
   - Superposition: disconnected components independent
   - Orientation: asymmetric 2-node test

   **d) Normalization invariants** (~100 LOC):
   - Support/zero diagonal unchanged
   - Signs preserved, exact zeros → zero
   - Each target row norm ≤ α after normalization
   - Already-stable rows unchanged
   - Spectral radius test on small dense examples

   **e) Contraction & convergence** (~200 LOC):
   - If normalized: ||F(x) - F(y)||_∞ ≤ α ||x - y||_∞ where F(g) = d + A g
   - Residuals decrease geometrically
   - Tolerance mode reaches stated threshold
   - Error bound: ||g* - g_K||_∞ ≤ α^(K+1) / (1-α) * ||d||_∞
   - Raw unstable: tolerance mode rejects, diagnostics expose unstable row norm

   **f) Differentiation** (~200 LOC):
   - `torch.autograd.gradcheck` in float64 for drive and raw weights
   - Normalized mode gradients away from min kink
   - Finite gradients at/near kink boundary
   - Sparse ≈ dense gradients
   - ChiRho world dimensions preserved
   - Serialization (state_dict round-trip), `.to(dtype/device)`

   **g) Scale** (~100 LOC):
   - Smoke test: 2000+ genes, modest E/K, no dense allocation
   - No CI timing assertions

**Blockers:**
- PyTorch version (require ≥1.13 for robust `index_add` + scatter ops)
- `torch.linalg.solve` or custom fixed-point solver for reference (test only)

---

### Phase 3: Integrate with SCANVI Model

**Files:**
- [scanvi/model.py](scanvi/model.py) (new, ~300–400 LOC)
- [scanvi/__init__.py](scanvi/__init__.py) (new, minimal)
- [test_scanvi_model.py](test_scanvi_model.py) (new, ~300 LOC)
- [test_scanvi_integration.py](test_scanvi_integration.py) (new, ~400 LOC)

**Tasks:**

1. **Extend base SCANVI architecture**:
   - Keep existing `Z2Decoder`, `XDecoder`, `Z2LEncoder`, `Classifier`, `Z1Encoder` from pyro
   - Add `RegulatoryVAE` subclass or new class combining SCANVI + sparse layer
   - Constructor parameter: `edge_index`, regulatory graph topology
   - Initialize `SparseRegulatoryLayer` with edge_index + learned param for weights

2. **Model method**:
   ```python
   def model(self, x, y=None, perturbation=None):
       pyro.module("regulatory_vae", self)
       # Existing: z1, y, z2, l, theta
       # New: basal_logits = decoder(z2) → shape (batch, num_genes)
       # New: if perturbation is not None:
       #        efficacy ∈ (num_genes,) as pyro.param or learned module
       #        drive = basal_logits + perturbation @ efficacy
       #      else:
       #        drive = basal_logits
       # New: regulatory_state = sparse_layer.forward(drive) → equilibrium
       # Existing: gate_logits, mu = split(decoder_x(regulatory_state))
       #           Observe x ~ ZINB(...)
   ```

3. **Guide method** (inference):
   - Initially: same as baseline SCANVI (may ignore perturbation)
   - Later: optionally encode perturbation if model performance warrants

4. **Tests** (`test_scanvi_model.py`):
   - Instantiation with valid edge_index
   - Preconditions on x, y, perturbation shapes
   - Forward pass produces finite loss/logits
   - One-epoch SVI convergence on mock data
   - Parameter update (efficacy, sparse layer weights) from gradients

5. **Integration tests** (`test_scanvi_integration.py`):
   - Baseline (zero perturbation) ≈ vanilla SCANVI
   - Non-zero perturbation affects only downstream genes
   - Regulatory layer disabled (zero weights/edges) ≈ baseline
   - Multi-batch perturbations shape preservation
   - Load/save checkpoint (state_dict)

**Blockers:**
- Pyro >= 1.8.5 (for TraceEnum_ELBO, config_enumerate)
- Ensure SCANVI encoders/decoders compatible with extra regulation dimension

---

### Phase 4: ChiRho Multi-World Orchestration (Optional)

**Files:**
- [scanvi/integration_chirho.py](scanvi/integration_chirho.py) (new, ~200 LOC)
- [test_scanvi_chirho.py](test_scanvi_chirho.py) (new, ~400 LOC) [optional test extra]

**Tasks:**

1. **`predict_counterfactuals()` function**:
   ```python
   def predict_counterfactuals(
       model, guide, x_obs, y_obs, factual_perturbation,
       intervention_scenarios, num_samples=1
   ):
       # Abduct latent state z1, z2, l from factual
       # Replay into model under:
       #   BatchedWorldCounterfactual(first_available_dim=-1)
       #   do(actions={"perturbation": BatchedAction(batched_scenarios_tensor, ...)})
       #   condition(data={"x": x_obs, "y": y_obs}) [factual only via SelectFactual]
       # Extract factual + counterfactual means via gather
   ```

2. **Mask & indexing**:
   - Use `SelectFactual()` to condition observation only on factual world
   - Use `gather(..., IndexSet(batched_interventions={0, 1, ...}), event_dim=1)`
     to extract per-world outputs

3. **Tests** (`test_scanvi_chirho.py`):
   - Factual world = vanilla model prediction
   - All worlds replay identical z1, z2, l
   - Scenarios differ only in perturbation (downstream effects)
   - Shapes: (batch, num_genes) → (batch, N_scenarios, num_genes)
   - Zero-dose scenario = factual structural mean
   - Intervention signs match edge signs
   - Gradients flow to edge weights / efficacy

**Blockers:**
- ChiRho >= 0.3.0 (for BatchedWorldCounterfactual, gather, SelectFactual)
- Pyro >= 1.9.1 (ChiRho dependency)
- Optional dependency: handle ImportError gracefully

---

### Phase 5: Documentation & Examples

**Files:**
- [docs/source/scanvi.rst](docs/source/scanvi.rst) (new, ~200 lines)
- [docs/source/regulatory_graphs.rst](docs/source/regulatory_graphs.rst) (new, ~150 lines)
- [docs/source/api/sparse_regulatory.rst](docs/source/api/sparse_regulatory.rst) (new)
- [docs/source/api/scanvi_model.rst](docs/source/api/scanvi_model.rst) (new)
- [examples/scanvi_regulatory.py](examples/scanvi_regulatory.py) (new, ~150 LOC)
- [examples/scanvi_regulatory_chirho.py](examples/scanvi_regulatory_chirho.py) (new optional, ~150 LOC)
- [examples/notebooks/regulatory_scanvi_tutorial.ipynb](examples/notebooks/regulatory_scanvi_tutorial.ipynb) (new)

**Content:**
1. **scanvi.rst**: Feature overview, architecture diagram, use cases
2. **regulatory_graphs.rst**: GraphML format, gene alignment, conversion workflow
3. **sparse_regulatory.rst**: Layer API, solver modes (fixed-step vs. tolerance), stability concepts
4. **scanvi_model.rst**: Model/guide structure, perturbation handling, efficacy parameterization
5. **examples/scanvi_regulatory.py**: CLI invocation, data loading, training loop
6. **examples/scanvi_regulatory_chirho.py**: Multi-world counterfactual prediction
7. **notebooks/regulatory_scanvi_tutorial.ipynb**: Walkthrough with synthetic/real data

---

## Dependency & Version Blockers

### Required (existing nocap)
- `torch >= 1.13.0` (robust scatter/index_add)
- `numpy >= 1.20`
- `pandas >= 1.0`
- `networkx >= 2.6`
- `y0 @ git+https://github.com/y0-causal-inference/y0.git@402-add-%CF%83-...` (σ-separation)
- `pyro-ppl >= 1.8.5` (TraceEnum_ELBO, modules)

### New (sparse regulatory)
- **No new hard dependencies** (sparse layer is pure PyTorch + y0 graph tools)

### Optional (ChiRho multi-world)
- `chirho >= 0.3.0` (BatchedWorldCounterfactual, gather, SelectFactual)
- `pyro-ppl >= 1.9.1` (ChiRho dependency)

**Handling:**
- Sparse layer tests: always run
- SCANVI model tests: always run (no ChiRho needed)
- ChiRho integration tests: skip if `ImportError` (mark with `@pytest.mark.skipif(...)`)
- Documentation: separate optional section for ChiRho features

---

## Reusable Symbols & Patterns

### From existing nocap

| Symbol | File | Purpose | Reuse Plan |
|--------|------|---------|-----------|
| `nx_digraph_to_y0()` | cyclic_single_door.py | Convert nx.DiGraph → y0 NxMixedGraph | Direct import + call in model validation |
| `same_scc()` | cyclic_single_door.py | Test if two nodes in same strongly-connected component | Diagnostic for cyclic regulatory structures |
| `classify_edge()` | cyclic_single_door.py | σ-separation single-door criterion → identifiability + adjustment_set | Compute confidence intervals on edge coefficients |
| `estimate_path_coefficient_for_edge()` | cyclic_single_door.py | Estimate β via OLS w/ optimal adjustment set | Baseline edge weight initialization |
| `build_intervened_graph()` | scc_perturb.py | Simulate `do(S)` by removing in-edges to S | Counterfactual propagation reference |
| `_break_cycles()` | graphml_to_semopy.py | Minimal feedback-arc-set detection | Optional DAG conversion for comparison |

### From Pyro (examples/scanvi/)

| Symbol | File | Purpose | Reuse Plan |
|--------|------|---------|-----------|
| `Z2Decoder` | scanvi.py | Decoder p(z2 \| z1, y) | Inherit or reuse as-is |
| `XDecoder` | scanvi.py | Decoder p(x \| z2) → gate_logits, mu | Inherit or wrap |
| `Z2LEncoder` | scanvi.py | Encoder q(z2, l \| x) | Inherit or reuse |
| `Classifier` | scanvi.py | q(y \| z2) for semi-supervised label pred | Inherit or reuse |
| `Z1Encoder` | scanvi.py | Encoder q(z1 \| z2, y) | Inherit or reuse |
| `make_fc()` | scanvi.py | Fully-connected net factory | Reuse or copy as internal util |
| SCANVI model/guide structure | scanvi.py | Semi-supervised VAE template | Extend with regulatory layer |

### From ChiRho (if using)

| Symbol | Module | Purpose | Reuse Plan |
|--------|--------|---------|-----------|
| `BatchedWorldCounterfactual` | counterfactual.handlers | Multi-world execution | Primary orchestration |
| `do()` + `BatchedAction` | interventional.handlers + counterfactual.handlers | Soft interventions on `perturbation` site | Direct use in model |
| `gather()`, `IndexSet` | indexed.ops | Extract per-world values | Extract factual + counterfactual outputs |
| `SelectFactual` | counterfactual.handlers.selection | Mask sites to factual world | Condition observations only on factual world |

---

## API Stability & Backward Compatibility

1. **Existing nocap modules unchanged:**
   - `cyclic_single_door.py`, `scc_perturb.py`, `graphml_to_semopy.py`, `scm.py` remain stable
   - New code is additive (new submodule `nocap.scanvi`, `nocap.sparse_regulatory`)

2. **Export boundaries:**
   - `nocap.__init__.py`: keep existing `from .api import *`, add optional `from .scanvi import ...` if needed
   - Avoid promoting `SparseRegulatoryLayer` to top-level nocap API initially; live at `nocap.sparse_regulatory.SparseRegulatoryLayer`
   - ChiRho orchestration: optional, gated by import try/except

3. **Versioning:**
   - nocap: 0.1.0 → 0.2.0 (feature release: SCANVI extension)
   - Add classifiers: "Development Status :: 3 - Alpha" for SCANVI submodule

---

## Testing Strategy

### Test Execution Order

1. **Unit tests** (no ChiRho):
   ```bash
   pytest tests/test_graphml_to_sparse.py -q
   pytest tests/test_sparse_regulatory.py -q  # Core operator
   pytest tests/test_scanvi_model.py -q       # SCANVI model
   pytest tests/test_scanvi_integration.py -q # Model + operator integration
   ```

2. **Optional ChiRho tests**:
   ```bash
   pytest tests/test_scanvi_chirho.py -q --optional-chirho  # If installed
   ```

3. **Existing tests** (regression):
   ```bash
   pytest tests/ -k "not scanvi and not sparse and not graphml_to_sparse" -q
   ```

4. **Coverage** (tox):
   ```bash
   tox -e py311  # Standard Python 3.11 + pytest + coverage
   ```

### CI/CD Integration

- GitHub Actions: add step to run nocap tests on all supported Python versions
- Optionally: separate workflow for ChiRho-enabled tests (only if ChiRho installed)
- Codecov: track coverage for new modules

---

## File Reference Summary

### New Files to Create

```
nocap/
├── scanvi/
│   ├── __init__.py                          (~20 LOC)
│   ├── model.py                             (~350 LOC)
│   ├── data.py                              (~150 LOC, optional)
│   └── integration_chirho.py                (~200 LOC, optional)
├── sparse_regulatory.py                     (~450 LOC)
└── graphml_to_sparse.py                     (~250 LOC)

tests/
├── test_sparse_regulatory.py                (~1500 LOC)
├── test_graphml_to_sparse.py                (~300 LOC)
├── test_scanvi_model.py                     (~300 LOC)
├── test_scanvi_integration.py               (~400 LOC)
└── [OPTIONAL] test_scanvi_chirho.py         (~400 LOC)

docs/source/
├── scanvi.rst                               (~200 lines)
├── regulatory_graphs.rst                    (~150 lines)
└── api/
    ├── sparse_regulatory.rst                (~100 lines)
    └── scanvi_model.rst                     (~100 lines)

examples/
├── scanvi_regulatory.py                     (~150 LOC)
├── [OPTIONAL] scanvi_regulatory_chirho.py   (~150 LOC)
└── notebooks/
    └── regulatory_scanvi_tutorial.ipynb     (~500 cells)
```

### Modified Files

```
nocap/
├── __init__.py                              (add optional scanvi import)
└── pyproject.toml                           (add scanvi extras, update version)

tests/
└── test_examples.py                         (add regulatory smoke test)

docs/source/
└── index.rst                                (add scanvi section to TOC)
```

---

## Success Criteria

1. ✅ All precondition/postcondition tests pass (sparse layer)
2. ✅ Algebraic invariants hold (linearity, locality, permutation equivariance)
3. ✅ Convergence & contraction verified (tolerance mode, error bounds)
4. ✅ Gradients correct (autograd + gradcheck)
5. ✅ SCANVI baseline mode (zero regulatory effect) reproducible
6. ✅ Factual SCANVI + ChiRho counterfactual round-trip + extraction shape correct
7. ✅ All tests pass in CI (Python 3.11, coverage > 80%)
8. ✅ Documentation complete (API, tutorial, examples)
9. ✅ No regression in existing nocap tests
10. ✅ Performance: >2000 genes, sparse layer forward < 10 ms per batch on CPU
