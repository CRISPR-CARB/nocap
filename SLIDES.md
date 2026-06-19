# Science Snapshots — NOCAP / Cyclic ID
### *E. coli* Gene Regulatory Network Identifiability Study
**Branch:** `28-explore-identifiability-of-e-coli-network`

---

## Slide 1 — What have we demonstrated?

### Implementation of the Generalized Causal Cyclic ID Algorithm

Traditional causal-discovery methods assume **directed acyclic graphs (DAGs)** and fail
when applied to biological systems with feedback loops and self-regulation. We have
developed the first end-to-end implementation of a **generalized cyclic identification
algorithm** that can determine whether a perturbation's effect on gene expression is
identifiable — even in networks with cycles.

Given a query `P(outcome | do(TF))` on the *E. coli* transcriptional regulatory network
(TRN), the algorithm mathematically proves whether the causal effect can be uniquely
determined from observational data, or whether it is fundamentally unidentifiable given
the current graph structure.

---

### Validation with Existing Data

We grounded our causal queries using an existing *E. coli* gene regulatory network
derived from **EcoCyc**. To validate without requiring new experimental data, we
constrained our intervention space to an existing **50-gene set** from Jake Brandner's
single-guide pilot mapSPLiT experiments.

This lets us ask: *"Of the causal queries we care about biologically, how many are
answerable from observational data alone — and which experiments would unlock the rest?"*

---

### A Perturbation-Nomination Engine

New in this branch: `perturbation_optimizer.py` + `build_coverage_matrix.py` turn
*"which experiments help?"* into a formal optimization problem with provable guarantees.
37 unit tests back the logic.

---

### Conditional Independence Testing for Network Repair *(framing)*

We have established a **falsification-based refinement** workflow:

1. Generate computational predictions from the causal model
2. Validate experimentally
3. If a prediction fails → use conditional independence testing to repair the network

This branch delivers the **structural-ID half** of the loop. CI-test-driven repair is
the next integration step (see Slide 3).

---

## Slide 2 — What is the strongest current result?

### Cyclic-ID Study on the *E. coli* 50-Gene Set

> **Figure:** Information Gain from Second Perturbation
> (`notebooks/Ecoli_Analysis_Notebooks/information_gain_dual_regulatory.png`)

---

### The Baseline Problem

Of the causal queries tested on the *E. coli* TRN using the 50-gene experimental set:

| | |
|---|---|
| Total single-perturbation queries | **67** |
| **Unidentifiable at baseline** | **~82% (55 queries)** |
| Identifiable at baseline | ~18% (12 queries) |

The network's cyclic "hairball" topology — dense feedback loops between transcription
factors — makes the vast majority of queries algebraically under-determined. Traditional
DAG-based methods would simply fail silently here.

---

### The Rescue — Information Gain from a Second Perturbation

By simulating an *additional* background perturbation across **2,640 new query
combinations**, we found a **7.3% information gain**:

> **Example:** `do(phoB) → argP` is strictly unidentifiable on its own.
> It *becomes* identifiable when we simultaneously intervene via `do(rpoH)`.

---

### The New Structural Insight — Cycle-Breaking Score

This branch adds a mechanistic explanation for *why* certain background perturbations
rescue queries:

> A hard intervention `do(g)` **removes all incoming edges to `g`**, breaking every
> cycle that passes through `g`. TFs that participate in more cycles are therefore the
> highest-priority background perturbation candidates.

We quantify this with the **cycle-breaking score** (number of simple cycles containing
a node; SCC-size proxy for the full TRN) and show it **strongly predicts empirical
coverage** — the correlation between cycle score and queries rescued is visible in the
`Perturbation_Optimization.ipynb` §2.5 scatter plot.

This turns an expensive empirical search into a cheap structural pre-ranking, so the
most valuable candidates are evaluated first.

---

## Slide 3 — What is missing by September 15, and what decision do we need today?

### What is missing

| Gap | Why it matters |
|---|---|
| **Soft interventions** | Current `do()` is a hard knockout. CRISPRa/i are *soft* — the parent node still partially influences an intervened TF. We need to model this to accurately represent the biology. |
| **Data-driven cyclic discovery** | Integrating *inspre* (inverse sparse regression) to refine the network from interventional data — closing the falsification → CI-test → repair loop. |
| **Coverage matrix completion** | `build_coverage_matrix.py` is ready to run; the full ~13k `cyclic_id` evaluations need to complete to produce the final nomination table. |

---

### Decision needed today

To hit the **September 15 model-to-experiment handoff**, we need to decide:

1. **Which genes are intervenable?**
   → This defines the optimizer's candidate set. The nomination tables are only as good
   as the intervenable gene list fed into them.

2. **How many genes can be intervened on simultaneously in a combinatorial mapSPLiT screen (budget *k*)?**
   → The optimizer produces per-budget nomination tables for k = 2, 5, 10, 15, 20, 25.
   The marginal-gain curve shows exactly how many unidentifiable queries are rescued at
   each budget level.

3. **Can we achieve 100–1000s of replicates?**
   → Note: replicate count governs **estimation precision** once a query is identifiable,
   not whether it *is* identifiable (which is structural). Both matter for the handoff,
   but they are independent questions.

---

## Appendix — How NOCAP/Cyclic ID Nominates Perturbations

### Goal: minimal set of perturbations for maximum identifiability

Instead of a massive, unfocused combinatorial screen, the algorithm nominates
perturbations by treating experimental planning as a **mathematical optimization
problem**:

```
Step 1 — Identify all unidentifiable causal relationships in the baseline
          observational network (cyclic ID, Phase 1).

Step 2 — Build a coverage matrix M[candidate_tf][query]:
          M[g][q] = True  if adding do(g) as background makes query q identifiable
          M[g][q] = False otherwise
          Candidates are pre-ranked by cycle-breaking score (structural proxy)
          so the most cycle-breaking TFs are evaluated first.

Step 3 — Run the greedy optimizer:
          • Budgeted max-coverage: choose ≤ k TFs to maximize queries resolved
            (provable (1 − 1/e) ≈ 63% of optimal guarantee)
          • Min-set-cover: fewest TFs to cover all resolvable queries
            (ln(n) approximation guarantee)
```

**Why cycle-breaking score drives the ranking:**

> `do(g)` severs all incoming edges to `g`, removing `g` from every cycle it
> participates in. A single, highly-strategic intervention on a hub TF can break
> multiple feedback loops simultaneously, unlocking identifiability for many
> downstream query pairs at once.

**Output:** A small, high-information perturbation panel — e.g., the top-10 greedy
nominations — that maximizes what the next round of mapSPLiT data can identify,
saving both time and experimental resources.

---

### Key files

| File | Purpose |
|---|---|
| `scripts/build_coverage_matrix.py` | Builds M[candidate][query] via cyclic_id; resumable checkpointing |
| `scripts/perturbation_optimizer.py` | Greedy max-coverage, min-set-cover, cycle-breaking score/ranking |
| `notebooks/Ecoli_Analysis_Notebooks/Perturbation_Optimization.ipynb` | Full analysis: cycle-breaking §2.5, marginal-gain curve, nomination tables, heatmap, decision-support table |
| `tests/test_perturbation_optimizer.py` | 37 unit tests (all passing) |

---

*Generated from branch `28-explore-identifiability-of-e-coli-network` — June 2026*
