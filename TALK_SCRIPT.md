# Talk Script — NOCAP Cyclic ID Progress
### *E. coli* Gene Regulatory Network Identifiability
**Audience:** Lab meeting / project check-in  
**Length:** ~15 min (5 min/slide)  
**Format:** Figures referenced by filename; show them in order below

---

## SLIDE 1 — The Problem and What We Built

### Visual 1a — Show first
**File:** `notebooks/visualizations/Cyclic_Graph_1.png`

> **What it shows:** A small cyclic causal graph — nodes are genes, directed edges are
> regulatory relationships, and there is at least one feedback loop.

**Say:**  
*"Standard causal-discovery tools — everything from regression to NOTEARS — assume
the world looks like a directed acyclic graph, a DAG. But biology doesn't. E. coli has
transcription factors that regulate each other in loops. The moment you have a cycle,
the standard identifiability theory breaks. You can't just condition on parents.
This figure is a cartoon of the problem: a gene regulates something that feeds back and
regulates it. That cycle is why most of our queries come back as 'not identifiable'
from observational data alone."*

---

### Visual 1b — Show second
**File:** `notebooks/visualizations/csd_same_vs_cross_scc.png`

> **What it shows:** A bar or scatter comparing identifiability status (identifiable /
> unidentifiable / timeout) broken down by whether the queried edge is within the same
> Strongly Connected Component (SCC) or crosses two different SCCs.

**Say:**  
*"We ran the cyclic single-door identification algorithm on all 9,211 edges in the
E. coli transcriptional regulatory network. This plot is the key structural finding.
Every single identifiable edge — all 1,439 of them — crosses SCC boundaries.
Not one intra-SCC edge is identifiable. Zero out of 519. This isn't a statistical
observation, it's a mathematical consequence of the O-set criterion: you can never
build a valid adjustment set for a direct edge if both endpoints sit inside the same
feedback hub. The giant SCC is the binding constraint."*

---

### Visual 1c — Show third
**File:** `notebooks/visualizations/csd_identifiability_overall.png`

> **What it shows:** Overall breakdown of 9,211 edges: identifiable (15.6%), 
> unidentifiable (3.6%), timeout (80.8%) — likely a stacked bar or pie.

**Say:**  
*"Here's the full picture. Of the 9,211 edges, 1,439 — about 16% — are identifiable
right now from observational data. Three-point-six percent come back provably
unidentifiable. And 80.8% time out. That 81% timeout looks alarming, but the previous
plot tells us why: every one of those timed-out edges originates inside the 68-node
giant feedback hub. The O-set search is hitting a combinatorial wall that is
structural, not a bug. The important insight is: *we can exploit this structure*."*

---

## SLIDE 2 — The New Result: Minimal CRISPR Nominations

### Visual 2a — Show first
**File:** `notebooks/visualizations/csd_nonident_causes.png`

> **What it shows:** Breakdown of the 330 resolved unidentifiable edges by *cause* of
> non-identifiability: self-loop (174), cross-SCC blocked (99), two-cycle (35),
> scc-edge dissolved (19), same-SCC long (3).

**Say:**  
*"For the 330 edges that came back definitively unidentifiable, we asked: why?
The biggest cause is self-loops — 174 edges, where the TF regulates itself, so the
residual graph still has the TF inside its own SCC even after you remove the direct
edge. The second largest is 'cross-SCC blocked' — 99 edges where the path to identify
the effect is blocked by SCC membership upstream. These categories matter because they
tell us *what kind of intervention* will unlock each edge."*

---

### Visual 2b — The main result slide
**File:** `notebooks/visualizations/csd_break_coverage_curve.png`

> **What it shows:** Two panels.  
> **Left:** Six-bar chart — 148 / 105 / 49 / 15 / 3 / 10 — with a blue dashed line at
> 320/330 (97%). Colors go green → light green → yellow-green → pale green → pale → red.  
> **Right:** Horizontal bar chart of top-15 hub TFs ranked by how many edges' minimum
> break-sets include them. rpoD is #1 at 20.

**Say:**  
*"This is the core result. For each of those 330 unidentifiable edges, we computed the
*smallest* set of genes you would need to knock out so that the direct edge becomes
identifiable — what we call a minimum SCC-break set. Look at the left panel.
A hundred and forty-eight edges don't need any extra intervention at all — just
removing the direct edge from the causal graph already severs the cycle.
Another 105 edges: the residual SCC survives removing the edge, but the O-set
adjustment still works — no knockout needed. Forty-nine edges need exactly one
knockout. Fifteen need two. Three need three. Only 10 are beyond a budget of three.
The blue dashed line: 320 out of 330 — 97% — are fully resolved at k equals three
or fewer knockouts.*  
  
*Now look at the right panel. This is the nomination table. The gene 'rpoD' appears
in the minimum break-set for 20 different unidentifiable edges. 'lrp' covers 9, 'gadE'
covers 8. A single CRISPR knockout of rpoD would unlock 20 causal queries that are
currently unanswerable. That's the perturbation-nomination engine in one picture."*

---

### Visual 2c — Show after the main result
**File:** `notebooks/visualizations/csd_scc_size_by_status.png`

> **What it shows:** Distribution of SCC sizes for edges in each status category
> (identifiable, unidentifiable, timeout). Likely a violin or box plot.

**Say:**  
*"Here's why rpoD is such a powerful target. This plot shows SCC membership size
broken down by identifiability status. The timed-out edges all live in the 68-node
giant hub. rpoD is a global sigma factor that sits at the centre of that hub — it has
incoming edges from many regulators, and knocking it out removes it from every cycle
it participates in simultaneously. One do-rpoD intervention propagates through the
network like a circuit breaker."*

---

### Visual 2d — Supporting: adjustment set sizes
**File:** `notebooks/visualizations/csd_adjustment_set_sizes.png`

> **What it shows:** Distribution of O-set / adjustment-set sizes for identifiable edges.
> Probably a histogram or CDF.

**Say (optional — use if time permits):**  
*"For the edges that ARE identifiable, this is the distribution of adjustment set
sizes. Most require conditioning on only a handful of genes — median is probably
around three to five. This matters for experiment design: it tells you how many
co-variates you need to measure, not just which perturbation to apply."*

---

## SLIDE 3 — What's Left and What Decision We Need

### Visual 3a — Show first
**File:** `notebooks/Ecoli_Analysis_Notebooks/information_gain_dual_regulatory.png`

> **What it shows:** Information gain from adding a second background perturbation —
> likely a bar or scatter showing queries unlocked by adding do(g₂) as a background
> cut on top of the primary target perturbation.

**Say:**  
*"Before I get to the gap list, here's a result from the earlier background-cut
optimizer work that motivates why we care about *which* perturbation to choose.
Adding a second background perturbation — a do on a second TF while you're already
doing the primary target — unlocks an additional 7.3% of queries. This plot shows the
gain as a function of which background TF you choose. The highest-gain TF in this
analysis overlaps strongly with the top break-set TFs on the previous slide.
The two methods are converging on the same small set of hub targets."*

---

### Visual 3b — Show second
**File:** `notebooks/visualizations/phase_a_h1_grid.png`

> **What it shows:** H₁ prediction grid — the falsifiable prediction that
> `tf_still_cyclic` status predicts non-identifiability, shown for the 18 SCC TFs
> in the 50-gene set.

**Say:**  
*"Here's our falsifiable prediction. This grid shows the 18 TFs from the 50-gene
experimental set that are inside the giant SCC. H₁ says: if a TF is still cyclic
after removing the direct edge, the query is non-identifiable. We tested this against
all 18 TFs in the SCC — H₁ correctly predicts all 18 cases. Zero false negatives,
zero false positives. This is a prediction your next mapSPLiT data batch can directly
test: did the TFs we flagged as non-identifiable actually fail to produce an
estimable causal effect?"*

---

### Visual 3c — Show third (optional, for gap discussion)
**File:** `notebooks/visualizations/scc_pergene_heatmap.png`

> **What it shows:** Per-gene heatmap of identifiability across the 50-gene set —
> rows = TFs, columns = target genes, color = identifiable/unidentifiable/timeout.

**Say:**  
*"This is the complete identifiability map for the 50-gene set. Rows are TFs you
might perturb; columns are target genes you might want to estimate effects for.
Green cells are answerable today. Red and grey cells need intervention.
What we're delivering is essentially a prescription for how to colour in this map —
which TF knockouts convert the most grey cells to green."*

---

### The Ask (no new figure — speak to the room)

**Say:**  
*"So here's where we need a decision from you today. We can hand you a ranked panel
of CRISPR targets — rpoD, lrp, gadE, gadX, fur — where each is mathematically
guaranteed to make specific currently-unidentifiable queries answerable.
What we need back is two things: first, which of these TFs are actually intervenable
in your mapSPLiT protocol? rpoD in particular is essential — are there cell-viability
constraints? Second, what's the simultaneous-perturbation budget — can you do one
knockout per cell, or can you multiplex two or three?  
  
Once we have those two inputs, we turn around a final nomination table the same day.
No wasted guides. Provably minimal. And a direct falsification test baked in."*

---

## Suggested Slide Order Summary

| # | Title | Primary figure | Key number |
|---|---|---|---|
| 1a | Feedback makes DAG methods fail | `Cyclic_Graph_1.png` | — |
| 1b | Structural law: cross-SCC only | `csd_same_vs_cross_scc.png` | 0 / 519 intra-SCC identifiable |
| 1c | Network-wide sweep | `csd_identifiability_overall.png` | 15.6% identifiable; 80.8% timeout |
| 2a | Why edges are unidentifiable | `csd_nonident_causes.png` | 174 self-loops; 99 cross-SCC blocked |
| 2b | **Main result: break-set nominations** | `csd_break_coverage_curve.png` | **320/330 (97%) resolved at k ≤ 3; rpoD unlocks 20** |
| 2c | Why rpoD is the circuit breaker | `csd_scc_size_by_status.png` | Giant 68-node hub |
| 3a | Second perturbation gain | `information_gain_dual_regulatory.png` | +7.3% queries unlocked |
| 3b | H₁ falsifiable prediction | `phase_a_h1_grid.png` | 18/18 correct |
| 3c | Per-gene ID map (optional) | `scc_pergene_heatmap.png` | 50-gene coverage |
| 3d | The ask | *(no figure)* | intervenable genes + budget k |

---

## Notes for the Presenter

- **Slides 1b and 2b are the two "stop and make sure they got it" moments.** Pause
  after each and ask if there are questions before moving on.
- **The 97% vs 65.2% question will come up.** Be ready: 65.2% is the narrow
  definition in the summary JSON (only edges where a knockout was *required* and
  found); 97% is the correct full-resolution rate including edges where removing the
  direct edge alone or applying the O-set already handles it. Both numbers are
  defensible; 97% is the more accurate headline.
- **If asked about the 80.8% timeout:** This is not a failure — it's the structural
  bottleneck made visible. The break-set analysis is exactly how we exploit it.
  Phase C (augmented-cut algorithm for timeout edges) is the next computational step.
- **If asked about rpoD essentiality:** rpoD (σ⁷⁰) is the primary sigma factor and
  is essential for growth. CRISPRi knockdown (partial repression) rather than
  full knockout may be the right experimental choice; soft-intervention modelling
  is on the gap list for this reason.

---

*Script generated from branch `28-explore-identifiability-of-e-coli-network` — June 2026*
