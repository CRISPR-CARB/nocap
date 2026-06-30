"""
patch_notebook_part4.py
========================
Programmatically updates the SCC_Perturbation_Analysis.ipynb notebook:
1. Replace section 5.2 header + text (DAG TFs trivially identifiable → identifiability boundary)
2. Replace section 5.3 text (joint vs per-gene → 3-state classification)
3. Update 9.A Results and Interpretation headline table/numbers
4. Update 9.A Summary caveat line about 12/18
5. Add Future Work section (§10) before References
"""

import json
from pathlib import Path

NB_PATH = (
    Path(__file__).parent.parent
    / "notebooks/Ecoli_Analysis_Notebooks/SCC_Perturbation_Analysis.ipynb"
)

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

# ── helpers ──────────────────────────────────────────────────────────────────


def find_markdown_cell_with(text):
    """Return index of first markdown cell whose source contains text."""
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown":
            src = "".join(cell["source"])
            if text in src:
                return i
    return None


def cell_source_str(idx):
    return "".join(cells[idx]["source"])


def set_cell_source(idx, new_source_str):
    """Replace a cell's source with a new string (stored as list of lines)."""
    lines = new_source_str.splitlines(keepends=True)
    # last line must not have trailing newline
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    cells[idx]["source"] = lines


# ── 1. Replace section 5.2 + 5.3 ────────────────────────────────────────────

idx_52 = find_markdown_cell_with("### 5.2 DAG TFs are Trivially Identifiable")
assert idx_52 is not None, "Could not find section 5.2 cell"
print(f"Found section 5.2 at cell index {idx_52}")

new_52_53 = """\
### 5.2 Identifiability Boundary and Observability Assumptions

The 32 DAG TFs (those not in any SCC) do not participate in regulatory feedback.  Their causal effects are identifiable from standard observational data without any additional interventional conditions.  This includes specialised metabolic regulators (aroF, aroH, fadR, lacI, trpR, tyrR, etc.).

For the 18 **SCC TFs**, the identifiability boundary is governed by the topology of the regulatory network after the background intervention `do(B(t))`.  Two structural properties drive the classification:

1. **Cut completeness** — whether `do(B(t))` actually severs every feedback return path from the in-SCC children of `t` back to `t`.  Under **Interpretation A**, the minimum cut `B(t)` excludes the TF itself and its direct in-SCC children, targeting only *intermediate* SCC nodes.  When the TF participates in a **short 2-cycle** (`t ↔ c`) with no intermediate nodes, no valid cut target exists and `B(t) = ∅`; the cut is structurally incomplete.

2. **tf_still_cyclic** — whether the TF itself remains in a non-trivial SCC of the post-intervention graph.  The Phase A experiment (§ 9.A) shows this to be the decisive predictor of unidentifiability.

**Latent TF confounding.**  The analysis assumes the *E. coli* regulatory graph is fully observed.  In practice, a TF with ≥ 2 observed target genes may act as an *unobserved common cause* if the TF itself is not measured or not directly perturbed.  The `cyclic_id` algorithm (Forré & Mooij, 2019, ref. 6) is designed to handle confounding and selection bias in cyclic graphs — but only when the confounders are *explicitly represented* in the causal model as bidirected edges.  If a latent TF creates unmeasured confounding between its targets, the current analysis would need to be extended with bidirected-edge representations of those latent paths.

**MNAR boundary.**  The `cyclic_id` procedure covers unobserved confounding (bidirected edges) and selection bias (selection nodes) but assumes data are not Missing Not At Random (MNAR).  Single-cell RNA-seq data exhibit significant *zero inflation*: a measured count of zero may reflect true biological absence or a technical dropout that depends on the gene's own expression level.  The latter is a plausibly MNAR mechanism.  The current pipeline does not model missingness explicitly, and the results should be interpreted with that caveat in mind (see also Future Work, § 10).

### 5.3 Three-State Classification of SCC TFs

The 3-state classification (`scripts/classify_scc_states.py`) recomputes cut completeness and `tf_still_cyclic` freshly from the graph and manifest for all 18 SCC TFs:

| State | Count | Description |
|---|---|---|
| **identifiable** | 6 | Cut complete, `tf_still_cyclic=False`, joint `cyclic_id` succeeded: argP, cpxR, cra, glnG, phoB, torR |
| **unidentifiable** | 0 | Cut complete, `tf_still_cyclic=False`, joint `cyclic_id` failed (clean structural result) |
| **cut_incomplete** | 12 | `do(B(t))` could not sever all return paths — `tf_still_cyclic=True`; all have direct in-SCC children that form 2-cycles with the TF |

**Key finding.** All 12 cut-incomplete TFs have `tf_still_cyclic=True` after `do(B(t))`.  The refined hypothesis H₁ from Phase A (§ 9.A) — *identifiability is achieved when and only when the TF is extracted from all cycles by the background perturbation* — is perfectly consistent with the 3-state classification: the 6 identifiable TFs all have `tf_still_cyclic=False` (cut was effective), and the 12 cut-incomplete TFs all have `tf_still_cyclic=True` (cut could not reach the direct return path).  The 0-count *unidentifiable* cell (cut complete + `tf_still_cyclic=False` + oracle fails) is notable: on this cohort, every TF with a structurally complete cut was found identifiable by `cyclic_id`.

**Interpretation A limitation.** The Interpretation A cut strategy is a fundamental constraint, not a software bug.  When a TF has a direct 2-cycle with one of its observed children (e.g. marR ↔ marA), perturbing that child would remove it from the outcome set, conflating the background intervention with the query.  A more aggressive cut strategy (Phase C) that allows augmenting B(t) with the in-SCC children *at the cost of dropping them from the outcome set Y* would resolve these cases."""

set_cell_source(idx_52, new_52_53)
print(f"  → Updated cell {idx_52} with new 5.2 + 5.3 text")

# ── 2. Update 9.A Results headline table ────────────────────────────────────

idx_9a = find_markdown_cell_with("### 9.A Results and Interpretation")
assert idx_9a is not None, "Could not find 9.A Results cell"
print(f"Found 9.A Results cell at index {idx_9a}")

old_9a_src = cell_source_str(idx_9a)

# Replace the old 2x2 table with new 3-state table + updated narrative
OLD_TABLE = (
    "| | Unidentifiable (12) | Identifiable (6) |\n"
    "|---|---|---|\n"
    "| **HAS residual cluster** | 10 (consistent) | **5 (VIOLATION, Check 2)** |\n"
    "| **NO residual cluster** | **2 (VIOLATION, Check 1)** | 1 (consistent) |"
)

NEW_TABLE = (
    "| State | Count | `tf_still_cyclic` | `cut_verified` | `joint_identifiable` |\n"
    "|---|---|---|---|---|\n"
    "| **identifiable** | 6 | False | True | True |\n"
    "| **unidentifiable** | 0 | False | True | False |\n"
    "| **cut_incomplete** | 12 | True | False | N/A — oracle not meaningful when cut fails |"
)

new_9a_src = old_9a_src.replace(OLD_TABLE, NEW_TABLE)

# Also replace the opening sentence to reflect the 3-state framing
OLD_OPENING = "Running `scripts/residual_scc_experiment.py` against all 18 SCC-TF shards produced the following headline result:"
NEW_OPENING = "Running `scripts/residual_scc_experiment.py` and `scripts/classify_scc_states.py` against all 18 SCC-TF shards produced the following headline 3-state classification:"
new_9a_src = new_9a_src.replace(OLD_OPENING, NEW_OPENING)

# Replace "The naïve verdict is *DISPROVEN*" opening with corrected framing
OLD_VERDICT = "The naïve verdict is *DISPROVEN*.  However, mechanistic diagnosis (`scripts/diagnose_phase_a.py`) reveals that both classes of violation have explanations that motivate a **refined hypothesis** rather than outright rejection."
NEW_VERDICT = "The original hypothesis — that residual child-cycles predict unidentifiability — is **disproven as stated**.  However, mechanistic diagnosis (`scripts/diagnose_phase_a.py`) reveals that the violations have structural explanations that motivate a **refined hypothesis** (H₁) rather than outright rejection.  Most strikingly, the *unidentifiable* row is **empty** (0 TFs): on this cohort, every TF for which the background cut succeeded was found identifiable by `cyclic_id`."
new_9a_src = new_9a_src.replace(OLD_VERDICT, NEW_VERDICT)

set_cell_source(idx_9a, new_9a_src)
print(f"  → Updated cell {idx_9a} with revised 9.A Results table and framing")

# ── 3. Update 9.A Summary caveat ─────────────────────────────────────────────

idx_9a_sum = find_markdown_cell_with("### 9.A Summary")
assert idx_9a_sum is not None, "Could not find 9.A Summary cell"
print(f"Found 9.A Summary cell at index {idx_9a_sum}")

old_sum_src = cell_source_str(idx_9a_sum)

OLD_CAVEAT = "- 12/18 TFs have `cut_verified=False` — the B(t) computed by `compute_min_cut_b` does not always eliminate all feedback to the TF (it targets intermediate SCC nodes, not direct child→tf edges).  A more aggressive cut strategy (Phase C) would address this."
NEW_CAVEAT = "- 12/18 TFs are classified **cut_incomplete** (`cut_verified=False`) — confirmed by `verify_cut_complete` in `nocap.scc_perturb` and independently by `classify_scc_states.py`.  The Interpretation A cut strategy targets *intermediate* SCC nodes; it cannot sever direct 2-cycle return paths.  A Phase C experiment with augmented B(t) (allowing in-SCC children to be included at the cost of reducing the outcome set Y) would address this."

new_sum_src = old_sum_src.replace(OLD_CAVEAT, NEW_CAVEAT)

# Also update the first row of the summary table
OLD_ROW1 = "| Original hypothesis (residual child-cluster predicts unidentifiability of `P(Y | do(t))` from `P(V | do(B(t)))`) | **Disproven** as stated |"
NEW_ROW1 = "| Original hypothesis (residual child-cluster predicts unidentifiability of `P(Y | do(t))` from `P(V | do(B(t)))`) | **Disproven** as stated — but violations are structural cut failures, not clean counterexamples |"
new_sum_src = new_sum_src.replace(OLD_ROW1, NEW_ROW1)

# Update "18/18 consistent" → be more precise
OLD_H1_ROW = "| Refined hypothesis H₁ | `tf_still_cyclic` perfectly predicts identifiability on this cohort (18/18 consistent) |"
NEW_H1_ROW = "| Refined hypothesis H₁ | `tf_still_cyclic` perfectly predicts identifiability on this cohort: 6 identifiable TFs all have `tf_still_cyclic=False`; 12 cut-incomplete TFs all have `tf_still_cyclic=True`; 0 clean counterexamples |"
new_sum_src = new_sum_src.replace(OLD_H1_ROW, NEW_H1_ROW)

set_cell_source(idx_9a_sum, new_sum_src)
print(f"  → Updated cell {idx_9a_sum} with revised 9.A Summary caveat")

# ── 4. Add Future Work section before References ─────────────────────────────

idx_refs = find_markdown_cell_with("## References")
assert idx_refs is not None, "Could not find References cell"
print(f"Found References cell at index {idx_refs}")

future_work_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## 10. Future Work\n",
        "\n",
        "The Phase A experiment establishes a refined hypothesis (H₁) and exposes three concrete research directions:\n",
        "\n",
        "### 10.1 Phase B — H₁ prospective validation\n",
        "\n",
        "The current analysis is *post hoc*: H₁ was formulated after observing the data.  Phase B should test H₁ prospectively on a **held-out cohort** — either a different organism's regulatory network (e.g. *B. subtilis* or yeast) or a synthetic network generated by a causal Bayesian network simulator.  This would establish whether the `tf_still_cyclic` criterion generalises beyond the *E. coli* RegulonDB network.\n",
        "\n",
        "### 10.2 Phase C — Augmented cut strategy\n",
        "\n",
        "12/18 SCC TFs have **cut_incomplete** status under Interpretation A.  Phase C should implement an **augmented cut** that allows perturbing direct in-SCC children of the TF (accepting those children as background variables rather than outcome variables).  Concretely:\n",
        "\n",
        "- For each cut-incomplete TF `t`, identify the in-SCC children `C_inSCC = {c : c → t, c ∈ SCC(t)}`.\n",
        "- Re-run `cyclic_id` with `B(t)' = B(t) ∪ C_inSCC`, restricting the outcome set `Y' = Y \\ C_inSCC`.\n",
        "- If `cyclic_id` now succeeds, the TF is *conditionally identifiable* given the augmented perturbation; if it still fails, unidentifiability is intrinsic even under maximal background intervention.\n",
        "\n",
        "This would resolve whether the 12 cut-incomplete TFs are genuinely unidentifiable or merely require a larger experimental perturbation.\n",
        "\n",
        "### 10.3 Missing Data and Zero Inflation\n",
        "\n",
        "Single-cell RNA-seq data exhibit *dropout* — technical zeros that depend on the gene's own expression level.  This is a potentially MNAR (Missing Not At Random) mechanism.  The `cyclic_id` algorithm (and do-calculus identification generally) does not accommodate MNAR data; identifiability results obtained here assume that missingness is either MCAR or MAR.  Future work should:\n",
        "\n",
        "- Model dropouts explicitly using the **M-graph / missingness graph** framework (Mohan & Pearl, 2021; ref. 4).\n",
        "- Determine whether the identifiability conditions derived here remain valid after adding missingness nodes to the causal graph.\n",
        "- Assess whether the `cyclic_id` oracle can be extended to handle m-graphs with cycles, or whether a separate recovery condition must be checked.\n",
        "\n",
        "### 10.4 Oracle Reliability and Timeout Tracking\n",
        "\n",
        "Several shards report `joint_identifiable=False` that may reflect **oracle timeout or numerical failure** rather than true structural unidentifiability.  Phase A would benefit from:\n",
        "\n",
        "- Recording `oracle_status` (SUCCEEDED / TIMED_OUT / ERROR) separately from the identifiability result.\n",
        "- Re-running timed-out shards with extended wall-time budgets.\n",
        "- Cross-validating results using an alternative identification algorithm (e.g. ID algorithm of Shpitser & Pearl, 2006) where applicable.\n",
        "\n",
        "### 10.5 Network Uncertainty\n",
        "\n",
        "The RegulonDB *E. coli* network is an *evidence-curated* but **incomplete** graph.  Missing edges (false negatives) could alter both the SCC structure and the min-cut topology.  Sensitivity analysis varying edge confidence thresholds — and comparing results across RegulonDB confidence tiers — would bound the impact of network uncertainty on the identifiability conclusions.",
    ],
}

# Insert the future work cell before the References cell
cells.insert(idx_refs, future_work_cell)
print(f"  → Inserted Future Work cell at index {idx_refs} (before References)")

# ── Save ─────────────────────────────────────────────────────────────────────

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nNotebook saved: {NB_PATH}")
print("Done.")
