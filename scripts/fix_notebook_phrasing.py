"""
Apply targeted text fixes to SCC_Perturbation_Analysis.ipynb.
Operates directly on parsed notebook cells (avoids raw JSON escape issues).
"""
import json
from pathlib import Path

NB = Path("notebooks/Ecoli_Analysis_Notebooks/SCC_Perturbation_Analysis.ipynb")

nb = json.load(open(NB))

# Each entry: (cell_index, old_substring, new_substring)
fixes = [
    # ── Cell 22, Section 5.3: model assumption paragraph ──────────────────
    (
        22,
        "`cyclic_id` reports non-identifiability when it cannot derive "
        "`P(Y | do(t, B(t)))` as a functional of the observational distribution under "
        "the do-calculus, given the (still possibly cyclic) graph that remains after "
        "the `do(B(t))` intervention.",
        "`cyclic_id` reports non-identifiability when it cannot express "
        "`P(Y | do(t))` as a functional of the background interventional distribution "
        "`P(V | do(B(t)))` under the do-calculus, given the (still possibly cyclic) "
        "graph that remains after the `do(B(t))` intervention.",
    ),
    # ── Cell 22, Section 5.3: jointly identifiable bullet ─────────────────
    (
        22,
        "— the `do(B(t))` intervention is sufficient for `cyclic_id` to identify the "
        "joint query over all children of `t`.",
        "— the background distribution `P(V | do(B(t)))` is sufficient for `cyclic_id` "
        "to identify `P(Y | do(t))` over all direct children of `t`.",
    ),
    # ── Cell 22, Section 5.3: jointly unidentifiable bullet ───────────────
    (
        22,
        "the joint query `P(Y₁,…,Yₙ | do(t, B(t)))` is not identifiable, yet "
        "individual queries `P(Yᵢ | do(t, B(t)))` succeed for a subset of children.",
        "the joint query `P(Y₁,…,Yₙ | do(t))` is not identifiable from "
        "`P(V | do(B(t)))`, yet individual queries `P(Yᵢ | do(t))` are identifiable "
        "for a subset of children.",
    ),
    # ── Cell 36, Section 9.A: H₁ biconditional ────────────────────────────
    (
        36,
        "> **H₁:**  `P(Y | do(t, B(t)))` is **unidentifiable** if and only if "
        "`tf_still_cyclic = True` after `do(B(t))`.",
        "> **H₁:**  `P(Y | do(t))` is **unidentifiable from `P(V | do(B(t)))`** "
        "if and only if `tf_still_cyclic = True` after `do(B(t))`.",
    ),
    # ── Cell 39, Section 9.A summary: original hypothesis row ─────────────
    (
        39,
        "| Original hypothesis (residual child-cluster ↔ unidentifiable) | "
        "**Disproven** as stated |",
        "| Original hypothesis (residual child-cluster predicts unidentifiability of "
        "`P(Y | do(t))` from `P(V | do(B(t)))`) | **Disproven** as stated |",
    ),
]

changed = 0
for cell_idx, old, new in fixes:
    cell = nb['cells'][cell_idx]
    src = cell['source']
    joined = ''.join(src)
    if old in joined:
        joined = joined.replace(old, new, 1)
        # Re-split on newlines, preserving them
        lines = joined.splitlines(keepends=True)
        cell['source'] = lines
        changed += 1
        print(f"  Fixed cell {cell_idx}: {old[:60]!r}...")
    else:
        print(f"  WARNING: not found in cell {cell_idx}: {old[:60]!r}...")

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Validate
nb2 = json.load(open(NB))
print(f"\nApplied {changed}/{len(fixes)} fixes.")
print(f"Notebook valid: nbformat {nb2['nbformat']}, {len(nb2['cells'])} cells.")
