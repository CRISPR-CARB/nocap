"""Verify all phrasing fixes were applied to the notebook."""
import json
from pathlib import Path

nb = json.load(open("notebooks/Ecoli_Analysis_Notebooks/SCC_Perturbation_Analysis.ipynb"))

# 1. No stale do(t, B(t)) in any markdown cell
stale = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"])
        if "do(t, B(t))" in src:
            stale.append(i)

if stale:
    print(f"FAIL: stale 'do(t, B(t))' found in cells: {stale}")
else:
    print("PASS: no stale 'do(t, B(t))' in any markdown cell")

# 2. Spot-check key corrected phrases
checks = [
    (0,  "identifiable from the background interventional distribution"),
    (0,  "P(Y | do(t))` for the direct children"),
    (0,  "3. Tests whether the target query `P(Y | do(t))`"),
    (7,  "cyclic_id` with minimum in-edge cut"),
    (7,  "minimum in-edge cut"),
    (22, "cannot express\n`P(Y | do(t))` as a functional of the background"),
    (22, "background distribution `P(V | do(B(t)))` is sufficient"),
    (22, "joint query `P(Y\u2081,\u2026,Y\u2099 | do(t))` is not identifiable from"),
    (36, "P(Y | do(t))` is **unidentifiable from `P(V | do(B(t)))`**"),
    (39, "residual child-cluster predicts unidentifiability of"),
]

all_ok = True
for cell_idx, snippet in checks:
    src = "".join(nb["cells"][cell_idx]["source"])
    if snippet in src:
        print(f"  OK  cell {cell_idx:>2}: {snippet[:70]!r}")
    else:
        print(f"  MISS cell {cell_idx:>2}: {snippet[:70]!r}")
        all_ok = False

print()
if all_ok and not stale:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED")
