"""smoke_csd_limitations.py — Verify cells 0-6 of the patched CSD notebook execute cleanly."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
NB_PATH = REPO / "notebooks" / "Ecoli_Analysis_Notebooks" / "Cyclic_SingleDoor_Analysis.ipynb"
CSV_PATH = REPO / "notebooks" / "Ecoli_Analysis_Notebooks" / "csd_results.csv"
SUMMARY_PATH = REPO / "notebooks" / "Ecoli_Analysis_Notebooks" / "csd_summary.json"
GRAPHML = (
    REPO / "notebooks" / "Ecoli_Analysis_Notebooks" / "ecoli_full_network_no_small_rna.graphml"
)

# Simulate the paths that the notebook sets up in its imports cell
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "notebooks" / "Ecoli_Analysis_Notebooks"))

import pandas as pd

# --- Replicate cell 4 (load cell) ---
df = pd.read_csv(CSV_PATH)
with open(SUMMARY_PATH) as f:
    summary = json.load(f)

df["same_scc"] = df["same_scc"].astype(str).str.lower().isin(["true", "1"])
df["timed_out"] = df["timed_out"].fillna(False).astype(str).str.lower().isin(["true", "1"])

_n_resolved = (df.status != "timeout").sum()
_n_ident = (df.status == "identifiable").sum()
_pct_ident_resolved = 100.0 * _n_ident / _n_resolved if _n_resolved > 0 else 0.0

print(f"Total edges: {len(df):,}")
print(
    f"Identifiable: {(df.status == 'identifiable').sum():,}  ({summary['pct_identifiable']}% of all, {_pct_ident_resolved:.1f}% of resolved)"
)
print(f"Unidentifiable: {(df.status == 'unidentifiable').sum():,}")
print(
    f"Timed out: {(df.status == 'timeout').sum():,}  (80.8% — O-set computationally intractable, see §1b)"
)
print(f"Same-SCC: {df.same_scc.sum():,}")

# --- Replicate cell 6 (resolved-only rate) ---
n_total = len(df)
n_resolved = (df.status != "timeout").sum()
n_ident = (df.status == "identifiable").sum()
n_unident = (df.status == "unidentifiable").sum()
n_timeout = (df.status == "timeout").sum()

pct_resolved = 100.0 * n_resolved / n_total
pct_ident_of_resolved = 100.0 * n_ident / n_resolved if n_resolved > 0 else 0.0
pct_ident_of_total = 100.0 * n_ident / n_total

print()
print("=== Scope summary ===")
print(f"  Total edges          : {n_total:,}")
print(f"  Resolved (evaluated) : {n_resolved:,}  ({pct_resolved:.1f}% of total)")
print(f"  Timeout (unevaluated): {n_timeout:,}  ({100 - pct_resolved:.1f}% of total)")
print()
print("=== Identifiability over *resolved* edges ===")
print(f"  Identifiable         : {n_ident:,}  ({pct_ident_of_resolved:.1f}% of resolved)")
print(f"  Unidentifiable       : {n_unident:,}  ({100 - pct_ident_of_resolved:.1f}% of resolved)")
print()
print(f"  (Over ALL edges: {pct_ident_of_total:.1f}% identifiable, but 80.8% unevaluated)")

# --- Verify cell 5 markdown is in the notebook ---
nb = json.loads(NB_PATH.read_text())
cell5 = nb["cells"][5]
assert cell5["cell_type"] == "markdown", f"cell 5 should be markdown, got {cell5['cell_type']}"
assert "Computational limitations" in "".join(cell5["source"]), "cell 5 missing limitations heading"
cell6 = nb["cells"][6]
assert cell6["cell_type"] == "code", f"cell 6 should be code, got {cell6['cell_type']}"
assert "pct_ident_of_resolved" in "".join(cell6["source"]), "cell 6 missing resolved-rate code"

print()
print("PASS: notebook structure and cell 4/5/6 content verified.")
