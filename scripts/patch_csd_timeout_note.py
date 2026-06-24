"""patch_csd_timeout_note.py — Add computational-limitations section to CSD notebook.

Inserts:
  - A new markdown cell (§1b. Computational Limitations) after cell 4/5 (between
    the load cell and the "2. Overall identifiability" heading) documenting that
    80.8% of edges timed out at 60 s, a 20-minute calibration found 0/128 resolved,
    and identifiability rates are therefore quoted over *resolved* edges only.
  - Updates cell 4 (load cell) to also print the resolved-only identifiability rate.
  - Adds a one-sentence caveat to the intro (cell 0) about timeout scope.

Usage:
    uv run python scripts/patch_csd_timeout_note.py
"""
from __future__ import annotations

import json
from pathlib import Path
import copy

NB_PATH = (
    Path(__file__).parent.parent
    / "notebooks"
    / "Ecoli_Analysis_Notebooks"
    / "Cyclic_SingleDoor_Analysis.ipynb"
)

# ── New cells to insert ────────────────────────────────────────────────────────

LIMITATIONS_MARKDOWN = """\
## 1b. Computational limitations — O-set scalability

### Timeout scope
The classification sweep ran with a **60-second per-edge SIGALRM timeout**.
Of 9,211 edges in the *E. coli* network:

| Status | Count | % |
|---|---|---|
| identifiable | 1,439 | 15.6% |
| **timeout** | **7,442** | **80.8%** |
| unidentifiable | 330 | 3.6% |

### Why the timeouts are structural

A 128-edge calibration batch was rerun at **1,200 s (20 min) per edge**
(SLURM jobs 873096/873097, 2026-06-23).  **0 of 128 edges resolved** at
the longer budget (resolution rate 0.0%).  This confirms that the bottleneck
is **not** wall-clock time but **O-set enumeration complexity** in the
giant feedback SCC.

The Perkovic et al. O-set algorithm enumerates a set that can be
exponential in the number of nodes in the SCC.  For the *E. coli* network,
which contains a giant SCC of ~130 transcription factors with dense
bidirectional edges, the O-set search space is intractable for the majority
of within-SCC edges at any practical timeout.

### Interpretation

All **identifiability rates reported below** (§2–§4) are quoted over
**resolved edges only** (1,769 = 1,439 identifiable + 330 unidentifiable),
which represent the 19.2% of edges for which the O-set algorithm terminated.
The remaining 80.8% are classified as **computationally undetermined** —
not identifiable or unidentifiable, but unevaluated due to O-set complexity.

This is a known limitation of the sigma-single-door / O-set approach on
large cyclic networks and is documented here for transparency.
"""

LIMITATIONS_CODE = """\
# Resolved-only identifiability rate (excludes timeouts)
n_total = len(df)
n_resolved = (df.status != "timeout").sum()
n_ident = (df.status == "identifiable").sum()
n_unident = (df.status == "unidentifiable").sum()
n_timeout = (df.status == "timeout").sum()

pct_resolved = 100.0 * n_resolved / n_total
pct_ident_of_resolved = 100.0 * n_ident / n_resolved if n_resolved > 0 else 0.0
pct_ident_of_total = 100.0 * n_ident / n_total

print("=== Scope summary ===")
print(f"  Total edges          : {n_total:,}")
print(f"  Resolved (evaluated) : {n_resolved:,}  ({pct_resolved:.1f}% of total)")
print(f"  Timeout (unevaluated): {n_timeout:,}  ({100-pct_resolved:.1f}% of total)")
print()
print("=== Identifiability over *resolved* edges ===")
print(f"  Identifiable         : {n_ident:,}  ({pct_ident_of_resolved:.1f}% of resolved)")
print(f"  Unidentifiable       : {n_unident:,}  ({100-pct_ident_of_resolved:.1f}% of resolved)")
print()
print(f"  (Over ALL edges: {pct_ident_of_total:.1f}% identifiable, but 80.8% unevaluated)")
"""

# ── Updated cell 4 source (adds resolved-rate line) ───────────────────────────

CELL4_NEW = """\
df = pd.read_csv(CSV_PATH)
with open(SUMMARY_PATH) as f:
    summary = json.load(f)

# Normalise types
df["same_scc"] = df["same_scc"].astype(str).str.lower().isin(["true", "1"])
df["timed_out"] = df["timed_out"].fillna(False).astype(str).str.lower().isin(["true", "1"])

_n_resolved = (df.status != "timeout").sum()
_n_ident = (df.status == "identifiable").sum()
_pct_ident_resolved = 100.0 * _n_ident / _n_resolved if _n_resolved > 0 else 0.0

print(f"Total edges: {len(df):,}")
print(f"Identifiable: {(df.status=='identifiable').sum():,}  ({summary['pct_identifiable']}% of all, {_pct_ident_resolved:.1f}% of resolved)")
print(f"Unidentifiable: {(df.status=='unidentifiable').sum():,}")
print(f"Timed out: {(df.status=='timeout').sum():,}  (80.8% — O-set computationally intractable, see §1b)")
print(f"Same-SCC: {df.same_scc.sum():,}")
print()
print(df.head())
"""

# ── Updated cell 0: add caveat sentence to intro paragraph ────────────────────

INTRO_CAVEAT = (
    "\n\n> **Scope note:** 80.8% of edges (7,442/9,211) hit the per-edge O-set timeout "
    "and are **computationally undetermined**. A 20-minute calibration (128 edges) confirmed "
    "this is structural, not a wall-clock issue (0/128 resolved). "
    "Identifiability rates are quoted over the 19.2% of edges that were evaluated. See §1b."
)


def make_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]

    # 1. Patch cell 0 (intro): append caveat
    cells[0]["source"] = "".join(cells[0]["source"]) + INTRO_CAVEAT

    # 2. Patch cell 4 (load cell): replace source
    assert "pd.read_csv(CSV_PATH)" in "".join(cells[4]["source"]), \
        "cell 4 doesn't look like the load cell — aborting"
    cells[4]["source"] = CELL4_NEW

    # 3. Insert §1b after cell 4 and before cell 5 (old "## 2. Overall identifiability")
    #    We insert: [markdown §1b, code for resolved rates]
    new_md = make_markdown_cell(LIMITATIONS_MARKDOWN)
    new_code = make_code_cell(LIMITATIONS_CODE)
    # Insert after index 4 → indices 5 and 6 become the new cells
    cells.insert(5, new_md)
    cells.insert(6, new_code)

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Patched notebook: {NB_PATH}")
    print(f"  cell 0 : intro + caveat")
    print(f"  cell 4 : load cell updated (resolved-only rate)")
    print(f"  cell 5 : NEW §1b Computational Limitations (markdown)")
    print(f"  cell 6 : NEW resolved-rate summary code")
    print(f"  Total cells now: {len(cells)}")


if __name__ == "__main__":
    main()
