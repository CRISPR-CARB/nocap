"""csd_diagnose_nonident.py — Pre-compute non-identifiability cause taxonomy.

Reads csd_results.csv + the GraphML, classifies every unidentifiable edge into
one of five structural categories, and writes csd_nonident_diagnosis.csv.

This is designed to run as a standalone script (or a fast SLURM job) so that
the notebook only needs to load the result rather than run the graph computation.

Usage:
    uv run python scripts/csd_diagnose_nonident.py

Outputs:
    notebooks/Ecoli_Analysis_Notebooks/csd_nonident_diagnosis.csv
        columns: cause, effect, nonident_cause
    notebooks/Ecoli_Analysis_Notebooks/csd_nonident_summary.json
        keys: cause_counts, total_unidentifiable
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from csd_rescue_worker import CAUSE_CATEGORIES, classify_nonident_cause  # noqa: E402

NB_DIR = REPO / "notebooks" / "Ecoli_Analysis_Notebooks"
GRAPHML = NB_DIR / "ecoli_full_network_no_small_rna.graphml"
CSV_PATH = NB_DIR / "csd_results.csv"
OUT_CSV = NB_DIR / "csd_nonident_diagnosis.csv"
OUT_JSON = NB_DIR / "csd_nonident_summary.json"


def main() -> None:
    assert CSV_PATH.exists(), f"PRE: csd_results.csv must exist at {CSV_PATH}"
    assert GRAPHML.exists(), f"PRE: GraphML must exist at {GRAPHML}"

    df = pd.read_csv(CSV_PATH)
    unident = df[df.status == "unidentifiable"][["cause", "effect"]].copy()
    print(f"Loading graph from {GRAPHML} ...")
    g = nx.read_graphml(str(GRAPHML))
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print(f"Classifying {len(unident):,} unidentifiable edges ...")

    cause_labels: list[str] = []
    for _, row in unident.iterrows():
        if g.has_edge(row.cause, row.effect):
            cause_labels.append(classify_nonident_cause(g, row.cause, row.effect))
        else:
            cause_labels.append("unknown")

    unident = unident.copy()
    unident["nonident_cause"] = cause_labels
    cause_counts = unident["nonident_cause"].value_counts()
    print(cause_counts.to_string())

    assert all(
        c in list(CAUSE_CATEGORIES) + ["unknown"] for c in cause_counts.index
    ), "POST: all causes must be valid category labels"

    unident.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(unident):,} rows to {OUT_CSV}")

    summary = {
        "total_unidentifiable": int(len(unident)),
        "cause_counts": {k: int(v) for k, v in cause_counts.items()},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {OUT_JSON}")


if __name__ == "__main__":
    main()
