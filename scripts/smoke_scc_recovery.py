"""smoke_scc_recovery.py — End-to-end smoke test for scc_recovery_bank.py.

Builds a small in-memory graph with known SCC structure, runs the full
bank pipeline (n=2, k=2), and asserts expected outputs.

Usage:
    uv run python scripts/smoke_scc_recovery.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import networkx as nx

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

import scc_recovery_bank as srb

# ---------------------------------------------------------------------------
# Build test graph
# ---------------------------------------------------------------------------
# Graph:
#   SCC1: A->B->C->A  (3 TFs unidentifiable; B is the min-cut for A and C)
#   DAG:  D->E        (D identifiable; E is a sink)

G = nx.DiGraph()
G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("D", "E")])

# ---------------------------------------------------------------------------
# Step 1: TF enumeration
# ---------------------------------------------------------------------------
tfs = srb._enumerate_tfs(G)
assert set(tfs) == {"A", "B", "C", "D"}, f"Unexpected TFs: {tfs}"
assert tfs == sorted(tfs), "TFs must be sorted"
print(f"PASS enumerate_tfs: {tfs}")

# ---------------------------------------------------------------------------
# Step 2: Baseline classification
# ---------------------------------------------------------------------------
identifiable, unidentifiable = srb._classify_tfs(G, tfs)
assert set(identifiable) == {"D"}, f"Expected D identifiable, got {identifiable}"
assert set(unidentifiable) == {"A", "B", "C"}, (
    f"Expected A,B,C unidentifiable, got {unidentifiable}"
)
assert len(identifiable) + len(unidentifiable) == len(tfs)
print(f"PASS classify_tfs: identifiable={identifiable}, unidentifiable={unidentifiable}")

# ---------------------------------------------------------------------------
# Step 3: Candidate pool
# ---------------------------------------------------------------------------
pool = srb._build_candidate_pool(G, unidentifiable)
assert isinstance(pool, set)
assert len(pool) >= 1, f"Pool should be non-empty, got {pool}"
print(f"PASS build_candidate_pool: pool={sorted(pool)}")

# ---------------------------------------------------------------------------
# Step 4: Greedy bank
# ---------------------------------------------------------------------------
bank = srb._greedy_bank(G, unidentifiable, pool, n=2, k=2, verbose=True)
assert len(bank) == 2, f"Expected 2 sets, got {len(bank)}"
assert bank[0]["set_index"] == 1
assert bank[1]["set_index"] == 2
# Coverage monotone
assert bank[1]["proxy_covered_cumulative"] >= bank[0]["proxy_covered_cumulative"]
# At least one TF recovered
assert bank[-1]["proxy_covered_cumulative"] >= 1, "At least one TF must be recovered"
# All chosen genes are in pool
for item in bank:
    for gene in item["genes"]:
        assert gene in pool, f"Chosen gene {gene!r} not in pool"
print(f"PASS _greedy_bank: {[(item['genes'], item['proxy_covered_cumulative']) for item in bank]}")

# ---------------------------------------------------------------------------
# Step 5: Per-TF scoring
# ---------------------------------------------------------------------------
tf_results = srb._score_per_tf(unidentifiable, bank, G)
assert len(tf_results) == len(unidentifiable)
n_recovered = sum(1 for e in tf_results if e["recovered"])
assert n_recovered == bank[-1]["proxy_covered_cumulative"], (
    f"Per-TF recovered count {n_recovered} != bank cumulative {bank[-1]['proxy_covered_cumulative']}"
)
print(f"PASS _score_per_tf: {n_recovered}/{len(unidentifiable)} recovered")

# ---------------------------------------------------------------------------
# Step 6: Output writers
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    out = Path(tmpdir)
    bank_csv = out / "scc_recovery_n2_k2.csv"
    tf_csv = out / "scc_recovery_tfs_n2_k2.csv"
    summary_json = out / "scc_recovery_summary.json"

    srb._write_bank_csv(bank, bank_csv)
    srb._write_tf_csv(tf_results, tf_csv)
    srb._update_summary(
        summary_json,
        n=2,
        k=2,
        n_tfs=len(tfs),
        n_identifiable=len(identifiable),
        n_unidentifiable=len(unidentifiable),
        bank=bank,
        tf_results=tf_results,
    )

    # Verify CSV contents
    bank_lines = bank_csv.read_text().splitlines()
    assert bank_lines[0].startswith("set_index"), "Bank CSV missing header"
    assert len(bank_lines) == 3, f"Expected 1 header + 2 data rows, got {len(bank_lines)}"

    tf_lines = tf_csv.read_text().splitlines()
    assert tf_lines[0].startswith("tf"), "TF CSV missing header"
    assert len(tf_lines) == len(unidentifiable) + 1

    with open(summary_json) as f:
        summary = json.load(f)
    assert "n2_k2" in summary
    assert summary["n2_k2"]["n_total_tfs"] == len(tfs)
    assert summary["n2_k2"]["n_identifiable_baseline"] == len(identifiable)
    assert summary["n2_k2"]["n_unidentifiable_baseline"] == len(unidentifiable)
    assert summary["n2_k2"]["n_recovered"] == bank[-1]["proxy_covered_cumulative"]

    print("PASS output writers: CSV and JSON correct")

# ---------------------------------------------------------------------------
# Step 7: Direct break verification
# ---------------------------------------------------------------------------
# Perturbing C (removes B->C) should break the cycle entirely
scc_map, scc_sizes = srb._build_do_scc_info(G, frozenset(["C"]))
assert srb._is_singleton_scc("A", scc_map, scc_sizes), "A should be singleton after do({C})"
assert srb._is_singleton_scc("B", scc_map, scc_sizes), "B should be singleton after do({C})"
assert srb._is_singleton_scc("C", scc_map, scc_sizes), "C should be singleton after do({C})"
print("PASS break verification: do({C}) recovers A, B, C")

print("\nALL SMOKE TESTS PASSED")
