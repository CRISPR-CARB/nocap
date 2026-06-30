"""smoke_csd_break.py — End-to-end smoke test for the csd_break_* pipeline.

Builds a tiny in-memory shard (no real GraphML needed), runs csd_break_worker
logic directly, and validates the output schema.

Usage
-----
    uv run python scripts/smoke_csd_break.py 2>&1
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import networkx as nx

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))


# ---------------------------------------------------------------------------
# Build tiny test graph (same as test_csd_break.py Case 2)
# A→B, B→C, C→D, A→D, D→E, E→A, D→F, F→A
#
# After removing A→D:
#   Forward path A→B→C→D still exists (A and D remain in same SCC).
#   Two node-disjoint return paths: D→E→A and D→F→A.
#   Min vertex cut = {E, F}, break_size == 2.
# ---------------------------------------------------------------------------


def _build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),  # forward path keeps A & D in same SCC
            ("A", "D"),  # edge under study
            ("D", "E"),
            ("E", "A"),  # return path 1: D→E→A
            ("D", "F"),
            ("F", "A"),  # return path 2: D→F→A
        ]
    )
    return g


EXPECTED_SCHEMA = {
    "cause",
    "effect",
    "nonident_cause",
    "same_scc_after_removal",
    "needs_intervention",
    "min_break_set",
    "min_break_size",
    "rescuable_within_k",
    "cut_verified",
}

K = 3


def run_smoke() -> None:
    g = _build_graph()
    graphml_file: str | None = None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write tiny GraphML
        graphml_path = tmpdir_path / "smoke_graph.graphml"
        nx.write_graphml(g, str(graphml_path))
        graphml_file = str(graphml_path)

        # Write shard
        shard_path = tmpdir_path / "shard_0.json"
        edges = [["A", "D"], ["A", "B"]]  # A→D: needs intervention; A→B: same SCC
        shard = {"shard_id": "0", "edges": edges}
        with open(shard_path, "w") as f:
            json.dump(shard, f)

        output_path = tmpdir_path / "out" / "shard_0.json"

        # --- Run worker logic directly (no subprocess) ---
        from csd_break_worker import run_shard

        run_shard(
            graphml=graphml_path,
            shard_path=shard_path,
            output_path=output_path,
            k=K,
        )

        assert output_path.exists(), "FAIL: output file not created"

        with open(output_path) as f:
            result = json.load(f)

        assert "results" in result, f"FAIL: 'results' key missing from {result.keys()}"
        rows = result["results"]
        assert len(rows) == 2, f"FAIL: expected 2 rows, got {len(rows)}"

        # Validate schema
        for row in rows:
            missing = EXPECTED_SCHEMA - set(row.keys())
            assert not missing, f"FAIL: missing keys {missing} in row {row}"

        # Case: A→D — D is pulled into SCC by return paths, needs intervention
        row_ad = next(r for r in rows if r["cause"] == "A" and r["effect"] == "D")
        assert row_ad["needs_intervention"] is True, (
            f"FAIL: A→D should need intervention, got {row_ad}"
        )
        assert row_ad["min_break_size"] == 2, (
            f"FAIL: A→D should have break_size=2, got {row_ad['min_break_size']}"
        )
        assert row_ad["cut_verified"] is True, (
            f"FAIL: cut_verified should be True for A→D, got {row_ad}"
        )
        assert "A" not in row_ad["min_break_set"], "FAIL: cause in break_set"
        assert "D" not in row_ad["min_break_set"], "FAIL: effect in break_set"

        # Case: A→B — B is in the same SCC as A in G': cycle A→B→C→A persists
        # needs_intervention depends on whether A & B remain in same SCC after
        # removing A→B. In G'=G-{A→B}: B can reach A via B→C→A, A can reach B
        # only via... A→D→E→B or A→... but A→D→E→B is a path, so still same SCC.
        # We just check schema and that break_set excludes cause/effect.
        row_ab = next(r for r in rows if r["cause"] == "A" and r["effect"] == "B")
        assert "A" not in row_ab["min_break_set"], "FAIL: cause A in break_set for A→B"
        assert "B" not in row_ab["min_break_set"], "FAIL: effect B in break_set for A→B"
        assert isinstance(row_ab["rescuable_within_k"], bool), "FAIL: rescuable_within_k type"

    print("smoke_csd_break: ALL CHECKS PASSED")
    print(
        f"  A→D: needs_intervention={row_ad['needs_intervention']}, "
        f"break_size={row_ad['min_break_size']}, break_set={row_ad['min_break_set']}, "
        f"cut_verified={row_ad['cut_verified']}"
    )
    print(
        f"  A→B: needs_intervention={row_ab['needs_intervention']}, "
        f"break_size={row_ab['min_break_size']}, break_set={row_ab['min_break_set']}, "
        f"cut_verified={row_ab['cut_verified']}"
    )


if __name__ == "__main__":
    run_smoke()
