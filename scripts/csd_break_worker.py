"""csd_break_worker.py — Compute minimum SCC-break intervention sets for unidentifiable edges.

For each unidentifiable edge cause→effect in a shard this worker:
  1. Classifies the *cause* of non-identifiability (reuses classify_nonident_cause).
  2. Calls ``min_scc_break_set`` to find the minimum vertex set B such that
     do(B) on G' = G − {cause→effect} separates cause and effect into
     different SCCs, making the edge single-door identifiable.
  3. Reports whether the edge is rescuable within budget k (default k=3).

If cause and effect already fall into different SCCs of G' (i.e. removing the
direct edge already breaks the cycle), ``needs_intervention=False`` and no
intervention is needed — the O-adjustment set already identifies the edge.

Resumable: completed rows are written to <output>.partial (one JSON per line).
On re-run any edges already present in .partial are skipped; on completion
.partial is atomically moved to the final output path.

Usage
-----
    python scripts/csd_break_worker.py \\
        --graphml notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml \\
        --shard results/cyclic_single_door/break_shards/shard_0.json \\
        --output results/cyclic_single_door/break_classified/shard_0.json \\
        --k 3

Output schema per row
---------------------
    {
        "cause":                 str,
        "effect":                str,
        "nonident_cause":        one of CAUSE_CATEGORIES,
        "same_scc_after_removal": bool,   # same SCC in G' = G − {cause→effect}
        "needs_intervention":    bool,    # False → O-adjustment suffices
        "min_break_set":         list[str],
        "min_break_size":        int,
        "rescuable_within_k":    bool,    # (not needs_intervention) or (break_size <= k)
        "cut_verified":          bool,    # verification that break_set actually works
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import networkx as nx

# ---------------------------------------------------------------------------
# Add project src/ to path so nocap can be imported without installing
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "src"))

from nocap.scc_perturb import min_scc_break_set  # noqa: E402

# Reuse the non-identifiability cause classifier from the rescue pipeline
sys.path.insert(0, str(_REPO / "scripts"))
from csd_rescue_worker import classify_nonident_cause  # noqa: E402

# ---------------------------------------------------------------------------
# Shard runner
# ---------------------------------------------------------------------------


def _load_checkpoint(partial_path: Path) -> set[tuple[str, str]]:
    """Load already-computed (cause, effect) pairs from a .partial file."""
    done: set[tuple[str, str]] = set()
    if not partial_path.exists():
        return done
    with open(partial_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add((obj["cause"], obj["effect"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def run_shard(graphml: Path, shard_path: Path, output_path: Path, k: int) -> None:
    """Process one break shard: diagnose cause + find min SCC-break set for each edge."""
    # Load graph
    g = nx.read_graphml(str(graphml))

    # Load shard edges
    with open(shard_path) as f:
        shard = json.load(f)
    edges: list[tuple[str, str]] = [tuple(e) for e in shard["edges"]]  # type: ignore[misc]
    shard_id = shard.get("shard_id", "?")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(".json.partial")

    # Skip if already done
    if output_path.exists():
        print(f"break worker: shard {shard_id} already complete, skipping.", file=sys.stderr)
        return

    # Load checkpoint
    done = _load_checkpoint(partial_path)
    remaining = [(c, e) for c, e in edges if (c, e) not in done]

    print(
        f"break worker: shard {shard_id}: {len(edges)} edges, "
        f"{len(done)} already done, {len(remaining)} remaining.",
        file=sys.stderr,
    )

    # --- Process remaining edges ---
    with open(partial_path, "a") as pf:
        for cause, effect in remaining:
            nonident_cause = classify_nonident_cause(g, cause, effect)
            info = min_scc_break_set(cause, effect, g)

            rescuable = (not info["needs_intervention"]) or (
                0 < info["break_size"] <= k
            )

            row = {
                "cause": cause,
                "effect": effect,
                "nonident_cause": nonident_cause,
                "same_scc_after_removal": info["same_scc_after_removal"],
                "needs_intervention": info["needs_intervention"],
                "min_break_set": info["break_set"],
                "min_break_size": info["break_size"],
                "rescuable_within_k": rescuable,
                "cut_verified": info["cut_verified"],
            }
            pf.write(json.dumps(row) + "\n")
            pf.flush()

    # Reload all rows (checkpoint + newly computed) in shard order
    done_rows: dict[tuple[str, str], dict] = {}
    with open(partial_path) as pf:
        for line in pf:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done_rows[(obj["cause"], obj["effect"])] = obj
            except (json.JSONDecodeError, KeyError):
                pass

    # Write final output in original shard edge order
    results = [done_rows[e] for e in edges if e in done_rows]
    final = {"shard_id": shard_id, "k": k, "results": results}
    tmp = output_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(final, f)
    os.replace(tmp, output_path)

    # Clean up partial
    if partial_path.exists():
        partial_path.unlink()

    print(
        f"break worker: shard {shard_id} done. {len(results)} rows written → {output_path}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--graphml", required=True, help="Path to the GraphML network file.")
    p.add_argument("--shard", required=True, help="Path to input break shard JSON.")
    p.add_argument("--output", required=True, help="Path to write classified break shard JSON.")
    p.add_argument("--k", type=int, default=3, help="Rescue budget (default 3).")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_shard(
        graphml=Path(args.graphml),
        shard_path=Path(args.shard),
        output_path=Path(args.output),
        k=args.k,
    )
