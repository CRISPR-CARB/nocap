"""csd_rescue_worker.py — Diagnose non-identifiable edges and compute single-node rescue interventions.

For each unidentifiable edge cause→effect in a shard this worker:
  1. Classifies the *cause* of non-identifiability (cause taxonomy).
  2. Enumerates single-node do(v) interventions from the SCC min-cut pool and
     records which ones flip the edge to identifiable.

Resumable: completed rows are written to <output>.partial (one JSON per line).
On re-run any edges already present in .partial are skipped; on completion
.partial is atomically moved to the final output path.

Usage
-----
    python scripts/csd_rescue_worker.py \\
        --graphml notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml \\
        --shard results/cyclic_single_door/rescue_shards/shard_0.json \\
        --output results/cyclic_single_door/rescue_classified/shard_0.json

Output schema per row
---------------------
    {
        "cause":            str,
        "effect":           str,
        "nonident_cause":   one of CAUSE_CATEGORIES,
        "rescue_nodes":     list[str],   # do(v) nodes that flip this edge to identifiable
        "n_rescue_nodes":   int,
    }

Non-identifiability cause categories
-------------------------------------
    "self_loop"           -- cause == effect
    "two_cycle"           -- reverse edge effect->cause also present
    "same_scc_long"       -- same SCC after removing cause->effect (long feedback)
    "scc_edge_dissolved"  -- removing cause->effect breaks the SCC (edge is the link)
    "cross_scc_blocked"   -- same_scc=False but still unidentifiable (O-set blocked)
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

from nocap.cyclic_single_door import (
    classify_edge,
    nx_digraph_to_y0,
)
from nocap.scc_perturb import (
    build_intervened_graph,
    compute_min_cut_b,
    find_in_scc_children,
)

try:
    from y0.algorithm.separation.sigma_extension import sigma_extension
except ImportError as exc:
    sys.exit(f"csd_rescue_worker: missing y0 dependency: {exc}")

# ---------------------------------------------------------------------------
# Cause taxonomy
# ---------------------------------------------------------------------------

CAUSE_CATEGORIES = (
    "self_loop",
    "two_cycle",
    "same_scc_long",
    "scc_edge_dissolved",
    "cross_scc_blocked",
)


def classify_nonident_cause(graph: nx.DiGraph, cause: str, effect: str) -> str:
    """Return the primary structural reason why cause->effect is unidentifiable.

    Categories are mutually exclusive and exhaustive (for unidentifiable edges):

    PRE: graph.has_edge(cause, effect)
    PRE: cause and effect are strings
    """
    assert graph.has_edge(cause, effect), f"PRE: edge {cause!r}->{effect!r} must exist"
    assert isinstance(cause, str) and isinstance(effect, str), "PRE: string nodes required"

    # 1. Self-loop
    if cause == effect:
        return "self_loop"

    # 2. 2-cycle: reverse edge exists
    if graph.has_edge(effect, cause):
        return "two_cycle"

    # 3. Check same-SCC-related categories (cause != effect, no 2-cycle)
    # Is cause reachable from effect (and vice versa) in the full graph?
    sccs = {frozenset(s) for s in nx.strongly_connected_components(graph)}
    in_same_scc = any(cause in s and effect in s for s in sccs)

    if in_same_scc:
        # Remove the edge and check whether they're still in the same SCC
        g_tmp = graph.copy()
        g_tmp.remove_edge(cause, effect)
        still_same_scc = nx.has_path(g_tmp, cause, effect) and nx.has_path(g_tmp, effect, cause)
        if still_same_scc:
            return "same_scc_long"
        else:
            return "scc_edge_dissolved"

    # 4. Cross-SCC but blocked (same_scc=False yet unidentifiable)
    return "cross_scc_blocked"


# ---------------------------------------------------------------------------
# Candidate intervention pool
# ---------------------------------------------------------------------------


def _candidate_pool(graph: nx.DiGraph) -> set:
    """Nodes in any SCC's min-cut B(t). Same logic as cyclic_single_door._candidate_pool."""
    candidates: set = set()
    for scc in nx.strongly_connected_components(graph):
        if len(scc) <= 1:
            continue
        scc_frozen = frozenset(scc)
        for tf in scc:
            in_scc_ch = find_in_scc_children(tf, scc_frozen, graph)
            if not in_scc_ch:
                continue
            cut = compute_min_cut_b(tf, scc_frozen, in_scc_ch, graph)
            candidates.update(cut)
    return candidates


# ---------------------------------------------------------------------------
# Per-edge rescue computation
# ---------------------------------------------------------------------------


def compute_rescue_nodes(
    graph: nx.DiGraph,
    cause: str,
    effect: str,
    candidates: set[str] | None = None,
) -> list[str]:
    """Return the list of single-node do(v) interventions that make cause->effect identifiable.

    Tries each node in the SCC min-cut pool (or *candidates* if supplied).
    do(v) = remove in-edges to v (preserves v and its out-edges).

    PRE: graph.has_edge(cause, effect)
    POST: result is a list of strings
    """
    assert graph.has_edge(cause, effect), f"PRE: edge {cause!r}->{effect!r} must exist"

    if candidates is None:
        candidates = _candidate_pool(graph)

    rescue: list[str] = []
    for node in sorted(candidates):  # sorted for determinism
        g_int = build_intervened_graph(graph, [node])
        if not g_int.has_edge(cause, effect):
            # Intervention removed the target edge entirely — not a valid rescue
            continue
        row = classify_edge(g_int, cause, effect)
        if row["status"] == "identifiable":
            rescue.append(node)
    return rescue


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


def run_shard(graphml: Path, shard_path: Path, output_path: Path) -> None:
    """Process one rescue shard: diagnose cause + find rescue nodes for each edge."""
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
        print(f"rescue worker: shard {shard_id} already complete, skipping.", file=sys.stderr)
        return

    # Load checkpoint
    done = _load_checkpoint(partial_path)
    remaining = [(c, e) for c, e in edges if (c, e) not in done]

    print(
        f"rescue worker: shard {shard_id}: {len(edges)} edges, "
        f"{len(done)} already done, {len(remaining)} remaining.",
        file=sys.stderr,
    )

    # Pre-compute candidate pool once
    candidates = _candidate_pool(g)

    # Pre-build sigma extension once for classify_edge
    g_y0 = nx_digraph_to_y0(g)
    g_sigma = sigma_extension(g_y0)

    # --- Process remaining edges ---
    with open(partial_path, "a") as pf:
        for cause, effect in remaining:
            nonident_cause = classify_nonident_cause(g, cause, effect)

            # Only try rescue for edges that are genuinely in the candidate pool's
            # scope (i.e. same-SCC or cross-SCC-blocked).  Self-loops are never
            # rescuable by do() on other nodes; 2-cycles require removing the
            # back-edge specifically.
            if nonident_cause in ("self_loop",):
                rescue_nodes: list[str] = []
            elif nonident_cause == "two_cycle":
                # The only effective rescue is do(effect) which removes effect->cause
                # OR do(cause) which removes cause->effect — check both
                rescue_nodes = compute_rescue_nodes(g, cause, effect, candidates | {cause, effect})
            else:
                rescue_nodes = compute_rescue_nodes(g, cause, effect, candidates)

            row = {
                "cause": cause,
                "effect": effect,
                "nonident_cause": nonident_cause,
                "rescue_nodes": rescue_nodes,
                "n_rescue_nodes": len(rescue_nodes),
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
    final = {"shard_id": shard_id, "results": results}
    tmp = output_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(final, f)
    os.replace(tmp, output_path)

    # Clean up partial
    if partial_path.exists():
        partial_path.unlink()

    print(
        f"rescue worker: shard {shard_id} done. {len(results)} rows written → {output_path}",
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
    p.add_argument("--shard", required=True, help="Path to input rescue shard JSON.")
    p.add_argument("--output", required=True, help="Path to write classified rescue shard JSON.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_shard(
        graphml=Path(args.graphml),
        shard_path=Path(args.shard),
        output_path=Path(args.output),
    )
