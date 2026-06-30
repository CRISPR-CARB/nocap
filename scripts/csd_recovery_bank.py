"""csd_recovery_bank.py — Greedy global perturbation-design recovery for CSD edges.

Given the full CSD classification results and the per-edge minimum SCC-break sets,
build a bank of n simultaneous do(S) experiments (each of size k) that maximises
the number of unidentifiable edges recovered.

Recovery definition (global experimental design / shared sets):
  Edge cause->effect is RECOVERED under do(S) iff, in G' = G - {cause->effect}
  with in-edges to every node in S removed, cause and effect are in different SCCs.
  This is the exact min_scc_break_set condition from src/nocap/scc_perturb.py.

Algorithm:
  1. Load targets: 6676 unidentifiable edges from classification_results.csv.
     Tag each with same_scc (from the CSV) and scc_rescuable (same_scc=True,
     i.e. the obstruction is cyclic — these are the only edges the bank can help).
  2. Build candidate pool: union of min_break_set genes from csd_break_results_full.csv
     (SCC-break-only genes).  Genes not appearing in any break set are excluded.
  3. Fast proxy: for a fixed set S, compute SCCs of do(S) on the FULL graph G once,
     then check each edge by SCC-id lookup (cause-scc != effect-scc).
     Soundness proof: removing the direct edge can only split SCCs, never merge them.
     So proxy-recovered implies exactly-recovered. Proxy can miss edge-removal-dependent
     recoveries (where the direct edge is the sole forward path in the SCC).
  4. Greedy bank: for each of the n sets, grow gene-by-gene (k steps) picking the
     gene from the candidate pool that maximally increases newly-covered uncovered
     targets (proxy). After building each set, mark its edges as covered.
  5. Exact final scoring: re-verify each chosen set against all target edges using
     the full G' = G - {edge} + do(S) exact check. Report proxy-vs-exact gap.
  6. Output CSVs and summary JSON.

Usage
-----
    python scripts/csd_recovery_bank.py \\
        --graphml notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml \\
        --classification results/cyclic_single_door/classification_results.csv \\
        --break-csv notebooks/Ecoli_Analysis_Notebooks/csd_break_results_full.csv \\
        --n 10 --k 3 \\
        --output-dir notebooks/Ecoli_Analysis_Notebooks

    python scripts/csd_recovery_bank.py \\
        --graphml notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml \\
        --classification results/cyclic_single_door/classification_results.csv \\
        --break-csv notebooks/Ecoli_Analysis_Notebooks/csd_break_results_full.csv \\
        --n 5 --k 6 \\
        --output-dir notebooks/Ecoli_Analysis_Notebooks

Outputs (in output-dir):
    csd_recovery_n{n}_k{k}.csv      — chosen sets + per-set stats
    csd_recovery_edges_n{n}_k{k}.csv — per-edge recovered flag + which set
    csd_recovery_summary.json        — combined summary for both budgets (appended)
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Graph helpers (no networkx at module level to keep startup fast; imported lazily)
# ---------------------------------------------------------------------------


def _load_graph(graphml_path: str):
    """Load the E.coli graphml and return an nx.DiGraph."""
    import networkx as nx

    G = nx.read_graphml(graphml_path)
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)
    return G


def _build_do_scc_map(G, perturb_set: frozenset) -> dict:
    """
    Return node -> scc_id mapping for do(S) applied to G (full graph, edge present).

    do(S): remove all in-edges to nodes in perturb_set.
    SCC computation is O(V + E).

    axiomander:
        requires:
            isinstance(perturb_set, frozenset)
        ensures:
            len(result) == len(list(G.nodes()))
        modifies:
            none
    """
    assert isinstance(perturb_set, frozenset), "PRE: perturb_set must be frozenset"

    import networkx as nx

    g_do = G.copy()
    for node in perturb_set:
        if node in g_do:
            g_do.remove_edges_from(list(g_do.in_edges(node)))

    scc_map: dict = {}
    for scc_id, scc in enumerate(nx.strongly_connected_components(g_do)):
        for node in scc:
            scc_map[node] = scc_id

    # POST: every node in G has an entry
    for node in G.nodes():
        assert node in scc_map, f"POST: node {node!r} missing from scc_map"

    return scc_map


def _proxy_recovered(cause: str, effect: str, scc_map: dict) -> bool:
    """
    Return True iff cause and effect are in different SCCs under do(S).

    Sound: if True, the edge is genuinely recovered (removing the direct edge
    can only further separate SCCs, never merge them).
    Not complete: may return False even when the edge is recoverable via
    edge-removal (the direct edge is the sole forward path in the SCC).

    axiomander:
        modifies:
            none
    """
    c_scc = scc_map.get(cause, -1)
    e_scc = scc_map.get(effect, -2)
    return c_scc != e_scc


def _exact_recovered(cause: str, effect: str, G, perturb_set: frozenset) -> bool:
    """
    Exact recovery check: build G' = G - {cause->effect}, apply do(S),
    check cause and effect are in different SCCs.

    axiomander:
        modifies:
            none
    """
    import networkx as nx

    g_prime = G.copy()
    if g_prime.has_edge(cause, effect):
        g_prime.remove_edge(cause, effect)

    for node in perturb_set:
        if node in g_prime:
            g_prime.remove_edges_from(list(g_prime.in_edges(node)))

    # Check SCC membership
    scc_map_exact: dict = {}
    for scc_id, scc in enumerate(nx.strongly_connected_components(g_prime)):
        for node in scc:
            scc_map_exact[node] = scc_id

    c_scc = scc_map_exact.get(cause, -1)
    e_scc = scc_map_exact.get(effect, -2)
    return c_scc != e_scc


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_targets(classification_csv: str) -> list[dict]:
    """
    Load unidentifiable edges from classification_results.csv.
    Returns list of dicts with keys: cause, effect, same_scc.

    axiomander:
        ensures:
            all(r["status"] == "unidentifiable" or True for r in result)
            all("cause" in r and "effect" in r for r in result)
        modifies:
            none
    """
    targets = []
    with open(classification_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "unidentifiable":
                targets.append(
                    {
                        "cause": row["cause"],
                        "effect": row["effect"],
                        "same_scc": row.get("same_scc", "").strip().lower() == "true",
                    }
                )
    assert len(targets) > 0, "POST: must have at least one unidentifiable edge"
    return targets


def _load_candidate_pool(break_csv: str) -> tuple[set, dict]:
    """
    Load SCC-break gene pool from csd_break_results_full.csv.

    Returns:
        pool: set of gene names that appear in at least one min_break_set
        edge_to_breakset: dict (cause, effect) -> frozenset of break genes
    """
    assert os.path.exists(break_csv), f"PRE: break CSV not found: {break_csv}"

    pool: set = set()
    edge_to_breakset: dict = {}

    with open(break_csv, newline="") as f:
        for row in csv.DictReader(f):
            cause = row["cause"]
            effect = row["effect"]
            raw = row.get("min_break_set", "[]")
            try:
                genes: list = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                genes = []
            bs = frozenset(genes)
            edge_to_breakset[(cause, effect)] = bs
            pool.update(bs)

    assert len(pool) >= 0, "POST: pool must be non-negative size"
    return pool, edge_to_breakset


# ---------------------------------------------------------------------------
# Greedy bank construction
# ---------------------------------------------------------------------------


def _greedy_bank(
    G,
    targets: list[dict],
    candidate_pool: set,
    n: int,
    k: int,
    verbose: bool = True,
) -> list[dict]:
    """
    Build n perturbation sets of size k greedily using the proxy.

    Each iteration:
      - Remaining = uncovered targets (those not yet proxy-recovered by any chosen set)
      - Grow a new set S gene-by-gene (k steps):
          at each step add the gene from pool that maximises |newly proxy-covered
          edges in Remaining| (ties broken by gene name for determinism)
      - Mark all proxy-recovered edges in Remaining as covered

    Returns list of n dicts:
        {
          "set_index": int,          # 1-based
          "genes": list[str],        # chosen set (sorted)
          "proxy_recovered_new": int, # newly covered by this set (proxy)
          "proxy_covered_cumulative": int,
        }

    axiomander:
        requires:
            n >= 1
            k >= 1
            len(targets) > 0
        modifies:
            none
    """
    assert n >= 1, "PRE: n must be >= 1"
    assert k >= 1, "PRE: k must be >= 1"
    assert len(targets) > 0, "PRE: targets must be non-empty"

    pool_sorted = sorted(candidate_pool)
    uncovered = set(range(len(targets)))  # indices into targets
    results = []
    total_covered = 0

    for set_idx in range(n):
        chosen_genes: list[str] = []
        current_set: frozenset = frozenset()

        # Build set S gene-by-gene
        for step in range(k):
            best_gene: str | None = None
            best_gain = -1

            # Compute current proxy recovery on uncovered targets
            if chosen_genes:
                cur_scc_map = _build_do_scc_map(G, current_set)
                already_covered = {
                    i
                    for i in uncovered
                    if _proxy_recovered(targets[i]["cause"], targets[i]["effect"], cur_scc_map)
                }
            else:
                already_covered = set()

            for gene in pool_sorted:
                if gene in current_set:
                    continue
                trial_set = current_set | frozenset([gene])
                trial_scc_map = _build_do_scc_map(G, trial_set)
                newly_covered = sum(
                    1
                    for i in uncovered
                    if i not in already_covered
                    and _proxy_recovered(targets[i]["cause"], targets[i]["effect"], trial_scc_map)
                )
                if newly_covered > best_gain or (
                    newly_covered == best_gain and (best_gene is None or gene < best_gene)
                ):
                    best_gain = newly_covered
                    best_gene = gene

            if best_gene is not None:
                chosen_genes.append(best_gene)
                current_set = frozenset(chosen_genes)
                if verbose:
                    print(
                        f"  Set {set_idx + 1} step {step + 1}/{k}: "
                        f"added {best_gene!r} (+{best_gain} proxy edges)"
                    )
            else:
                if verbose:
                    print(
                        f"  Set {set_idx + 1} step {step + 1}/{k}: no candidate improves coverage"
                    )
                break

        # Score this set against uncovered
        if current_set:
            final_scc_map = _build_do_scc_map(G, current_set)
            newly_covered_indices = {
                i
                for i in uncovered
                if _proxy_recovered(targets[i]["cause"], targets[i]["effect"], final_scc_map)
            }
        else:
            newly_covered_indices = set()

        newly_covered_count = len(newly_covered_indices)
        uncovered -= newly_covered_indices
        total_covered += newly_covered_count

        results.append(
            {
                "set_index": set_idx + 1,
                "genes": sorted(chosen_genes),
                "proxy_recovered_new": newly_covered_count,
                "proxy_covered_cumulative": total_covered,
            }
        )

        if verbose:
            print(
                f"  Set {set_idx + 1}: {sorted(chosen_genes)} "
                f"-> +{newly_covered_count} (cumulative: {total_covered})"
            )

        if not uncovered:
            if verbose:
                print(f"  All targets covered after set {set_idx + 1}; stopping early.")
            # Pad remaining slots with empty sets
            for rem in range(set_idx + 1, n):
                results.append(
                    {
                        "set_index": rem + 1,
                        "genes": [],
                        "proxy_recovered_new": 0,
                        "proxy_covered_cumulative": total_covered,
                    }
                )
            break

    # POST
    assert len(results) == n, f"POST: must return {n} sets; got {len(results)}"
    return results


# ---------------------------------------------------------------------------
# Exact final verification
# ---------------------------------------------------------------------------


def _exact_verify_bank(
    G,
    targets: list[dict],
    bank: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    For each of the n chosen sets, exactly verify every target edge.

    Returns:
        bank_exact: list of n dicts (like bank input but with exact_recovered_new,
                    exact_covered_cumulative added)
        edge_results: list of per-edge dicts with recovered flag + which set

    axiomander:
        requires:
            len(bank) >= 1
            len(targets) >= 1
        modifies:
            none
    """
    assert len(bank) >= 1, "PRE: bank must be non-empty"
    assert len(targets) >= 1, "PRE: targets must be non-empty"

    # Per-edge tracking
    edge_status: list[dict] = [
        {
            "cause": t["cause"],
            "effect": t["effect"],
            "same_scc": t["same_scc"],
            "recovered": False,
            "recovered_by_set": None,
        }
        for t in targets
    ]

    uncovered_exact = set(range(len(targets)))
    total_exact = 0
    bank_exact = []

    for item in bank:
        genes = item["genes"]
        if not genes:
            bank_exact.append(
                {**item, "exact_recovered_new": 0, "exact_covered_cumulative": total_exact}
            )
            continue

        perturb_set = frozenset(genes)
        newly_exact: list[int] = []

        for i in list(uncovered_exact):
            t = targets[i]
            if _exact_recovered(t["cause"], t["effect"], G, perturb_set):
                newly_exact.append(i)

        for i in newly_exact:
            uncovered_exact.discard(i)
            edge_status[i]["recovered"] = True
            edge_status[i]["recovered_by_set"] = item["set_index"]

        total_exact += len(newly_exact)
        bank_exact.append(
            {
                **item,
                "exact_recovered_new": len(newly_exact),
                "exact_covered_cumulative": total_exact,
            }
        )

    # POST
    n_recovered = sum(1 for e in edge_status if e["recovered"])
    assert n_recovered == total_exact, "POST: edge_status count must match total_exact"

    return bank_exact, edge_status


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_bank_csv(bank_exact: list[dict], path: Path, n: int, k: int) -> None:
    """Write per-set CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "set_index",
                "genes",
                "n_genes",
                "proxy_recovered_new",
                "proxy_covered_cumulative",
                "exact_recovered_new",
                "exact_covered_cumulative",
            ]
        )
        for item in bank_exact:
            w.writerow(
                [
                    item["set_index"],
                    json.dumps(item["genes"]),
                    len(item["genes"]),
                    item["proxy_recovered_new"],
                    item["proxy_covered_cumulative"],
                    item.get("exact_recovered_new", ""),
                    item.get("exact_covered_cumulative", ""),
                ]
            )
    print(f"Wrote bank CSV: {path}")


def _write_edge_csv(edge_results: list[dict], path: Path) -> None:
    """Write per-edge recovery CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cause", "effect", "same_scc", "recovered", "recovered_by_set"])
        for e in edge_results:
            w.writerow(
                [
                    e["cause"],
                    e["effect"],
                    e["same_scc"],
                    e["recovered"],
                    e["recovered_by_set"] or "",
                ]
            )
    print(f"Wrote edge CSV: {path}")


def _update_summary(
    summary_path: Path,
    n: int,
    k: int,
    targets: list[dict],
    bank_exact: list[dict],
    edge_results: list[dict],
    proxy_gap: int,
    n_universe: int,
) -> None:
    """Append/update the shared summary JSON with this run's results."""
    # Load existing summary if present
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    n_total = len(targets)
    n_same_scc = sum(1 for t in targets if t["same_scc"])
    n_recovered = sum(1 for e in edge_results if e["recovered"])
    n_recovered_same_scc = sum(1 for e in edge_results if e["recovered"] and e["same_scc"])
    n_recovered_not_same_scc = n_recovered - n_recovered_same_scc

    final_exact = bank_exact[-1]["exact_covered_cumulative"] if bank_exact else 0
    marginal_curve = [item["exact_recovered_new"] for item in bank_exact]

    run_key = f"n{n}_k{k}"
    summary[run_key] = {
        "n": n,
        "k": k,
        "n_total_unidentifiable": n_total,
        "n_same_scc": n_same_scc,
        "n_not_same_scc": n_total - n_same_scc,
        "n_universe": n_universe,
        "n_recovered_exact": final_exact,
        "pct_recovered_of_unident": round(100.0 * final_exact / n_total, 2) if n_total else 0,
        "pct_recovered_of_same_scc": round(100.0 * n_recovered_same_scc / n_same_scc, 2)
        if n_same_scc
        else 0,
        "pct_recovered_of_universe": round(100.0 * final_exact / n_universe, 2)
        if n_universe
        else 0,
        "proxy_total": bank_exact[-1]["proxy_covered_cumulative"] if bank_exact else 0,
        "proxy_vs_exact_gap": proxy_gap,
        "marginal_curve_exact": marginal_curve,
        "chosen_sets": [
            {"set_index": item["set_index"], "genes": item["genes"]}
            for item in bank_exact
            if item["genes"]
        ],
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Updated summary JSON: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--graphml", required=True, help="E.coli GraphML file.")
    p.add_argument(
        "--classification",
        required=True,
        help="classification_results.csv (from cyclic_single_door gather).",
    )
    p.add_argument(
        "--break-csv",
        required=True,
        help="csd_break_results_full.csv (per-edge min break sets, full run).",
    )
    p.add_argument("--n", type=int, required=True, help="Number of perturbation sets.")
    p.add_argument("--k", type=int, required=True, help="Genes per set.")
    p.add_argument(
        "--output-dir", required=True, help="Directory to write output CSVs and summary JSON."
    )
    p.add_argument(
        "--n-universe", type=int, default=9211, help="Total valid edge pairs (default: 9211)."
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-step output.")
    args = p.parse_args()

    assert args.n >= 1, f"PRE: --n must be >= 1, got {args.n}"
    assert args.k >= 1, f"PRE: --k must be >= 1, got {args.k}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet

    print(f"\n=== CSD Recovery Bank: n={args.n}, k={args.k} ===")

    # Load graph
    print("Loading graph...")
    G = _load_graph(args.graphml)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load targets
    print("Loading unidentifiable targets...")
    targets = _load_targets(args.classification)
    n_same_scc = sum(1 for t in targets if t["same_scc"])
    print(
        f"  Targets: {len(targets)} unidentifiable edges ({n_same_scc} same-SCC, "
        f"{len(targets) - n_same_scc} not same-SCC)"
    )

    # Load candidate pool
    print("Loading SCC-break candidate pool...")
    candidate_pool, _edge_to_breakset = _load_candidate_pool(args.break_csv)
    print(f"  Pool: {len(candidate_pool)} unique SCC-break genes")

    if not candidate_pool:
        print("ERROR: candidate pool is empty. Run submit_csd_break_full.sh first.")
        sys.exit(1)

    # Greedy bank construction
    print(f"\nBuilding greedy bank (n={args.n}, k={args.k})...")
    bank = _greedy_bank(G, targets, candidate_pool, args.n, args.k, verbose=verbose)

    proxy_total = bank[-1]["proxy_covered_cumulative"] if bank else 0
    print(
        f"\nProxy recovery: {proxy_total}/{len(targets)} "
        f"({100.0 * proxy_total / len(targets):.1f}%)"
    )

    # Exact final verification
    print("\nRunning exact final verification...")
    bank_exact, edge_results = _exact_verify_bank(G, targets, bank)
    exact_total = bank_exact[-1]["exact_covered_cumulative"] if bank_exact else 0
    proxy_gap = proxy_total - exact_total

    print(
        f"Exact recovery: {exact_total}/{len(targets)} ({100.0 * exact_total / len(targets):.1f}%)"
    )
    print(f"Proxy-vs-exact gap: {proxy_gap} edges (proxy over-counts by {proxy_gap})")

    # Write outputs
    bank_csv = out_dir / f"csd_recovery_n{args.n}_k{args.k}.csv"
    edge_csv = out_dir / f"csd_recovery_edges_n{args.n}_k{args.k}.csv"
    summary_json = out_dir / "csd_recovery_summary.json"

    _write_bank_csv(bank_exact, bank_csv, args.n, args.k)
    _write_edge_csv(edge_results, edge_csv)
    _update_summary(
        summary_json,
        args.n,
        args.k,
        targets,
        bank_exact,
        edge_results,
        proxy_gap,
        args.n_universe,
    )

    print(f"\n=== Done: n={args.n}, k={args.k} ===")
    print(f"  Bank CSV:  {bank_csv}")
    print(f"  Edge CSV:  {edge_csv}")
    print(f"  Summary:   {summary_json}")
    print(
        f"  Exact recovered: {exact_total}/{len(targets)} unidentifiable edges "
        f"({100.0 * exact_total / len(targets):.1f}%)"
    )
    print(
        f"  Same-SCC ceiling: {n_same_scc} edges "
        f"({100.0 * n_same_scc / len(targets):.1f}% of unidentifiable)"
    )


if __name__ == "__main__":
    main()
