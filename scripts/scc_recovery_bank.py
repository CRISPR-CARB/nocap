"""scc_recovery_bank.py — Greedy global perturbation-design recovery for causal TF queries.

For every transcription factor (TF) in the E. coli network — defined as any node
with at least one direct child (out-degree >= 1) — we study the causal query

    P(children(t) | do(t))

A TF's causal query is **identifiable** iff the TF is in a trivial (singleton) SCC
of the graph.  TFs that sit inside a non-trivial SCC have unidentifiable causal
queries and are the recovery targets.

Recovery under do(S):
    TF t is RECOVERED by an experiment do(S) iff, after removing all in-edges
    to every gene in S, the TF t lands in a singleton SCC (is no longer cyclic).
    Justification: §5.3 of the SCC-perturbation analysis — identifiability is
    achieved when and only when do(B) extracts the TF from all cycles.

Algorithm:
  1. Load graph; find every node with out-degree >= 1 (TFs).
  2. Split into already-identifiable (singleton SCC) and unidentifiable (non-trivial SCC).
  3. Build candidate pool: union of min_break_set genes for each unidentifiable TF,
     computed via compute_min_cut_b from src/nocap/scc_perturb.py.
  4. Greedy bank: for each of the n sets, grow gene-by-gene (k steps) picking the
     gene from the candidate pool that maximally recovers newly-uncovered TFs.
     One SCC computation per candidate per step.
  5. Output CSVs and summary JSON.

Usage
-----
    python scripts/scc_recovery_bank.py \\
        --graphml notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml \\
        --n 10 --k 3 \\
        --output-dir notebooks/Ecoli_Analysis_Notebooks

    python scripts/scc_recovery_bank.py \\
        --graphml notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml \\
        --n 5 --k 6 \\
        --output-dir notebooks/Ecoli_Analysis_Notebooks

Outputs (in output-dir):
    scc_recovery_n{n}_k{k}.csv           — chosen sets + per-set stats
    scc_recovery_tfs_n{n}_k{k}.csv       — per-TF recovered flag + which set
    scc_recovery_summary.json            — combined summary for both budgets (appended)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Graph helpers (networkx imported lazily for fast startup)
# ---------------------------------------------------------------------------


def _load_graph(graphml_path: str):
    """Load the E. coli graphml and return an nx.DiGraph."""
    import networkx as nx

    G = nx.read_graphml(graphml_path)
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)
    return G


def _build_do_scc_map(G, perturb_set: frozenset) -> dict:
    """
    Return node -> scc_id mapping for do(S) applied to G.

    do(S): remove all in-edges to nodes in perturb_set.

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

    for node in G.nodes():
        assert node in scc_map, f"POST: node {node!r} missing from scc_map"

    return scc_map


def _tf_recovered(tf: str, scc_map: dict) -> bool:
    """
    Return True iff tf is in a singleton SCC under do(S).

    A TF's causal query P(children | do(t)) is identifiable iff t is extracted
    from all cycles, i.e. t is a singleton SCC in the intervened graph.

    axiomander:
        modifies:
            none
    """
    # We need to know the SCC size for tf.  scc_map maps node -> scc_id.
    # We need the scc_sizes dict.  However this function receives only scc_map.
    # Use a separate helper that takes the full info.
    # (This function is a convenience wrapper; see _tf_recovered_from_scc_info.)
    return scc_map.get(tf) is not None  # always True if node exists; see below


def _is_singleton_scc(tf: str, scc_map: dict, scc_sizes: dict) -> bool:
    """
    Return True iff tf maps to a singleton SCC in the intervened graph.

    axiomander:
        modifies:
            none
    """
    scc_id = scc_map.get(tf, None)
    if scc_id is None:
        return True  # node absent from graph -> trivially isolated
    return scc_sizes.get(scc_id, 1) == 1


def _build_do_scc_info(G, perturb_set: frozenset) -> tuple[dict, dict]:
    """
    Build (scc_map, scc_sizes) for do(S) on G.

    Returns:
        scc_map  : node -> scc_id
        scc_sizes: scc_id -> size

    axiomander:
        requires:
            isinstance(perturb_set, frozenset)
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
    scc_sizes: dict = {}
    for scc_id, scc in enumerate(nx.strongly_connected_components(g_do)):
        scc_sizes[scc_id] = len(scc)
        for node in scc:
            scc_map[node] = scc_id

    return scc_map, scc_sizes


# ---------------------------------------------------------------------------
# TF enumeration and baseline classification
# ---------------------------------------------------------------------------


def _enumerate_tfs(G) -> list[str]:
    """
    Return sorted list of all nodes with out-degree >= 1 (i.e., TFs / regulators).

    axiomander:
        ensures:
            result == sorted(result)
            all(G.out_degree(t) >= 1 for t in result)
        modifies:
            none
    """
    tfs = sorted(n for n in G.nodes() if G.out_degree(n) >= 1)
    assert tfs == sorted(tfs), "POST: result must be sorted"
    return tfs


def _classify_tfs(G, tfs: list[str]) -> tuple[list[str], list[str]]:
    """
    Classify each TF as already identifiable (singleton SCC) or unidentifiable.

    Returns:
        identifiable    : list of TFs in singleton SCCs
        unidentifiable  : list of TFs in non-trivial SCCs

    axiomander:
        ensures:
            len(result[0]) + len(result[1]) == len(tfs)
        modifies:
            none
    """
    import networkx as nx

    # One SCC pass over the full graph
    scc_map: dict = {}
    scc_sizes_map: dict = {}
    for scc_id, scc in enumerate(nx.strongly_connected_components(G)):
        scc_sizes_map[scc_id] = len(scc)
        for node in scc:
            scc_map[node] = scc_id

    identifiable = []
    unidentifiable = []
    for t in tfs:
        scc_id = scc_map.get(t)
        if scc_id is None or scc_sizes_map.get(scc_id, 1) == 1:
            identifiable.append(t)
        else:
            unidentifiable.append(t)

    assert len(identifiable) + len(unidentifiable) == len(tfs), "POST: all TFs must be classified"
    return identifiable, unidentifiable


# ---------------------------------------------------------------------------
# Candidate pool construction
# ---------------------------------------------------------------------------


def _build_candidate_pool(G, unidentifiable_tfs: list[str]) -> set:
    """
    Compute the union of B(t) min-cut genes for all unidentifiable TFs.

    Uses compute_min_cut_b and find_in_scc_children from src/nocap/scc_perturb.py.
    TFs with empty B(t) (e.g. 2-cycle with no intermediate nodes) contribute
    nothing to the pool; those TFs cannot be recovered by this bank.

    axiomander:
        ensures:
            isinstance(result, set)
        modifies:
            none
    """
    import os
    import sys

    import networkx as nx

    # Ensure src/ is on the path
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from nocap.scc_perturb import compute_min_cut_b, find_in_scc_children

    # Compute SCC structure once
    scc_map: dict = {}
    scc_nodes_map: dict = {}
    for scc_id, scc in enumerate(nx.strongly_connected_components(G)):
        scc_nodes_map[scc_id] = frozenset(scc)
        for node in scc:
            scc_map[node] = scc_id

    pool: set = set()
    for tf in unidentifiable_tfs:
        scc_id = scc_map.get(tf)
        if scc_id is None:
            continue
        scc_nodes = scc_nodes_map[scc_id]
        in_scc_children = find_in_scc_children(tf, scc_nodes, G)
        if not in_scc_children:
            continue
        b_t = compute_min_cut_b(tf, scc_nodes, in_scc_children, G)
        pool.update(b_t)

    assert isinstance(pool, set), "POST: pool must be a set"
    return pool


# ---------------------------------------------------------------------------
# Greedy bank construction
# ---------------------------------------------------------------------------


def _greedy_bank(
    G,
    targets: list[str],
    candidate_pool: set,
    n: int,
    k: int,
    verbose: bool = True,
) -> list[dict]:
    """
    Build n perturbation sets of size k greedily using the TF-recovery proxy.

    Each iteration:
      - Remaining = uncovered TFs (not yet recovered by any chosen set)
      - Grow a new set S gene-by-gene (k steps):
          at each step add the gene from pool that maximises |newly recovered
          TFs in Remaining| (ties broken by gene name for determinism)
      - Mark all recovered TFs in Remaining as covered

    Returns list of n dicts:
        {
          "set_index": int,               # 1-based
          "genes": list[str],             # chosen set (sorted)
          "proxy_recovered_new": int,     # newly covered by this set
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
                cur_scc_map, cur_scc_sizes = _build_do_scc_info(G, current_set)
                already_covered = {
                    i
                    for i in uncovered
                    if _is_singleton_scc(targets[i], cur_scc_map, cur_scc_sizes)
                }
            else:
                already_covered = set()

            for gene in pool_sorted:
                if gene in current_set:
                    continue
                trial_set = current_set | frozenset([gene])
                trial_scc_map, trial_scc_sizes = _build_do_scc_info(G, trial_set)
                newly_covered = sum(
                    1
                    for i in uncovered
                    if i not in already_covered
                    and _is_singleton_scc(targets[i], trial_scc_map, trial_scc_sizes)
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
                        f"added {best_gene!r} (+{best_gain} TF queries recovered)"
                    )
            else:
                if verbose:
                    print(
                        f"  Set {set_idx + 1} step {step + 1}/{k}: no candidate improves coverage"
                    )
                break

        # Score this set against uncovered
        if current_set:
            final_scc_map, final_scc_sizes = _build_do_scc_info(G, current_set)
            newly_covered_indices = {
                i
                for i in uncovered
                if _is_singleton_scc(targets[i], final_scc_map, final_scc_sizes)
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
                f"-> +{newly_covered_count} TFs (cumulative: {total_covered})"
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
# Per-TF recovery status
# ---------------------------------------------------------------------------


def _score_per_tf(
    targets: list[str],
    bank: list[dict],
    G,
) -> list[dict]:
    """
    For each chosen set (in order), mark which previously-uncovered TFs it recovers.

    Returns list of per-TF dicts:
        {
          "tf": str,
          "recovered": bool,
          "recovered_by_set": int | None,
          "scc_size": int,
        }

    axiomander:
        requires:
            len(bank) >= 1
            len(targets) >= 1
        modifies:
            none
    """
    assert len(bank) >= 1, "PRE: bank must be non-empty"
    assert len(targets) >= 1, "PRE: targets must be non-empty"

    import networkx as nx

    # Pre-compute SCC sizes for the original graph (for reporting)
    orig_scc_map: dict = {}
    orig_scc_sizes: dict = {}
    for scc_id, scc in enumerate(nx.strongly_connected_components(G)):
        orig_scc_sizes[scc_id] = len(scc)
        for node in scc:
            orig_scc_map[node] = scc_id

    tf_status: list[dict] = [
        {
            "tf": t,
            "recovered": False,
            "recovered_by_set": None,
            "scc_size": orig_scc_sizes.get(orig_scc_map.get(t, -1), 1),
        }
        for t in targets
    ]

    uncovered_idx = set(range(len(targets)))

    for item in bank:
        genes = item["genes"]
        if not genes:
            continue

        perturb_set = frozenset(genes)
        scc_map, scc_sizes = _build_do_scc_info(G, perturb_set)

        newly_recovered = []
        for i in list(uncovered_idx):
            if _is_singleton_scc(targets[i], scc_map, scc_sizes):
                newly_recovered.append(i)

        for i in newly_recovered:
            uncovered_idx.discard(i)
            tf_status[i]["recovered"] = True
            tf_status[i]["recovered_by_set"] = item["set_index"]

    return tf_status


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_bank_csv(bank: list[dict], path: Path) -> None:
    """Write per-set CSV."""
    import csv as csv_mod

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(
            [
                "set_index",
                "genes",
                "n_genes",
                "proxy_recovered_new",
                "proxy_covered_cumulative",
            ]
        )
        for item in bank:
            w.writerow(
                [
                    item["set_index"],
                    json.dumps(item["genes"]),
                    len(item["genes"]),
                    item["proxy_recovered_new"],
                    item["proxy_covered_cumulative"],
                ]
            )
    print(f"Wrote bank CSV: {path}")


def _write_tf_csv(tf_results: list[dict], path: Path) -> None:
    """Write per-TF recovery CSV."""
    import csv as csv_mod

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(["tf", "scc_size", "recovered", "recovered_by_set"])
        for e in tf_results:
            w.writerow(
                [
                    e["tf"],
                    e["scc_size"],
                    e["recovered"],
                    e["recovered_by_set"] or "",
                ]
            )
    print(f"Wrote TF CSV: {path}")


def _update_summary(
    summary_path: Path,
    n: int,
    k: int,
    n_tfs: int,
    n_identifiable: int,
    n_unidentifiable: int,
    bank: list[dict],
    tf_results: list[dict],
) -> None:
    """Append/update the shared summary JSON with this run's results."""
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    n_recovered = sum(1 for e in tf_results if e["recovered"])
    marginal_curve = [item["proxy_recovered_new"] for item in bank]
    final_covered = bank[-1]["proxy_covered_cumulative"] if bank else 0

    run_key = f"n{n}_k{k}"
    summary[run_key] = {
        "n": n,
        "k": k,
        "n_total_tfs": n_tfs,
        "n_identifiable_baseline": n_identifiable,
        "n_unidentifiable_baseline": n_unidentifiable,
        "n_recovered": final_covered,
        "n_still_unrecovered": n_unidentifiable - final_covered,
        "pct_recovered_of_unidentifiable": (
            round(100.0 * final_covered / n_unidentifiable, 2) if n_unidentifiable else 0
        ),
        "pct_identifiable_after_recovery": (
            round(100.0 * (n_identifiable + final_covered) / n_tfs, 2) if n_tfs else 0
        ),
        "marginal_curve": marginal_curve,
        "chosen_sets": [
            {"set_index": item["set_index"], "genes": item["genes"]}
            for item in bank
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
    p.add_argument(
        "--graphml",
        default="notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml",
        help="E. coli GraphML file.",
    )
    p.add_argument("--n", type=int, required=True, help="Number of perturbation sets.")
    p.add_argument("--k", type=int, required=True, help="Genes per set.")
    p.add_argument(
        "--output-dir",
        default="notebooks/Ecoli_Analysis_Notebooks",
        help="Directory to write output CSVs and summary JSON.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-step output.")
    args = p.parse_args()

    assert args.n >= 1, f"PRE: --n must be >= 1, got {args.n}"
    assert args.k >= 1, f"PRE: --k must be >= 1, got {args.k}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet

    print(f"\n=== SCC TF Recovery Bank: n={args.n}, k={args.k} ===")

    # Load graph
    print("Loading graph...")
    G = _load_graph(args.graphml)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Enumerate TFs
    print("Enumerating TFs (out-degree >= 1)...")
    tfs = _enumerate_tfs(G)
    print(f"  TFs: {len(tfs)}")

    # Classify baseline
    print("Classifying baseline identifiability...")
    identifiable, unidentifiable = _classify_tfs(G, tfs)
    print(f"  Already identifiable (singleton SCC): {len(identifiable)}")
    print(f"  Unidentifiable (non-trivial SCC):     {len(unidentifiable)}")

    if not unidentifiable:
        print("All TF queries are already identifiable. Nothing to recover.")
        sys.exit(0)

    # Build candidate pool
    print("Building candidate pool (B(t) min-cut genes)...")
    candidate_pool = _build_candidate_pool(G, unidentifiable)
    print(f"  Pool: {len(candidate_pool)} unique genes")

    if not candidate_pool:
        print(
            "WARNING: candidate pool is empty (all unidentifiable TFs have "
            "structural 2-cycles with no intermediate nodes — cannot be broken "
            "under Interpretation A). Exiting."
        )
        sys.exit(1)

    # Greedy bank construction
    print(f"\nBuilding greedy bank (n={args.n}, k={args.k})...")
    bank = _greedy_bank(G, unidentifiable, candidate_pool, args.n, args.k, verbose=verbose)

    proxy_total = bank[-1]["proxy_covered_cumulative"] if bank else 0
    print(
        f"\nProxy recovery: {proxy_total}/{len(unidentifiable)} unidentifiable TFs "
        f"({100.0 * proxy_total / len(unidentifiable):.1f}%)"
    )

    # Per-TF recovery scoring
    print("Computing per-TF recovery status...")
    tf_results = _score_per_tf(unidentifiable, bank, G)

    # Write outputs
    bank_csv = out_dir / f"scc_recovery_n{args.n}_k{args.k}.csv"
    tf_csv = out_dir / f"scc_recovery_tfs_n{args.n}_k{args.k}.csv"
    summary_json = out_dir / "scc_recovery_summary.json"

    _write_bank_csv(bank, bank_csv)
    _write_tf_csv(tf_results, tf_csv)
    _update_summary(
        summary_json,
        args.n,
        args.k,
        len(tfs),
        len(identifiable),
        len(unidentifiable),
        bank,
        tf_results,
    )

    print(f"\n=== Done: n={args.n}, k={args.k} ===")
    print(f"  Total TFs:           {len(tfs)}")
    print(
        f"  Baseline identifiable:    {len(identifiable)} ({100.0 * len(identifiable) / len(tfs):.1f}%)"
    )
    print(
        f"  Baseline unidentifiable:  {len(unidentifiable)} ({100.0 * len(unidentifiable) / len(tfs):.1f}%)"
    )
    print(
        f"  Recovered by bank:        {proxy_total} ({100.0 * proxy_total / len(unidentifiable):.1f}% of unidentifiable)"
    )
    print(
        f"  Identifiable after recovery: {len(identifiable) + proxy_total} / {len(tfs)} ({100.0 * (len(identifiable) + proxy_total) / len(tfs):.1f}%)"
    )
    print(f"  Bank CSV: {bank_csv}")
    print(f"  TF CSV:   {tf_csv}")
    print(f"  Summary:  {summary_json}")


if __name__ == "__main__":
    main()
