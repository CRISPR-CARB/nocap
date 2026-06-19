"""
perturbation_optimizer.py
=========================
Greedy submodular optimizer for perturbation panel design.

Given a boolean coverage matrix M[candidate_tf][query] (from build_coverage_matrix.py),
finds the minimal / most informative set of background perturbations to maximize
causal identifiability.

Two algorithms:
  1. greedy_max_coverage(M, k)   -- best k TFs for maximum query coverage
                                    (budgeted max-coverage, (1-1/e) guarantee)
  2. greedy_min_set_cover(M)     -- fewest TFs to cover all resolvable queries
                                    (greedy set cover, ln(n) approximation)

Usage (standalone):
  uv run python scripts/perturbation_optimizer.py \
    --matrix notebooks/Ecoli_Analysis_Notebooks/coverage_matrix.csv \
    --budgets 2,5,10,15,20,25 \
    --output-dir notebooks/Ecoli_Analysis_Notebooks/

Outputs:
  nomination_k{k}.csv          -- ordered nomination list for each budget k
  nomination_min_cover.csv     -- greedy min-set-cover result
  marginal_gain_curve.csv      -- cumulative coverage vs. # perturbations (up to max k)
"""

import argparse
import csv
import os

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_coverage_matrix(path: str) -> tuple[list[str], list[str], dict[str, list[bool]]]:
    """
    Load coverage_matrix.csv.

    Returns:
        candidates  -- list of candidate TF names (rows)
        queries     -- list of query labels (columns, e.g. "tf1->outcome")
        matrix      -- dict: candidate -> list[bool] aligned with queries
    """
    candidates = []
    queries = []
    matrix: dict[str, list[bool]] = {}

    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        queries = header[1:]  # skip "candidate_tf" column
        for row in reader:
            cand = row[0]
            vals = [bool(int(v)) for v in row[1:]]
            candidates.append(cand)
            matrix[cand] = vals

    return candidates, queries, matrix


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------


def greedy_max_coverage(
    candidates: list[str],
    queries: list[str],
    matrix: dict[str, list[bool]],
    budget_k: int,
    intervenable: set[str] | None = None,
) -> list[tuple[str, int, int]]:
    """
    Greedy budgeted maximum coverage.

    At each step, pick the candidate TF that resolves the most
    still-unresolved queries. Repeat up to budget_k times.

    Args:
        candidates:   all candidate TF names
        queries:      query labels
        matrix:       coverage matrix
        budget_k:     maximum number of TFs to select
        intervenable: optional subset of candidates to restrict to

    Returns:
        List of (selected_tf, marginal_gain, cumulative_coverage) tuples,
        one per selection step.
    """
    pool = [c for c in candidates if intervenable is None or c in intervenable]
    n_queries = len(queries)
    unresolved: set[int] = set(range(n_queries))
    selected = []

    for step in range(budget_k):
        if not unresolved:
            break

        best_tf = None
        best_gain = -1
        for cand in pool:
            if cand in [s for s, _, _ in selected]:
                continue
            gain = sum(1 for qi in unresolved if matrix[cand][qi])
            if gain > best_gain:
                best_gain = gain
                best_tf = cand

        if best_tf is None or best_gain == 0:
            break

        # update unresolved
        newly_resolved = {qi for qi in unresolved if matrix[best_tf][qi]}
        unresolved -= newly_resolved
        cumulative = n_queries - len(unresolved)
        selected.append((best_tf, best_gain, cumulative))

    return selected


def greedy_min_set_cover(
    candidates: list[str],
    queries: list[str],
    matrix: dict[str, list[bool]],
    intervenable: set[str] | None = None,
) -> list[tuple[str, int, int]]:
    """
    Greedy minimum set cover.

    Repeatedly pick the TF that covers the most uncovered queries until
    all resolvable queries are covered (or no further progress is possible).

    Returns:
        List of (selected_tf, marginal_gain, cumulative_coverage) tuples.
    """
    pool = [c for c in candidates if intervenable is None or c in intervenable]
    n_queries = len(queries)

    # Only try to cover queries that are resolvable by at least one TF
    resolvable: set[int] = set()
    for cand in pool:
        for qi, val in enumerate(matrix[cand]):
            if val:
                resolvable.add(qi)

    unresolved = set(resolvable)
    selected = []

    while unresolved:
        best_tf = None
        best_gain = -1
        for cand in pool:
            if cand in [s for s, _, _ in selected]:
                continue
            gain = sum(1 for qi in unresolved if matrix[cand][qi])
            if gain > best_gain:
                best_gain = gain
                best_tf = cand

        if best_tf is None or best_gain == 0:
            break

        newly_resolved = {qi for qi in unresolved if matrix[best_tf][qi]}
        unresolved -= newly_resolved
        cumulative = len(resolvable) - len(unresolved)
        selected.append((best_tf, best_gain, cumulative))

    return selected


def build_marginal_gain_curve(
    candidates: list[str],
    queries: list[str],
    matrix: dict[str, list[bool]],
    max_k: int,
    intervenable: set[str] | None = None,
) -> list[tuple[int, int, float]]:
    """
    Run greedy_max_coverage up to max_k and return the cumulative coverage
    curve for plotting.

    Returns:
        List of (k, cumulative_queries_resolved, fraction_resolved) tuples.
    """
    results = greedy_max_coverage(candidates, queries, matrix, max_k, intervenable)
    n_queries = len(queries)
    curve = [(0, 0, 0.0)]
    for step_idx, (tf, gain, cumulative) in enumerate(results):
        curve.append((step_idx + 1, cumulative, cumulative / n_queries))
    return curve


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_nomination_csv(path: str, selected: list[tuple[str, int, int]], n_queries: int):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "candidate_tf",
                "marginal_gain",
                "cumulative_coverage",
                "pct_of_unidentifiable",
            ]
        )
        for rank, (tf, gain, cumulative) in enumerate(selected, start=1):
            pct = cumulative / n_queries * 100
            writer.writerow([rank, tf, gain, cumulative, f"{pct:.1f}"])


def write_curve_csv(path: str, curve: list[tuple[int, int, float]], n_queries: int):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "queries_resolved", "fraction_resolved", "pct_resolved"])
        for k, resolved, frac in curve:
            writer.writerow([k, resolved, f"{frac:.4f}", f"{frac * 100:.1f}"])


# ---------------------------------------------------------------------------
# Cycle-breaking heuristics
# ---------------------------------------------------------------------------


def cycle_breaking_score(node: str, graph) -> int:
    """
    Count the number of simple cycles in *graph* that contain *node*.

    A "hard" intervention do(node) removes all incoming edges to *node*,
    which breaks every cycle that passes through it.  Nodes with higher
    cycle-membership scores are therefore higher-priority candidates for
    background perturbations because they are more likely to resolve
    unidentifiable queries caused by cyclic confounding.

    For large graphs (>~500 nodes) this can be slow; use
    rank_candidates_by_cycle_score which falls back to SCC-size weighting
    automatically.

    Args:
        node:  node name (must be present in graph)
        graph: networkx DiGraph

    Returns:
        Number of simple cycles containing *node* (0 if node not in any cycle).
    """
    import networkx as nx

    count = 0
    for cycle in nx.simple_cycles(graph):
        if node in cycle:
            count += 1
    return count


def rank_candidates_by_cycle_score(
    candidates: list[str],
    graph,
    use_scc_fallback: bool = True,
    scc_fallback_threshold: int = 500,
) -> list[str]:
    """
    Return *candidates* sorted descending by cycle-breaking score.

    For graphs with more than *scc_fallback_threshold* nodes, computing all
    simple cycles is expensive.  The fallback uses the size of each node's
    strongly-connected component (SCC) as a proxy: nodes in larger SCCs
    participate in more cycles and are better cycle-breakers.

    Args:
        candidates:              list of candidate TF names to rank
        graph:                   networkx DiGraph
        use_scc_fallback:        if True, use SCC-size proxy for large graphs
        scc_fallback_threshold:  node count above which fallback is used

    Returns:
        Candidates sorted by cycle-breaking score (descending).
        Ties are broken alphabetically for determinism.
    """
    import networkx as nx

    if not candidates:
        return []

    n_nodes = graph.number_of_nodes()

    if use_scc_fallback and n_nodes > scc_fallback_threshold:
        # SCC-size proxy: size of the SCC containing each node minus 1
        # (subtract 1 so singleton SCCs — not in any cycle — score 0)
        scc_map: dict[str, int] = {}
        for scc in nx.strongly_connected_components(graph):
            scc_size = len(scc)
            for node in scc:
                scc_map[node] = scc_size - 1  # 0 for singletons

        scores = {c: scc_map.get(c, 0) for c in candidates}
    else:
        scores = {c: cycle_breaking_score(c, graph) for c in candidates}

    return sorted(candidates, key=lambda c: (-scores[c], c))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Greedy perturbation panel optimizer")
    parser.add_argument("--matrix", required=True, help="Path to coverage_matrix.csv")
    parser.add_argument(
        "--budgets", default="2,5,10,15,20,25", help="Comma-separated list of k budgets"
    )
    parser.add_argument("--output-dir", default=".", help="Directory for output CSVs")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    max_k = max(budgets)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading coverage matrix from: {args.matrix}")
    candidates, queries, matrix = load_coverage_matrix(args.matrix)
    n_queries = len(queries)
    n_candidates = len(candidates)
    print(f"  Candidates: {n_candidates}, Queries: {n_queries}")

    # Count resolvable queries
    resolvable = set()
    for cand in candidates:
        for qi, val in enumerate(matrix[cand]):
            if val:
                resolvable.add(qi)
    print(f"  Resolvable queries (by >= 1 TF): {len(resolvable)} / {n_queries}")

    # --- Budgeted max-coverage for each k ---
    print(f"\nRunning greedy max-coverage for k in {budgets}...")
    # Run once up to max_k, then slice for each budget
    full_greedy = greedy_max_coverage(candidates, queries, matrix, max_k)

    for k in budgets:
        selected_k = full_greedy[:k]
        out_path = os.path.join(args.output_dir, f"nomination_k{k:02d}.csv")
        write_nomination_csv(out_path, selected_k, n_queries)
        if selected_k:
            cov = selected_k[-1][2]
            pct = cov / n_queries * 100
            print(
                f"  k={k:2d}: {cov}/{n_queries} queries resolved ({pct:.1f}%) "
                f"| top TF: {selected_k[0][0]} (gain={selected_k[0][1]})"
            )
        else:
            print(f"  k={k:2d}: no TFs selected (no resolvable queries?)")
        print(f"         -> {out_path}")

    # --- Marginal gain curve ---
    print(f"\nBuilding marginal gain curve (up to k={max_k})...")
    curve = build_marginal_gain_curve(candidates, queries, matrix, max_k)
    curve_path = os.path.join(args.output_dir, "marginal_gain_curve.csv")
    write_curve_csv(curve_path, curve, n_queries)
    print(f"  Curve written to: {curve_path}")

    # --- Min set cover ---
    print("\nRunning greedy min-set-cover...")
    min_cover = greedy_min_set_cover(candidates, queries, matrix)
    min_cover_path = os.path.join(args.output_dir, "nomination_min_cover.csv")
    write_nomination_csv(min_cover_path, min_cover, n_queries)
    if min_cover:
        final_cov = min_cover[-1][2]
        print(
            f"  Min cover: {len(min_cover)} TFs cover {final_cov}/{len(resolvable)} "
            f"resolvable queries ({final_cov / len(resolvable) * 100:.1f}%)"
        )
    print(f"  -> {min_cover_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
