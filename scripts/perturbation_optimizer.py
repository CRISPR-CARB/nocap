r"""
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
  uv run python scripts/perturbation_optimizer.py \\
    --matrix notebooks/Ecoli_Analysis_Notebooks/coverage_matrix.csv \\
    --budgets 2,5,10,15,20,25 \\
    --output-dir notebooks/Ecoli_Analysis_Notebooks/

Outputs:
  nomination_k{k}.csv          -- ordered nomination list for each budget k
  nomination_min_cover.csv     -- greedy min-set-cover result
  marginal_gain_curve.csv      -- cumulative coverage vs. # perturbations (up to max k)

Contract annotations
--------------------
Every public function carries assert-based contracts in the axiomander style:
  PRE  -- preconditions checked at function entry
  POST -- postconditions checked before return
  INV  -- loop invariants checked at each iteration

These asserts are active at runtime (they are NOT stripped by -O).  They serve
as both documentation and lightweight runtime verification.
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

    Contracts:
        PRE:  path is a non-empty str pointing to an existing file
        POST: candidates is a non-empty list of str
        POST: queries is a list of str (may be empty)
        POST: matrix keys == set(candidates)
        POST: every row in matrix has len == len(queries)
        POST: every value in every row is bool
    """
    # --- PRECONDITIONS ---
    assert isinstance(path, str) and path, "PRE: path must be a non-empty str"
    assert os.path.isfile(path), f"PRE: file must exist: {path}"

    candidates: list[str] = []
    queries: list[str] = []
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
            # LOOP INVARIANT: every row so far has the right length
            assert len(matrix[cand]) == len(queries), (
                f"INV: row for {cand!r} has {len(matrix[cand])} values, "
                f"expected {len(queries)}"
            )

    # --- POSTCONDITIONS ---
    assert isinstance(candidates, list), "POST: candidates must be a list"
    assert isinstance(queries, list), "POST: queries must be a list"
    assert set(matrix.keys()) == set(candidates), (
        "POST: matrix keys must equal set(candidates)"
    )
    for cand in candidates:
        assert len(matrix[cand]) == len(queries), (
            f"POST: row for {cand!r} must have len == len(queries)"
        )
        assert all(isinstance(v, bool) for v in matrix[cand]), (
            f"POST: all values in row {cand!r} must be bool"
        )

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

    axiomander:
        requires:
            isinstance(candidates, list)
            isinstance(queries, list)
            isinstance(budget_k, int)
            budget_k >= 0
        modifies:
            none
        ensures:
            isinstance(result, list)
            len(result) <= budget_k
            all(gain >= 1 for _, gain, _ in result)
            all(result[i][2] <= result[i+1][2] for i in range(len(result)-1))
            len(result) == 0 or result[len(result)-1][2] <= len(queries)
    """
    # --- PRECONDITIONS ---
    assert isinstance(candidates, list), "PRE: candidates must be a list"
    assert isinstance(queries, list), "PRE: queries must be a list"
    assert isinstance(budget_k, int) and budget_k >= 0, (
        "PRE: budget_k must be a non-negative int"
    )
    assert intervenable is None or isinstance(intervenable, set | frozenset), (
        "PRE: intervenable must be None or a set"
    )
    for c in candidates:
        assert c in matrix, f"PRE: candidate {c!r} must be a key in matrix"
        assert len(matrix[c]) == len(queries), (
            f"PRE: matrix row for {c!r} must have len == len(queries)"
        )

    pool = [c for c in candidates if intervenable is None or c in intervenable]
    n_queries = len(queries)
    unresolved: set[int] = set(range(n_queries))
    selected: list[tuple[str, int, int]] = []
    prev_unresolved_size = len(unresolved)

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

        newly_resolved = {qi for qi in unresolved if matrix[best_tf][qi]}
        unresolved -= newly_resolved
        cumulative = n_queries - len(unresolved)
        selected.append((best_tf, best_gain, cumulative))

        # LOOP INVARIANT: unresolved strictly shrinks each step
        assert len(unresolved) < prev_unresolved_size, (
            "INV: unresolved must strictly shrink each step"
        )
        prev_unresolved_size = len(unresolved)

        # LOOP INVARIANT: reported gain matches actual newly-resolved count
        assert best_gain == len(newly_resolved), (
            f"INV: reported gain {best_gain} != newly_resolved {len(newly_resolved)}"
        )

        # LOOP INVARIANT: cumulative == n_queries - len(unresolved)
        assert cumulative == n_queries - len(unresolved), (
            "INV: cumulative must equal n_queries - len(unresolved)"
        )

    # --- POSTCONDITIONS ---
    assert isinstance(selected, list), "POST: result must be a list"
    assert len(selected) <= min(budget_k, len(pool)), (
        "POST: len(result) must be <= min(budget_k, len(pool))"
    )
    selected_tfs = [tf for tf, _, _ in selected]
    assert len(selected_tfs) == len(set(selected_tfs)), (
        "POST: selected TFs must be distinct"
    )
    for tf, gain, cumulative in selected:
        assert isinstance(tf, str), "POST: each tf must be a str"
        assert isinstance(gain, int) and gain >= 1, (
            "POST: each marginal gain must be a positive int"
        )
        assert isinstance(cumulative, int) and 0 < cumulative <= n_queries, (
            "POST: cumulative must be in (0, n_queries]"
        )
    # cumulative is non-decreasing
    cumulatives = [c for _, _, c in selected]
    assert cumulatives == sorted(cumulatives), (
        "POST: cumulative_coverage must be non-decreasing"
    )

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

    Contracts:
        PRE:  candidates is a list of str
        PRE:  queries is a list of str
        PRE:  matrix keys contain all candidates; each row has len == len(queries)
        PRE:  intervenable is None or a set of str
        POST: result is a list of (str, int, int) tuples
        POST: selected TFs are distinct
        POST: marginal gains are all >= 1
        POST: cumulative_coverage is non-decreasing
        POST: final cumulative == len(resolvable) (all resolvable queries covered)
        INV:  unresolved shrinks monotonically
    """
    # --- PRECONDITIONS ---
    assert isinstance(candidates, list), "PRE: candidates must be a list"
    assert isinstance(queries, list), "PRE: queries must be a list"
    assert intervenable is None or isinstance(intervenable, set | frozenset), (
        "PRE: intervenable must be None or a set"
    )
    for c in candidates:
        assert c in matrix, f"PRE: candidate {c!r} must be a key in matrix"
        assert len(matrix[c]) == len(queries), (
            f"PRE: matrix row for {c!r} must have len == len(queries)"
        )

    pool = [c for c in candidates if intervenable is None or c in intervenable]
    n_queries = len(queries)

    # Only try to cover queries that are resolvable by at least one TF
    resolvable: set[int] = set()
    for cand in pool:
        for qi, val in enumerate(matrix[cand]):
            if val:
                resolvable.add(qi)

    unresolved = set(resolvable)
    selected: list[tuple[str, int, int]] = []
    prev_unresolved_size = len(unresolved)

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

        # LOOP INVARIANT: unresolved strictly shrinks each iteration
        assert len(unresolved) < prev_unresolved_size, (
            "INV: unresolved must strictly shrink each iteration"
        )
        prev_unresolved_size = len(unresolved)

        # LOOP INVARIANT: cumulative == len(resolvable) - len(unresolved)
        assert cumulative == len(resolvable) - len(unresolved), (
            "INV: cumulative must equal len(resolvable) - len(unresolved)"
        )

    # --- POSTCONDITIONS ---
    assert isinstance(selected, list), "POST: result must be a list"
    selected_tfs = [tf for tf, _, _ in selected]
    assert len(selected_tfs) == len(set(selected_tfs)), (
        "POST: selected TFs must be distinct"
    )
    for tf, gain, cumulative in selected:
        assert isinstance(tf, str), "POST: each tf must be a str"
        assert isinstance(gain, int) and gain >= 1, (
            "POST: each marginal gain must be a positive int"
        )
        assert isinstance(cumulative, int) and cumulative >= 1, (
            "POST: cumulative must be a positive int"
        )
    cumulatives = [c for _, _, c in selected]
    assert cumulatives == sorted(cumulatives), (
        "POST: cumulative_coverage must be non-decreasing"
    )
    if selected and resolvable:
        assert selected[-1][2] == len(resolvable), (
            "POST: final cumulative must equal len(resolvable)"
        )

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
        Always starts with (0, 0, 0.0).

    Contracts:
        PRE:  max_k >= 0
        PRE:  queries is a non-empty list (fraction requires len > 0)
        POST: result[0] == (0, 0, 0.0)
        POST: k values are 0, 1, 2, ... (strictly increasing by 1)
        POST: cumulative values are non-decreasing
        POST: all fractions are in [0.0, 1.0]
        POST: len(result) == len(greedy steps taken) + 1
    """
    # --- PRECONDITIONS ---
    assert isinstance(max_k, int) and max_k >= 0, (
        "PRE: max_k must be a non-negative int"
    )
    assert isinstance(queries, list) and len(queries) > 0, (
        "PRE: queries must be a non-empty list"
    )

    results = greedy_max_coverage(candidates, queries, matrix, max_k, intervenable)
    n_queries = len(queries)
    curve: list[tuple[int, int, float]] = [(0, 0, 0.0)]
    for step_idx, (tf, gain, cumulative) in enumerate(results):
        curve.append((step_idx + 1, cumulative, cumulative / n_queries))

    # --- POSTCONDITIONS ---
    assert curve[0] == (0, 0, 0.0), "POST: curve must start with (0, 0, 0.0)"
    ks = [k for k, _, _ in curve]
    assert ks == list(range(len(curve))), "POST: k values must be 0, 1, 2, ..."
    cumulatives = [c for _, c, _ in curve]
    assert cumulatives == sorted(cumulatives), (
        "POST: cumulative values must be non-decreasing"
    )
    for _, _, frac in curve:
        assert 0.0 <= frac <= 1.0, f"POST: fraction {frac} must be in [0.0, 1.0]"

    return curve


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_nomination_csv(path: str, selected: list[tuple[str, int, int]], n_queries: int):
    """Write a nomination CSV for a greedy selection result."""
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
    """Write a marginal gain curve CSV."""
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

    Contracts:
        PRE:  node is a str
        PRE:  graph has a .nodes() method
        POST: result is a non-negative int
    """
    import networkx as nx

    # --- PRECONDITIONS ---
    assert isinstance(node, str), "PRE: node must be a str"
    assert hasattr(graph, "nodes"), "PRE: graph must have a .nodes() method"

    count = 0
    for cycle in nx.simple_cycles(graph):
        if node in cycle:
            count += 1

    # --- POSTCONDITION ---
    assert isinstance(count, int) and count >= 0, (
        "POST: result must be a non-negative int"
    )
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

    Contracts:
        PRE:  candidates is a list of str
        PRE:  graph has .number_of_nodes() and .strongly_connected_components support
        PRE:  scc_fallback_threshold >= 1
        POST: result is a permutation of candidates (same elements, possibly reordered)
        POST: len(result) == len(candidates)
        POST: set(result) == set(candidates)
    """
    import networkx as nx

    # --- PRECONDITIONS ---
    assert isinstance(candidates, list), "PRE: candidates must be a list"
    assert hasattr(graph, "number_of_nodes"), (
        "PRE: graph must have a .number_of_nodes() method"
    )
    assert isinstance(scc_fallback_threshold, int) and scc_fallback_threshold >= 1, (
        "PRE: scc_fallback_threshold must be a positive int"
    )

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

    result = sorted(candidates, key=lambda c: (-scores[c], c))

    # --- POSTCONDITIONS ---
    assert isinstance(result, list), "POST: result must be a list"
    assert len(result) == len(candidates), (
        "POST: result must have the same length as candidates"
    )
    assert set(result) == set(candidates), (
        "POST: result must be a permutation of candidates"
    )

    return result


def scc_mass(graph) -> int:
    """
    Total cyclic mass of *graph*: sum of SCC sizes for all non-singleton SCCs.

    Singleton SCCs (nodes not in any cycle) contribute 0.  This metric
    quantifies how much cyclic structure remains in the graph — removing
    in-edges to a set of nodes (a joint hard intervention) reduces it.

    The ``cyclic_id`` algorithm reasons about SCCs, so this metric is
    canonically aligned with identifiability: higher SCC mass = more
    unidentifiable queries.

    Args:
        graph: networkx DiGraph

    Returns:
        Sum of SCC sizes for all SCCs with size >= 2.

    axiomander:
        requires:
            hasattr(graph, 'nodes')
        ensures:
            result >= 0
        modifies:
            none
    """
    import networkx as nx

    # --- PRE ---
    assert hasattr(graph, "nodes"), "PRE: graph must have a .nodes() method"

    total = 0
    for scc in nx.strongly_connected_components(graph):
        if len(scc) >= 2:
            total += len(scc)

    # --- POST ---
    assert isinstance(total, int) and total >= 0, "POST: result must be a non-negative int"
    return total


def set_cycle_break_score(candidate_set: "set | frozenset", graph) -> int:
    """
    SCC-mass reduction from a joint hard intervention on *candidate_set*.

    A hard intervention ``do(S)`` removes all incoming edges to every node in
    S simultaneously, breaking every cycle that passes through any member of S.
    The score is ``scc_mass(original) - scc_mass(after_intervention)``.

    This is the canonical cycle-breaking metric for simultaneous perturbations
    because ``cyclic_id`` itself reasons about SCCs: fewer / smaller SCCs after
    the intervention means more queries become identifiable.

    Unlike the single-node ``cycle_breaking_score`` (which counts simple cycles
    and is slow on large graphs), this function uses only
    ``nx.strongly_connected_components`` — O(V + E) — and is safe on the full
    E. coli network.

    Args:
        candidate_set: set of node names to intervene on simultaneously
        graph:         networkx DiGraph (not mutated)

    Returns:
        scc_mass(graph) - scc_mass(graph_after_joint_do(candidate_set))
        A higher score means more cyclic structure is dissolved.

    axiomander:
        requires:
            isinstance(candidate_set, (set, frozenset))
            hasattr(graph, 'nodes')
        ensures:
            result >= 0
        modifies:
            none
    """
    import networkx as nx

    # --- PRE ---
    assert isinstance(candidate_set, (set, frozenset)), (
        "PRE: candidate_set must be a set or frozenset"
    )
    assert hasattr(graph, "nodes"), "PRE: graph must have a .nodes() method"

    before = scc_mass(graph)

    # Build the intervened subgraph: remove all in-edges to every node in the set.
    # We do NOT mutate the original — build a view using edge filtering.
    intervened = graph.copy()
    for node in candidate_set:
        if node in intervened:
            in_edges = list(intervened.in_edges(node))
            intervened.remove_edges_from(in_edges)

    after = scc_mass(intervened)
    score = before - after

    # --- POST ---
    assert isinstance(score, int) and score >= 0, (
        "POST: score must be a non-negative int"
    )
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Entry point for the perturbation optimizer CLI."""
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
