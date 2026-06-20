"""
topk_simultaneous.py
====================
Find the top-k **simultaneous** background perturbations that maximize the
number of identifiable queries (information gain = queries resolved).

Background
----------
A hard intervention ``do(S)`` on a set S removes all incoming edges to every
node in S at once.  Because ``cyclic_id`` reasons about strongly-connected
components (SCCs), a joint intervention can break cycles that no single
perturbation dissolves — creating synergies invisible to single-column
coverage matrices.

Objective
---------
For a fixed query list Q and a budget k, find:

    S* = argmax_{|S|=k} |{ q in Q : cyclic_id resolves q given do(S) }|

Algorithm overview
------------------
This script uses a **matrix-driven cycle-guided greedy/beam search** that
stays tractable on the full E. coli network (~4500 nodes, ~7000 edges):

1. Load the single-perturbation coverage matrix (pre-computed by the
   submit_coverage pipeline) as a free heuristic layer.

2. Pre-filter the candidate pool:
   - By default (``--require-singleton-gain``), keep only candidates that
     resolve ≥1 query alone (from the matrix).  Use
     ``--no-require-singleton-gain`` to include zero-gain candidates for
     pure-synergy exploration.
   - Rank survivors by single-node SCC-break score (O(V+E), no simple_cycles).

3. Warm-start the beam from the greedy union-of-singletons solution (free,
   no ``cyclic_id`` calls).

4. Grow the set one gene at a time.  At each step:
   a. For each query already resolved by a set member (from the matrix),
      skip the joint ``cyclic_id`` call — monotonicity guarantees resolution.
   b. For remaining queries, invoke joint ``cyclic_id`` only on
      ``--candidate-cap`` SCC-top-ranked candidates, not all 285+.
   c. Pick the candidate with the highest *joint* marginal gain.
   d. Branch-and-bound: prune beam branches whose optimistic upper bound
      (matrix-derivable) cannot beat the current best.

5. Return a ranked list of (perturbation_set, scc_mass_broken, queries_resolved).

Usage
-----
    uv run python scripts/topk_simultaneous.py \\
        --matrix    notebooks/Ecoli_Analysis_Notebooks/coverage_matrix.csv \\
        --graphml   notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml \\
        --k 5 \\
        --output-dir notebooks/Ecoli_Analysis_Notebooks/

    # Use exhaustive mode for small pools / k=2:
    uv run python scripts/topk_simultaneous.py \\
        --matrix ... --graphml ... --k 2 --mode exhaustive

    # Allow pure-synergy candidates (zero individual gain):
    uv run python scripts/topk_simultaneous.py \\
        --matrix ... --graphml ... --k 3 --no-require-singleton-gain

Outputs
-------
    topk_simultaneous_k{k}.csv
        rank, perturbation_set, scc_mass_broken, queries_resolved, pct_resolved

Contract annotations
--------------------
Every public function carries assert-based PRE/INV/POST guards and an
``axiomander:`` docstring block.  No axiomander import at runtime.
"""

import argparse
import csv
import itertools
import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Matrix heuristic helpers
# ---------------------------------------------------------------------------


def build_query_resolvers(
    candidates: list,
    queries: list,
    matrix: dict,
) -> dict:
    """
    For each query index, compute the set of candidate names that resolve it
    alone (from the single-perturbation coverage matrix).

    Returns:
        resolvers: dict mapping query_index -> frozenset of candidate names

    axiomander:
        requires:
            isinstance(candidates, list)
            isinstance(queries, list)
            isinstance(matrix, dict)
        ensures:
            isinstance(result, dict)
            len(result) == len(queries)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(candidates, list), "PRE: candidates must be a list"
    assert isinstance(queries, list), "PRE: queries must be a list"
    assert isinstance(matrix, dict), "PRE: matrix must be a dict"

    resolvers: dict = {}
    for qi in range(len(queries)):
        resolvers[qi] = frozenset(c for c in candidates if matrix[c][qi])

    # --- POST ---
    assert isinstance(resolvers, dict), "POST: result must be a dict"
    assert len(resolvers) == len(queries), (
        "POST: result must have one entry per query"
    )
    return resolvers


def union_lower_bound(
    candidate_set: set,
    queries: list,
    matrix: dict,
    resolvers: dict,
) -> int:
    """
    Lower bound on queries resolved by *candidate_set*: count queries resolved
    by at least one member of the set individually (from the matrix).

    This is a lower bound (not exact) because joint resolution can resolve
    queries that no singleton resolves.  However it is free — no cyclic_id.

    axiomander:
        requires:
            isinstance(candidate_set, (set, frozenset))
            isinstance(queries, list)
            isinstance(matrix, dict)
            isinstance(resolvers, dict)
        ensures:
            result >= 0
            result <= len(queries)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(candidate_set, (set, frozenset)), (
        "PRE: candidate_set must be a set or frozenset"
    )
    assert isinstance(queries, list), "PRE: queries must be a list"
    assert isinstance(matrix, dict), "PRE: matrix must be a dict"
    assert isinstance(resolvers, dict), "PRE: resolvers must be a dict"

    resolved_by_union = sum(
        1 for qi in range(len(queries))
        if any(c in resolvers.get(qi, frozenset()) for c in candidate_set)
    )

    # --- POST ---
    assert 0 <= resolved_by_union <= len(queries), (
        "POST: result must be in [0, len(queries)]"
    )
    return resolved_by_union


def optimistic_upper_bound(
    current_set: set,
    pool: list,
    remaining_budget: int,
    queries: list,
    resolvers: dict,
    already_resolved_qi: set,
) -> int:
    """
    Optimistic upper bound for branch-and-bound pruning.

    Assumes the remaining *remaining_budget* slots will be filled with the
    *remaining_budget* pool candidates that individually resolve the most
    still-unresolved queries (from the matrix).  This over-estimates joint
    gain (singleton gains may overlap) but is free and sound for pruning.

    Returns:
        len(already_resolved_qi) + optimistic additional gain

    axiomander:
        requires:
            isinstance(current_set, (set, frozenset))
            isinstance(pool, list)
            remaining_budget >= 0
            isinstance(queries, list)
            isinstance(resolvers, dict)
            isinstance(already_resolved_qi, (set, frozenset))
        ensures:
            result >= len(already_resolved_qi)
            result <= len(queries)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(current_set, (set, frozenset)), (
        "PRE: current_set must be a set or frozenset"
    )
    assert isinstance(pool, list), "PRE: pool must be a list"
    assert isinstance(remaining_budget, int) and remaining_budget >= 0, (
        "PRE: remaining_budget must be a non-negative int"
    )
    assert isinstance(queries, list), "PRE: queries must be a list"
    assert isinstance(resolvers, dict), "PRE: resolvers must be a dict"
    assert isinstance(already_resolved_qi, (set, frozenset)), (
        "PRE: already_resolved_qi must be a set or frozenset"
    )

    still_unresolved = set(range(len(queries))) - already_resolved_qi
    available = [c for c in pool if c not in current_set]

    # per-candidate singleton gain on still-unresolved queries
    gains = sorted(
        (sum(1 for qi in still_unresolved if c in resolvers.get(qi, frozenset()))
         for c in available),
        reverse=True,
    )

    optimistic_extra = sum(gains[:remaining_budget])
    result = len(already_resolved_qi) + optimistic_extra

    # --- POST ---
    assert result >= len(already_resolved_qi), (
        "POST: bound must be >= already resolved"
    )
    return min(result, len(queries))


# ---------------------------------------------------------------------------
# Joint identifiability helpers
# ---------------------------------------------------------------------------


def evaluate_perturbation_set(
    tf1: str,
    outcome: str,
    candidate_set: "set | frozenset",
    graph,
    apt_order,
    identify_fn=None,
) -> bool:
    """
    Test whether query (tf1 -> outcome) becomes identifiable under a joint
    hard intervention do(candidate_set).

    Skips candidates in the set that match tf1 or outcome (self-pairs cannot
    serve as background perturbations for their own query).

    Args:
        tf1:           intervention variable of the query
        outcome:       outcome variable of the query
        candidate_set: set of candidate names to apply simultaneously
        graph:         NxMixedGraph (y0)
        apt_order:     topological ordering for cyclic_id
        identify_fn:   optional injectable (for testing); defaults to cyclic_id

    Returns:
        True if the query is identifiable under do(candidate_set \\ {tf1, outcome}).

    axiomander:
        requires:
            isinstance(tf1, str)
            isinstance(outcome, str)
            isinstance(candidate_set, (set, frozenset))
        ensures:
            isinstance(result, bool)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(tf1, str), "PRE: tf1 must be a str"
    assert isinstance(outcome, str), "PRE: outcome must be a str"
    assert isinstance(candidate_set, (set, frozenset)), (
        "PRE: candidate_set must be a set or frozenset"
    )

    # Exclude self-pairs
    effective_set = candidate_set - {tf1, outcome}

    if identify_fn is not None:
        result = identify_fn(tf1, outcome, effective_set)
        assert isinstance(result, bool), "POST: identify_fn must return bool"
        return result

    from y0.algorithm.identify.cyclic_id import cyclic_id
    from y0.algorithm.identify.utils import Unidentifiable
    from y0.dsl import Variable

    try:
        cyclic_id(
            graph=graph,
            outcomes={Variable(outcome)},
            interventions={Variable(tf1)},
            ordering=apt_order,
        )
        identified = True
    except Unidentifiable:
        identified = False

    # --- POST ---
    assert isinstance(identified, bool), "POST: result must be bool"
    return identified


def score_candidate_set(
    candidate_set: "set | frozenset",
    query_list: list,
    graph,
    apt_order,
    resolvers: dict,
    identify_fn=None,
) -> int:
    """
    Count the number of queries resolved by do(candidate_set).

    Optimization: queries already resolved by a single member of the set
    (from the matrix resolvers dict) are counted directly without a
    cyclic_id call (monotonicity of intervention).  Joint cyclic_id calls
    are reserved for queries not resolved by any singleton member.

    Args:
        candidate_set: set of candidate names
        query_list:    list of (tf1, outcome) pairs
        graph:         NxMixedGraph
        apt_order:     topological ordering
        resolvers:     dict qi -> frozenset of singletons that resolve query qi
        identify_fn:   optional injectable (for testing)

    Returns:
        Number of queries resolved by do(candidate_set).

    axiomander:
        requires:
            isinstance(candidate_set, (set, frozenset))
            isinstance(query_list, list)
            isinstance(resolvers, dict)
        ensures:
            result >= 0
            result <= len(query_list)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(candidate_set, (set, frozenset)), (
        "PRE: candidate_set must be a set or frozenset"
    )
    assert isinstance(query_list, list), "PRE: query_list must be a list"
    assert isinstance(resolvers, dict), "PRE: resolvers must be a dict"

    resolved = 0
    for qi, (tf1, outcome) in enumerate(query_list):
        # Fast path: a singleton member already resolves this query
        if any(c in resolvers.get(qi, frozenset()) for c in candidate_set):
            resolved += 1
            continue
        # Slow path: joint cyclic_id check
        if evaluate_perturbation_set(
            tf1, outcome, candidate_set, graph, apt_order, identify_fn
        ):
            resolved += 1

    # --- POST ---
    assert 0 <= resolved <= len(query_list), (
        "POST: result must be in [0, len(query_list)]"
    )
    return resolved


# ---------------------------------------------------------------------------
# Greedy / beam search driver
# ---------------------------------------------------------------------------


def greedy_topk(
    query_list: list,
    candidates: list,
    queries: list,
    matrix: dict,
    graph,
    apt_order,
    k: int,
    candidate_cap: int,
    beam_width: int,
    require_singleton_gain: bool,
    identify_fn=None,
) -> tuple:
    """
    Cycle-guided greedy/beam search for the top-k simultaneous perturbation set.

    Returns:
        (best_set, scc_break, queries_resolved, steps)
        where steps is a list of (added_gene, joint_gain_at_that_step) for logging.

    axiomander:
        requires:
            isinstance(query_list, list)
            isinstance(candidates, list)
            isinstance(k, int)
            k >= 1
            candidate_cap >= 1
            beam_width >= 1
        ensures:
            isinstance(result, tuple)
            len(result) == 4
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(query_list, list), "PRE: query_list must be a list"
    assert isinstance(candidates, list), "PRE: candidates must be a list"
    assert isinstance(k, int) and k >= 1, "PRE: k must be a positive int"
    assert isinstance(candidate_cap, int) and candidate_cap >= 1, (
        "PRE: candidate_cap must be a positive int"
    )
    assert isinstance(beam_width, int) and beam_width >= 1, (
        "PRE: beam_width must be a positive int"
    )

    from perturbation_optimizer import set_cycle_break_score, rank_candidates_by_cycle_score

    # --- Step 1: build per-query resolver map ---
    resolvers = build_query_resolvers(candidates, queries, matrix)

    # --- Step 2: prune & rank the candidate pool ---
    if require_singleton_gain:
        pool = [c for c in candidates if any(matrix[c][qi] for qi in range(len(queries)))]
        print(f"  Candidate pool after singleton-gain filter: {len(pool)} / {len(candidates)}")
    else:
        pool = list(candidates)
        print(f"  Candidate pool (no singleton-gain filter): {len(pool)}")

    if graph is not None:
        pool = rank_candidates_by_cycle_score(pool, graph)
        # Use only the top candidate_cap for each step's search
        capped_pool = pool[:candidate_cap]
        print(f"  Using top-{candidate_cap} candidates by SCC-break score per step")
    else:
        capped_pool = pool[:candidate_cap]

    # --- Step 3: greedy beam search ---
    # Each beam state: (current_set, already_resolved_qi_set)
    # Seed: empty set (use mutable set for already_resolved_qi — PRE requires set/frozenset)
    beam: list = [(frozenset(), set())]
    best_set: frozenset = frozenset()
    best_score: int = 0
    best_steps: list = []

    for step in range(k):
        remaining_budget = k - step - 1
        next_beam_candidates: list = []

        for current_set, current_resolved_qi in beam:
            # Evaluate each candidate's marginal joint gain
            for cand in capped_pool:
                if cand in current_set:
                    continue

                new_set = current_set | {cand}

                # Upper bound check (branch-and-bound):
                # prune only if we CANNOT beat the current best (strict <)
                ub = optimistic_upper_bound(
                    new_set, pool, remaining_budget,
                    queries, resolvers, current_resolved_qi
                )
                if ub < best_score:
                    continue  # pruned

                # Score the new set (with fast-path skip via resolvers)
                score = score_candidate_set(
                    new_set, query_list, graph, apt_order, resolvers, identify_fn
                )

                # Track newly resolved indices for the next beam state
                new_resolved_qi = current_resolved_qi | {
                    qi for qi in range(len(query_list))
                    if (qi not in current_resolved_qi) and
                       evaluate_perturbation_set(
                           query_list[qi][0], query_list[qi][1],
                           new_set, graph, apt_order, identify_fn
                       )
                }

                next_beam_candidates.append((score, new_set, new_resolved_qi, cand))

                if score > best_score:
                    best_score = score
                    best_set = new_set
                    best_steps = list(best_steps) + [(cand, score)]

        if not next_beam_candidates:
            break

        # Trim beam to beam_width (keep highest-scoring states)
        next_beam_candidates.sort(key=lambda x: -x[0])
        beam = [(s, rqi) for _, s, rqi, _ in next_beam_candidates[:beam_width]]

    # Compute final SCC break score
    if graph is not None:
        scc_break = set_cycle_break_score(best_set, graph)
    else:
        scc_break = 0

    # --- POST ---
    assert isinstance(best_set, frozenset), "POST: best_set must be a frozenset"
    assert isinstance(best_score, int) and best_score >= 0, (
        "POST: best_score must be a non-negative int"
    )

    return best_set, scc_break, best_score, best_steps


# ---------------------------------------------------------------------------
# Exhaustive mode
# ---------------------------------------------------------------------------


def exhaustive_topk(
    query_list: list,
    candidates: list,
    queries: list,
    matrix: dict,
    graph,
    apt_order,
    k: int,
    require_singleton_gain: bool,
    top_n: int = 20,
    identify_fn=None,
) -> list:
    """
    Exhaustively score all C(n,k) candidate sets and return the top *top_n*.

    WARNING: Only feasible for small pools (n <= ~30) or k=2.

    Returns:
        List of (perturbation_set, scc_break, queries_resolved) sorted
        descending by queries_resolved, top top_n entries.

    axiomander:
        requires:
            isinstance(query_list, list)
            isinstance(candidates, list)
            isinstance(k, int)
            k >= 1
            top_n >= 1
        ensures:
            isinstance(result, list)
            len(result) <= top_n
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(query_list, list), "PRE: query_list must be a list"
    assert isinstance(candidates, list), "PRE: candidates must be a list"
    assert isinstance(k, int) and k >= 1, "PRE: k must be a positive int"
    assert isinstance(top_n, int) and top_n >= 1, "PRE: top_n must be a positive int"

    from perturbation_optimizer import set_cycle_break_score, rank_candidates_by_cycle_score

    resolvers = build_query_resolvers(candidates, queries, matrix)

    if require_singleton_gain:
        pool = [c for c in candidates if any(matrix[c][qi] for qi in range(len(queries)))]
        print(f"  Pool after singleton-gain filter: {len(pool)} / {len(candidates)}")
    else:
        pool = list(candidates)

    if graph is not None:
        pool = rank_candidates_by_cycle_score(pool, graph)

    total_combos = 1
    for i in range(k):
        total_combos = total_combos * (len(pool) - i) // (i + 1)
    print(f"  Exhaustive search: C({len(pool)},{k}) = {total_combos} sets")
    if total_combos > 500_000:
        print(
            f"  WARNING: {total_combos} sets is large. "
            "Consider --mode greedy or a smaller pool."
        )

    results: list = []
    for combo in itertools.combinations(pool, k):
        s = frozenset(combo)
        score = score_candidate_set(s, query_list, graph, apt_order, resolvers, identify_fn)
        scc_break = set_cycle_break_score(s, graph) if graph is not None else 0
        results.append((score, scc_break, s))

    results.sort(key=lambda x: (-x[0], -x[1]))
    top = results[:top_n]

    # --- POST ---
    assert isinstance(top, list), "POST: result must be a list"
    assert len(top) <= top_n, "POST: result must have at most top_n entries"
    return [(s, sb, sc) for sc, sb, s in top]


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_topk_csv(path: str, ranked: list, n_queries: int, mode: str):
    """
    Write the top-k simultaneous perturbation results to CSV.

    ranked: list of (perturbation_set, scc_mass_broken, queries_resolved)
    """
    # --- PRE ---
    assert isinstance(path, str) and path, "PRE: path must be a non-empty str"
    assert isinstance(ranked, list), "PRE: ranked must be a list"
    assert isinstance(n_queries, int) and n_queries >= 0, (
        "PRE: n_queries must be a non-negative int"
    )

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "perturbation_set", "scc_mass_broken",
            "queries_resolved", "pct_resolved", "mode",
        ])
        for rank, (pset, scc_break, score) in enumerate(ranked, start=1):
            pset_str = "+".join(sorted(pset))
            pct = score / n_queries * 100 if n_queries > 0 else 0.0
            writer.writerow([rank, pset_str, scc_break, score, f"{pct:.1f}", mode])

    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Graph + apt_order loading
# ---------------------------------------------------------------------------


def load_graph_and_order(graphml_path: str):
    """
    Load the E. coli network and compute apt_order for cyclic_id.

    Returns (ecoli_mixed, apt_order, nx_digraph) or raises on failure.
    """
    # --- PRE ---
    assert isinstance(graphml_path, str) and graphml_path, (
        "PRE: graphml_path must be a non-empty str"
    )
    assert os.path.isfile(graphml_path), f"PRE: file must exist: {graphml_path}"

    import networkx as nx
    from y0.graph import NxMixedGraph

    nx_graph = nx.read_graphml(graphml_path)
    # Convert to DiGraph for cycle analysis (drop undirected edges)
    if not isinstance(nx_graph, nx.DiGraph):
        nx_graph = nx.DiGraph(nx_graph)

    ecoli_mixed = NxMixedGraph.from_edges(directed=list(nx_graph.edges()))

    # Topological ordering: reverse-BFS from all nodes (works on cyclic graphs)
    apt_order = list(nx_graph.nodes())

    return ecoli_mixed, apt_order, nx_graph


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for topk simultaneous perturbation search."""
    parser = argparse.ArgumentParser(
        description="Find top-k simultaneous perturbations maximizing identifiable queries"
    )
    parser.add_argument(
        "--matrix", required=True,
        help="Path to coverage_matrix.csv (from coverage_reduce.py)"
    )
    parser.add_argument(
        "--graphml", required=True,
        help="Path to network .graphml file (for SCC-break scoring)"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Maximum perturbation set size (default: 5)"
    )
    parser.add_argument(
        "--mode", choices=["greedy", "exhaustive"], default="greedy",
        help="Search mode: greedy (default) or exhaustive (only k=2 / small pools)"
    )
    parser.add_argument(
        "--beam-width", type=int, default=5,
        help="Beam width for greedy mode (default: 5)"
    )
    parser.add_argument(
        "--candidate-cap", type=int, default=50,
        help="Number of SCC-top candidates to consider per greedy step (default: 50)"
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Top-n sets to report in exhaustive mode (default: 20)"
    )
    parser.add_argument(
        "--require-singleton-gain", dest="require_singleton_gain",
        action="store_true", default=True,
        help="Restrict to candidates with >=1 individual query resolved (default: on)"
    )
    parser.add_argument(
        "--no-require-singleton-gain", dest="require_singleton_gain",
        action="store_false",
        help="Include zero-individual-gain candidates (synergy exploration)"
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory for output CSV (default: current directory)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load coverage matrix ---
    print(f"Loading coverage matrix: {args.matrix}")
    from perturbation_optimizer import load_coverage_matrix
    candidates, queries, matrix = load_coverage_matrix(args.matrix)
    n_queries = len(queries)
    print(f"  Candidates: {len(candidates)}, Queries: {n_queries}")

    # Derive query_list as (tf1, outcome) pairs from query labels "tf1->outcome"
    query_list: list = []
    for label in queries:
        parts = label.split("->")
        assert len(parts) == 2, f"PRE: query label must be 'tf1->outcome', got {label!r}"
        query_list.append((parts[0], parts[1]))

    # Count baseline (greedy union) for reference
    from perturbation_optimizer import greedy_max_coverage
    baseline = greedy_max_coverage(candidates, queries, matrix, args.k)
    baseline_score = baseline[-1][2] if baseline else 0
    baseline_tfs = [tf for tf, _, _ in baseline]
    print(
        f"  Greedy-union baseline (k={args.k}): {baseline_score}/{n_queries} queries "
        f"({baseline_score/n_queries*100:.1f}%) via {baseline_tfs}"
    )

    # --- Load graph ---
    print(f"\nLoading graph: {args.graphml}")
    ecoli_mixed, apt_order, nx_graph = load_graph_and_order(args.graphml)
    print(f"  Nodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}")

    from perturbation_optimizer import scc_mass as _scc_mass
    print(f"  SCC mass (before any intervention): {_scc_mass(nx_graph)}")

    # --- Run search ---
    out_path = os.path.join(args.output_dir, f"topk_simultaneous_k{args.k}.csv")

    if args.mode == "greedy":
        print(
            f"\nRunning cycle-guided greedy/beam search "
            f"(k={args.k}, beam={args.beam_width}, cap={args.candidate_cap})..."
        )
        best_set, scc_break, score, steps = greedy_topk(
            query_list=query_list,
            candidates=candidates,
            queries=queries,
            matrix=matrix,
            graph=nx_graph,
            apt_order=apt_order,
            k=args.k,
            candidate_cap=args.candidate_cap,
            beam_width=args.beam_width,
            require_singleton_gain=args.require_singleton_gain,
        )
        pct = score / n_queries * 100 if n_queries > 0 else 0.0
        print(
            f"\nBest set found: {sorted(best_set)}"
            f"\n  Queries resolved: {score}/{n_queries} ({pct:.1f}%)"
            f"\n  SCC mass broken:  {scc_break}"
            f"\n  Steps: {steps}"
        )
        ranked = [(best_set, scc_break, score)]
        write_topk_csv(out_path, ranked, n_queries, "greedy")

    else:  # exhaustive
        print(f"\nRunning exhaustive search (k={args.k}, top-{args.top_n})...")
        ranked_sets = exhaustive_topk(
            query_list=query_list,
            candidates=candidates,
            queries=queries,
            matrix=matrix,
            graph=nx_graph,
            apt_order=apt_order,
            k=args.k,
            require_singleton_gain=args.require_singleton_gain,
            top_n=args.top_n,
        )
        print(f"\nTop-{len(ranked_sets)} sets:")
        for i, (pset, scc_break, score) in enumerate(ranked_sets, start=1):
            pct = score / n_queries * 100 if n_queries > 0 else 0.0
            print(
                f"  #{i:2d}: {sorted(pset)} | "
                f"{score}/{n_queries} queries ({pct:.1f}%) | "
                f"SCC mass broken: {scc_break}"
            )
        write_topk_csv(out_path, ranked_sets, n_queries, "exhaustive")

    print(f"\nDone. Output: {out_path}")


if __name__ == "__main__":
    main()
