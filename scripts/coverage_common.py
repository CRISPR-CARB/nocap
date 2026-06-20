"""
coverage_common.py
==================
Shared helpers for the parallelised coverage-matrix pipeline.

Imported by:
  coverage_prepare.py  -- serial manifest builder
  coverage_worker.py   -- SLURM array task
  coverage_reduce.py   -- shard merger / CSV writer
  build_coverage_matrix.py -- legacy serial entry-point (backward-compat)

Pure helpers (no I/O, no y0 calls):
  assign_work(n_queries, task_id, n_tasks) -> list[int]
  merge_shards(shard_list)                 -> list[list]
  rows_to_matrix(rows, query_labels)       -> tuple[list[str], list[str], dict]

I/O helpers (graph loading, CSV parsing, y0 calls):
  load_valid_genes(supptable_path, network_nodes) -> set[str]
  build_baseline_queries(ecoli_graph, valid_genes) -> list[tuple[str,str]]
  run_phase1(ecoli_mixed, query_pairs, apt_order)  -> tuple[list, list]
  get_candidate_tfs(ecoli_graph, valid_genes)      -> list[str]

Contract annotations
--------------------
Every public function carries:
  - Inline assert PRE / INV / POST statements (executable at runtime)
  - Docstring ``axiomander:`` blocks for frame conditions, ghost bindings,
    and quantified postconditions that are verifier-only.

Production scripts are dependency-free (no axiomander import).
"""

import csv
import os
import sys as _sys
import os as _os

_sys.path.insert(0, _os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def assign_work(n_queries: int, task_id: int, n_tasks: int) -> list:
    """
    Return the list of query indices assigned to *task_id* in a round-robin
    partition of ``range(n_queries)`` across ``n_tasks`` tasks.

    Round-robin ensures near-equal load even when n_queries is not divisible
    by n_tasks.  Task 0 gets indices [0, n_tasks, 2*n_tasks, ...], task 1
    gets [1, n_tasks+1, ...], etc.

    axiomander:
        requires:
            n_queries >= 0
            n_tasks >= 1
            task_id >= 0
            task_id < n_tasks
        ensures:
            len(result) >= 0
            all(0 <= result[i] < n_queries for i in range(len(result)))
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(n_queries, int), "PRE: n_queries must be int"
    assert isinstance(n_tasks, int), "PRE: n_tasks must be int"
    assert isinstance(task_id, int), "PRE: task_id must be int"
    assert n_queries >= 0, "PRE: n_queries >= 0"
    assert n_tasks >= 1, "PRE: n_tasks >= 1"
    assert task_id >= 0, "PRE: task_id >= 0"
    assert task_id < n_tasks, "PRE: task_id < n_tasks"

    indices = list(range(task_id, n_queries, n_tasks))

    # --- POST ---
    assert isinstance(indices, list), "POST: result must be a list"
    assert all(isinstance(i, int) for i in indices), "POST: all indices must be int"
    assert all(0 <= i < n_queries for i in indices), "POST: all indices in [0, n_queries)"

    return indices


def merge_shards(shard_list: list) -> list:
    """
    Merge a list of shard row-lists into a single deduplicated list of rows.

    Each shard is a list of rows ``[tf1, candidate, outcome, found]``.
    Deduplication is by the key ``(tf1, candidate, outcome)``; the first
    occurrence wins (shards are assumed to be internally consistent).

    axiomander:
        requires:
            isinstance(shard_list, list)
        ensures:
            isinstance(result, list)
            all(len(result[i]) == 4 for i in range(len(result)))
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(shard_list, list), "PRE: shard_list must be a list"

    seen: set = set()
    merged: list = []
    for shard in shard_list:
        # INV: shard is a list of rows
        assert isinstance(shard, list), "INV: each shard must be a list"
        for row in shard:
            assert isinstance(row, (list, tuple)) and len(row) == 4, (
                "INV: each row must be a 4-element list/tuple"
            )
            key = (row[0], row[1], row[2])
            if key not in seen:
                seen.add(key)
                merged.append(list(row))

    # --- POST ---
    assert isinstance(merged, list), "POST: result must be a list"
    assert all(len(r) == 4 for r in merged), "POST: every row has 4 elements"
    # No duplicate keys
    keys = [(r[0], r[1], r[2]) for r in merged]
    assert len(keys) == len(set(keys)), "POST: no duplicate (tf1, candidate, outcome) keys"

    return merged


def rows_to_matrix(rows: list, query_labels: list) -> tuple:
    """
    Convert a flat list of ``[tf1, candidate, outcome, found]`` rows into a
    coverage matrix representation.

    Returns:
        candidate_set  -- sorted list of candidate TF names
        query_labels   -- the input query_labels list (passed through)
        lookup         -- dict: (candidate, query_label) -> bool

    axiomander:
        requires:
            isinstance(rows, list)
            isinstance(query_labels, list)
        ensures:
            isinstance(result, tuple)
            len(result) == 3
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(rows, list), "PRE: rows must be a list"
    assert isinstance(query_labels, list), "PRE: query_labels must be a list"

    lookup: dict = {}
    for row in rows:
        # INV: each row is a 4-element sequence
        assert isinstance(row, (list, tuple)) and len(row) == 4, (
            "INV: each row must be a 4-element list/tuple"
        )
        tf1, candidate, outcome, found = row
        label = f"{tf1}->{outcome}"
        lookup[(candidate, label)] = bool(found)

    candidate_set = sorted(set(r[1] for r in rows))

    # --- POST ---
    assert isinstance(candidate_set, list), "POST: candidate_set must be a list"
    assert isinstance(lookup, dict), "POST: lookup must be a dict"
    assert all(isinstance(c, str) for c in candidate_set), (
        "POST: all candidates must be str"
    )
    # candidate_set is sorted
    assert candidate_set == sorted(candidate_set), "POST: candidate_set is sorted"

    return candidate_set, query_labels, lookup


# ---------------------------------------------------------------------------
# I/O helpers (shared with build_coverage_matrix.py)
# ---------------------------------------------------------------------------


def load_valid_genes(supptable_path: str, network_nodes: set) -> set:
    """Return the set of experimental genes present in the network.

    axiomander:
        requires:
            isinstance(supptable_path, str)
            len(supptable_path) > 0
            isinstance(network_nodes, (set, frozenset))
        ensures:
            isinstance(result, set)
            result <= set(network_nodes)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(supptable_path, str), "PRE: supptable_path must be str"
    assert len(supptable_path) > 0, "PRE: supptable_path must be non-empty"
    assert os.path.isfile(supptable_path), f"PRE: file must exist: {supptable_path}"
    assert isinstance(network_nodes, (set, frozenset)), (
        "PRE: network_nodes must be a set or frozenset"
    )

    csv_genes: set = set()
    with open(supptable_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Perturbation Name"]
            if "off" not in name.lower() and "AAV" not in name:
                csv_genes.add(name)

    valid = csv_genes & set(network_nodes)

    # --- POST ---
    assert isinstance(valid, set), "POST: result must be a set"
    assert valid <= set(network_nodes), "POST: result must be a subset of network_nodes"
    assert all("off" not in g.lower() for g in valid), (
        "POST: result must not contain any 'off' gene"
    )
    assert all("AAV" not in g for g in valid), (
        "POST: result must not contain any 'AAV' gene"
    )

    print(f"  Experimental genes in CSV: {len(csv_genes)}")
    print(f"  Valid genes in network:    {len(valid)}")
    return valid


def build_baseline_queries(ecoli_graph, valid_genes: set) -> list:
    """
    Reproduce Biology_Analysis.ipynb Phase 1 query construction:
    directed pairs (intervention, outcome) where outcome is a direct
    downstream neighbour of intervention, both in valid_genes, no self-loops.

    axiomander:
        requires:
            isinstance(valid_genes, (set, frozenset))
        ensures:
            isinstance(result, list)
            all(len(result[i]) == 2 for i in range(len(result)))
            all(result[i][0] != result[i][1] for i in range(len(result)))
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(valid_genes, (set, frozenset)), (
        "PRE: valid_genes must be a set or frozenset"
    )
    assert hasattr(ecoli_graph, "successors"), (
        "PRE: ecoli_graph must support .successors()"
    )

    pairs: list = []
    for gene in valid_genes:
        if gene not in ecoli_graph:
            continue
        for target in ecoli_graph.successors(gene):
            if target in valid_genes and target != gene:
                pairs.append((gene, target))

    # --- POST ---
    assert isinstance(pairs, list), "POST: result must be a list"
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs), (
        "POST: every element must be a 2-tuple"
    )
    assert all(src != tgt for src, tgt in pairs), "POST: no self-loops"
    assert len(pairs) == len(set(pairs)), "POST: no duplicate pairs"

    return pairs


def run_phase1(ecoli_mixed, query_pairs: list, apt_order) -> tuple:
    """Return (identifiable, unidentifiable) lists from Phase 1.

    axiomander:
        requires:
            isinstance(query_pairs, list)
        ensures:
            isinstance(result, tuple)
            len(result) == 2
        modifies:
            none
    """
    from y0.algorithm.identify.cyclic_id import cyclic_id
    from y0.algorithm.identify.utils import Unidentifiable
    from y0.dsl import Variable

    # --- PRE ---
    assert isinstance(query_pairs, list), "PRE: query_pairs must be a list"

    identifiable: list = []
    unidentifiable: list = []
    for intervention, outcome in query_pairs:
        try:
            result = cyclic_id(
                graph=ecoli_mixed,
                outcomes={Variable(outcome)},
                interventions={Variable(intervention)},
                ordering=apt_order,
            )
            identifiable.append((intervention, outcome, result))
        except Unidentifiable:
            unidentifiable.append((intervention, outcome))

    # --- POST ---
    assert isinstance(identifiable, list), "POST: identifiable must be a list"
    assert isinstance(unidentifiable, list), "POST: unidentifiable must be a list"
    assert len(identifiable) + len(unidentifiable) == len(query_pairs), (
        "POST: every query is classified"
    )

    return identifiable, unidentifiable


def get_candidate_tfs(ecoli_graph, valid_genes: set) -> list:
    """
    Candidate background perturbations: every node with out-degree >= 1,
    ranked by cycle-breaking score (descending).

    axiomander:
        requires:
            isinstance(valid_genes, (set, frozenset))
        ensures:
            isinstance(result, list)
        modifies:
            none
    """
    from perturbation_optimizer import rank_candidates_by_cycle_score

    # --- PRE ---
    assert isinstance(valid_genes, (set, frozenset)), (
        "PRE: valid_genes must be a set or frozenset"
    )

    raw = [n for n in ecoli_graph.nodes() if ecoli_graph.out_degree(n) >= 1]
    print(f"  Candidate TFs (out-degree >= 1): {len(raw)}")
    print("  Ranking candidates by cycle-breaking score...")
    ranked = rank_candidates_by_cycle_score(raw, ecoli_graph)
    print(f"  Top 5 by cycle score: {ranked[:5]}")

    # --- POST ---
    assert isinstance(ranked, list), "POST: result must be a list"

    return ranked
