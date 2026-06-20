"""
coverage_worker.py
==================
SLURM array task for the parallelised coverage-matrix pipeline.

Each task reads ``$SLURM_ARRAY_TASK_ID`` (or ``--task-id``), picks its
assigned unidentifiable queries from the manifest via round-robin
``assign_work``, and evaluates every (query, candidate) pair by calling
``cyclic_id`` with the candidate as background perturbation.

Results are written to ``<shards_dir>/shard_<task_id>.json``.
Per-task checkpointing means a preempted/requeued task resumes from where
it left off.

Usage (SLURM sets SLURM_ARRAY_TASK_ID automatically):
  uv run python scripts/coverage_worker.py \\
    --manifest  notebooks/Ecoli_Analysis_Notebooks/coverage_job.json \\
    --shards-dir notebooks/Ecoli_Analysis_Notebooks/shards \\
    --n-tasks   <total array size>

Manual / local test run:
  uv run python scripts/coverage_worker.py \\
    --manifest  notebooks/Ecoli_Analysis_Notebooks/coverage_job.json \\
    --shards-dir /tmp/shards \\
    --n-tasks 1 --task-id 0
"""

import argparse
import json
import os
import sys

import networkx as nx
from y0.algorithm.identify.cyclic_id import cyclic_id
from y0.algorithm.identify.utils import Unidentifiable
from y0.algorithm.ioscm.utils import get_apt_order
from y0.dsl import P, Variable
from y0.graph import NxMixedGraph

sys.path.insert(0, os.path.dirname(__file__))
from coverage_common import assign_work


def evaluate_query(
    tf1: str,
    outcome: str,
    candidates: list,
    ecoli_mixed,
    apt_order,
    all_network_vars: set,
    completed: set,
) -> list:
    """
    Evaluate all (tf1, outcome, candidate) triples for one query.

    Returns a list of [tf1, candidate, outcome, found] rows.
    Skips candidates equal to tf1 or outcome, and already-completed keys.

    axiomander:
        requires:
            isinstance(tf1, str)
            isinstance(outcome, str)
            isinstance(candidates, list)
            isinstance(completed, set)
        ensures:
            isinstance(result, list)
            all(len(result[i]) == 4 for i in range(len(result)))
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(tf1, str), "PRE: tf1 must be str"
    assert isinstance(outcome, str), "PRE: outcome must be str"
    assert isinstance(candidates, list), "PRE: candidates must be a list"
    assert isinstance(completed, set), "PRE: completed must be a set"

    rows: list = []
    for candidate in candidates:
        if candidate == tf1 or candidate == outcome:
            continue
        key = (tf1, candidate, outcome)
        if key in completed:
            continue

        try:
            cyclic_id(
                graph=ecoli_mixed,
                outcomes={Variable(outcome)},
                interventions={Variable(tf1)},
                ordering=apt_order,
                base_distribution=P[{Variable(candidate)}](all_network_vars),
            )
            found = True
        except Unidentifiable:
            found = False

        rows.append([tf1, candidate, outcome, found])

    # --- POST ---
    assert isinstance(rows, list), "POST: result must be a list"
    assert all(len(r) == 4 for r in rows), "POST: every row has 4 elements"

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Coverage-matrix worker (one SLURM array task)"
    )
    parser.add_argument("--manifest", required=True, help="Path to coverage_job.json")
    parser.add_argument(
        "--shards-dir",
        required=True,
        help="Directory to write shard_<task_id>.json",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        required=True,
        help="Total number of array tasks (== array size)",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Task index (0-based). Defaults to $SLURM_ARRAY_TASK_ID.",
    )
    args = parser.parse_args()

    # Resolve task_id
    if args.task_id is not None:
        task_id = args.task_id
    else:
        env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_id is None:
            print("ERROR: --task-id not given and $SLURM_ARRAY_TASK_ID not set.")
            sys.exit(1)
        task_id = int(env_id)

    n_tasks = args.n_tasks
    assert task_id >= 0 and task_id < n_tasks, (
        f"task_id {task_id} out of range [0, {n_tasks})"
    )

    # --- Load manifest ---
    manifest_path = os.path.abspath(args.manifest)
    assert os.path.isfile(manifest_path), f"Manifest not found: {manifest_path}"
    with open(manifest_path) as f:
        manifest = json.load(f)

    unidentifiable = [tuple(q) for q in manifest["unidentifiable"]]
    candidates = manifest["candidates"]
    graphml_path = manifest["graphml"]
    n_queries = manifest["n_queries"]

    assert os.path.isfile(graphml_path), f"graphml not found: {graphml_path}"

    # --- Assign work ---
    my_query_indices = assign_work(n_queries, task_id, n_tasks)
    my_queries = [unidentifiable[i] for i in my_query_indices]
    print(
        f"[task {task_id}/{n_tasks}] Assigned {len(my_queries)} queries: "
        f"indices {my_query_indices}"
    )

    if not my_queries:
        print(f"[task {task_id}] No queries assigned. Exiting.")
        sys.exit(0)

    # --- Shard checkpoint ---
    os.makedirs(args.shards_dir, exist_ok=True)
    shard_path = os.path.join(args.shards_dir, f"shard_{task_id}.json")

    if os.path.exists(shard_path):
        with open(shard_path) as f:
            ckpt = json.load(f)
        rows = ckpt["rows"]
        completed = set(tuple(x) for x in ckpt["completed"])
        print(f"[task {task_id}] Resuming: {len(completed)} pairs already done.")
    else:
        rows = []
        completed = set()
        print(f"[task {task_id}] Starting fresh.")

    # --- Load graph (each worker loads independently — no shared state) ---
    print(f"[task {task_id}] Loading graph...")
    ecoli_graph = nx.read_graphml(graphml_path)
    network_nodes = set(ecoli_graph.nodes())
    ecoli_mixed = NxMixedGraph.from_edges(directed=list(ecoli_graph.edges()))
    apt_order = get_apt_order(ecoli_mixed)
    all_network_vars = {Variable(g) for g in network_nodes}
    print(f"[task {task_id}] Graph loaded. Nodes: {len(network_nodes)}")

    # --- Main loop ---
    total_pairs = len(my_queries) * len(candidates)
    done = len(completed)
    print(
        f"[task {task_id}] Evaluating {total_pairs} pairs "
        f"({len(my_queries)} queries × {len(candidates)} candidates)."
    )

    for q_num, (tf1, outcome) in enumerate(my_queries):
        new_rows = evaluate_query(
            tf1=tf1,
            outcome=outcome,
            candidates=candidates,
            ecoli_mixed=ecoli_mixed,
            apt_order=apt_order,
            all_network_vars=all_network_vars,
            completed=completed,
        )
        for row in new_rows:
            rows.append(row)
            completed.add((row[0], row[1], row[2]))
            done += 1

        # Checkpoint after each query
        with open(shard_path, "w") as f:
            json.dump({"rows": rows, "completed": [list(k) for k in completed]}, f)

        pct = done / max(total_pairs, 1) * 100
        print(
            f"[task {task_id}] Query {q_num + 1}/{len(my_queries)}: "
            f"do({tf1})->{outcome} | {done}/{total_pairs} ({pct:.1f}%)"
        )

    # --- Final shard save ---
    with open(shard_path, "w") as f:
        json.dump({"rows": rows, "completed": [list(k) for k in completed]}, f)

    print(
        f"[task {task_id}] Done. {len(rows)} rows written to {shard_path}"
    )


if __name__ == "__main__":
    main()
