"""
scc_perturb_worker.py
=====================
SLURM array task for the SCC-perturbation pipeline.

Each task reads ``$SLURM_ARRAY_TASK_ID`` (or ``--task-id``) and processes the
corresponding TF entry from the manifest produced by ``scc_perturb_prepare.py``.

For the assigned TF *t*:
  1. Build the intervened graph  do(B(t))  by removing all in-edges to every
     node in B(t) = task["min_cut"].
  2. Compute O(t) = descendants of t in the post-intervention graph.
  3. Issue **one** joint ``cyclic_id`` call:
       cyclic_id(interventions={t}, outcomes=O(t), base_distribution=do(B(t)))
  4. If the joint call raises ``Unidentifiable`` and ``--per-gene-on-failure``
     is set, fall back to individual calls for each g in O(t).

Results are written to ``<shards_dir>/scc_perturb_shard_<tf>.json``.
Idempotent: if the shard already exists it is skipped (re-run safe).

Usage (SLURM sets SLURM_ARRAY_TASK_ID automatically):
  uv run python scripts/scc_perturb_worker.py \\
    --manifest   notebooks/Ecoli_Analysis_Notebooks/scc_perturb_job.json \\
    --shards-dir notebooks/Ecoli_Analysis_Notebooks/scc_perturb_shards \\
    --n-tasks    <total array size>

Manual / local test run:
  uv run python scripts/scc_perturb_worker.py \\
    --manifest   notebooks/Ecoli_Analysis_Notebooks/scc_perturb_job.json \\
    --shards-dir /tmp/scc_shards \\
    --n-tasks 1 --task-id 0
"""

import argparse
import json
import os
import sys

import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Core helpers (injectable for testing)
# ---------------------------------------------------------------------------


def build_intervened_graph(graph, min_cut: list):
    """
    Return a copy of *graph* with all in-edges to every node in *min_cut*
    removed (hard intervention do(B(t))).

    axiomander:
        ensures:
            all(result.in_degree(n) == 0 for n in min_cut if n in result.nodes())
            result.number_of_nodes() == graph.number_of_nodes()
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(min_cut, list), "PRE: min_cut must be a list"

    intervened = graph.copy()
    for node in min_cut:
        if node in intervened:
            in_edges = list(intervened.in_edges(node))
            intervened.remove_edges_from(in_edges)

    # --- POST ---
    for node in min_cut:
        if node in intervened:
            assert intervened.in_degree(node) == 0, (
                f"POST: node {node!r} must have in-degree 0 after intervention"
            )
    return intervened


def get_descendants(tf: str, intervened_graph) -> list:
    """
    Return the sorted list of descendants of *tf* in *intervened_graph*
    (nodes reachable from *tf* via directed edges, excluding *tf* itself).

    axiomander:
        ensures:
            tf not in result
            result == sorted(result)
            all(isinstance(g, str) for g in result)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(tf, str), "PRE: tf must be a str"

    if tf not in intervened_graph:
        return []

    desc = sorted(nx.descendants(intervened_graph, tf))

    # --- POST ---
    assert isinstance(desc, list), "POST: result must be a list"
    assert tf not in desc, "POST: tf must not be in its own descendants"
    return desc


def run_joint_cyclic_id(
    tf: str,
    outcome_set: set,
    min_cut: list,
    ecoli_mixed,
    apt_order,
    all_network_vars: set,
    identify_fn=None,
) -> bool:
    """
    Issue one joint ``cyclic_id`` call for ``do(tf)`` over *outcome_set*
    under base distribution ``do(B(t))`` = ``P[{B(t)}](...)``.

    Returns True if jointly identifiable, False otherwise.

    Args:
        tf:              TF to intervene on
        outcome_set:     set of Variable objects (post-intervention descendants)
        min_cut:         B(t) node list (used as base distribution perturbation)
        ecoli_mixed:     NxMixedGraph
        apt_order:       topological ordering
        all_network_vars: set of all Variable objects in the network
        identify_fn:     optional injectable; called as
                         identify_fn(tf, outcome_set, min_cut) -> bool

    axiomander:
        ensures:
            isinstance(result, bool)
            implies(len(outcome_set) == 0, result == True)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(tf, str), "PRE: tf must be a str"
    assert isinstance(outcome_set, (set, frozenset)), (
        "PRE: outcome_set must be a set or frozenset"
    )
    assert isinstance(min_cut, list), "PRE: min_cut must be a list"

    if identify_fn is not None:
        result = identify_fn(tf, outcome_set, min_cut)
        assert isinstance(result, bool), "POST: identify_fn must return bool"
        return result

    from y0.algorithm.identify.cyclic_id import cyclic_id
    from y0.algorithm.identify.utils import Unidentifiable
    from y0.dsl import P, Variable

    base_dist_vars = {Variable(n) for n in min_cut}

    try:
        cyclic_id(
            graph=ecoli_mixed,
            outcomes=outcome_set,
            interventions={Variable(tf)},
            ordering=apt_order,
            base_distribution=P[base_dist_vars](all_network_vars),
        )
        identifiable = True
    except Unidentifiable:
        identifiable = False

    # --- POST ---
    assert isinstance(identifiable, bool), "POST: result must be bool"
    return identifiable


def run_per_gene_cyclic_id(
    tf: str,
    outcome_vars: list,
    min_cut: list,
    ecoli_mixed,
    apt_order,
    all_network_vars: set,
    identify_fn=None,
) -> dict:
    """
    Fallback: run individual ``cyclic_id`` calls for each outcome variable.

    Returns a dict mapping gene name -> bool (identifiable or not).

    axiomander:
        ensures:
            len(result) == len(outcome_vars)
            all(isinstance(v, bool) for v in result.values())
            implies(len(outcome_vars) == 0, len(result) == 0)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(tf, str), "PRE: tf must be a str"
    assert isinstance(outcome_vars, list), "PRE: outcome_vars must be a list"
    assert isinstance(min_cut, list), "PRE: min_cut must be a list"

    from y0.dsl import Variable

    results: dict = {}
    for var in outcome_vars:
        gene = var.name if hasattr(var, "name") else str(var)
        identified = run_joint_cyclic_id(
            tf=tf,
            outcome_set={var},
            min_cut=min_cut,
            ecoli_mixed=ecoli_mixed,
            apt_order=apt_order,
            all_network_vars=all_network_vars,
            identify_fn=identify_fn,
        )
        results[gene] = identified

    # --- POST ---
    assert isinstance(results, dict), "POST: result must be a dict"
    assert len(results) == len(outcome_vars), (
        "POST: result must have one entry per outcome variable"
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="SCC-perturbation worker (one SLURM array task)"
    )
    parser.add_argument("--manifest", required=True, help="Path to scc_perturb_job.json")
    parser.add_argument(
        "--shards-dir",
        required=True,
        help="Directory to write scc_perturb_shard_<tf>.json",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        required=True,
        help="Total number of array tasks (== array size == n_tasks in manifest)",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Task index (0-based). Defaults to $SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--per-gene-on-failure",
        action="store_true",
        default=False,
        help=(
            "If the joint cyclic_id call is Unidentifiable, fall back to "
            "individual per-gene calls (slower, but gives per-gene results)."
        ),
    )
    args = parser.parse_args()

    # --- Resolve task_id ---
    if args.task_id is not None:
        task_id = args.task_id
    else:
        env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_id is None:
            print("ERROR: --task-id not given and $SLURM_ARRAY_TASK_ID not set.")
            sys.exit(1)
        task_id = int(env_id)

    n_tasks = args.n_tasks
    assert 0 <= task_id < n_tasks, (
        f"task_id {task_id} out of range [0, {n_tasks})"
    )

    # --- Load manifest ---
    manifest_path = os.path.abspath(args.manifest)
    assert os.path.isfile(manifest_path), f"Manifest not found: {manifest_path}"
    with open(manifest_path) as f:
        manifest = json.load(f)

    tasks = manifest["tasks"]
    assert 0 <= task_id < len(tasks), (
        f"task_id {task_id} out of range for {len(tasks)} tasks"
    )
    task = tasks[task_id]

    tf = task["tf"]
    min_cut: list = task["min_cut"]
    scc_size: int = task["scc_size"]
    in_scc_children: list = task["in_scc_children"]

    graphml_path: str = manifest["graphml"]
    assert os.path.isfile(graphml_path), f"graphml not found: {graphml_path}"

    # --- Idempotency: skip if shard already written ---
    os.makedirs(args.shards_dir, exist_ok=True)
    shard_path = os.path.join(args.shards_dir, f"scc_perturb_shard_{tf}.json")
    if os.path.exists(shard_path):
        print(f"[task {task_id} | {tf}] Shard already exists — skipping.")
        sys.exit(0)

    print(f"[task {task_id} | {tf}] Starting.")
    print(f"  SCC size:          {scc_size}")
    print(f"  In-SCC children:   {in_scc_children}")
    print(f"  B(t) (min cut):    {min_cut}")

    # --- Load graph ---
    print(f"[task {task_id} | {tf}] Loading graph...")
    raw_graph = nx.read_graphml(graphml_path)
    if not isinstance(raw_graph, nx.DiGraph):
        raw_graph = nx.DiGraph(raw_graph)
    network_nodes = set(raw_graph.nodes())

    from y0.algorithm.ioscm.utils import get_apt_order
    from y0.dsl import Variable
    from y0.graph import NxMixedGraph

    ecoli_mixed = NxMixedGraph.from_edges(directed=list(raw_graph.edges()))
    apt_order = get_apt_order(ecoli_mixed)
    all_network_vars = {Variable(g) for g in network_nodes}
    print(f"[task {task_id} | {tf}] Graph loaded. Nodes: {len(network_nodes)}")

    # --- Build intervened graph and compute descendants ---
    intervened_graph = build_intervened_graph(raw_graph, min_cut)
    descendants = get_descendants(tf, intervened_graph)
    print(f"[task {task_id} | {tf}] Descendants in do(B(t)) graph: {len(descendants)}")

    if not descendants:
        print(f"[task {task_id} | {tf}] No descendants — writing empty shard.")
        shard = {
            "tf": tf,
            "min_cut": min_cut,
            "scc_size": scc_size,
            "in_scc_children": in_scc_children,
            "n_descendants": 0,
            "outcomes": [],
            "joint_identifiable": None,
            "per_gene": {},
            "note": "no_descendants",
        }
        with open(shard_path, "w") as f:
            json.dump(shard, f, indent=2)
        sys.exit(0)

    outcome_vars = {Variable(g) for g in descendants}
    outcome_var_list = [Variable(g) for g in descendants]

    # --- Joint cyclic_id call ---
    print(
        f"[task {task_id} | {tf}] Running joint cyclic_id "
        f"(interventions={{{tf}}}, |outcomes|={len(outcome_vars)}, "
        f"|B(t)|={len(min_cut)})..."
    )
    joint_identifiable = run_joint_cyclic_id(
        tf=tf,
        outcome_set=outcome_vars,
        min_cut=min_cut,
        ecoli_mixed=ecoli_mixed,
        apt_order=apt_order,
        all_network_vars=all_network_vars,
    )
    print(
        f"[task {task_id} | {tf}] Joint identifiable: {joint_identifiable}"
    )

    # --- Optional per-gene fallback ---
    per_gene: dict = {}
    if not joint_identifiable and args.per_gene_on_failure:
        print(
            f"[task {task_id} | {tf}] Joint Unidentifiable — "
            f"running per-gene fallback ({len(outcome_var_list)} calls)..."
        )
        per_gene = run_per_gene_cyclic_id(
            tf=tf,
            outcome_vars=outcome_var_list,
            min_cut=min_cut,
            ecoli_mixed=ecoli_mixed,
            apt_order=apt_order,
            all_network_vars=all_network_vars,
        )
        n_id = sum(1 for v in per_gene.values() if v)
        print(
            f"[task {task_id} | {tf}] Per-gene: {n_id}/{len(per_gene)} identifiable"
        )

    # --- Write shard ---
    shard = {
        "tf": tf,
        "min_cut": min_cut,
        "scc_size": scc_size,
        "in_scc_children": in_scc_children,
        "n_descendants": len(descendants),
        "outcomes": sorted(descendants),
        "joint_identifiable": joint_identifiable,
        "per_gene": per_gene,
        "note": "",
    }
    with open(shard_path, "w") as f:
        json.dump(shard, f, indent=2)

    print(f"[task {task_id} | {tf}] Done. Shard written to {shard_path}")


if __name__ == "__main__":
    main()
