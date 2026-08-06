"""Create a deterministic CSD experiment manifest and task table.

The setup workflow can optionally expand a recovery-bank intervention CSV into
one hard-intervention task per recovery set. For intervention runs, ``--graphml``
is the source graph used to validate gene names and write deterministic,
task-specific perturbed GraphML artifacts. The generated task records point
``csd_estimate.py`` at those artifacts and include intervention provenance;
recovery-bank status fields are not used to make estimation decisions.

Usage
-----
Run from the repository root with the project environment, for example::

    uv run python scripts/setup_csd_experiment.py \
        --output results/experiment.json \
        --tasks-dir results/tasks \
        --output-dir results/csv \
        --experiment-id ecoli-csd-v1 \
        --design-mode paired_hierarchical \
        --base-seed 123 \
        --scm-replicates 2 \
        --data-replicates 3 \
        --n-samples-list 100,500,1000 \
        --missing-edge-rates 0.0,0.2,0.4 \
        --missing-data-rates 0.0,0.3 \
        --mechanisms biological_error,instrument_error \
        --use-latent-expression-hat

Intervention task generation::

    uv run python scripts/setup_csd_experiment.py \
        --output results/interventions.json \
        --tasks-dir results/intervention-tasks \
        --output-dir results/intervention-csv \
        --experiment-id ecoli-csd-interventions \
        --graphml data/network.graphml \
        --intervention-csv notebooks/Ecoli_Analysis_Notebooks/csd_recovery_n5_k6.csv \
        --intervention-graph-dir results/intervention-graphs

The command writes one manifest to ``--output`` and, when ``--tasks-dir`` is
provided, one JSON record per complete simulation job. Each task record can be
executed directly by ``csd_estimate.py``::

    uv run python scripts/csd_estimate.py \
        --task-json results/tasks/<task>.json \
        --graphml path/to/network.graphml \
        --adjustments-csv path/to/adjustments.csv

The SLURM wrapper ``scripts/slurm/submit_csd_estimate.sh`` invokes this setup
script automatically. Use ``--output-dir`` to add an ``output_csv`` path to
each task; without it, the manifest still contains the complete task
identities and seed assignments but is not directly runnable by the estimator.

The generated records include stable experiment, replicate, condition, and job
identifiers. Intervention jobs additionally include the intervention ID, source
set index, normalized genes, hard-intervention semantics, perturbed GraphML
path, and forbidden incoming edges. They intentionally permit repeated seed
pairs across conditions when the design requires them; complete job identity,
not seed-pair uniqueness, is used for scheduling and resume behavior.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import networkx as nx

from nocap.experiment import (
    canonical_condition_id,
    canonical_job_id,
    derive_seed,
    experiment_conditions,
)
from nocap.scc_perturb import build_intervened_graph


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _interventions(path: str, graph: nx.DiGraph) -> list[dict]:
    result = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("genes")
            if raw is None:
                raise ValueError("Intervention CSV must contain a genes column")
            try:
                genes = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    genes = ast.literal_eval(raw.replace('""', '"'))
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(f"Malformed genes list: {raw!r}") from exc
            if not isinstance(genes, (list, tuple)) or not all(isinstance(g, str) for g in genes):
                raise ValueError(f"Malformed genes list: {raw!r}")
            genes = sorted(set(genes))
            unknown = set(genes) - {str(n) for n in graph.nodes}
            if unknown:
                raise ValueError(f"Unknown intervention genes: {sorted(unknown)}")
            set_index = row.get("set_index", "")
            canonical = f"{set_index}|{','.join(genes)}"
            intervention_id = "int-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]
            result.append({"id": intervention_id, "set_index": set_index, "genes": genes})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tasks-dir", default=None)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--design-mode", default="paired_hierarchical")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--scm-replicates", type=int, default=1)
    parser.add_argument("--data-replicates", type=int, default=1)
    parser.add_argument("--n-samples-list", default="100")
    parser.add_argument("--missing-edge-rates", default="0")
    parser.add_argument("--missing-data-rates", default="0")
    parser.add_argument("--mechanisms", default="biological_error")
    parser.add_argument("--graphml", default=None)
    parser.add_argument("--intervention-csv", default=None)
    parser.add_argument("--intervention-graph-dir", default=None)
    parser.add_argument("--include-observational", action="store_true")
    parser.add_argument("--fixed-intervention-value", type=float, default=None)
    latent_hat_group = parser.add_mutually_exclusive_group()
    latent_hat_group.add_argument(
        "--use-latent-expression-hat",
        dest="use_latent_expression_hat",
        action="store_true",
        type=bool,
        help="Use count-derived expression estimates as estimator input (default).",
    )
    latent_hat_group.add_argument(
        "--no-latent-expression-hat",
        dest="use_latent_expression_hat",
        action="store_false",
        type=bool,
        help="Use true latent expression as estimator input.",
    )
    parser.set_defaults(use_latent_expression_hat=True)
    args = parser.parse_args()

    source_graph = None
    interventions = []
    if args.intervention_csv:
        if not args.graphml:
            raise SystemExit("--graphml is required with --intervention-csv")
        source_graph = nx.DiGraph(nx.read_graphml(args.graphml))
        source_graph = nx.relabel_nodes(source_graph, {n: str(n) for n in source_graph.nodes})
        interventions = _interventions(args.intervention_csv, source_graph)
        graph_dir = Path(
            args.intervention_graph_dir or Path(args.output).parent / "intervention_graphs"
        )
        graph_dir.mkdir(parents=True, exist_ok=True)
        for intervention in interventions:
            graph = build_intervened_graph(source_graph, intervention["genes"])
            path = graph_dir / f"{intervention['id']}.graphml"
            ordered = nx.DiGraph()
            ordered.add_nodes_from(sorted(graph.nodes()))
            ordered.add_edges_from(sorted((str(u), str(v)) for u, v in graph.edges()))
            nx.write_graphml(ordered, path)

    designs = list(interventions)
    if args.include_observational or not args.intervention_csv:
        designs.append(None)

    conditions = experiment_conditions(
        _ints(args.n_samples_list),
        _floats(args.missing_edge_rates),
        _floats(args.missing_data_rates),
        args.mechanisms.split(","),
    )

    # The cell samples sizes for multiple experiments is
    # the number of total cells divided by the number of experiments.
    # So if there are n=5 experiments and 100k total cells, then
    # each experiment will have 20k total cells
    if len(designs) > 1:
        for condition in conditions:
            condition["n_samples"] //= len(designs)

    tasks = []

    for scm_rep in range(args.scm_replicates):
        for data_rep in range(args.data_replicates):
            data_replicate_id = str(data_rep)
            for condition in conditions:
                for intervention in designs:
                    condition_id = canonical_condition_id(condition)
                    intervention_suffix = intervention["id"] if intervention else "observational"
                    condition_token = (
                        condition["missing_data_mechanism"],
                        condition["missing_edge_rate"],
                        condition["missing_data_rate"],
                        condition["n_samples"],
                        intervention_suffix,
                    )
                    if args.design_mode == "paired_hierarchical":
                        scm_seed = derive_seed(args.base_seed, "scm_replicate", scm_rep)
                        data_seed = derive_seed(
                            args.base_seed, "data_replicate", scm_rep, data_rep, intervention_suffix
                        )
                    elif args.design_mode == "fixed_scm":
                        scm_seed = derive_seed(
                            args.base_seed, "scm_block", scm_rep, condition["missing_edge_rate"]
                        )
                        data_seed = derive_seed(
                            args.base_seed, "data_cell", scm_rep, data_rep, *condition_token
                        )
                    else:
                        scm_seed = derive_seed(
                            args.base_seed, "scm_cell", scm_rep, data_rep, *condition_token
                        )
                        data_seed = derive_seed(
                            args.base_seed, "data_cell", scm_rep, data_rep, *condition_token
                        )
                    job_id = canonical_job_id(
                        args.experiment_id,
                        args.design_mode,
                        str(scm_rep),
                        data_replicate_id,
                        condition_id,
                        intervention_suffix,
                    )
                    task = {
                        "experiment_id": args.experiment_id,
                        "design_mode": args.design_mode,
                        "scm_replicate_id": str(scm_rep),
                        "data_replicate_id": data_replicate_id,
                        "scm_seed": scm_seed,
                        "data_seed": data_seed,
                        "parameter_condition_id": condition_id,
                        "job_id": job_id,
                        "seed_scope": "scm_block"
                        if args.design_mode == "fixed_scm"
                        else ("parameter_cell" if args.design_mode == "full" else "replicate"),
                        "use_latent_expression_hat": args.use_latent_expression_hat,
                        **condition,
                    }
                    if intervention:
                        task.update(
                            {
                                "graphml": str(graph_dir / f"{intervention['id']}.graphml"),
                                "intervention_id": intervention["id"],
                                "intervention_set_index": intervention["set_index"],
                                "intervention_genes": intervention["genes"],
                                "intervention_semantics": "hard_do",
                                "fixed_intervention_values": (
                                    dict.fromkeys(intervention["genes"], args.fixed_intervention_value)
                                    if args.fixed_intervention_value is not None
                                    else {}
                                ),
                                "forbidden_edges": [
                                    (str(u), str(v))
                                    for u in source_graph.nodes
                                    for v in intervention["genes"]
                                    if u != v
                                ],
                            }
                        )
                    elif args.graphml:
                        task["graphml"] = args.graphml
                    if args.output_dir:
                        task["output_csv"] = str(
                            Path(args.output_dir) / f"{job_id.replace(':', '_')}.csv"
                        )
                    tasks.append(task)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version": "1", "tasks": tasks}, indent=2) + "\n")
    if args.tasks_dir:
        tasks_dir = Path(args.tasks_dir)
        tasks_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            (tasks_dir / f"{task['job_id'].replace(':', '_')}.json").write_text(
                json.dumps(task, sort_keys=True) + "\n"
            )


if __name__ == "__main__":
    main()
