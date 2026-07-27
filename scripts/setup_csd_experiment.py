"""Create a deterministic CSD experiment manifest and task table.

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
        --mechanisms biological_error,instrument_error

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
identifiers. They intentionally permit repeated seed pairs across conditions
when the design requires them; complete job identity, not seed-pair uniqueness,
is used for scheduling and resume behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nocap.experiment import (
    canonical_condition_id,
    canonical_job_id,
    derive_seed,
    experiment_conditions,
)


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


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
    args = parser.parse_args()

    conditions = experiment_conditions(
        _ints(args.n_samples_list),
        _floats(args.missing_edge_rates),
        _floats(args.missing_data_rates),
        args.mechanisms.split(","),
    )
    tasks = []
    for scm_rep in range(args.scm_replicates):
        for data_rep in range(args.data_replicates):
            data_replicate_id = str(data_rep)
            for condition in conditions:
                condition_id = canonical_condition_id(condition)
                condition_token = (
                    condition["missing_data_mechanism"],
                    condition["missing_edge_rate"],
                    condition["missing_data_rate"],
                    condition["n_samples"],
                )
                if args.design_mode == "paired_hierarchical":
                    scm_seed = derive_seed(args.base_seed, "scm_replicate", scm_rep)
                    data_seed = derive_seed(args.base_seed, "data_replicate", scm_rep, data_rep)
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
                    **condition,
                }
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
