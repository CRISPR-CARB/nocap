"""
clear_unident_shards.py
=======================
Delete shards that need to be recomputed with --per-gene-on-failure:
  - missing shards (not yet computed at all)
  - shards where joint_identifiable=False AND per_gene is empty {}

Shards that are already good (joint_identifiable=True, OR per_gene non-empty)
are left untouched.

This is the preparation step before resubmitting the SLURM array with
PER_GENE_ON_FAILURE=1 so that the per-gene fallback populates the shards.

Usage:
    uv run python scripts/clear_unident_shards.py [--dry-run] 2>&1 | tail -30

Options:
    --dry-run   Print what would be deleted without actually deleting.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from shard_io import load_first_json_object

MANIFEST = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "notebooks", "Ecoli_Analysis_Notebooks", "scc_perturb_job.json"
))
SHARDS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "notebooks", "Ecoli_Analysis_Notebooks", "scc_perturb_shards"
))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print deletions without performing them.")
    args = parser.parse_args()

    with open(MANIFEST) as f:
        manifest = json.load(f)

    tasks = manifest["tasks"]

    to_delete = []
    keep = []

    for i, task in enumerate(tasks):
        tf = task["tf"]
        shard_path = os.path.join(SHARDS_DIR, f"scc_perturb_shard_{tf}.json")

        if not os.path.exists(shard_path):
            to_delete.append((i, tf, shard_path, "missing"))
            continue

        try:
            s = load_first_json_object(shard_path)
            joint = s.get("joint_identifiable")
            per_gene = s.get("per_gene", {})
            if joint is False and len(per_gene) == 0:
                to_delete.append((i, tf, shard_path, "joint=False, per_gene={}"))
            else:
                keep.append((i, tf, joint, len(per_gene)))
        except Exception as e:
            to_delete.append((i, tf, shard_path, f"read-error: {e}"))

    print(f"Shards to DELETE ({len(to_delete)}):")
    for i, tf, path, reason in to_delete:
        print(f"  task {i:>2}  {tf:<12}  [{reason}]")

    print()
    print(f"Shards to KEEP ({len(keep)}):")
    for i, tf, joint, pg_n in keep:
        print(f"  task {i:>2}  {tf:<12}  joint={joint}  per_gene_n={pg_n}")

    if args.dry_run:
        print()
        print("DRY-RUN: no files deleted.")
        return

    print()
    for i, tf, path, reason in to_delete:
        if os.path.exists(path):
            os.remove(path)
            print(f"  Deleted: {os.path.basename(path)}")
        else:
            print(f"  Already absent: {os.path.basename(path)}")

    print()
    print(f"Done. {len(to_delete)} shards cleared. Ready for SLURM resubmit with PER_GENE_ON_FAILURE=1.")


if __name__ == "__main__":
    main()
