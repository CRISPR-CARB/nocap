r"""
coverage_reduce.py
==================
Final step of the parallelised coverage-matrix pipeline.

Globs all ``shard_*.json`` files from the shards directory, merges them
into a single deduplicated row list, and writes the final
``coverage_matrix.csv`` in the same format as the legacy serial script.

Usage:
  uv run python scripts/coverage_reduce.py \\
    --manifest   notebooks/Ecoli_Analysis_Notebooks/coverage_job.json \\
    --shards-dir notebooks/Ecoli_Analysis_Notebooks/shards \\
    --output     notebooks/Ecoli_Analysis_Notebooks/coverage_matrix.csv

Run this after all array tasks have completed (SLURM dependency:
  --dependency=afterok:<array_job_id>).
"""

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from coverage_common import merge_shards, rows_to_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Merge coverage-matrix shards into final CSV"
    )
    parser.add_argument("--manifest", required=True, help="Path to coverage_job.json")
    parser.add_argument(
        "--shards-dir",
        required=True,
        help="Directory containing shard_*.json files",
    )
    parser.add_argument(
        "--output",
        default="coverage_matrix.csv",
        help="Output CSV path (default: coverage_matrix.csv)",
    )
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    assert os.path.isfile(manifest_path), f"Manifest not found: {manifest_path}"
    with open(manifest_path) as f:
        manifest = json.load(f)

    unidentifiable = [tuple(q) for q in manifest["unidentifiable"]]
    query_labels = [f"{tf1}->{outcome}" for tf1, outcome in unidentifiable]

    # --- Collect shards ---
    shard_pattern = os.path.join(args.shards_dir, "shard_*.json")
    shard_files = sorted(glob.glob(shard_pattern))
    if not shard_files:
        print(f"ERROR: No shard files found matching: {shard_pattern}")
        sys.exit(1)
    print(f"Found {len(shard_files)} shard file(s).")

    shard_list = []
    for path in shard_files:
        with open(path) as f:
            ckpt = json.load(f)
        shard_list.append(ckpt["rows"])
        print(f"  {os.path.basename(path)}: {len(ckpt['rows'])} rows")

    # --- Merge ---
    rows = merge_shards(shard_list)
    print(f"Merged: {len(rows)} unique (tf1, candidate, outcome) triples.")

    # --- Build matrix ---
    candidate_set, query_labels_out, lookup = rows_to_matrix(rows, query_labels)

    # --- Write CSV ---
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_tf", *query_labels_out])
        for cand in candidate_set:
            row = [cand]
            for qlabel in query_labels_out:
                row.append(int(lookup.get((cand, qlabel), False)))
            writer.writerow(row)

    # --- Summary ---
    resolved_queries = set()
    for tf1, candidate, outcome, found in rows:
        if found:
            resolved_queries.add(f"{tf1}->{outcome}")

    print("\n--- Coverage Matrix Summary ---")
    print(f"Unidentifiable queries:          {len(unidentifiable)}")
    print(f"Queries resolvable by >= 1 TF:   {len(resolved_queries)}")
    print(f"Queries with no rescue found:    {len(unidentifiable) - len(resolved_queries)}")
    print(f"Candidate TFs in matrix:         {len(candidate_set)}")
    print(f"Output written to:               {output_path}")


if __name__ == "__main__":
    main()
