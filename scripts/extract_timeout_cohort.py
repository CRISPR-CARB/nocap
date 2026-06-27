"""extract_timeout_cohort.py — Extract timed-out edges from csd_results.csv.

Reads notebooks/Ecoli_Analysis_Notebooks/csd_results.csv (or --input),
filters rows where status == 'timeout', and writes a shard-format JSON file
suitable for direct consumption by:

    cyclic_single_door_classify.py classify --shard <output> ...

Output format:
    {"shard_id": "timeout_cohort", "edges": [[cause, effect], ...]}

This avoids the need for a --restrict-edges flag on the classifier; the
shard format is the native classifier input and triggers a single
sigma-extension build reused for all edges in the shard.

Usage
-----
    uv run python scripts/extract_timeout_cohort.py
    uv run python scripts/extract_timeout_cohort.py \\
        --input notebooks/Ecoli_Analysis_Notebooks/csd_results.csv \\
        --output results/timeout_cohort_shard.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract timeout edges from csd_results.csv into shard format JSON",
    )
    parser.add_argument(
        "--input",
        default="notebooks/Ecoli_Analysis_Notebooks/csd_results.csv",
        help="Input CSV with columns: cause,effect,status,...",
    )
    parser.add_argument(
        "--output",
        default="results/timeout_cohort_shard.json",
        help="Output shard-format JSON path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    timeout_edges: list[list[str]] = []
    total_rows = 0

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if row.get("status") == "timeout":
                timeout_edges.append([row["cause"], row["effect"]])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shard = {
        "shard_id": "timeout_cohort",
        "edges": timeout_edges,
    }

    with open(output_path, "w") as f:
        json.dump(shard, f)

    n_timeout = len(timeout_edges)
    pct = 100.0 * n_timeout / total_rows if total_rows > 0 else 0.0

    print(f"Input  : {input_path}  ({total_rows} total edges)")
    print(f"Timeout: {n_timeout} ({pct:.1f}%)")
    print(f"Output : {output_path}")
    print()
    if n_timeout == 0:
        print("WARNING: no timeout edges found — check --input path and column names")
    else:
        print(f"Shard ready: {n_timeout} edges written in shard format.")
        print("Next step:")
        print(f"  uv run python scripts/cyclic_single_door_classify.py classify \\")
        print(f"      --graphml notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml \\")
        print(f"      --shard {output_path} \\")
        print(f"      --output results/timeout_cohort_reclassified.json \\")
        print(f"      --timeout 60")


if __name__ == "__main__":
    main()
