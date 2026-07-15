"""Gather node-packed CSD estimation CSVs into one CSV and summary.

Usage::

    uv run python scripts/csd_estimate_gather.py \
        --input-dir results/estimation/csv \
        --output-csv results/estimation/csd_estimates.csv \
        --output-summary results/estimation/csd_estimates_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ESTIMATE_FIELDS = [
    "trial",
    "n_samples",
    "missing_edge_rate",
    "missing_data_rate",
    "missing_data_mechanism",
    "seed",
    "cause",
    "effect",
    "same_scc",
    "status",
    "adjustment_set",
    "n_rows_used",
    "estimated_path_coefficient",
    "stderr",
    "residual_variance",
    "t_value",
    "ground_truth_beta",
    "scm_true_missing_edges_count",
    "error",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge packed CSD estimation CSVs and write summary statistics"
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing estimate CSVs")
    parser.add_argument("--output-csv", required=True, help="Merged CSV path")
    parser.add_argument("--output-summary", required=True, help="Summary JSON path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    output_summary = Path(args.output_summary)
    files = sorted(input_dir.glob("*.csv"))
    files = [p for p in files if p.resolve() != output_csv.resolve()]
    if not files:
        print(f"gather: no input CSVs found in {input_dir}", file=sys.stderr)
        raise SystemExit(1)

    rows: list[dict[str, str]] = []
    fieldnames = ESTIMATE_FIELDS
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                print(f"gather: missing header in {path}", file=sys.stderr)
                raise SystemExit(1)
            missing = [field for field in fieldnames if field not in reader.fieldnames]
            if missing:
                print(f"gather: {path} is missing columns: {', '.join(missing)}", file=sys.stderr)
                raise SystemExit(1)
            rows.extend(reader)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    statuses = Counter(row.get("status", "") for row in rows)
    summary = {
        "n_rows": len(rows),
        "n_input_files": len(files),
        "statuses": dict(sorted(statuses.items())),
        "n_identifiable": statuses.get("identifiable", 0),
        "n_unidentifiable": statuses.get("unidentifiable", 0),
        "n_estimation_error": statuses.get("estimation_error", 0),
        "n_insufficient_data": statuses.get("insufficient_data", 0),
        "pct_identifiable": round(100 * statuses.get("identifiable", 0) / len(rows), 2)
        if rows
        else 0.0,
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with output_summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"gather: {len(rows)} rows from {len(files)} files", file=sys.stderr)
    print(f"gather: CSV → {output_csv}", file=sys.stderr)
    print(f"gather: summary → {output_summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
