"""cyclic_single_door_gather.py — Merge classified shard JSONs into CSV + summary.

Called by the Snakemake 'gather' rule after all 'classify' shards complete.

Usage
-----
    python scripts/cyclic_single_door_gather.py \\
        --input-dir results/cyclic_single_door/classified \\
        --output-csv results/cyclic_single_door/classification_results.csv \\
        --output-summary results/cyclic_single_door/classification_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge classified shard JSONs into a single CSV and summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing classified shard_*.json files",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV path (one row per edge)",
    )
    parser.add_argument(
        "--output-summary",
        required=True,
        help="Output summary JSON path",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    shard_files = sorted(input_dir.glob("shard_*.json"))

    if not shard_files:
        print(
            f"gather: no shard files found in {input_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    all_rows: list[dict] = []
    for shard_path in shard_files:
        with open(shard_path) as f:
            shard_data = json.load(f)
        all_rows.extend(shard_data["results"])

    # Write CSV
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["cause", "effect", "status", "adjustment_set", "same_scc", "timed_out"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            # adjustment_set is a list (from JSON) or None; join for CSV
            adj = row.get("adjustment_set")
            row_out = {k: row.get(k) for k in fieldnames}
            row_out["adjustment_set"] = "|".join(sorted(adj)) if adj is not None else ""
            row_out["timed_out"] = row.get("timed_out", False)
            writer.writerow(row_out)

    # Compute summary statistics
    n_total = len(all_rows)
    n_identifiable = sum(1 for r in all_rows if r["status"] == "identifiable")
    n_unidentifiable = sum(1 for r in all_rows if r["status"] == "unidentifiable")
    n_timeout = sum(1 for r in all_rows if r["status"] == "timeout")
    n_same_scc = sum(1 for r in all_rows if r.get("same_scc") is True)

    summary = {
        "n_edges": n_total,
        "n_identifiable": n_identifiable,
        "n_unidentifiable": n_unidentifiable,
        "n_timeout": n_timeout,
        "pct_identifiable": round(100.0 * n_identifiable / n_total, 2) if n_total > 0 else 0.0,
        "n_same_scc": n_same_scc,
        "n_shards": len(shard_files),
    }

    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"gather: {n_total} edges, {n_identifiable} identifiable "
        f"({summary['pct_identifiable']}%), {n_same_scc} same-SCC",
        file=sys.stderr,
    )
    print(f"gather: CSV → {csv_path}", file=sys.stderr)
    print(f"gather: summary → {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
