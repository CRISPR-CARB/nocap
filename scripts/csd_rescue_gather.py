"""csd_rescue_gather.py — Merge rescue shard JSONs into CSV + summary.

Usage
-----
    python scripts/csd_rescue_gather.py \\
        --input-dir results/cyclic_single_door/rescue_classified \\
        --output-csv notebooks/Ecoli_Analysis_Notebooks/csd_rescue_results.csv \\
        --output-summary notebooks/Ecoli_Analysis_Notebooks/csd_rescue_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-dir", required=True, help="Directory of rescue shard JSONs.")
    p.add_argument("--output-csv", required=True, help="Output CSV path.")
    p.add_argument("--output-summary", required=True, help="Output summary JSON path.")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    shard_files = sorted(input_dir.glob("shard_*.json"))

    if not shard_files:
        print(f"rescue gather: no shard files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    all_rows: list[dict] = []
    for sf in shard_files:
        with open(sf) as f:
            data = json.load(f)
        all_rows.extend(data["results"])

    # Write CSV
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["cause", "effect", "nonident_cause", "rescue_nodes", "n_rescue_nodes"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            row_out = dict(row)
            nodes = row_out.get("rescue_nodes") or []
            row_out["rescue_nodes"] = "|".join(nodes) if nodes else ""
            writer.writerow(row_out)

    # Summary
    n_total = len(all_rows)
    from collections import Counter
    cause_counts = Counter(r["nonident_cause"] for r in all_rows)
    n_rescuable = sum(1 for r in all_rows if r.get("n_rescue_nodes", 0) > 0)
    rescue_node_counter: Counter = Counter()
    for r in all_rows:
        for node in (r.get("rescue_nodes") or []):
            rescue_node_counter[node] += 1
    top_rescue = rescue_node_counter.most_common(20)

    summary = {
        "n_unidentifiable": n_total,
        "n_rescuable": n_rescuable,
        "pct_rescuable": round(100.0 * n_rescuable / n_total, 2) if n_total > 0 else 0.0,
        "cause_counts": dict(cause_counts),
        "top_rescue_nodes": [{"node": n, "count": c} for n, c in top_rescue],
        "n_shards": len(shard_files),
    }
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"rescue gather: {n_total} rows, {n_rescuable} rescuable ({summary['pct_rescuable']}%)",
        file=sys.stderr,
    )
    print(f"rescue gather: CSV → {csv_path}", file=sys.stderr)
    print(f"rescue gather: summary → {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
