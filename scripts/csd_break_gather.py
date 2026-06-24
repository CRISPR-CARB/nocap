"""csd_break_gather.py — Collect break-sweep shards into a single CSV + summary.

Reads all classified shard JSON files produced by csd_break_worker.py and
merges them into:
  - csd_break_results.csv   (one row per unidentifiable edge)
  - csd_break_summary.json  (aggregate statistics)

Usage
-----
    python scripts/csd_break_gather.py \\
        --input-dir results/cyclic_single_door/break_classified \\
        --output-csv notebooks/Ecoli_Analysis_Notebooks/csd_break_results.csv \\
        --output-summary notebooks/Ecoli_Analysis_Notebooks/csd_break_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-dir", required=True, help="Directory of classified shard JSON files.")
    p.add_argument("--output-csv", required=True, help="Output CSV path.")
    p.add_argument("--output-summary", required=True, help="Output summary JSON path.")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    output_summary = Path(args.output_summary)

    shard_files = sorted(input_dir.glob("shard_*.json"))
    if not shard_files:
        print(f"break gather: no shard files found in {input_dir}")
        return

    rows = []
    k_values: list[int] = []
    n_shards_loaded = 0

    for shard_file in shard_files:
        with open(shard_file) as f:
            shard = json.load(f)
        k_values.append(shard.get("k", 3))
        for row in shard.get("results", []):
            rows.append({
                "cause":                 row["cause"],
                "effect":                row["effect"],
                "nonident_cause":        row["nonident_cause"],
                "same_scc_after_removal": row["same_scc_after_removal"],
                "needs_intervention":    row["needs_intervention"],
                "min_break_set":         json.dumps(row["min_break_set"]),
                "min_break_size":        row["min_break_size"],
                "rescuable_within_k":    row["rescuable_within_k"],
                "cut_verified":          row["cut_verified"],
            })
        n_shards_loaded += 1

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"break gather: {len(df):,} rows written to {output_csv}")

    # --- Summary statistics ---
    k = max(k_values) if k_values else 3

    n_total = len(df)
    n_no_intervention = int((~df["needs_intervention"]).sum())
    n_needs_intervention = int(df["needs_intervention"].sum())
    n_rescuable = int(df["rescuable_within_k"].sum())
    n_not_rescuable = n_total - n_rescuable

    break_size_counts: dict[str, int] = {
        str(k): int(v)
        for k, v in Counter(df["min_break_size"].tolist()).items()
    }
    nonident_cause_counts: dict[str, int] = {
        str(k): int(v)
        for k, v in Counter(df["nonident_cause"].tolist()).items()
    }
    n_cut_verified = int(df["cut_verified"].sum())
    n_cut_not_verified = n_total - n_cut_verified

    summary = {
        "n_unidentifiable_edges": n_total,
        "k_budget": k,
        "n_no_intervention_needed":   n_no_intervention,
        "n_needs_intervention":        n_needs_intervention,
        "n_rescuable_within_k":        n_rescuable,
        "n_not_rescuable_within_k":    n_not_rescuable,
        "pct_rescuable":               round(100.0 * n_rescuable / n_total, 2) if n_total else 0.0,
        "break_size_distribution":     break_size_counts,
        "nonident_cause_distribution": nonident_cause_counts,
        "n_cut_verified":              n_cut_verified,
        "n_cut_not_verified":          n_cut_not_verified,
        "n_shards_loaded":             n_shards_loaded,
    }

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"break gather: summary written to {output_summary}")
    print(f"  no_intervention_needed : {n_no_intervention:,}")
    print(f"  needs_intervention     : {n_needs_intervention:,}")
    print(f"  rescuable within k={k}  : {n_rescuable:,} ({100.0*n_rescuable/n_total:.1f}%)")
    print(f"  cut_verified           : {n_cut_verified:,} / {n_total:,}")


if __name__ == "__main__":
    main()
