"""check_residual_timeouts.py — Gate check after Phase A cohort reclassification.

Reads results/timeout_cohort_reclassified.json (or --input) and counts how
many edges still have status == 'timeout'.

Exit codes
----------
0  All edges resolved (no residual timeouts) — PASS, proceed to Phase B.
1  Some edges still timing out — WARN / investigate before full rerun.
2  Input file not found or malformed — check path.

Usage
-----
    uv run python scripts/check_residual_timeouts.py
    uv run python scripts/check_residual_timeouts.py \\
        --input results/timeout_cohort_reclassified.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate check: count residual timeouts after cohort reclassification",
    )
    parser.add_argument(
        "--input",
        default="results/timeout_cohort_reclassified.json",
        help="Classified shard JSON output from cyclic_single_door_classify.py classify",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        print("       Run the Phase A classify command first.", file=sys.stderr)
        sys.exit(2)

    try:
        data = json.loads(input_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: malformed JSON in {input_path}: {exc}", file=sys.stderr)
        sys.exit(2)

    results = data.get("results", [])
    n_total = len(results)

    if n_total == 0:
        print("ERROR: no results found in input file", file=sys.stderr)
        sys.exit(2)

    n_identifiable = sum(1 for r in results if r.get("status") == "identifiable")
    n_unidentifiable = sum(1 for r in results if r.get("status") == "unidentifiable")
    n_timeout = sum(1 for r in results if r.get("status") == "timeout")

    print(f"Cohort reclassification results ({n_total} edges):")
    print(f"  identifiable   : {n_identifiable:>6}  ({100 * n_identifiable / n_total:.1f}%)")
    print(f"  unidentifiable : {n_unidentifiable:>6}  ({100 * n_unidentifiable / n_total:.1f}%)")
    print(f"  still timeout  : {n_timeout:>6}  ({100 * n_timeout / n_total:.1f}%)")
    print()

    if n_timeout == 0:
        print("PASS: all cohort edges resolved — proceed to Phase B (full SLURM rerun).")
        sys.exit(0)
    else:
        print(f"WARN: {n_timeout} edges still timing out.")
        print()
        # Print the first few residual timeouts to help diagnose
        residual = [(r["cause"], r["effect"]) for r in results if r.get("status") == "timeout"]
        print("First up to 20 residual timeout edges:")
        for cause, effect in residual[:20]:
            print(f"  {cause} -> {effect}")
        if len(residual) > 20:
            print(f"  ... ({len(residual) - 20} more)")
        print()
        print("ACTION: investigate these edges before proceeding to Phase B.")
        print("        Consider increasing --timeout or checking the y0 O-set implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
