"""reconcile_csd.py — Strict diff between old and new CSD classification results.

Compares the original csd_results.csv (before the fast O-set rerun) against
the new classification_results.csv produced by the full Snakemake rerun.

Pass condition: zero regressions (no non-timeout edge changed verdict).
Allowed changes: timeout -> identifiable or timeout -> unidentifiable.

Exit codes
----------
0  No regressions — rerun is valid.
1  Regressions found — investigate before publishing.
2  Input file(s) not found.

Usage
-----
    uv run python scripts/reconcile_csd.py
    uv run python scripts/reconcile_csd.py \\
        --old notebooks/Ecoli_Analysis_Notebooks/csd_results.csv \\
        --new results/cyclic_single_door/classification_results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_csv(path: Path) -> dict[tuple[str, str], str]:
    """Return {(cause, effect): status} from a classification CSV."""
    mapping: dict[tuple[str, str], str] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["cause"], row["effect"])
            mapping[key] = row["status"]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict diff: verify no non-timeout verdicts changed between runs",
    )
    parser.add_argument(
        "--old",
        default="notebooks/Ecoli_Analysis_Notebooks/csd_results.csv",
        help="Old classification CSV (before fast O-set rerun)",
    )
    parser.add_argument(
        "--new",
        default="results/cyclic_single_door/classification_results.csv",
        help="New classification CSV (after fast O-set rerun)",
    )
    args = parser.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)

    for p in (old_path, new_path):
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(2)

    old = load_csv(old_path)
    new = load_csv(new_path)

    # Classify changes
    regressions: list[tuple[tuple[str, str], str, str]] = []  # (key, old_status, new_status)
    resolved: list[tuple[tuple[str, str], str]] = []           # (key, new_status)
    still_timeout: list[tuple[str, str]] = []
    missing_in_new: list[tuple[str, str]] = []
    new_in_new: list[tuple[str, str]] = []

    all_keys = set(old.keys()) | set(new.keys())

    for key in sorted(all_keys):
        old_status = old.get(key)
        new_status = new.get(key)

        if old_status is None:
            new_in_new.append(key)
            continue
        if new_status is None:
            missing_in_new.append(key)
            continue

        if old_status == new_status:
            continue  # unchanged — expected for identifiable/unidentifiable

        if old_status == "timeout" and new_status != "timeout":
            resolved.append((key, new_status))
        elif old_status == "timeout" and new_status == "timeout":
            still_timeout.append(key)
        else:
            # Non-timeout verdict changed — this is a regression
            regressions.append((key, old_status, new_status))

    # Report
    print(f"Old file : {old_path}  ({len(old)} edges)")
    print(f"New file : {new_path}  ({len(new)} edges)")
    print()
    print(f"Resolved timeouts        : {len(resolved)}")
    print(f"Still timing out         : {len(still_timeout)}")
    print(f"REGRESSIONS (non-timeout): {len(regressions)}")
    if missing_in_new:
        print(f"Missing in new           : {len(missing_in_new)}")
    if new_in_new:
        print(f"New edges in new run     : {len(new_in_new)}")
    print()

    if resolved:
        # Show breakdown of how timeouts resolved
        n_ident = sum(1 for _, s in resolved if s == "identifiable")
        n_unident = sum(1 for _, s in resolved if s == "unidentifiable")
        print(f"Resolved breakdown: {n_ident} identifiable, {n_unident} unidentifiable")
        print()

    if regressions:
        print("REGRESSIONS (first 20):")
        for (cause, effect), old_s, new_s in regressions[:20]:
            print(f"  {cause} -> {effect}:  {old_s} -> {new_s}")
        if len(regressions) > 20:
            print(f"  ... ({len(regressions) - 20} more)")
        print()
        print("FAIL: verdict regressions found — investigate the fast O-set rewrite.")
        sys.exit(1)

    if missing_in_new:
        print("WARNING: some old edges are missing from new results:")
        for cause, effect in missing_in_new[:10]:
            print(f"  {cause} -> {effect}")
        if len(missing_in_new) > 10:
            print(f"  ... ({len(missing_in_new) - 10} more)")
        print()

    print("PASS: zero regressions — fast O-set rerun is verdict-consistent.")
    if still_timeout:
        print(f"NOTE: {len(still_timeout)} edges still timing out after rerun.")
    sys.exit(0)


if __name__ == "__main__":
    main()
