"""csd_calibrate_report.py — Summarise calibration results after the 20-min SLURM job.

Reads all calib_classified/calib_*.json files (output of classify --timeout 1200),
tallies status counts, and prints a go/no-go recommendation for the full rerun.

Usage:
    uv run python scripts/csd_calibrate_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).parent.parent
CALIB_CLASSIFIED_DIR = REPO / "results" / "cyclic_single_door" / "calib_classified"
CALIB_SHARDS_DIR = REPO / "results" / "cyclic_single_door" / "calib_shards"


def main() -> None:
    classified = sorted(CALIB_CLASSIFIED_DIR.glob("calib_*.json"))
    total_shards = len(list(CALIB_SHARDS_DIR.glob("calib_*.json")))

    if not classified:
        print("No calibration results found yet.")
        print(f"Expected output in: {CALIB_CLASSIFIED_DIR}")
        return

    counts: dict[str, int] = {"identifiable": 0, "unidentifiable": 0, "timeout": 0, "other": 0}
    for f in classified:
        with open(f) as fh:
            data = json.load(fh)
        for row in data.get("results", []):
            status = row.get("status", "other")
            if status in counts:
                counts[status] += 1
            else:
                counts["other"] += 1

    n_done = len(classified)
    n_pending = total_shards - n_done
    n_resolved = counts["identifiable"] + counts["unidentifiable"]
    n_timeout = counts["timeout"]
    n_total_classified = sum(counts.values())

    resolution_rate = n_resolved / n_total_classified if n_total_classified > 0 else 0.0
    ident_rate = counts["identifiable"] / n_total_classified if n_total_classified > 0 else 0.0

    print("=" * 60)
    print("CSD Calibration Report (--timeout 1200 / 20 min)")
    print("=" * 60)
    print(f"  Calib shards total  : {total_shards}")
    print(f"  Results available   : {n_done}  ({n_pending} still pending)")
    print()
    print(f"  identifiable        : {counts['identifiable']}")
    print(f"  unidentifiable      : {counts['unidentifiable']}")
    print(f"  timeout (still)     : {counts['timeout']}")
    if counts["other"]:
        print(f"  other               : {counts['other']}")
    print()
    print(f"  Resolution rate     : {resolution_rate:.1%}  ({n_resolved}/{n_total_classified})")
    print(
        f"  Identifiable rate   : {ident_rate:.1%}  ({counts['identifiable']}/{n_total_classified})"
    )
    print()

    # --- Extrapolation to full 7,442-edge pool ---
    N_FULL = 7442
    est_resolved = round(resolution_rate * N_FULL)
    est_ident = round(ident_rate * N_FULL)

    print(f"  Extrapolated to full {N_FULL} timeout edges:")
    print(f"    ~{est_resolved} would resolve  (~{est_ident} identifiable)")
    print(f"    ~{N_FULL - est_resolved} would still time out at 1200s")
    print()

    # --- Go / No-go recommendation ---
    if resolution_rate >= 0.20:
        print("RECOMMENDATION: GO — >=20% resolve at 1200s.")
        print("  Run: bash scripts/slurm/submit_csd_timeout_rerun.sh")
    elif resolution_rate >= 0.05:
        print("RECOMMENDATION: MARGINAL — 5–20% resolve.")
        print("  Consider a longer timeout (e.g. 3600s) or restricting to")
        print("  edges outside the giant SCC before a full rerun.")
    else:
        print("RECOMMENDATION: NO-GO — <5% resolve at 1200s.")
        print("  Bottleneck is structural (large O-sets in the giant SCC).")
        print("  Document the timeout limitation in the notebook instead.")
    print("=" * 60)


if __name__ == "__main__":
    main()
