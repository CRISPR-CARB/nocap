"""csd_identified_edges.py — Live summarizer for the cyclic-single-door sweep.

Reads completed classified shards from results/cyclic_single_door/classified/
WITHOUT touching running jobs.  Works mid-run (partial shards OK).

Modes
-----
Default (no flag):
    Print aggregate counts: total edges seen so far, identifiable / unidentifiable,
    same-SCC breakdown, and % of shards completed.

--list:
    Stream every identifiable edge as:
        cause  →  effect    adj={node1, node2, ...}  [same_scc]

--errors:
    Cross-reference shard manifest vs classified/ to list missing/failed shard IDs.
    Also grep .err logs for Tracebacks and print the last 20 lines of each.

--csv PATH:
    Dump the partial identified-edge table to a CSV right now (no need to wait for
    the full gather step).  Columns: cause, effect, status, adjustment_set, same_scc.

Usage
-----
    # Aggregate counts (default)
    uv run python scripts/csd_identified_edges.py

    # List every identified edge so far
    uv run python scripts/csd_identified_edges.py --list | head -40

    # Check what's failing
    uv run python scripts/csd_identified_edges.py --errors

    # Partial CSV dump
    uv run python scripts/csd_identified_edges.py --csv /tmp/partial_edges.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_classified_dir() -> Path:
    """Return the default classified-shards directory relative to this script."""
    return Path(__file__).parent.parent / "results" / "cyclic_single_door" / "classified"


def _default_manifest_path() -> Path:
    return Path(__file__).parent.parent / "results" / "cyclic_single_door" / "shard_manifest.json"


def _default_log_dir() -> Path:
    return Path(__file__).parent.parent / "results" / "cyclic_single_door" / "logs"


def _load_all_results(classified_dir: Path) -> list[dict]:
    """Load every result record from all classified shards present on disk."""
    records: list[dict] = []
    for shard_file in sorted(classified_dir.glob("shard_*.json")):
        try:
            with open(shard_file) as f:
                data = json.load(f)
            for r in data.get("results", []):
                records.append(r)
        except (json.JSONDecodeError, KeyError):
            # Shard partially written or corrupt — skip silently
            pass
    return records


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def cmd_summary(classified_dir: Path, manifest_path: Path) -> None:
    """Print aggregate counts (default mode)."""
    records = _load_all_results(classified_dir)

    n_identifiable = sum(1 for r in records if r.get("status") == "identifiable")
    n_unidentifiable = sum(1 for r in records if r.get("status") == "unidentifiable")
    n_total = len(records)
    n_same_scc = sum(1 for r in records if r.get("same_scc"))
    n_same_scc_ident = sum(
        1 for r in records if r.get("same_scc") and r.get("status") == "identifiable"
    )

    # Shard counts
    n_classified_shards = len(list(classified_dir.glob("shard_*.json")))
    n_total_shards: int | str = "?"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            n_total_shards = manifest.get("n_shards", "?")
        except (json.JSONDecodeError, KeyError):
            pass

    pct_shards = (
        f"{n_classified_shards / n_total_shards * 100:.1f}%"
        if isinstance(n_total_shards, int) and n_total_shards > 0
        else "?%"
    )

    print(f"  Shards classified : {n_classified_shards} / {n_total_shards}  ({pct_shards})")
    print(f"  Edges seen so far : {n_total}")
    if n_total > 0:
        pct_id = n_identifiable / n_total * 100
        print(f"  Identifiable      : {n_identifiable}  ({pct_id:.1f}%)")
        print(f"  Unidentifiable    : {n_unidentifiable}  ({100 - pct_id:.1f}%)")
        print(f"  Same-SCC edges    : {n_same_scc}  ({n_same_scc_ident} identifiable)")
    else:
        print("  (no edge records yet)")


def cmd_list(classified_dir: Path, same_scc_only: bool = False) -> None:
    """Stream every identifiable edge to stdout."""
    records = _load_all_results(classified_dir)
    identifiable = [r for r in records if r.get("status") == "identifiable"]
    if same_scc_only:
        identifiable = [r for r in identifiable if r.get("same_scc")]

    if not identifiable:
        print("(no identifiable edges in completed shards yet)")
        return

    # Sort by cause then effect for stable output
    identifiable.sort(key=lambda r: (r.get("cause", ""), r.get("effect", "")))

    for r in identifiable:
        cause = r.get("cause", "?")
        effect = r.get("effect", "?")
        adj = r.get("adjustment_set")
        adj_str = "{" + ", ".join(sorted(adj)) + "}" if adj else "{}"
        scc_tag = "  [same-SCC]" if r.get("same_scc") else ""
        print(f"{cause}  ->  {effect}    adj={adj_str}{scc_tag}")


def cmd_errors(
    classified_dir: Path,
    manifest_path: Path,
    log_dir: Path,
) -> None:
    """Cross-reference manifest vs disk; show tracebacks from .err logs."""
    # --- Missing classified shards ---
    classified_ids: set[str] = set()
    for f in classified_dir.glob("shard_*.json"):
        sid = f.stem.removeprefix("shard_")
        classified_ids.add(sid)

    missing_ids: list[str] = []
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            all_ids = [str(sid) for sid in manifest.get("shard_ids", [])]
            missing_ids = [sid for sid in all_ids if sid not in classified_ids]
        except (json.JSONDecodeError, KeyError):
            print("WARNING: could not parse manifest", file=sys.stderr)

    print(f"Missing/unclassified shards: {len(missing_ids)}")
    if missing_ids:
        for sid in sorted(missing_ids, key=lambda x: int(x) if x.isdigit() else x):
            print(f"  shard_{sid}")
    else:
        print("  (none — all manifest shards have a classified output)")
    print()

    # --- FAIL markers in stdout logs ---
    print("=== [FAIL] markers in packed job logs ===")
    fail_found = False
    for log_file in sorted(log_dir.glob("packed_*.out")):
        try:
            lines = log_file.read_text(errors="replace").splitlines()
        except OSError:
            continue
        fail_lines = [ln for ln in lines if "[FAIL ]" in ln]
        if fail_lines:
            fail_found = True
            print(f"  {log_file.name}:")
            for ln in fail_lines:
                print(f"    {ln}")
    if not fail_found:
        print("  No [FAIL] markers found.")
    print()

    # --- Python tracebacks in .err logs ---
    print("=== Python Tracebacks in .err logs ===")
    tb_found = False
    for err_file in sorted(log_dir.glob("packed_*.err")):
        try:
            text = err_file.read_text(errors="replace")
        except OSError:
            continue
        if "Traceback" in text:
            tb_found = True
            lines = text.splitlines()
            print(f"  --- {err_file.name} (last 20 lines) ---")
            for ln in lines[-20:]:
                print(f"    {ln}")
            print()
    if not tb_found:
        print("  No Traceback found in any .err log so far.")


def cmd_csv(classified_dir: Path, output_path: Path) -> None:
    """Dump all classified results (partial) to a CSV."""
    records = _load_all_results(classified_dir)
    if not records:
        print("No records found — nothing to write.", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["cause", "effect", "status", "adjustment_set", "same_scc"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = dict(r)
            # Serialise adjustment_set list → space-separated string
            adj = row.get("adjustment_set")
            if isinstance(adj, list):
                row["adjustment_set"] = " ".join(sorted(adj)) if adj else ""
            elif adj is None:
                row["adjustment_set"] = ""
            writer.writerow(row)

    n_id = sum(1 for r in records if r.get("status") == "identifiable")
    print(f"Wrote {len(records)} rows ({n_id} identifiable) to {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live summarizer for cyclic-single-door classify sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--classified-dir",
        type=Path,
        default=None,
        help="Directory of classified shard JSONs (default: results/cyclic_single_door/classified)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Shard manifest JSON (default: results/cyclic_single_door/shard_manifest.json)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory containing packed_*.out/.err logs (default: results/cyclic_single_door/logs)",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--list",
        action="store_true",
        help="Stream every identifiable edge (cause -> effect, adj set)",
    )
    group.add_argument(
        "--list-same-scc",
        action="store_true",
        help="Like --list but restrict to same-SCC edges only",
    )
    group.add_argument(
        "--errors",
        action="store_true",
        help="Show missing shards + tracebacks from .err logs",
    )
    group.add_argument(
        "--csv",
        type=Path,
        metavar="PATH",
        default=None,
        help="Dump partial edge table to CSV at PATH",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    classified_dir = args.classified_dir or _default_classified_dir()
    manifest_path = args.manifest or _default_manifest_path()
    log_dir = args.log_dir or _default_log_dir()

    if not classified_dir.exists():
        print(
            f"ERROR: classified dir does not exist: {classified_dir}\n"
            "Run the classify sweep first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.list:
        cmd_list(classified_dir, same_scc_only=False)
    elif args.list_same_scc:
        cmd_list(classified_dir, same_scc_only=True)
    elif args.errors:
        cmd_errors(classified_dir, manifest_path, log_dir)
    elif args.csv is not None:
        cmd_csv(classified_dir, args.csv)
    else:
        cmd_summary(classified_dir, manifest_path)


if __name__ == "__main__":
    main()
