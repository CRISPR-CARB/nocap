"""csd_rescue_prepare.py — Shard unidentifiable edges for the rescue sweep.

Reads the classification CSV produced by cyclic_single_door_gather.py, filters
to unidentifiable edges, splits them into shards, and writes a manifest.

Usage
-----
    python scripts/csd_rescue_prepare.py \\
        --results-csv notebooks/Ecoli_Analysis_Notebooks/csd_results.csv \\
        --shard-dir results/cyclic_single_door/rescue_shards \\
        --manifest results/cyclic_single_door/rescue_manifest.json \\
        --shard-size 50
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _load_unidentifiable_edges(csv_path: Path) -> list[tuple[str, str]]:
    """Return (cause, effect) pairs where status is unidentifiable."""
    edges = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "unidentifiable":
                edges.append((row["cause"], row["effect"]))
    return edges


def _split_shards(edges: list, shard_size: int, n_shards: int | None) -> list[list]:
    """Split edges into shards of approximately shard_size (or exactly n_shards shards)."""
    if not edges:
        return []
    if n_shards is not None:
        n = min(n_shards, len(edges))
    else:
        n = max(1, math.ceil(len(edges) / shard_size))
    n = min(n, len(edges))
    base, extra = divmod(len(edges), n)
    chunks = []
    i = 0
    for j in range(n):
        size = base + (1 if j < extra else 0)
        chunks.append(edges[i : i + size])
        i += size
    return chunks


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-csv", required=True, help="csd_results.csv from gather step.")
    p.add_argument("--shard-dir", required=True, help="Directory to write shard JSON files.")
    p.add_argument("--manifest", required=True, help="Output manifest JSON path.")
    p.add_argument("--shard-size", type=int, default=50, help="Edges per shard (default 50).")
    p.add_argument("--n-shards", type=int, default=None, help="Override: total number of shards.")
    args = p.parse_args()

    results_csv = Path(args.results_csv)
    shard_dir = Path(args.shard_dir)
    manifest_path = Path(args.manifest)

    edges = _load_unidentifiable_edges(results_csv)
    print(f"rescue prepare: {len(edges)} unidentifiable edges to process.")

    shards = _split_shards(edges, args.shard_size, args.n_shards)
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_ids = []
    for idx, chunk in enumerate(shards):
        sid = str(idx)
        shard_file = shard_dir / f"shard_{sid}.json"
        with open(shard_file, "w") as f:
            json.dump({"shard_id": sid, "edges": [list(e) for e in chunk]}, f)
        shard_ids.append(sid)

    manifest = {
        "n_unidentifiable": len(edges),
        "n_shards": len(shards),
        "shard_ids": shard_ids,
        "shard_size": args.shard_size,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"rescue prepare: {len(shards)} shards written to {shard_dir}",
    )
    print(f"rescue prepare: manifest → {manifest_path}")


if __name__ == "__main__":
    main()
