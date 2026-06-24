"""csd_calibrate_prepare.py — Sample timeout edges and write single-edge calibration shards.

Reads csd_results.csv, filters to status == 'timeout', randomly samples
SAMPLE_SIZE edges (default 128), and writes each as its own single-edge shard
in the standard {"shard_id", "edges"} format used by cyclic_single_door_classify.py.

Usage:
    uv run python scripts/csd_calibrate_prepare.py [--n 128] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent.parent
DEFAULT_CSV = REPO / "notebooks" / "Ecoli_Analysis_Notebooks" / "csd_results.csv"
DEFAULT_OUTDIR = REPO / "results" / "cyclic_single_door" / "calib_shards"
DEFAULT_N = 128
DEFAULT_SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="csd_results.csv path")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTDIR), help="Output shard dir")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="Number of edges to sample")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.output_dir)
    assert csv_path.exists(), f"PRE: csd_results.csv not found at {csv_path}"

    df = pd.read_csv(csv_path)
    timeout_df = df[df["status"] == "timeout"][["cause", "effect"]].copy()

    n_timeout = len(timeout_df)
    n_sample = min(args.n, n_timeout)
    assert n_timeout > 0, "PRE: no timeout edges found in csd_results.csv"

    rng = random.Random(args.seed)
    sample = timeout_df.sample(n=n_sample, random_state=args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (_, row) in enumerate(sample.iterrows()):
        shard = {
            "shard_id": f"calib_{i:04d}",
            "edges": [[row["cause"], row["effect"]]],
        }
        shard_path = out_dir / f"calib_{i:04d}.json"
        with open(shard_path, "w") as f:
            json.dump(shard, f)

    # Write manifest
    manifest = {
        "total_shards": n_sample,
        "source_csv": str(csv_path),
        "n_timeout_total": n_timeout,
        "n_sampled": n_sample,
        "seed": args.seed,
        "shard_dir": str(out_dir),
        "shard_ids": [f"calib_{i:04d}" for i in range(n_sample)],
    }
    manifest_path = out_dir.parent / "calib_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"calibrate prepare: sampled {n_sample} of {n_timeout} timeout edges.",
        flush=True,
    )
    print(f"calibrate prepare: {n_sample} shards written to {out_dir}", flush=True)
    print(f"calibrate prepare: manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
