"""
scc_perturb_reduce.py
=====================
Final step of the SCC-perturbation pipeline.

Globs all ``scc_perturb_shard_*.json`` files from the shards directory,
merges them, and writes the final ``scc_perturbation_results.csv``.

Usage:
  uv run python scripts/scc_perturb_reduce.py \\
    --manifest   notebooks/Ecoli_Analysis_Notebooks/scc_perturb_job.json \\
    --shards-dir notebooks/Ecoli_Analysis_Notebooks/scc_perturb_shards \\
    --output     notebooks/Ecoli_Analysis_Notebooks/scc_perturbation_results.csv

Run this after all array tasks have completed (SLURM dependency:
  --dependency=afterok:<array_job_id>).

Output CSV columns:
  tf, scc_size, min_cut_size, min_cut_nodes, n_descendants,
  joint_identifiable, n_per_gene_identifiable, pct_per_gene_identifiable,
  note
"""

import argparse
import csv
import glob
import json
import os
import sys


def load_shards(shards_dir: str) -> list:
    """
    Load all scc_perturb_shard_*.json files from *shards_dir*.

    Returns a list of shard dicts (one per TF).

    axiomander:
        requires:
            isinstance(shards_dir, str)
            len(shards_dir) > 0
        ensures:
            isinstance(result, list)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(shards_dir, str) and shards_dir, (
        "PRE: shards_dir must be a non-empty str"
    )

    pattern = os.path.join(shards_dir, "scc_perturb_shard_*.json")
    shard_files = sorted(glob.glob(pattern))

    shards: list = []
    for path in shard_files:
        with open(path) as f:
            shards.append(json.load(f))
        print(f"  {os.path.basename(path)}: tf={shards[-1].get('tf', '?')}, "
              f"joint={shards[-1].get('joint_identifiable', '?')}, "
              f"n_desc={shards[-1].get('n_descendants', '?')}")

    # --- POST ---
    assert isinstance(shards, list), "POST: result must be a list"
    return shards


def merge_and_write(
    shards: list,
    manifest: dict,
    output_path: str,
) -> None:
    """
    Merge shard list and write the final CSV.

    axiomander:
        requires:
            isinstance(shards, list)
            isinstance(manifest, dict)
            isinstance(output_path, str)
            len(output_path) > 0
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(shards, list), "PRE: shards must be a list"
    assert isinstance(manifest, dict), "PRE: manifest must be a dict"
    assert isinstance(output_path, str) and output_path, (
        "PRE: output_path must be a non-empty str"
    )

    # Check for duplicate TFs (shouldn't happen but guard)
    tfs_seen: set = set()
    deduped: list = []
    for shard in shards:
        tf = shard.get("tf", "")
        if tf not in tfs_seen:
            tfs_seen.add(tf)
            deduped.append(shard)

    # Sort by TF name for deterministic output
    deduped.sort(key=lambda s: s.get("tf", ""))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tf",
            "scc_size",
            "min_cut_size",
            "min_cut_nodes",
            "n_descendants",
            "joint_identifiable",
            "n_per_gene_identifiable",
            "pct_per_gene_identifiable",
            "note",
        ])
        for shard in deduped:
            tf = shard.get("tf", "")
            scc_size = shard.get("scc_size", 0)
            min_cut = shard.get("min_cut", [])
            n_desc = shard.get("n_descendants", 0)
            joint_id = shard.get("joint_identifiable", None)
            per_gene = shard.get("per_gene", {})
            note = shard.get("note", "")

            n_per_gene = sum(1 for v in per_gene.values() if v)
            pct = n_per_gene / n_desc * 100 if n_desc > 0 and per_gene else ""

            writer.writerow([
                tf,
                scc_size,
                len(min_cut),
                "+".join(sorted(min_cut)),
                n_desc,
                joint_id,
                n_per_gene if per_gene else "",
                f"{pct:.1f}" if pct != "" else "",
                note,
            ])

    print(f"\nCSV written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge SCC-perturbation shards into final CSV"
    )
    parser.add_argument("--manifest", required=True, help="Path to scc_perturb_job.json")
    parser.add_argument(
        "--shards-dir",
        required=True,
        help="Directory containing scc_perturb_shard_*.json files",
    )
    parser.add_argument(
        "--output",
        default="scc_perturbation_results.csv",
        help="Output CSV path (default: scc_perturbation_results.csv)",
    )
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    assert os.path.isfile(manifest_path), f"Manifest not found: {manifest_path}"
    with open(manifest_path) as f:
        manifest = json.load(f)

    n_tasks = manifest.get("n_tasks", 0)

    # --- Collect shards ---
    pattern = os.path.join(args.shards_dir, "scc_perturb_shard_*.json")
    shard_files = glob.glob(pattern)
    if not shard_files:
        print(f"ERROR: No shard files found matching: {pattern}")
        sys.exit(1)
    print(f"Found {len(shard_files)} shard file(s) (expected {n_tasks}).")
    if len(shard_files) < n_tasks:
        print(f"WARNING: Only {len(shard_files)}/{n_tasks} shards found — some tasks may not have completed.")

    print("Loading shards...")
    shards = load_shards(args.shards_dir)

    # --- Merge and write ---
    output_path = os.path.abspath(args.output)
    merge_and_write(shards, manifest, output_path)

    # --- Summary ---
    joint_id_count = sum(
        1 for s in shards if s.get("joint_identifiable") is True
    )
    joint_unid_count = sum(
        1 for s in shards if s.get("joint_identifiable") is False
    )
    no_desc_count = sum(
        1 for s in shards if s.get("note") == "no_descendants"
    )
    dag_tfs = manifest.get("dag_tfs", [])

    print("\n--- SCC Perturbation Results Summary ---")
    print(f"TFs with SCC perturbation tasks:     {n_tasks}")
    print(f"Shards processed:                    {len(shards)}")
    print(f"Jointly identifiable:                {joint_id_count}")
    print(f"Jointly unidentifiable:              {joint_unid_count}")
    print(f"No descendants (empty shard):        {no_desc_count}")
    print(f"DAG TFs (no perturbation needed):    {len(dag_tfs)}")
    print(f"Output written to:                   {output_path}")


if __name__ == "__main__":
    main()
