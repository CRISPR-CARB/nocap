"""
residual_scc_experiment.py
==========================
Phase A: Deterministic falsification test of the residual-SCC hypothesis.

Hypothesis
----------
Residual child-cycle present (>= 2 children sharing a non-trivial SCC after
do(B(t))) <=> joint cyclic_id query unidentifiable.

This test has two arms, each checked as an exact categorical claim:

  Check 1 (necessity): every UNIDENTIFIABLE TF must have >= 1 residual cluster.
      Violation: any unidentifiable TF with zero residual clusters — including
      single-child unidentifiable TFs (by definition no child-pair SCC exists).

  Check 2 (specificity): every IDENTIFIABLE TF must have zero residual clusters.
      Violation: any identifiable TF with >= 1 residual cluster.

A single violation in either check falsifies the hypothesis.

Verdict
-------
  SUPPORTED  — both violation cells empty  -> Phase B (augmented-cut reruns) justified.
  DISPROVEN  — any violation found         -> violating TFs listed; mechanism unclear.

Inputs
------
  --manifest    path to scc_perturb_job.json   (default: notebooks/.../scc_perturb_job.json)
  --shards-dir  path to scc_perturb_shards/     (default: notebooks/.../scc_perturb_shards)
  --out-dir     directory for CSV outputs       (default: notebooks/Ecoli_Analysis_Notebooks)

Outputs
-------
  residual_scc_summary.csv      — one row per TF; all structural diagnostics
  residual_cluster_sizes.csv    — per-TF per-cluster child-count (long format)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import networkx as nx

from nocap.scc_perturb import (
    residual_scc_analysis,
    residual_cluster_size_distribution,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_graph(graphml_path: str) -> nx.DiGraph:
    assert os.path.isfile(graphml_path), f"graphml not found: {graphml_path}"
    raw = nx.read_graphml(graphml_path)
    if not isinstance(raw, nx.DiGraph):
        raw = nx.DiGraph(raw)
    return raw


def load_shard(shard_path: str) -> dict:
    with open(shard_path) as f:
        return json.load(f)


def write_csv(path: str, rows: list, fieldnames: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase A: residual-SCC falsification experiment"
    )
    parser.add_argument(
        "--manifest",
        default="notebooks/Ecoli_Analysis_Notebooks/scc_perturb_job.json",
        help="Path to scc_perturb_job.json",
    )
    parser.add_argument(
        "--shards-dir",
        default="notebooks/Ecoli_Analysis_Notebooks/scc_perturb_shards",
        help="Directory containing scc_perturb_shard_<tf>.json files",
    )
    parser.add_argument(
        "--out-dir",
        default="notebooks/Ecoli_Analysis_Notebooks",
        help="Directory to write CSV outputs",
    )
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    shards_dir = os.path.abspath(args.shards_dir)
    out_dir = os.path.abspath(args.out_dir)

    assert os.path.isfile(manifest_path), f"Manifest not found: {manifest_path}"
    assert os.path.isdir(shards_dir), f"Shards dir not found: {shards_dir}"

    with open(manifest_path) as f:
        manifest = json.load(f)

    graphml_path: str = manifest["graphml"]
    print(f"Loading graph from {graphml_path} ...")
    graph = load_graph(graphml_path)
    print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    tasks: list = manifest["tasks"]
    print(f"Manifest has {len(tasks)} tasks.\n")

    # --- Accumulate per-TF results ---
    summary_rows: list = []
    cluster_size_rows: list = []

    n_identifiable = 0
    n_unidentifiable = 0
    n_no_shard = 0
    n_null_joint = 0  # joint_identifiable == None (unexpected per our reasoning)

    # Falsification tracking
    violations_check1: list = []   # unidentifiable + no residual cluster
    violations_check2: list = []   # identifiable + has residual cluster

    for task in tasks:
        tf: str = task["tf"]
        min_cut: list = task["min_cut"]

        shard_path = os.path.join(shards_dir, f"scc_perturb_shard_{tf}.json")
        if not os.path.exists(shard_path):
            print(f"  [WARN] No shard for {tf} — skipping.")
            n_no_shard += 1
            continue

        shard = load_shard(shard_path)
        joint_identifiable = shard.get("joint_identifiable")

        # Guard: None should not occur per our analysis (B(t) preserves direct children)
        if joint_identifiable is None:
            n_null_joint += 1
            print(
                f"  [ERROR] {tf}: joint_identifiable is None "
                f"(note={shard.get('note', '')!r}) — unexpected; treating as error."
            )
            # Hard error: we committed to treating these as errors, not silent buckets
            sys.exit(
                f"HARD ERROR: shard for {tf} has joint_identifiable=None. "
                "This contradicts our analysis that B(t) preserves direct children. "
                "Investigate this shard before proceeding."
            )

        children: list = shard.get("outcomes", [])
        n_children: int = shard.get("n_children", len(children))

        print(f"  {tf}: joint_identifiable={joint_identifiable}, "
              f"n_children={n_children}, |B(t)|={len(min_cut)}")

        # --- Run residual_scc_analysis ---
        analysis = residual_scc_analysis(
            tf=tf,
            children=children,
            min_cut=min_cut,
            graph=graph,
        )

        # --- Summarise cluster size distribution ---
        dist = residual_cluster_size_distribution(analysis)

        has_residual = dist["has_residual_cluster"]
        n_clusters = dist["n_clusters"]
        cluster_sizes = dist["sizes"]
        max_cluster = dist["max_size"]
        n_in_clusters = dist["total_children_in_clusters"]
        cut_verified = analysis["cut_verified"]
        tf_still_cyclic = analysis["tf_still_cyclic"]
        n_cyclic_children = len(analysis["children_cyclic"])
        n_acyclic_children = len(analysis["children_acyclic"])

        # --- Falsification checks ---
        if joint_identifiable:
            n_identifiable += 1
            if has_residual:
                violations_check2.append(tf)
        else:
            n_unidentifiable += 1
            if not has_residual:
                violations_check1.append(tf)

        # --- Summary row ---
        summary_rows.append({
            "tf": tf,
            "scc_size": task.get("scc_size", ""),
            "joint_identifiable": joint_identifiable,
            "n_children": n_children,
            "n_cyclic_children": n_cyclic_children,
            "n_acyclic_children": n_acyclic_children,
            "has_residual_cluster": has_residual,
            "n_residual_clusters": n_clusters,
            "residual_cluster_sizes": str(cluster_sizes),
            "max_cluster_size": max_cluster,
            "n_children_in_clusters": n_in_clusters,
            "min_cut": str(min_cut),
            "n_min_cut": len(min_cut),
            "cut_verified": cut_verified,
            "tf_still_cyclic": tf_still_cyclic,
        })

        # --- Cluster size rows (long format) ---
        for i, sz in enumerate(cluster_sizes):
            cluster_size_rows.append({
                "tf": tf,
                "joint_identifiable": joint_identifiable,
                "cluster_index": i,
                "n_children_in_cluster": sz,
            })

    # ---------------------------------------------------------------------------
    # Write CSVs
    # ---------------------------------------------------------------------------
    summary_path = os.path.join(out_dir, "residual_scc_summary.csv")
    cluster_path = os.path.join(out_dir, "residual_cluster_sizes.csv")

    write_csv(
        summary_path,
        summary_rows,
        [
            "tf", "scc_size", "joint_identifiable",
            "n_children", "n_cyclic_children", "n_acyclic_children",
            "has_residual_cluster", "n_residual_clusters",
            "residual_cluster_sizes", "max_cluster_size",
            "n_children_in_clusters",
            "min_cut", "n_min_cut",
            "cut_verified", "tf_still_cyclic",
        ],
    )
    write_csv(
        cluster_path,
        cluster_size_rows,
        ["tf", "joint_identifiable", "cluster_index", "n_children_in_cluster"],
    )
    print(f"\nCSVs written:")
    print(f"  {summary_path}")
    print(f"  {cluster_path}")

    # ---------------------------------------------------------------------------
    # Print falsification grid and verdict
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE A RESULTS: RESIDUAL-SCC FALSIFICATION TEST")
    print("=" * 70)
    print(f"\nCohort summary:")
    print(f"  TFs with shards processed:  {len(summary_rows)}")
    print(f"  Jointly IDENTIFIABLE:        {n_identifiable}")
    print(f"  Jointly UNIDENTIFIABLE:      {n_unidentifiable}")
    if n_no_shard:
        print(f"  Missing shards (skipped):    {n_no_shard}")

    # 2x2 grid
    ident_has    = sum(1 for r in summary_rows if r["joint_identifiable"] and r["has_residual_cluster"])
    ident_nohas  = sum(1 for r in summary_rows if r["joint_identifiable"] and not r["has_residual_cluster"])
    unid_has     = sum(1 for r in summary_rows if not r["joint_identifiable"] and r["has_residual_cluster"])
    unid_nohas   = sum(1 for r in summary_rows if not r["joint_identifiable"] and not r["has_residual_cluster"])

    print()
    print(f"  Falsification grid (N per cell):")
    print(f"  {'':30s}  {'UNIDENTIFIABLE':>16s}  {'IDENTIFIABLE':>14s}")
    print(f"  {'HAS residual cluster':30s}  {'consistent':>16s}  {'VIOLATION(Chk2)':>14s}")
    print(f"  {'':30s}  {unid_has:>16d}  {ident_has:>14d}")
    print(f"  {'NO  residual cluster':30s}  {'VIOLATION(Chk1)':>16s}  {'consistent':>14s}")
    print(f"  {'':30s}  {unid_nohas:>16d}  {ident_nohas:>14d}")

    print()
    print("  Check 1 (necessity) — unidentifiable TFs with NO residual cluster:")
    if violations_check1:
        for v in violations_check1:
            row = next(r for r in summary_rows if r["tf"] == v)
            print(f"    VIOLATION: {v}  "
                  f"(n_children={row['n_children']}, "
                  f"n_residual_clusters={row['n_residual_clusters']})")
    else:
        print("    None — Check 1 HOLDS")

    print()
    print("  Check 2 (specificity) — identifiable TFs WITH residual cluster:")
    if violations_check2:
        for v in violations_check2:
            row = next(r for r in summary_rows if r["tf"] == v)
            print(f"    VIOLATION: {v}  "
                  f"(n_children={row['n_children']}, "
                  f"n_residual_clusters={row['n_residual_clusters']}, "
                  f"sizes={row['residual_cluster_sizes']})")
    else:
        print("    None — Check 2 HOLDS")

    print()
    any_violation = bool(violations_check1 or violations_check2)
    if not any_violation:
        verdict = "SUPPORTED"
        next_step = "Phase B (augmented-cut reruns) is JUSTIFIED."
    else:
        verdict = "DISPROVEN"
        next_step = "Phase B is NOT justified until mechanism is clarified."

    print(f"  VERDICT: {verdict}")
    print(f"  {next_step}")

    # Cluster size summary
    if cluster_size_rows:
        all_sizes = [r["n_children_in_cluster"] for r in cluster_size_rows]
        print()
        print("  Residual cluster child-count distribution (across all TFs):")
        print(f"    Total clusters:  {len(all_sizes)}")
        print(f"    Min size:        {min(all_sizes)}")
        print(f"    Max size:        {max(all_sizes)}")
        mean_sz = sum(all_sizes) / len(all_sizes)
        print(f"    Mean size:       {mean_sz:.1f}")
        size_freq: dict = {}
        for s in all_sizes:
            size_freq[s] = size_freq.get(s, 0) + 1
        print(f"    Size freq:       {dict(sorted(size_freq.items()))}")

    # cut_verified warning
    bad_cut = [r["tf"] for r in summary_rows if not r["cut_verified"]]
    if bad_cut:
        print()
        print(f"  [WARN] cut_verified=False for: {bad_cut}")
        print("  B(t) did NOT fully sever child->tf paths for these TFs!")
        print("  Their shard data is suspect; interpret Phase A results with caution.")

    print("=" * 70)


if __name__ == "__main__":
    main()
