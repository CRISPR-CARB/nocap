"""
Smoke-test the data loading logic from SCC_Perturbation_Analysis.ipynb.
Verifies all imports and key computations work with current shards.
"""

import glob
import json
import os

import pandas as pd

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "notebooks", "Ecoli_Analysis_Notebooks")
)
MANIFEST_PATH = os.path.join(BASE_DIR, "scc_perturb_job.json")
SHARDS_DIR = os.path.join(BASE_DIR, "scc_perturb_shards")
RESULTS_CSV = os.path.join(BASE_DIR, "scc_perturbation_results.csv")
SUPPTABLE = os.path.join(BASE_DIR, "supptable1.csv")

# 1. Manifest
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)
dag_tfs = set(manifest["dag_tfs"])
scc_tasks = manifest["tasks"]
n_tasks = manifest["n_tasks"]
print(f"Manifest OK: n_tasks={n_tasks}, dag_tfs={len(dag_tfs)}")

# 2. Shards
shard_files = sorted(glob.glob(os.path.join(SHARDS_DIR, "scc_perturb_shard_*.json")))
shards = {}
for path in shard_files:
    with open(path) as f:
        s = json.load(f)
    shards[s["tf"]] = s
missing = [t["tf"] for t in scc_tasks if t["tf"] not in shards]
print(f"Shards: {len(shards)}/{n_tasks} loaded. Missing: {missing or 'none'}")

# 3. DataFrame from shards
rows = []
for tf, s in shards.items():
    per_gene = s.get("per_gene", {})
    n_children = s.get("n_children", 0)
    n_pg_id = sum(1 for v in per_gene.values() if v)
    pct = n_pg_id / n_children * 100 if n_children > 0 and per_gene else None
    rows.append(
        {
            "tf": tf,
            "scc_size": s.get("scc_size", 0),
            "min_cut_size": len(s.get("min_cut", [])),
            "n_children": n_children,
            "joint_identifiable": s.get("joint_identifiable"),
            "n_per_gene_identifiable": n_pg_id if per_gene else None,
            "pct_per_gene_identifiable": round(pct, 1) if pct is not None else None,
        }
    )
df = pd.DataFrame(rows).sort_values("tf").reset_index(drop=True)
df["joint_identifiable"] = df["joint_identifiable"].astype(object)
df["n_per_gene_identifiable"] = pd.to_numeric(df["n_per_gene_identifiable"], errors="coerce")
df["pct_per_gene_identifiable"] = pd.to_numeric(df["pct_per_gene_identifiable"], errors="coerce")

total_tfs = n_tasks + len(dag_tfs)
n_jid = int((df["joint_identifiable"] == True).sum())
n_jnid = int((df["joint_identifiable"] == False).sum())
n_pg_any = int(
    ((df["joint_identifiable"] == False) & (df["n_per_gene_identifiable"].fillna(0) > 0)).sum()
)

print(f"DataFrame: {len(df)} rows")
print(f"Total TFs: {total_tfs}  (DAG={len(dag_tfs)}, SCC={n_tasks})")
print(f"Joint ID: {n_jid}  Joint unID: {n_jnid}  Any per-gene: {n_pg_any}")

# 4. Supptable
supp = pd.read_csv(SUPPTABLE)
supp.columns = [c.strip() for c in supp.columns]
supp = supp.rename(columns={"Perturbation Name": "tf", "CRISPR type": "crispr_type"})
supp_meta = supp.drop_duplicates("tf")[["tf", "crispr_type"]].copy()
print(f"Supptable: {len(supp_meta)} TFs")
print(supp["crispr_type"].value_counts().to_string())

# 5. Per-gene summary
print("\nPer-gene identifiability (current shards):")
for tf in sorted(shards):
    s = shards[tf]
    if s.get("joint_identifiable") is False:
        pg = s.get("per_gene", {})
        n_id = sum(1 for v in pg.values() if v)
        total = len(pg)
        pct_s = f"{n_id / total * 100:.1f}%" if total > 0 else "N/A"
        print(f"  {tf:<12}: {n_id:>4}/{total:<4} identifiable  ({pct_s})")
    else:
        print(f"  {tf:<12}: jointly identifiable")

print("\nSMOKE TEST PASSED")
