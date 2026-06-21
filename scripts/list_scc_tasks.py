"""
list_scc_tasks.py
=================
Inspector: prints each task in scc_perturb_job.json alongside the
current shard status (missing / joint_identifiable / per_gene filled).

Usage:
    uv run python scripts/list_scc_tasks.py 2>&1 | tail -60
"""
import json
import os
import sys

MANIFEST = os.path.join(
    os.path.dirname(__file__), "..",
    "notebooks", "Ecoli_Analysis_Notebooks", "scc_perturb_job.json"
)
SHARDS_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "notebooks", "Ecoli_Analysis_Notebooks", "scc_perturb_shards"
)

sys.path.insert(0, os.path.dirname(__file__))
from shard_io import load_first_json_object

with open(MANIFEST) as f:
    manifest = json.load(f)

tasks = manifest["tasks"]
print(f"Manifest: {len(tasks)} tasks")
print(f"{'idx':>4}  {'tf':<12}  {'|B(t)|':>6}  {'scc_sz':>6}  {'shard':>8}  {'joint_id':>10}  {'per_gene_n':>10}  {'n_children':>10}")
print("-" * 80)

missing = []
unident_no_pergene = []

for i, t in enumerate(tasks):
    tf = t["tf"]
    shard_path = os.path.join(SHARDS_DIR, f"scc_perturb_shard_{tf}.json")
    if not os.path.exists(shard_path):
        status = "MISSING"
        joint = "-"
        pg_n = "-"
        nc = "-"
        missing.append((i, tf))
    else:
        try:
            s = load_first_json_object(shard_path)
            joint = str(s.get("joint_identifiable"))
            pg_n = str(len(s.get("per_gene", {})))
            nc = str(s.get("n_children", "-"))
            status = "ok"
            if s.get("joint_identifiable") is False and len(s.get("per_gene", {})) == 0:
                unident_no_pergene.append((i, tf))
        except Exception as e:
            status = f"ERROR:{e}"
            joint = "-"
            pg_n = "-"
            nc = "-"
    print(f"{i:>4}  {tf:<12}  {len(t['min_cut']):>6}  {t['scc_size']:>6}  {status:>8}  {joint:>10}  {pg_n:>10}  {nc:>10}")

print()
print(f"Missing shards ({len(missing)}): {[tf for _, tf in missing]}")
print(f"Joint=False, per_gene empty ({len(unident_no_pergene)}): {[tf for _, tf in unident_no_pergene]}")
