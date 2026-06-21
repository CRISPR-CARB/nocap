"""Quick summary of all scc_perturb shards."""
import json
import glob
import os

SHARDS_DIR = "notebooks/Ecoli_Analysis_Notebooks/scc_perturb_shards"
MANIFEST = "notebooks/Ecoli_Analysis_Notebooks/scc_perturb_job.json"

with open(MANIFEST) as f:
    manifest = json.load(f)

all_tfs = [t["tf"] for t in manifest["tasks"]]
dag_tfs = manifest.get("dag_tfs", [])
n_tasks = manifest["n_tasks"]

shards = {}
for path in sorted(glob.glob(os.path.join(SHARDS_DIR, "scc_perturb_shard_*.json"))):
    with open(path) as f:
        raw = f.read()
    # Handle files with duplicate JSON objects (glnG race condition)
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        # Take only the first complete JSON object
        decoder = json.JSONDecoder()
        d, _ = decoder.raw_decode(raw)
        print(f"  WARNING: {os.path.basename(path)} has extra data — using first JSON object")
    shards[d["tf"]] = d

print(f"=== SCC Perturbation Run Status ===")
print(f"Expected tasks (SCC TFs):    {n_tasks}")
print(f"Shards present:              {len(shards)}")
print(f"DAG TFs (no perturb needed): {len(dag_tfs)}")
print()

missing = [tf for tf in all_tfs if tf not in shards]
print(f"Missing shards ({len(missing)}): {missing}")
print()

print(f"{'TF':<12} {'SCC_sz':>6} {'MinCut':>6} {'InSCCch':>8} {'N_child':>8} {'JointID':>9} {'PerGene':>8} {'Note'}")
print("-" * 80)

joint_true = []
joint_false = []
no_children = []

for tf in sorted(shards.keys()):
    d = shards[tf]
    ji = d["joint_identifiable"]
    nc = d["n_children"]
    mc = len(d["min_cut"])
    iscc = len(d.get("in_scc_children", []))
    pg = d.get("per_gene", {})
    n_pg = sum(1 for v in pg.values() if v) if pg else ""
    note = d.get("note", "")
    print(f"{tf:<12} {d['scc_size']:>6} {mc:>6} {iscc:>8} {nc:>8} {str(ji):>9} {str(n_pg):>8} {note}")
    if ji is True:
        joint_true.append(tf)
    elif ji is False:
        joint_false.append(tf)
    elif note == "no_children":
        no_children.append(tf)

print()
print(f"Jointly identifiable:   {len(joint_true)} -> {joint_true}")
print(f"Jointly unidentifiable: {len(joint_false)} -> {joint_false}")
print(f"No children:            {len(no_children)} -> {no_children}")
print()
print(f"=== Worker Log Summary ===")

LOGS_DIR = "notebooks/Ecoli_Analysis_Notebooks/logs"
err_files = sorted(glob.glob(os.path.join(LOGS_DIR, "scc_worker_*.err")))
cancelled = []
clean = []
for ef in err_files:
    content = open(ef).read().strip()
    idx = os.path.basename(ef).replace(".err","")
    if "CANCELLED" in content:
        cancelled.append(idx)
    elif content:
        clean.append((idx, content[:100]))

print(f"Workers cancelled (SLURM timeout/signal): {len(cancelled)}")
print(f"Workers with other stderr:                {len(clean)}")
for idx, c in clean:
    print(f"  {idx}: {c}")
