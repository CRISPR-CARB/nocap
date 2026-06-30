"""
rerun_unident_pergene.py
========================
Re-run all joint=False shards with --per-gene-on-failure in one pass.

For each TF whose shard is either:
  - missing, or
  - present but joint_identifiable=False and per_gene is empty

... this script deletes the shard (if present) and re-runs the worker
with --per-gene-on-failure, then calls list_scc_tasks to print a summary.

Run with:
    uv run python scripts/rerun_unident_pergene.py 2>&1 | tee /tmp/rerun_log.txt

Ordered by n_children ascending so fast tasks finish first and output
is visible quickly.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from shard_io import load_first_json_object

MANIFEST = os.path.join(
    os.path.dirname(__file__), "..", "notebooks", "Ecoli_Analysis_Notebooks", "scc_perturb_job.json"
)
SHARDS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "notebooks",
        "Ecoli_Analysis_Notebooks",
        "scc_perturb_shards",
    )
)
WORKER = os.path.join(os.path.dirname(__file__), "scc_perturb_worker.py")

with open(MANIFEST) as f:
    manifest = json.load(f)

tasks = manifest["tasks"]
n_tasks = len(tasks)

# -----------------------------------------------------------------------
# Identify which tasks need re-running
# -----------------------------------------------------------------------


def needs_rerun(task_idx: int, task: dict) -> bool:
    tf = task["tf"]
    shard_path = os.path.join(SHARDS_DIR, f"scc_perturb_shard_{tf}.json")
    if not os.path.exists(shard_path):
        return True
    try:
        s = load_first_json_object(shard_path)
        return s.get("joint_identifiable") is False and len(s.get("per_gene", {})) == 0
    except Exception:
        return True


# Get n_children from existing shard or manifest for ordering
def child_count(task_idx: int, task: dict) -> int:
    tf = task["tf"]
    shard_path = os.path.join(SHARDS_DIR, f"scc_perturb_shard_{tf}.json")
    if os.path.exists(shard_path):
        try:
            s = load_first_json_object(shard_path)
            return s.get("n_children", 9999)
        except Exception:
            pass
    return 9999


to_rerun = [(i, t) for i, t in enumerate(tasks) if needs_rerun(i, t)]
to_rerun.sort(key=lambda x: child_count(x[0], x[1]))

print(f"Tasks needing per-gene re-run: {len(to_rerun)}")
for i, t in to_rerun:
    print(f"  task {i:>2}  {t['tf']:<12}  scc_size={t['scc_size']}  |B(t)|={len(t['min_cut'])}")
print()

if not to_rerun:
    print("Nothing to do — all shards present and per_gene populated.")
    sys.exit(0)

# -----------------------------------------------------------------------
# Re-run each task
# -----------------------------------------------------------------------

python_exe = sys.executable

for i, task in to_rerun:
    tf = task["tf"]
    shard_path = os.path.join(SHARDS_DIR, f"scc_perturb_shard_{tf}.json")

    # Delete existing shard so idempotency guard doesn't skip it
    if os.path.exists(shard_path):
        os.remove(shard_path)
        print(f"[{tf}] Deleted existing shard.")

    print(f"[{tf}] Running worker (task {i})...")
    result = subprocess.run(
        [
            python_exe,
            WORKER,
            "--manifest",
            os.path.abspath(MANIFEST),
            "--shards-dir",
            SHARDS_DIR,
            "--n-tasks",
            str(n_tasks),
            "--task-id",
            str(i),
            "--per-gene-on-failure",
        ],
        capture_output=False,  # let stdout/stderr flow to terminal
        check=False,
    )
    if result.returncode != 0:
        print(f"[{tf}] ERROR: worker exited with code {result.returncode}", file=sys.stderr)
    else:
        # Quick confirmation read
        try:
            s = load_first_json_object(shard_path)
            pg = s.get("per_gene", {})
            n_id = sum(1 for v in pg.values() if v)
            print(
                f"[{tf}] Done: joint={s.get('joint_identifiable')}  per_gene={n_id}/{len(pg)} identifiable"
            )
        except Exception as e:
            print(f"[{tf}] Shard written but could not verify: {e}", file=sys.stderr)
    print()

print("All re-runs complete.")
print()

# -----------------------------------------------------------------------
# Final summary via list_scc_tasks
# -----------------------------------------------------------------------
print("=" * 60)
print("FINAL STATUS:")
print("=" * 60)
list_script = os.path.join(os.path.dirname(__file__), "list_scc_tasks.py")
subprocess.run([python_exe, list_script], check=False)
