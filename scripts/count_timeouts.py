"""Count timed-out edges across all classified shards."""
import json
import glob
from collections import Counter

classified = sorted(glob.glob("results/cyclic_single_door/classified/shard_*.json"))

total_edges = 0
total_timeouts = 0
timeout_edges = []

for f in classified:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    results = d.get("results", [])
    total_edges += len(results)
    for r in results:
        if r.get("timed_out"):
            total_timeouts += 1
            timeout_edges.append((r["cause"], r["effect"]))

print(f"Classified shards : {len(classified)}")
print(f"Total edges       : {total_edges}")
print(f"Timed out         : {total_timeouts}  ({100*total_timeouts/total_edges:.1f}%)")
print()

# Most common timeout causes
cause_counts = Counter(c for c, e in timeout_edges)
effect_counts = Counter(e for c, e in timeout_edges)
print("Top 10 causes in timeouts:")
for cause, cnt in cause_counts.most_common(10):
    print(f"  {cause}: {cnt}")
print()
print("Top 10 effects in timeouts:")
for effect, cnt in effect_counts.most_common(10):
    print(f"  {effect}: {cnt}")
