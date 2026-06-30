"""scripts/reconcile_fast_oset.py — Compare old csd_results.csv (pre-fast-O-set)
with the new classification_results.csv (post-fast-O-set rerun).

Reports:
  - Regressions: non-timeout verdicts that changed
  - Resolved timeouts: edges that were 'timeout' and now have a definite answer
  - Still timing out: edges that are still 'timeout'
  - New identifiable count vs old
"""
import csv
import sys

OLD = "notebooks/Ecoli_Analysis_Notebooks/csd_results.csv"
NEW = "results/cyclic_single_door/classification_results.csv"

def load(path):
    with open(path) as f:
        return {(r["cause"], r["effect"]): r["status"]
                for r in csv.DictReader(f)}

old = load(OLD)
new = load(NEW)

regressions  = [(k, old[k], new[k]) for k in old
                if old[k] not in ("timeout",) and k in new and new[k] != old[k]]
resolved     = [(k, new[k]) for k in old
                if old[k] == "timeout" and k in new and new[k] != "timeout"]
still_to     = [k for k in old
                if old[k] == "timeout" and (k not in new or new[k] == "timeout")]
only_in_new  = [k for k in new if k not in old]

old_ident = sum(1 for v in old.values() if v == "identifiable")
new_ident = sum(1 for v in new.values() if v == "identifiable")
old_unid  = sum(1 for v in old.values() if v == "unidentifiable")
new_unid  = sum(1 for v in new.values() if v == "unidentifiable")
old_to    = sum(1 for v in old.values() if v == "timeout")
new_to    = sum(1 for v in new.values() if v == "timeout")

print("=" * 60)
print("  Fast O-set rerun reconciliation")
print("=" * 60)
print(f"Old total edges : {len(old):>6}")
print(f"New total edges : {len(new):>6}")
print()
print(f"{'Status':<20} {'Old':>8} {'New':>8} {'Delta':>8}")
print("-" * 46)
print(f"{'identifiable':<20} {old_ident:>8} {new_ident:>8} {new_ident-old_ident:>+8}")
print(f"{'unidentifiable':<20} {old_unid:>8} {new_unid:>8} {new_unid-old_unid:>+8}")
print(f"{'timeout':<20} {old_to:>8} {new_to:>8} {new_to-old_to:>+8}")
print()
print(f"Regressions (non-timeout verdict changed) : {len(regressions)}")
print(f"Resolved timeouts                         : {len(resolved)}")
print(f"Still timing out                          : {len(still_to)}")
print(f"Edges only in new (not in old)            : {len(only_in_new)}")

if regressions:
    print("\n--- REGRESSIONS (first 20) ---")
    for k, o, n in regressions[:20]:
        print(f"  {k[0]:>20} -> {k[1]:<20}  {o} -> {n}")

if still_to:
    print(f"\n--- STILL TIMING OUT (first 10) ---")
    for k in still_to[:10]:
        print(f"  {k[0]:>20} -> {k[1]:<20}")

if regressions:
    print("\nRESULT: FAIL — regressions detected")
    sys.exit(1)
elif still_to:
    print("\nRESULT: PARTIAL — timeouts remain; check above")
    sys.exit(0)
else:
    print("\nRESULT: PASS — 0 regressions, all timeouts resolved")
    sys.exit(0)
