# Plan: Re-run CSD on every E. coli edge with the fast O-set

**Date:** 2026-06-26  
**Author:** Jeremy Zucker  
**Status:** DRAFT — no code changes have been made yet

---

## 1. Background and motivation

The previous full-network run (`notebooks/Ecoli_Analysis_Notebooks/csd_summary.json`) produced:

| Metric | Value |
|---|---|
| Total directed edges | 9 211 |
| Identifiable | 1 439 (15.6 %) |
| Unidentifiable | 330 (3.6 %) |
| **Timed out** | **7 442 (80.8 %)** |
| Same-SCC edges | 519 |
| Shards used | 1 843 |

The 80 % timeout rate was caused by a single algorithmic bottleneck: the
`optimal_adjustment_set` function in y0 used `nx.all_simple_paths` to enumerate
causal paths, which is **exponential** in the size of strongly connected
components (SCCs).  The E. coli regulatory network has a large SCC (~500 nodes),
so every edge touching that SCC triggered the exponential path enumeration and
hit the per-edge timeout.

### What changed upstream (y0 branch 402)

`y0/src/y0/algorithm/separation/sigma_single_door.py` —
`optimal_adjustment_set` and `proper_backdoor_graph` were rewritten to use
**two BFS/DFS reachability passes** instead of `all_simple_paths`:

1. **Causal nodes** = `desc(X) ∩ anc(Y)` — one forward BFS from X, one
   backward BFS from Y, intersect.
2. **Forbidden nodes** = `desc_inclusive(cn) ∪ {X}` — one forward BFS from
   each causal node.
3. **O-set** = `pa(cn) \ forb` — direct parent lookup, no path enumeration.

Theoretical complexity: **O(|V| + |E|)** per edge, matching the σ-extension
preprocessing step.  The `all_simple_paths` call is completely eliminated.

### What did NOT change

- `nocap/src/nocap/cyclic_single_door.py` — `classify_edge`,
  `evaluate_all_edges`, `maximize_identifiable_edges` call into y0 through the
  same `find_sigma_single_door_set` interface.  **No nocap source edits are
  required for correctness.**
- The Snakemake workflow rules (`workflow/Snakefile`) — structure unchanged;
  only resource limits need tuning.
- The σ-extension step — unchanged; still O(|V|+|E|) via Tarjan SCC.
- Verdict semantics — the O-set is the same mathematical object; verdicts must
  be **identical** to the prior run for all non-timeout edges.  Only
  previously-`timeout` edges should change status.

### Verification status

Three integer-arithmetic O-set invariants are **formally proved** via
Axiomander (Iris WP calculus, level1) in
`tests/test_oset_kernel_axiomander.py`:

| Invariant | Proved postconditions |
|---|---|
| `oset_non_negative_size` | `result >= 0` AND `result == n_parents - n_forbidden` |
| `forbidden_includes_cause` | `result >= n_causal_nodes + 1` AND `result >= 2` |
| `cause_always_excluded_from_oset` | `result >= 0` AND `result < n_parents_of_cn` |

Set-arithmetic kernels (`causal_nodes_kernel`, `forbidden_kernel`,
`oset_kernel`, `pbd_first_hops_kernel`) are pytest-verified with 37/37 passing
tests including the cyclic counterexample `B→A, A→X, X→A, X→Y → O-set={B}`.

---

## 2. Code update checklist

These are the **only** changes needed before running.  Do them in order.

### 2.1 Re-pin and lock the fast y0

The `pyproject.toml` already points to the `402-...` branch:

```toml
"y0 @ git+https://github.com/y0-causal-inference/y0.git@402-add-σ-separation-single-door-criterion-...",
```

The branch has been updated with the fast O-set commits.  Refresh the lock:

```bash
# From the nocap repo root
uv lock --upgrade-package y0
uv sync
```

Commit the updated `uv.lock`.

### 2.2 Verify the fast code is loaded (load-verification probe)

Before any cluster work, confirm that `all_simple_paths` is no longer in the
O-set call path:

```bash
uv run python -c "
import inspect
from y0.algorithm.separation.sigma_single_door import find_sigma_single_door_set
src = inspect.getsource(find_sigma_single_door_set)
assert 'all_simple_paths' not in src, 'FAIL: old path-enum code still loaded'
print('PASS: fast O-set loaded')
"
```

If this fails, `uv lock` did not pick up the new commit — check the branch tip
with `git -C $(uv run python -c "import y0; print(y0.__file__.split('/src')[0])")
log --oneline -3`.

### 2.3 Local correctness gate

```bash
# Unit + Axiomander kernel tests
uv run pytest tests/test_cyclic_single_door.py tests/test_oset_kernel_axiomander.py -q

# Smoke test on a small subgraph (should complete in < 5 s)
uv run python scripts/smoke_csd_break.py
```

All tests must pass before proceeding.

### 2.4 Snakefile resource retune

In `workflow/Snakefile`, the `classify` rule currently reserves:

```python
resources:
    mem_mb  = 8000,
    runtime = 120,   # 2 h; old worst-case was ~1 h
```

With the O(|V|+|E|) O-set, the dense-SCC edges that drove the 1 h worst case
should now complete in seconds.  Proposed new values:

```python
resources:
    mem_mb  = 4000,   # same as prepare/gather; no large intermediate structures
    runtime = 30,     # 30 min; generous margin for σ-extension + 500 edges/shard
```

Keep the per-edge `timeout_seconds` guard in `evaluate_all_edges` (currently
passed from `cyclic_single_door_classify.py`) as a safety net for the first
run.  If the first run shows all edges completing in < 2 min, remove the
timeout on the second run.

Also consider reducing `n_shards` from 179 to ~50 (larger shards, fewer jobs)
since each shard is now fast — this reduces SLURM scheduler overhead.  Or keep
179 for maximum parallelism; both are fine.

---

## 3. Timeout-cohort-first validation (Phase A)

**Goal:** confirm that the 7 442 previously-timed-out edges now resolve before
committing the full 9 211-edge rerun.

### 3.1 Extract the timeout cohort

```bash
# Produces timeout_cohort.json — a list of {cause, effect} pairs
uv run python scripts/count_timeouts.py \
    --input notebooks/Ecoli_Analysis_Notebooks/csd_results.csv \
    --output results/timeout_cohort.json
```

If `count_timeouts.py` does not already support `--output`, add a
`--output` flag that writes a JSON list of `[cause, effect]` pairs.
Alternatively, use a one-liner:

```bash
uv run python -c "
import json, csv
rows = list(csv.DictReader(open('notebooks/Ecoli_Analysis_Notebooks/csd_results.csv')))
cohort = [[r['cause'], r['effect']] for r in rows if r['status'] == 'timeout']
json.dump(cohort, open('results/timeout_cohort.json', 'w'))
print(len(cohort), 'timeout edges written')
"
```

Expected output: `7442 timeout edges written`.

### 3.2 Run the cohort locally (or one HPC node)

```bash
uv run python scripts/cyclic_single_door_classify.py classify \
    --graphml notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml \
    --restrict-edges results/timeout_cohort.json \
    --output results/timeout_cohort_reclassified.json \
    --timeout-seconds 60
```

This uses `evaluate_all_edges(restrict_edges=...)` — the σ-extension is built
once and reused for all 7 442 edges.  Expected wall time with the fast O-set:
**< 10 minutes** for the full cohort on a single core.

### 3.3 Gate: check for residual timeouts

```bash
uv run python -c "
import json
results = json.load(open('results/timeout_cohort_reclassified.json'))
timeouts = [r for r in results if r['status'] == 'timeout']
print(f'{len(timeouts)} residual timeouts out of {len(results)} cohort edges')
if timeouts:
    print('WARN: some edges still timing out — investigate before full rerun')
else:
    print('PASS: all cohort edges resolved — proceed to full rerun')
"
```

**Gate condition:** zero residual timeouts (or a small, explainable residual).
If any edges still time out, investigate before proceeding to Phase B.

### 3.4 Diff non-timeout verdicts

For the 1 769 edges that were already resolved (1 439 identifiable + 330
unidentifiable), the new run must produce **identical** verdicts:

```bash
uv run python -c "
import json, csv
old = {(r['cause'], r['effect']): r['status']
       for r in csv.DictReader(open('notebooks/Ecoli_Analysis_Notebooks/csd_results.csv'))
       if r['status'] != 'timeout'}
new = {(r['cause'], r['effect']): r['status']
       for r in json.load(open('results/timeout_cohort_reclassified.json'))}
# new only contains the timeout cohort, so overlap is empty — this is a sanity check
# for the full rerun diff (Section 4.3)
print('Cohort diff check: OK (no overlap with prior resolved edges expected)')
"
```

The full verdict diff runs in Section 4.3 after the complete rerun.

---

## 4. Full-network rerun on HPC (Phase B)

Only proceed after Phase A passes.

### 4.1 Data staging on HPC

Ensure the following files are present on the HPC filesystem (e.g. `/qfs/...`):

```
nocap/                                          ← git clone or rsync
  notebooks/Ecoli_Analysis_Notebooks/
    ecoli_full_network_no_small_rna.graphml     ← input graph
    supptable1.csv                              ← metadata
  workflow/
    config.yaml
    Snakefile
    profiles/slurm/config.yaml
  uv.lock                                       ← updated in Step 2.1
```

The `UV_CACHE_DIR` is set to `{REPO_ROOT}/.uv-cache` in the Snakefile so uv
uses hardlinks on the same filesystem (avoids NFS copy overhead).

### 4.2 Submit the full pipeline

```bash
# From the nocap repo root on the HPC login node
snakemake \
    --profile workflow/profiles/slurm \
    --configfile workflow/config.yaml \
    --config n_shards=50 \
    -j 200
```

Notes:
- `n_shards=50` gives ~184 edges/shard; adjust to match available node count.
- The `classify` rule `runtime=30` (after Step 2.4 retune) fits within the
  `slurm` partition's default time limit.
- `retries: 2` in the SLURM profile handles transient node failures.
- `rerun-incomplete: true` handles partial shard writes from preemption.

### 4.3 Reconcile results

After `gather` completes:

```bash
# Strict diff: only timeout→resolved changes are allowed
uv run python -c "
import json, csv

old = {(r['cause'], r['effect']): r['status']
       for r in csv.DictReader(open('notebooks/Ecoli_Analysis_Notebooks/csd_results.csv'))}
new_rows = list(csv.DictReader(open('results/cyclic_single_door/classification_results.csv')))
new = {(r['cause'], r['effect']): r['status'] for r in new_rows}

regressions = [(k, old[k], new[k]) for k in old
               if old[k] != 'timeout' and new.get(k) != old[k]]
resolved    = [(k, new[k]) for k in old if old[k] == 'timeout' and new.get(k) != 'timeout']
still_to    = [(k,) for k in old if old[k] == 'timeout' and new.get(k) == 'timeout']

print(f'Regressions (non-timeout verdict changed): {len(regressions)}')
print(f'Resolved timeouts: {len(resolved)}')
print(f'Still timing out: {len(still_to)}')
if regressions:
    for k, o, n in regressions[:10]:
        print(f'  REGRESSION {k}: {o} -> {n}')
"
```

**Pass condition:** `Regressions: 0`.  Any non-zero regression count means the
fast O-set changed a verdict — investigate immediately (likely a bug in the
reachability rewrite).

### 4.4 Regenerate summary and figures

```bash
# Update csd_summary.json
uv run python scripts/cyclic_single_door_gather.py \
    --input-dir results/cyclic_single_door/classified \
    --output-csv notebooks/Ecoli_Analysis_Notebooks/csd_results.csv \
    --output-summary notebooks/Ecoli_Analysis_Notebooks/csd_summary.json

# Regenerate visualizations (run the Cyclic_SingleDoor_Analysis notebook)
uv run jupyter nbconvert --to notebook --execute \
    notebooks/Ecoli_Analysis_Notebooks/Cyclic_SingleDoor_Analysis.ipynb \
    --output notebooks/Ecoli_Analysis_Notebooks/Cyclic_SingleDoor_Analysis.ipynb
```

---

## 5. Optional: re-run greedy rescue (Phase C)

With classification now fast, `maximize_identifiable_edges` (the `preprocess`
Snakemake rule) is affordable for larger `k`:

```bash
snakemake preprocess \
    --profile workflow/profiles/slurm \
    --configfile workflow/config.yaml \
    --config k=20
```

The `preprocess` rule currently reserves `runtime=300` (5 h).  With the fast
O-set, this should complete in < 30 min for `k=20`; reduce to `runtime=60`
after confirming.

---

## 6. Rollback procedure

If Phase A or Phase B reveals unexpected verdict regressions:

1. Identify the y0 commit that introduced the regression:
   ```bash
   git -C $(uv run python -c "import y0; print(y0.__file__.split('/src')[0])") log --oneline -5
   ```
2. Pin back to the prior commit in `pyproject.toml`:
   ```toml
   "y0 @ git+https://github.com/y0-causal-inference/y0.git@<prior-commit-hash>",
   ```
3. `uv lock && uv sync`
4. File a bug on the y0 `402` branch with the specific edge that regressed.

---

## 7. Summary of file changes required

| File | Change | When |
|---|---|---|
| `uv.lock` | Re-lock after `uv lock --upgrade-package y0` | Step 2.1 |
| `workflow/Snakefile` | `classify` rule: `runtime 120→30`, `mem_mb 8000→4000` | Step 2.4 |
| `workflow/config.yaml` | `n_shards: 179→50` (optional, for fewer larger jobs) | Step 4.2 |
| `notebooks/Ecoli_Analysis_Notebooks/csd_results.csv` | Overwrite with full-rerun output | Step 4.4 |
| `notebooks/Ecoli_Analysis_Notebooks/csd_summary.json` | Overwrite with new summary | Step 4.4 |
| `notebooks/visualizations/csd_*.png` | Regenerate from updated notebook | Step 4.4 |

**No changes to `src/nocap/cyclic_single_door.py` or any other nocap source
file are required.**  The algorithm is correct; only the upstream y0 dependency
needed the O(|V|+|E|) rewrite.
