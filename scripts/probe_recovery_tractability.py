"""probe_recovery_tractability.py — Estimate wall-time for csd_recovery_bank n=5 k=6.

The greedy bank algorithm is:
  - n outer loops (sets), k inner loops (gene picks)
  - Each gene pick: scan candidate_pool genes, for each build do(S) SCC map once
    + count proxy-recovered targets.
  - SCC map: O(V+E) where V~1500, E~3800 for ecoli graphml
  - Per step: |pool| SCC computations
  - Total greedy SCC calls: sum over sets of (k * |pool|)
  = n * k * |pool|  (upper bound — pool shrinks as we zero out targets)

After greedy: exact verify — n * |targets| SCC computations.

We estimate:
  - Time per SCC call (networkx DiGraph, ~1500 nodes)
  - |pool| from break CSV (if present) or estimate ~300 genes
  - |targets| = 6676 unidentifiable edges
"""

import sys
import time
from pathlib import Path

# --- Estimate SCC time on a small synthetic graph ---
import networkx as nx

# Build a random sparse digraph of similar size to ecoli
# ecoli: ~1500 nodes, ~3800 edges
rng = __import__("random").Random(42)
V, E = 1500, 3800
G_test = nx.DiGraph()
G_test.add_nodes_from(range(V))
edges_added = set()
while len(edges_added) < E:
    u = rng.randint(0, V-1)
    v = rng.randint(0, V-1)
    if u != v and (u,v) not in edges_added:
        G_test.add_edge(u, v)
        edges_added.add((u, v))

N_TRIALS = 200
t0 = time.perf_counter()
for _ in range(N_TRIALS):
    list(nx.strongly_connected_components(G_test))
t1 = time.perf_counter()
scc_time_ms = (t1 - t0) / N_TRIALS * 1000
print(f"SCC time per call (V={V}, E={E}): {scc_time_ms:.2f} ms")

# --- Pool size estimate ---
break_csv = Path("notebooks/Ecoli_Analysis_Notebooks/csd_break_results_full.csv")
if break_csv.exists():
    import csv, ast
    pool = set()
    with open(break_csv) as f:
        for row in csv.DictReader(f):
            raw = row.get("min_break_set", "[]")
            try:
                genes = ast.literal_eval(raw)
            except Exception:
                genes = []
            pool.update(genes)
    pool_size = len(pool)
    print(f"Pool size (from break CSV): {pool_size}")
else:
    pool_size = 300  # conservative estimate
    print(f"Pool size (estimated): {pool_size}  (break CSV not yet available)")

# --- Compute estimates ---
targets = 6676

for n, k in [(10, 3), (5, 6)]:
    # Greedy: n*k steps, each scanning pool_size candidates
    greedy_scc_calls = n * k * pool_size
    greedy_sec = greedy_scc_calls * (scc_time_ms / 1000)

    # Exact verify: n sets * targets edges, each an SCC call on G'
    exact_scc_calls = n * targets
    exact_sec = exact_scc_calls * (scc_time_ms / 1000)

    total_min = (greedy_sec + exact_sec) / 60
    print(f"\nn={n}, k={k}:")
    print(f"  Greedy SCC calls: {greedy_scc_calls:,}  -> {greedy_sec:.1f}s ({greedy_sec/60:.1f} min)")
    print(f"  Exact verify calls: {exact_scc_calls:,} -> {exact_sec:.1f}s ({exact_sec/60:.1f} min)")
    print(f"  Total estimate: {total_min:.1f} min")
    print(f"  Fits in 4h budget: {'YES' if total_min < 240 else 'NO — need longer'}")
