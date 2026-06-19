"""
Smoke test: verify perturbation_optimizer.py functions work correctly
on a tiny synthetic coverage matrix.
"""
import sys
import os

# Ensure scripts/ is on path
sys.path.insert(0, os.path.dirname(__file__))

from perturbation_optimizer import (
    greedy_max_coverage,
    greedy_min_set_cover,
    build_marginal_gain_curve,
)

# Synthetic 4-candidate x 6-query coverage matrix
# g1 covers q0,q1,q2
# g2 covers q2,q3,q4
# g3 covers q4,q5
# g4 covers nothing
candidates = ["g1", "g2", "g3", "g4"]
queries    = ["q0", "q1", "q2", "q3", "q4", "q5"]
matrix = {
    "g1": [True,  True,  True,  False, False, False],
    "g2": [False, False, True,  True,  True,  False],
    "g3": [False, False, False, False, True,  True ],
    "g4": [False, False, False, False, False, False],
}

# --- greedy_max_coverage ---
result = greedy_max_coverage(candidates, queries, matrix, budget_k=3)
print("greedy_max_coverage (k=3):")
for rank, (tf, gain, cum) in enumerate(result, 1):
    print(f"  {rank}. {tf}  gain={gain}  cumulative={cum}")

assert result[0][0] == "g1", f"Expected g1 first, got {result[0][0]}"
assert result[0][1] == 3,    f"Expected gain=3, got {result[0][1]}"
assert result[0][2] == 3,    f"Expected cumulative=3, got {result[0][2]}"
assert result[1][0] == "g2", f"Expected g2 second, got {result[1][0]}"
assert result[1][1] == 2,    f"Expected marginal gain=2 (q3,q4), got {result[1][1]}"
assert result[1][2] == 5,    f"Expected cumulative=5, got {result[1][2]}"
assert result[2][0] == "g3", f"Expected g3 third, got {result[2][0]}"
assert result[2][1] == 1,    f"Expected marginal gain=1 (q5), got {result[2][1]}"
assert result[2][2] == 6,    f"Expected cumulative=6, got {result[2][2]}"
print("  PASS")

# --- greedy_min_set_cover ---
cover = greedy_min_set_cover(candidates, queries, matrix)
print("greedy_min_set_cover:")
for rank, (tf, gain, cum) in enumerate(cover, 1):
    print(f"  {rank}. {tf}  gain={gain}  cumulative={cum}")

assert len(cover) == 3, f"Expected 3 TFs in min cover, got {len(cover)}"
assert cover[-1][2] == 6, f"Expected all 6 resolvable covered, got {cover[-1][2]}"
print("  PASS")

# --- build_marginal_gain_curve ---
curve = build_marginal_gain_curve(candidates, queries, matrix, max_k=3)
print("build_marginal_gain_curve:")
for k, resolved, frac in curve:
    print(f"  k={k}  resolved={resolved}  frac={frac:.3f}")

assert curve[0] == (0, 0, 0.0), f"Expected (0,0,0.0) at k=0, got {curve[0]}"
assert curve[1][1] == 3, f"Expected 3 resolved at k=1, got {curve[1][1]}"
assert curve[3][1] == 6, f"Expected 6 resolved at k=3, got {curve[3][1]}"
print("  PASS")

print("\nAll smoke tests passed.")
