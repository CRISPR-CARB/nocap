"""probe_fast_oset.py — Verify the fast O-set (BFS-based) code is loaded from y0.

Checks two things:
1. The locked commit in uv.lock matches the current remote HEAD of the 402 branch.
2. find_sigma_single_door_set does NOT contain 'all_simple_paths' in its source
   (confirms the exponential path-enumeration code was replaced).

Exit code 0 = both PASS; exit code 1 = at least one FAIL.
"""
from __future__ import annotations

import inspect
import subprocess
import sys
import re


# ---------------------------------------------------------------------------
# Check 1: locked commit == remote 402 branch tip
# ---------------------------------------------------------------------------

BRANCH = "402-add-\u03c3-separation-single-door-criterion-for-linear-scm-coefficient-estimation-in-cyclic-directed-graphs"
REMOTE_URL = "https://github.com/y0-causal-inference/y0.git"

def get_remote_tip() -> str | None:
    """Fetch the remote branch tip via git ls-remote (no clone needed)."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", REMOTE_URL, f"refs/heads/{BRANCH}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.strip().splitlines()
        if not lines:
            return None
        return lines[0].split()[0]
    except Exception as exc:
        print(f"  WARNING: could not fetch remote tip: {exc}", file=sys.stderr)
        return None


def get_locked_commit() -> str | None:
    """Read the y0 commit hash from uv.lock.

    The uv.lock format for git deps is:
        name = "y0"
        ...
        source = { git = "https://...y0.git?rev=<branch>#<commit-hash>" }
    The 40-char commit hash is embedded after '#' inside the git URL.
    """
    try:
        with open("uv.lock") as f:
            content = f.read()
    except FileNotFoundError:
        return None

    # Find the [[package]] block for y0 and extract the commit hash after '#'
    # in the git source URL.
    pattern = re.compile(
        r'^name\s*=\s*"y0"\s*\n'          # name = "y0" line
        r'.*?'                              # anything (version, etc.)
        r'source\s*=\s*\{[^}]*y0\.git[^}]*#([0-9a-f]{40})',
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(content)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Check 2: all_simple_paths absent from find_sigma_single_door_set
# ---------------------------------------------------------------------------

def check_no_all_simple_paths() -> bool:
    """Return True if find_sigma_single_door_set does NOT use all_simple_paths."""
    try:
        from y0.algorithm.separation.sigma_single_door import find_sigma_single_door_set
        src = inspect.getsource(find_sigma_single_door_set)
        return "all_simple_paths" not in src
    except Exception as exc:
        print(f"  ERROR importing find_sigma_single_door_set: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    all_pass = True

    # --- Check 1: commit match ---
    print("Check 1: locked y0 commit == remote 402 branch tip")
    remote_tip = get_remote_tip()
    locked = get_locked_commit()

    if remote_tip is None:
        print("  WARNING: could not determine remote tip (network unavailable?) — skipping commit check")
    elif locked is None:
        print("  FAIL: could not find y0 commit in uv.lock — run 'uv lock --upgrade-package y0' first")
        all_pass = False
    elif remote_tip == locked:
        print(f"  PASS: locked commit {locked[:12]} == remote tip {remote_tip[:12]}")
    else:
        print(f"  FAIL: locked commit {locked[:12]} != remote tip {remote_tip[:12]}")
        print(f"        Run: uv lock --upgrade-package y0 && uv sync")
        all_pass = False

    # --- Check 2: no all_simple_paths ---
    print("Check 2: find_sigma_single_door_set does not use all_simple_paths")
    if check_no_all_simple_paths():
        print("  PASS: all_simple_paths not found — fast BFS O-set is loaded")
    else:
        print("  FAIL: all_simple_paths still present — old exponential code is loaded")
        all_pass = False

    print()
    if all_pass:
        print("OVERALL: PASS — fast O-set code confirmed loaded from latest 402 branch tip")
    else:
        print("OVERALL: FAIL — see above; resolve before running cohort or SLURM job")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
