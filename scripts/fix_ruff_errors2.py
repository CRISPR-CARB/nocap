"""Fix all remaining ruff lint errors (round 2)."""

from pathlib import Path

ROOT = Path(__file__).parent.parent


def fix_file(rel_path, replacements):
    """Apply list of (old, new) string replacements to a file."""
    p = ROOT / rel_path
    text = p.read_text()
    for old, new in replacements:
        if old not in text:
            print(f"  WARNING: pattern not found in {rel_path}: {old!r}")
            continue
        text = text.replace(old, new, 1)
    p.write_text(text)
    print(f"Fixed: {rel_path}")


# -----------------------------------------------------------------------
# 1. scripts/fix_ruff_errors.py: F401 (unused re), RUF003 (EN DASH in comment),
#    RUF003 (MINUS SIGN in comment), E741 (ambiguous l in f-string)
# -----------------------------------------------------------------------
p = ROOT / "scripts/fix_ruff_errors.py"
text = p.read_text()
# Remove unused 'import re'
text = text.replace("import re\n", "", 1)
# Replace EN DASH in comment
text = text.replace(
    "# Replace EN DASH (\u2013) with hyphen-minus in the docstring lines",
    "# Replace EN DASH (U+2013) with hyphen-minus in the docstring lines",
)
# Replace MINUS SIGN in comment
text = text.replace(
    "# Replace MINUS SIGN (\u2212, U+2212) with hyphen-minus",
    "# Replace MINUS SIGN (U+2212) with hyphen-minus",
)
# Fix E741 - ambiguous variable 'l' in f-strings (lines 143, 146)
text = text.replace(
    '    print(f"smoke_csd_scc_residual.py line 140: {text.splitlines()[139]!r}")\n',
    '    print(f"smoke_csd_scc_residual.py line 140: {text.splitlines()[139]!r}")\n',
)
# Fix the actual E741 - f-strings with {l} (ambiguous name)
# These are the print statements using f"  {i}: {ln!r}" - already fixed actually
# Lines 143, 146 are the for ln enumerate lines - need to check actual content
p.write_text(text)
print("Fixed: scripts/fix_ruff_errors.py (partial)")

# Re-read and check lines 140-150
p = ROOT / "scripts/fix_ruff_errors.py"
lines = p.read_text().splitlines()
for i, ln in enumerate(lines[138:152], 139):
    print(f"  {i}: {ln!r}")

# -----------------------------------------------------------------------
# 2. scripts/smoke_csd_scc_residual.py: S607 - partial executable
# -----------------------------------------------------------------------
fix_file(
    "scripts/smoke_csd_scc_residual.py",
    [
        (
            '        ["uv", "run", "python", str(fig_script)],',
            '        [shutil.which("uv") or "uv", "run", "python", str(fig_script)],',
        ),
    ],
)
# Also need to add 'import shutil' if not present
p = ROOT / "scripts/smoke_csd_scc_residual.py"
text = p.read_text()
if "import shutil" not in text:
    # Add after the first import block
    text = text.replace("import subprocess\n", "import shutil\nimport subprocess\n", 1)
    p.write_text(text)
    print("  Added import shutil to scripts/smoke_csd_scc_residual.py")

# -----------------------------------------------------------------------
# 3. tests/test_csd_recovery_bank.py: E402, C405
# -----------------------------------------------------------------------
p = ROOT / "tests/test_csd_recovery_bank.py"
text = p.read_text()
# C405: set(["A"]) -> {"A"}
text = text.replace('crb._build_do_scc_map(G, set(["A"]))', 'crb._build_do_scc_map(G, {"A"})')
p.write_text(text)
print("Fixed: tests/test_csd_recovery_bank.py (C405)")
# E402 is about import after non-import code - we can't easily fix without seeing more context
# Check if there's a noqa option needed
lines = p.read_text().splitlines()
print(f"  Line 28: {lines[27]!r}")

# -----------------------------------------------------------------------
# 4. tests/test_csd_rescue.py: D401 - imperative mood docstrings (lines 47, 54, 61)
# -----------------------------------------------------------------------
fix_file(
    "tests/test_csd_rescue.py",
    [
        (
            '    """A -> B -> A: both edges are in a 2-cycle."""',
            '    """Build graph with A -> B -> A (both edges in a 2-cycle)."""',
        ),
        (
            '    """A -> C -> D -> A (long feedback loop, removing A->C still leaves A in same SCC via D)."""',
            '    """Build long feedback loop A -> C -> D -> A (removing A->C leaves A in same SCC via D)."""',
        ),
        (
            '    """A -> B is the only link making A and B in the same SCC.\n    (B -> A is NOT present; instead B -> C -> A)\n    Removing A->B dissolves the SCC.\n    """',
            '    """Build graph where removing A->B dissolves the SCC.\n\n    A -> B is the only link making A and B in the same SCC\n    (B -> A is NOT present; instead B -> C -> A).\n    """',
        ),
    ],
)

# -----------------------------------------------------------------------
# 5. tests/test_csd_scc_residual.py: D401 line 23
# -----------------------------------------------------------------------
p = ROOT / "tests/test_csd_scc_residual.py"
lines = p.read_text().splitlines()
print(f"test_csd_scc_residual.py line 23: {lines[22]!r}")

# -----------------------------------------------------------------------
# 6. tests/test_scc_recovery_bank.py: E402 (line 27), D401 (lines 34, 41, 48), C405 (line 175)
# -----------------------------------------------------------------------
p = ROOT / "tests/test_scc_recovery_bank.py"
text = p.read_text()
# C405: set(["A"]) -> {"A"} etc.
text = text.replace('srb._build_do_scc_map(G, set(["A"]))', 'srb._build_do_scc_map(G, {"A"})')
p.write_text(text)
print("Fixed: tests/test_scc_recovery_bank.py (C405)")
# Read docstrings on lines 34, 41, 48
lines = p.read_text().splitlines()
print(f"  Line 34: {lines[33]!r}")
print(f"  Line 41: {lines[40]!r}")
print(f"  Line 48: {lines[47]!r}")

# -----------------------------------------------------------------------
# 7. tests/test_cyclic_single_door_axiomander.py: F541 (lines 307, 308)
# -----------------------------------------------------------------------
p = ROOT / "tests/test_cyclic_single_door_axiomander.py"
lines = p.read_text().splitlines()
print(f"test_cyclic_single_door_axiomander.py line 307: {lines[306]!r}")
print(f"test_cyclic_single_door_axiomander.py line 308: {lines[307]!r}")

# -----------------------------------------------------------------------
# 8. tests/test_cyclic_single_door_classify.py: E741 line 654
# -----------------------------------------------------------------------
p = ROOT / "tests/test_cyclic_single_door_classify.py"
lines = p.read_text().splitlines()
print(f"test_cyclic_single_door_classify.py line 654: {lines[653]!r}")

# -----------------------------------------------------------------------
# 9. tests/test_oset_kernel_axiomander.py: RUF002 (line 92), F541 (lines 579-580)
# -----------------------------------------------------------------------
p = ROOT / "tests/test_oset_kernel_axiomander.py"
lines = p.read_text().splitlines()
print(f"test_oset_kernel_axiomander.py line 92: {lines[91]!r}")
print(f"test_oset_kernel_axiomander.py line 579: {lines[578]!r}")
print(f"test_oset_kernel_axiomander.py line 580: {lines[579]!r}")

print("\nDone round 2 (partial). See output above for remaining fixes needed.")
