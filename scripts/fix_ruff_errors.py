"""Fix all 46 ruff lint errors across the codebase."""

from pathlib import Path

ROOT = Path(__file__).parent.parent


def fix_file(rel_path, replacements):
    """Apply list of (old, new) string replacements to a file."""
    p = ROOT / rel_path
    text = p.read_text()
    for old, new in replacements:
        assert old in text, f"Pattern not found in {rel_path}: {old!r}"
        text = text.replace(old, new, 1)
    p.write_text(text)
    print(f"Fixed: {rel_path}")


# 1. scripts/check_failures.py: S607 - use full executable paths via sys.executable
fix_file(
    "scripts/check_failures.py",
    [
        (
            '"""Check manifest, mypy, docstr-coverage failures."""\nimport subprocess',
            '"""Check manifest, mypy, docstr-coverage failures."""\nimport subprocess\nimport sys',
        ),
        (
            'r = subprocess.run(["check-manifest", "--verbose"], capture_output=True, text=True)',
            'r = subprocess.run([sys.executable, "-m", "check_manifest", "--verbose"], capture_output=True, text=True)',
        ),
        (
            'r = subprocess.run(["mypy", "--ignore-missing-imports", "src/"], capture_output=True, text=True)',
            'r = subprocess.run([sys.executable, "-m", "mypy", "--ignore-missing-imports", "src/"], capture_output=True, text=True)',
        ),
        (
            'r = subprocess.run(["docstr-coverage", "src/", "tests/", "--skip-private", "--skip-magic"], capture_output=True, text=True)',
            'r = subprocess.run([sys.executable, "-m", "docstr_coverage", "src/", "tests/", "--skip-private", "--skip-magic"], capture_output=True, text=True)',
        ),
    ],
)

# 2. scripts/classify_scc_states.py: RUF002 - EN DASH -> HYPHEN-MINUS
p = ROOT / "scripts/classify_scc_states.py"
text = p.read_text()
# Replace EN DASH (U+2013) with hyphen-minus in the docstring lines
text = text.replace(
    "    identifiable   \u2013 joint cyclic_id query succeeded",
    "    identifiable   - joint cyclic_id query succeeded",
)
text = text.replace(
    "    unidentifiable \u2013 joint query failed AND cut is complete (a real result)",
    "    unidentifiable - joint query failed AND cut is complete (a real result)",
)
text = text.replace(
    "    cut_incomplete \u2013 do(B(t)) did NOT sever every in-SCC child\u2192t return path",
    "    cut_incomplete - do(B(t)) did NOT sever every in-SCC child->t return path",
)
p.write_text(text)
print("Fixed: scripts/classify_scc_states.py")

# 3. scripts/csd_break_worker.py: RUF002 - MINUS SIGN -> HYPHEN-MINUS
p = ROOT / "scripts/csd_break_worker.py"
text = p.read_text()
# Replace MINUS SIGN (U+2212) with hyphen-minus
text = text.replace("\u2212", "-")
p.write_text(text)
print("Fixed: scripts/csd_break_worker.py")

# 4. scripts/csd_calibrate_report.py: RUF046 - redundant int() casts
fix_file(
    "scripts/csd_calibrate_report.py",
    [
        (
            "    est_resolved = int(round(resolution_rate * N_FULL))\n    est_ident = int(round(ident_rate * N_FULL))",
            "    est_resolved = round(resolution_rate * N_FULL)\n    est_ident = round(ident_rate * N_FULL)",
        ),
    ],
)

# 5. scripts/csd_diagnose_nonident.py: RUF005 - list concatenation
fix_file(
    "scripts/csd_diagnose_nonident.py",
    [
        ('c in list(CAUSE_CATEGORIES) + ["unknown"]', 'c in [*list(CAUSE_CATEGORIES), "unknown"]'),
    ],
)

# 6. scripts/csd_identified_edges.py: E741 - ambiguous variable name `l`
p = ROOT / "scripts/csd_identified_edges.py"
text = p.read_text()
text = text.replace(
    '        fail_lines = [l for l in lines if "[FAIL ]" in l]\n        if fail_lines:\n            fail_found = True\n            print(f"  {log_file.name}:")\n            for l in fail_lines:\n                print(f"    {l}")',
    '        fail_lines = [ln for ln in lines if "[FAIL ]" in ln]\n        if fail_lines:\n            fail_found = True\n            print(f"  {log_file.name}:")\n            for ln in fail_lines:\n                print(f"    {ln}")',
)
text = text.replace("            for l in lines[-20:]:", "            for ln in lines[-20:]:")
text = text.replace('                print(f"    {l}")', '                print(f"    {ln}")')
p.write_text(text)
print("Fixed: scripts/csd_identified_edges.py")

# 7. scripts/patch_notebook_part4.py: RUF003 - MULTIPLICATION SIGN in comment
p = ROOT / "scripts/patch_notebook_part4.py"
text = p.read_text()
text = text.replace("\u00d7", "x")
p.write_text(text)
print("Fixed: scripts/patch_notebook_part4.py")

# 8. scripts/probe_fast_oset.py: S607
p = ROOT / "scripts/probe_fast_oset.py"
text = p.read_text()
# Need to see the context
print(f"probe_fast_oset.py line 28 content: {text.splitlines()[27]!r}")

# 9. scripts/smoke_csd_break.py: RUF003 - MINUS SIGN in comment
p = ROOT / "scripts/smoke_csd_break.py"
text = p.read_text()
text = text.replace("\u2212", "-")
p.write_text(text)
print("Fixed: scripts/smoke_csd_break.py")

# 10. scripts/smoke_csd_notebook.py: RUF005
fix_file(
    "scripts/smoke_csd_notebook.py",
    [
        ('list(CAUSE_CATEGORIES) + ["unknown"]', '[*list(CAUSE_CATEGORIES), "unknown"]'),
    ],
)

# 11. scripts/smoke_csd_scc_residual.py: S607
p = ROOT / "scripts/smoke_csd_scc_residual.py"
text = p.read_text()
print(f"smoke_csd_scc_residual.py line 140: {text.splitlines()[139]!r}")

# 12. tests/test_csd_break.py: RUF002 - MINUS SIGN
p = ROOT / "tests/test_csd_break.py"
text = p.read_text()
text = text.replace("\u2212", "-")
p.write_text(text)
print("Fixed: tests/test_csd_break.py")

# 13. tests/test_csd_recovery_bank.py: E402 + C405
p = ROOT / "tests/test_csd_recovery_bank.py"
text = p.read_text()
print("test_csd_recovery_bank.py lines 25-30:")
for i, ln in enumerate(text.splitlines()[24:32], 25):
    print(f"  {i}: {ln!r}")
print("test_csd_recovery_bank.py line 135-140:")
for i, ln in enumerate(text.splitlines()[134:142], 135):
    print(f"  {i}: {ln!r}")

print("\nDone reading context. Now applying remaining fixes...")
