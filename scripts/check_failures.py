"""Check manifest, mypy, docstr-coverage failures."""

import subprocess
import sys

print("=" * 60)
print("MANIFEST")
print("=" * 60)
r = subprocess.run(
    [sys.executable, "-m", "check_manifest", "--verbose"], capture_output=True, text=True
)
print(r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout)
print(r.stderr[-1000:] if len(r.stderr) > 1000 else r.stderr)

print("=" * 60)
print("MYPY")
print("=" * 60)
r = subprocess.run(
    [sys.executable, "-m", "mypy", "--ignore-missing-imports", "src/"],
    capture_output=True,
    text=True,
)
print(r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout)
print(r.stderr[-500:] if len(r.stderr) > 500 else r.stderr)

print("=" * 60)
print("DOCSTR-COVERAGE")
print("=" * 60)
r = subprocess.run(
    [sys.executable, "-m", "docstr_coverage", "src/", "tests/", "--skip-private", "--skip-magic"],
    capture_output=True,
    text=True,
)
print(r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout)
print(r.stderr[-500:] if len(r.stderr) > 500 else r.stderr)
