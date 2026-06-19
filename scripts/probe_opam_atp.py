"""Check opam for ATP packages needed by coq-hammer (Level 2)."""
import subprocess
import sys

result = subprocess.run(
    ["/opt/homebrew/bin/opam", "list"],
    capture_output=True,
    text=True,
    env={
        "HOME": "/Users/zuck016",
        "PATH": "/Users/zuck016/.opam/default/bin:/usr/bin:/bin",
        "OPAMROOT": "/Users/zuck016/.opam",
    },
)
lines = result.stdout.splitlines()
atp_lines = [l for l in lines if any(k in l.lower() for k in ["z3", "eprover", "vampire", "cvc4", "cvc5", "alt-ergo"])]
print("ATP-related opam packages:")
for l in atp_lines:
    print(" ", l)
if not atp_lines:
    print("  (none found)")
