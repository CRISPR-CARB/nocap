"""
probe_axiomander_mcp.py
=======================
Smoke-test the axiomander MCP server by sending the full Cline handshake
sequence over stdin/stdout and checking the responses.

Run from the repo root:
  uv run python scripts/probe_axiomander_mcp.py 2>&1 | tail -30
"""

import json
import os
import subprocess
import sys

AXIOMANDER_ROOT = "/Users/zuck016/Projects/CausalInference/Vericoding/axiomander"
PYTHON = f"{AXIOMANDER_ROOT}/.venv/bin/python"

env = os.environ.copy()
env["PYTHONPATH"] = f"{AXIOMANDER_ROOT}/py"
env["AXIOMANDER_ROOT"] = AXIOMANDER_ROOT
env["VIRTUAL_ENV"] = f"{AXIOMANDER_ROOT}/.venv"
env["PATH"] = "/opt/homebrew/bin:/Users/zuck016/.opam/default/bin:/usr/bin:/bin:/usr/local/bin"

# Full Cline MCP handshake sequence:
# 1. initialize (with id=1)
# 2. notifications/initialized (no id — notification, should be silently ignored)
# 3. tools/list (with id=2)
messages = [
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "cline", "version": "3.0"},
        },
    },
    {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
]

stdin_data = "\n".join(json.dumps(m) for m in messages) + "\n"
stdin_bytes = stdin_data.encode()

print(f"Launching: {PYTHON} -m axiomander.oracle.mcp_server")
print(f"Sending {len(messages)} messages (initialize, notifications/initialized, tools/list)")
print()

try:
    result = subprocess.run(
        [PYTHON, "-m", "axiomander.oracle.mcp_server"],
        input=stdin_bytes,
        capture_output=True,
        timeout=15,
        cwd=AXIOMANDER_ROOT,
        env=env,
    )
    stdout = result.stdout.decode(errors="replace")
    stderr = result.stderr.decode(errors="replace")

    print("=== stdout (JSON-RPC responses) ===")
    responses = []
    for line in stdout.strip().splitlines():
        if line.strip():
            try:
                obj = json.loads(line)
                responses.append(obj)
                print(f"  {json.dumps(obj, indent=None)[:200]}")
            except json.JSONDecodeError:
                print(f"  (non-JSON): {line[:100]}")

    print()
    if stderr.strip():
        print("=== stderr ===")
        print(stderr[:500])
        print()

    print(f"=== exit code: {result.returncode} ===")
    print()

    # Validate responses
    errors = []
    # Should have exactly 2 responses (initialize + tools/list), NOT 3
    # (the notification should be silently ignored)
    if len(responses) != 2:
        errors.append(f"Expected 2 responses, got {len(responses)}")
    else:
        r1 = responses[0]
        if r1.get("id") != 1:
            errors.append(f"Response 1 id should be 1, got {r1.get('id')}")
        if "error" in r1:
            errors.append(f"Response 1 has error: {r1['error']}")
        if r1.get("result", {}).get("serverInfo", {}).get("name") != "axiomander":
            errors.append(f"Response 1 serverInfo.name wrong: {r1.get('result')}")

        r2 = responses[1]
        if r2.get("id") != 2:
            errors.append(f"Response 2 id should be 2, got {r2.get('id')}")
        if "error" in r2:
            errors.append(f"Response 2 has error: {r2['error']}")
        tools = r2.get("result", {}).get("tools", [])
        if not tools:
            errors.append("tools/list returned no tools")
        else:
            print(f"Tools available: {[t['name'] for t in tools]}")

    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("PASS: Full Cline handshake succeeded — notification silently ignored, tools listed.")

except subprocess.TimeoutExpired as e:
    print("Server did not exit within 15 s.")
    print("stdout so far:")
    print(e.stdout.decode(errors="replace") if e.stdout else "(empty)")
    print("stderr so far:")
    print(e.stderr.decode(errors="replace") if e.stderr else "(empty)")
    sys.exit(1)
except Exception as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)
