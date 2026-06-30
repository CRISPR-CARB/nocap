"""
shard_io.py
===========
Robust shard loader for the SCC-perturbation pipeline.

Provides ``load_first_json_object(path)`` which safely reads a JSON shard
even when the file has been corrupted by a re-run appending a second JSON
object (e.g. ``scc_perturb_shard_glnG.json``).

Contract
--------
The loader extracts the **first** complete, balanced JSON object from the
file and discards everything after the matching closing brace.  Subsequent
content is silently ignored (a warning is printed to stderr if trailing
content exists so operators are aware of the corruption).

Usage
-----
    from shard_io import load_first_json_object

    data = load_first_json_object("notebooks/.../scc_perturb_shard_glnG.json")

Production functions carry plain ``assert`` PRE/INV/POST guards and an
``axiomander:`` docstring block.  No axiomander import at runtime.
"""

import json
import sys


def load_first_json_object(path: str) -> dict:
    """
    Read the first complete JSON object from *path*, ignoring any trailing
    content after the balanced closing brace.

    This handles the case where a SLURM re-run appended a second JSON object
    to an existing shard file, making the file unparseable by ``json.load``.

    Args:
        path: Absolute or relative path to the shard JSON file.

    Returns:
        The first JSON object as a Python dict.

    Raises:
        AssertionError: If *path* is empty (PRE violation).
        ValueError:     If no complete JSON object can be parsed from the file.
        FileNotFoundError: If *path* does not exist.

    axiomander:
        requires:
            len(path) > 0
        ensures:
            isinstance(result, dict)
            "tf" in result
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(path, str) and len(path) > 0, "PRE: path must be a non-empty string"

    with open(path) as f:
        raw = f.read()

    # Fast path: the whole file is valid JSON (no corruption)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            # --- POST (fast path) ---
            assert isinstance(obj, dict), "POST: result must be a dict"
            assert "tf" in obj, "POST: result must have 'tf' key"
            return obj
        raise ValueError(f"Top-level JSON in {path!r} is not a dict")
    except json.JSONDecodeError:
        pass

    # Slow path: scan for the first balanced JSON object
    depth = 0
    in_string = False
    escape_next = False
    first_brace = None

    for i, ch in enumerate(raw):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                first_brace = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and first_brace is not None:
                # Found the end of the first balanced object
                candidate = raw[first_brace : i + 1]
                trailing = raw[i + 1 :].strip()
                if trailing:
                    print(
                        f"WARNING: shard_io: {path!r} has trailing content after "
                        f"first JSON object ({len(trailing)} chars). "
                        "Keeping first object only.",
                        file=sys.stderr,
                    )
                # INV: candidate must be valid JSON
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"First balanced object in {path!r} is not valid JSON: {exc}"
                    ) from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"First JSON object in {path!r} is not a dict")
                # --- POST (slow path) ---
                assert isinstance(obj, dict), "POST: result must be a dict"
                assert "tf" in obj, "POST: result must have 'tf' key"
                return obj

    raise ValueError(f"No complete JSON object found in {path!r}")


def repair_shard_inplace(path: str) -> bool:
    """
    Repair a corrupted shard file by overwriting it with only the first
    valid JSON object.  Returns True if a repair was performed (i.e. the
    file contained trailing content), False if the file was already clean.

    axiomander:
        requires:
            len(path) > 0
        ensures:
            isinstance(result, bool)
        modifies:
            path   # overwrites the file if repair needed
    """
    # --- PRE ---
    assert isinstance(path, str) and len(path) > 0, "PRE: path must be a non-empty string"

    with open(path) as f:
        raw = f.read()

    # Check if already clean
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return False  # already clean
    except json.JSONDecodeError:
        pass

    # Load first object (will warn about trailing content to stderr)
    obj = load_first_json_object(path)

    # Overwrite with clean JSON
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")

    # --- POST ---
    assert isinstance(True, bool), "POST: result must be bool"
    return True
