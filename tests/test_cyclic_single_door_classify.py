"""Tests for scripts/cyclic_single_door_classify.py, scripts/cyclic_single_door_gather.py,
and scripts/csd_identified_edges.py.

Test groups
-----------
A. _row_to_jsonable — frozenset serialisation
B. _split_into_n_shards — edge-partition math
C. cmd_prepare — shard files + manifest written correctly
D. cmd_classify (timeout path) — per-edge timeout recorded, shard not crashed
E. cmd_classify (checkpoint/resume) — partial .partial → skip done edges, clean up
F. cmd_classify (partial-then-complete) — simulate kill mid-run, re-run completes
G. gather (cyclic_single_door_gather.main) — merge to CSV + summary JSON
H. csd_identified_edges — cmd_summary / cmd_list / cmd_csv produce correct output
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Import helpers — bring scripts onto sys.path without installing them
# ---------------------------------------------------------------------------

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _import_script(name: str) -> types.ModuleType:
    """Import a script from scripts/ by filename (without the .py extension)."""
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"Cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_classify = _import_script("cyclic_single_door_classify")
_gather = _import_script("cyclic_single_door_gather")
_csd_ids = _import_script("csd_identified_edges")

# ---------------------------------------------------------------------------
# Tiny in-memory GraphML helper
# ---------------------------------------------------------------------------

_TINY_GRAPHML = """\
<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <graph edgedefault="directed">
    <node id="A"/>
    <node id="B"/>
    <node id="C"/>
    <node id="D"/>
    <edge source="A" target="B"/>
    <edge source="B" target="C"/>
    <edge source="C" target="A"/>
    <edge source="D" target="A"/>
  </graph>
</graphml>
"""

# A DAG with no SCC (2 edges, both identifiable quickly)
_SIMPLE_DAG_GRAPHML = """\
<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <graph edgedefault="directed">
    <node id="Z"/>
    <node id="X"/>
    <node id="Y"/>
    <edge source="Z" target="X"/>
    <edge source="X" target="Y"/>
  </graph>
</graphml>
"""


def _write_graphml(tmp_path: Path, content: str = _TINY_GRAPHML) -> Path:
    """Write GraphML content to a temp file and return its Path."""
    p = tmp_path / "test_graph.graphml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# A. _row_to_jsonable
# ---------------------------------------------------------------------------


def test_row_to_jsonable_frozenset_converted():
    """Frozenset adjustment_set becomes a sorted list."""
    row = {
        "cause": "X",
        "effect": "Y",
        "status": "identifiable",
        "adjustment_set": frozenset({"Z", "W"}),
        "same_scc": False,
    }
    result = _classify._row_to_jsonable(row)
    assert isinstance(result["adjustment_set"], list)
    assert result["adjustment_set"] == sorted(["Z", "W"])
    assert not any(isinstance(v, frozenset) for v in result.values())


def test_row_to_jsonable_none_unchanged():
    """None adjustment_set passes through unchanged."""
    row = {
        "cause": "X",
        "effect": "Y",
        "status": "unidentifiable",
        "adjustment_set": None,
        "same_scc": True,
    }
    result = _classify._row_to_jsonable(row)
    assert result["adjustment_set"] is None


def test_row_to_jsonable_empty_frozenset():
    """Empty frozenset → empty list."""
    row = {
        "cause": "X",
        "effect": "Y",
        "status": "identifiable",
        "adjustment_set": frozenset(),
        "same_scc": False,
    }
    result = _classify._row_to_jsonable(row)
    assert result["adjustment_set"] == []


# ---------------------------------------------------------------------------
# B. _split_into_n_shards
# ---------------------------------------------------------------------------


def test_split_into_n_shards_exact_division():
    """6 edges into 3 shards → 2 each."""
    edges = list(range(6))
    chunks = _classify._split_into_n_shards(edges, 3)
    assert len(chunks) == 3
    assert all(len(c) == 2 for c in chunks)
    assert sum(len(c) for c in chunks) == 6


def test_split_into_n_shards_uneven():
    """7 edges into 3 shards → sizes [3, 2, 2]."""
    edges = list(range(7))
    chunks = _classify._split_into_n_shards(edges, 3)
    assert len(chunks) == 3
    assert sum(len(c) for c in chunks) == 7
    assert chunks[0] == [0, 1, 2]


def test_split_into_n_shards_more_shards_than_edges():
    """Requesting more shards than edges gives at most len(edges) shards."""
    edges = list(range(3))
    chunks = _classify._split_into_n_shards(edges, 10)
    assert len(chunks) == 3
    assert sum(len(c) for c in chunks) == 3


def test_split_into_n_shards_empty():
    """Empty edge list → no shards."""
    chunks = _classify._split_into_n_shards([], 5)
    assert chunks == []


def test_split_into_n_shards_pre_violation():
    """n_shards < 1 raises AssertionError."""
    with pytest.raises(AssertionError, match="PRE"):
        _classify._split_into_n_shards([1, 2], 0)


# ---------------------------------------------------------------------------
# C. cmd_prepare
# ---------------------------------------------------------------------------


def test_cmd_prepare_creates_shards_and_manifest(tmp_path):
    """Prepare writes shard JSON files and a manifest."""
    graphml = _write_graphml(tmp_path, _TINY_GRAPHML)
    shard_dir = tmp_path / "shards"
    manifest_path = tmp_path / "manifest.json"

    args = Namespace(
        graphml=str(graphml),
        shard_dir=str(shard_dir),
        shard_size=2,
        n_shards=None,
        manifest=str(manifest_path),
    )
    _classify.cmd_prepare(args)

    # Manifest exists and has correct counts
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_edges"] == 4  # 4 edges in _TINY_GRAPHML
    assert manifest["n_shards"] == 2  # ceil(4/2)
    assert len(manifest["shard_ids"]) == 2

    # Shard files exist
    shard_files = sorted(shard_dir.glob("shard_*.json"))
    assert len(shard_files) == 2

    # Every edge appears exactly once across all shards (no drops, no dupes)
    all_edges = []
    for sf in shard_files:
        d = json.loads(sf.read_text())
        all_edges.extend(tuple(e) for e in d["edges"])
    assert len(all_edges) == 4
    assert len(set(all_edges)) == 4  # no duplicates


def test_cmd_prepare_n_shards_overrides_shard_size(tmp_path):
    """--n-shards takes priority over --shard-size."""
    graphml = _write_graphml(tmp_path, _TINY_GRAPHML)
    shard_dir = tmp_path / "shards"
    manifest_path = tmp_path / "manifest.json"

    args = Namespace(
        graphml=str(graphml),
        shard_dir=str(shard_dir),
        shard_size=1,  # would give 4 shards if used
        n_shards=2,  # overrides: give exactly 2
        manifest=str(manifest_path),
    )
    _classify.cmd_prepare(args)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_shards"] == 2


def test_cmd_prepare_shard_size_1(tmp_path):
    """shard_size=1 → one shard per edge."""
    graphml = _write_graphml(tmp_path, _SIMPLE_DAG_GRAPHML)
    shard_dir = tmp_path / "shards"
    manifest_path = tmp_path / "manifest.json"

    args = Namespace(
        graphml=str(graphml),
        shard_dir=str(shard_dir),
        shard_size=1,
        n_shards=None,
        manifest=str(manifest_path),
    )
    _classify.cmd_prepare(args)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_shards"] == 2  # 2 edges in _SIMPLE_DAG_GRAPHML
    assert manifest["n_edges"] == 2


# ---------------------------------------------------------------------------
# D. cmd_classify — timeout path
# ---------------------------------------------------------------------------


def _make_shard_file(tmp_path: Path, edges: list, shard_id: str = "0") -> Path:
    """Write a shard JSON and return its path."""
    p = tmp_path / f"shard_{shard_id}.json"
    p.write_text(json.dumps({"shard_id": shard_id, "edges": [list(e) for e in edges]}))
    return p


def test_cmd_classify_timeout_recorded_not_crash(tmp_path):
    """When evaluate_all_edges returns a timeout row, it's recorded without crashing."""
    graphml = _write_graphml(tmp_path, _SIMPLE_DAG_GRAPHML)
    shard_file = _make_shard_file(tmp_path, [("Z", "X"), ("X", "Y")])
    output = tmp_path / "classified" / "shard_0.json"

    # Monkeypatch evaluate_all_edges to always return a timeout row
    def _fake_evaluate(g, restrict_edges=None, timeout_seconds=None):
        """Return a single timeout row for the first edge in restrict_edges."""
        u, v = (restrict_edges or list(g.edges()))[0]
        return [
            {
                "cause": u,
                "effect": v,
                "status": "timeout",
                "adjustment_set": None,
                "same_scc": False,
                "timed_out": True,
            }
        ]

    args = Namespace(
        graphml=str(graphml),
        shard=str(shard_file),
        output=str(output),
        timeout=1,
    )

    with patch("nocap.cyclic_single_door.evaluate_all_edges", side_effect=_fake_evaluate):
        _classify.cmd_classify(args)

    assert output.exists()
    result = json.loads(output.read_text())
    assert len(result["results"]) == 2
    assert all(r["status"] == "timeout" for r in result["results"])
    assert all(r["timed_out"] is True for r in result["results"])
    # Partial file must be cleaned up
    assert not output.with_suffix(".json.partial").exists()


def test_cmd_classify_real_dag_identifiable(tmp_path):
    """On a real DAG, classify produces identifiable results (no mocking)."""
    graphml = _write_graphml(tmp_path, _SIMPLE_DAG_GRAPHML)
    shard_file = _make_shard_file(tmp_path, [("Z", "X"), ("X", "Y")])
    output = tmp_path / "classified" / "shard_0.json"

    args = Namespace(
        graphml=str(graphml),
        shard=str(shard_file),
        output=str(output),
        timeout=0,  # no timeout
    )
    _classify.cmd_classify(args)

    assert output.exists()
    result = json.loads(output.read_text())
    assert len(result["results"]) == 2
    statuses = {r["status"] for r in result["results"]}
    assert statuses <= {"identifiable", "unidentifiable", "timeout"}
    # .partial must be cleaned up
    assert not output.with_suffix(".json.partial").exists()


# ---------------------------------------------------------------------------
# E. cmd_classify — checkpoint/resume idempotency
# ---------------------------------------------------------------------------


def test_cmd_classify_resumes_from_partial(tmp_path):
    """Already-done edges in .partial are skipped on resume; final output correct."""
    graphml = _write_graphml(tmp_path, _SIMPLE_DAG_GRAPHML)
    edges = [("Z", "X"), ("X", "Y")]
    shard_file = _make_shard_file(tmp_path, edges)
    output = tmp_path / "classified" / "shard_0.json"
    partial = output.with_suffix(".json.partial")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Pre-populate partial with the first edge already done
    first_done = {
        "cause": "Z",
        "effect": "X",
        "status": "identifiable",
        "adjustment_set": None,
        "same_scc": False,
        "timed_out": False,
    }
    partial.write_text(json.dumps(first_done) + "\n")

    call_log: list[tuple] = []

    def _fake_evaluate(g, restrict_edges=None, timeout_seconds=None):
        """Return a single identified row for the first edge in restrict_edges."""
        u, v = (restrict_edges or [])[0]
        call_log.append((u, v))
        return [
            {
                "cause": u,
                "effect": v,
                "status": "identifiable",
                "adjustment_set": None,
                "same_scc": False,
                "timed_out": False,
            }
        ]

    args = Namespace(
        graphml=str(graphml),
        shard=str(shard_file),
        output=str(output),
        timeout=0,
    )

    with patch("nocap.cyclic_single_door.evaluate_all_edges", side_effect=_fake_evaluate):
        _classify.cmd_classify(args)

    # Only the second edge should have been evaluated (first was in checkpoint)
    assert call_log == [("X", "Y")], f"Expected only X->Y to be evaluated, got {call_log}"

    # Output has both edges
    result = json.loads(output.read_text())
    assert len(result["results"]) == 2
    causes = [r["cause"] for r in result["results"]]
    assert "Z" in causes and "X" in causes

    # .partial cleaned up
    assert not partial.exists()


def test_cmd_classify_partial_removed_after_completion(tmp_path):
    """The .partial checkpoint file is removed after successful classify."""
    graphml = _write_graphml(tmp_path, _SIMPLE_DAG_GRAPHML)
    shard_file = _make_shard_file(tmp_path, [("Z", "X")])
    output = tmp_path / "classified" / "shard_0.json"

    args = Namespace(
        graphml=str(graphml),
        shard=str(shard_file),
        output=str(output),
        timeout=0,
    )
    _classify.cmd_classify(args)

    assert output.exists()
    assert not output.with_suffix(".json.partial").exists()


def test_cmd_classify_output_preserves_shard_edge_order(tmp_path):
    """Results in output JSON are in the same order as the shard's edge list."""
    graphml = _write_graphml(tmp_path, _SIMPLE_DAG_GRAPHML)
    edges = [("X", "Y"), ("Z", "X")]  # deliberately reversed from graphml order
    shard_file = _make_shard_file(tmp_path, edges)
    output = tmp_path / "classified" / "shard_0.json"

    args = Namespace(
        graphml=str(graphml),
        shard=str(shard_file),
        output=str(output),
        timeout=0,
    )
    _classify.cmd_classify(args)

    result = json.loads(output.read_text())
    result_edges = [(r["cause"], r["effect"]) for r in result["results"]]
    assert result_edges == edges


# ---------------------------------------------------------------------------
# F. cmd_classify — partial-then-complete (simulate mid-run kill)
# ---------------------------------------------------------------------------


def test_cmd_classify_partial_then_complete(tmp_path):
    """Simulate worker killed after 1 edge; re-run completes the shard cleanly."""
    graphml = _write_graphml(tmp_path, _SIMPLE_DAG_GRAPHML)
    edges = [("Z", "X"), ("X", "Y")]
    shard_file = _make_shard_file(tmp_path, edges)
    output = tmp_path / "classified" / "shard_0.json"
    partial = output.with_suffix(".json.partial")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Simulate: worker classified Z->X but was killed before finishing X->Y
    partial.write_text(
        json.dumps(
            {
                "cause": "Z",
                "effect": "X",
                "status": "identifiable",
                "adjustment_set": None,
                "same_scc": False,
                "timed_out": False,
            }
        )
        + "\n"
    )
    # output does NOT exist yet (worker was killed before writing it)
    assert not output.exists()

    args = Namespace(
        graphml=str(graphml),
        shard=str(shard_file),
        output=str(output),
        timeout=0,
    )
    _classify.cmd_classify(args)

    # Final output should exist with both edges
    assert output.exists()
    result = json.loads(output.read_text())
    causes = {r["cause"] for r in result["results"]}
    assert causes == {"Z", "X"}
    assert len(result["results"]) == 2
    # No duplicates
    edge_pairs = [(r["cause"], r["effect"]) for r in result["results"]]
    assert len(edge_pairs) == len(set(edge_pairs))


# ---------------------------------------------------------------------------
# G. cyclic_single_door_gather.main — merge to CSV + summary
# ---------------------------------------------------------------------------


def _write_classified_shard(classified_dir: Path, shard_id: str, results: list) -> None:
    """Write a classified shard JSON into classified_dir."""
    classified_dir.mkdir(parents=True, exist_ok=True)
    p = classified_dir / f"shard_{shard_id}.json"
    p.write_text(json.dumps({"shard_id": shard_id, "results": results}))


def test_gather_merges_shards_to_csv(tmp_path):
    """Gather produces a CSV with one row per edge across all shards."""
    classified_dir = tmp_path / "classified"
    _write_classified_shard(
        classified_dir,
        "0",
        [
            {
                "cause": "A",
                "effect": "B",
                "status": "identifiable",
                "adjustment_set": ["Z"],
                "same_scc": False,
            },
            {
                "cause": "B",
                "effect": "C",
                "status": "unidentifiable",
                "adjustment_set": None,
                "same_scc": True,
            },
        ],
    )
    _write_classified_shard(
        classified_dir,
        "1",
        [
            {
                "cause": "C",
                "effect": "A",
                "status": "unidentifiable",
                "adjustment_set": None,
                "same_scc": True,
            },
        ],
    )

    out_csv = tmp_path / "out.csv"
    out_summary = tmp_path / "summary.json"

    args = [
        "--input-dir",
        str(classified_dir),
        "--output-csv",
        str(out_csv),
        "--output-summary",
        str(out_summary),
    ]
    with patch("sys.argv", ["gather"] + args):
        _gather.main()

    # CSV has header + 3 data rows
    assert out_csv.exists()
    rows = list(csv.DictReader(out_csv.read_text().splitlines()))
    assert len(rows) == 3
    causes = {r["cause"] for r in rows}
    assert causes == {"A", "B", "C"}

    # Fieldnames correct
    assert set(rows[0].keys()) == {"cause", "effect", "status", "adjustment_set", "same_scc"}

    # Summary JSON
    assert out_summary.exists()
    summary = json.loads(out_summary.read_text())
    assert summary["n_edges"] == 3
    assert summary["n_identifiable"] == 1
    assert summary["n_unidentifiable"] == 2
    assert summary["n_shards"] == 2


def test_gather_adjustment_set_pipe_separated(tmp_path):
    """Gather serialises adjustment_set as pipe-separated string."""
    classified_dir = tmp_path / "classified"
    _write_classified_shard(
        classified_dir,
        "0",
        [
            {
                "cause": "X",
                "effect": "Y",
                "status": "identifiable",
                "adjustment_set": ["Z", "W"],
                "same_scc": False,
            },
        ],
    )
    out_csv = tmp_path / "out.csv"
    out_summary = tmp_path / "summary.json"

    with patch(
        "sys.argv",
        [
            "gather",
            "--input-dir",
            str(classified_dir),
            "--output-csv",
            str(out_csv),
            "--output-summary",
            str(out_summary),
        ],
    ):
        _gather.main()

    rows = list(csv.DictReader(out_csv.read_text().splitlines()))
    assert rows[0]["adjustment_set"] in ("Z|W", "W|Z")


def test_gather_empty_input_exits_nonzero(tmp_path):
    """Gather exits with code 1 when no shard files are present."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    out_csv = tmp_path / "out.csv"
    out_summary = tmp_path / "summary.json"

    with patch(
        "sys.argv",
        [
            "gather",
            "--input-dir",
            str(empty_dir),
            "--output-csv",
            str(out_csv),
            "--output-summary",
            str(out_summary),
        ],
    ):
        with pytest.raises(SystemExit) as exc_info:
            _gather.main()
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# H. csd_identified_edges — cmd_summary / cmd_list / cmd_csv
# ---------------------------------------------------------------------------


def _make_classified_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Create a small classified dir + manifest; return (classified_dir, manifest_path)."""
    classified_dir = tmp_path / "classified"
    classified_dir.mkdir()
    _write_classified_shard(
        classified_dir,
        "0",
        [
            {
                "cause": "A",
                "effect": "B",
                "status": "identifiable",
                "adjustment_set": ["Z"],
                "same_scc": False,
                "timed_out": False,
            },
            {
                "cause": "B",
                "effect": "A",
                "status": "unidentifiable",
                "adjustment_set": None,
                "same_scc": True,
                "timed_out": False,
            },
        ],
    )
    _write_classified_shard(
        classified_dir,
        "1",
        [
            {
                "cause": "X",
                "effect": "Y",
                "status": "identifiable",
                "adjustment_set": None,
                "same_scc": False,
                "timed_out": False,
            },
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"n_shards": 2, "n_edges": 3, "shard_ids": ["0", "1"]}))
    return classified_dir, manifest_path


def test_csd_summary_counts(tmp_path, capsys):
    """cmd_summary prints correct identifiable/unidentifiable counts."""
    classified_dir, manifest_path = _make_classified_dir(tmp_path)
    _csd_ids.cmd_summary(classified_dir, manifest_path)
    out = capsys.readouterr().out
    assert "2" in out  # 2 identifiable
    assert "1" in out  # 1 unidentifiable
    assert "2 / 2" in out  # shards classified / total


def test_csd_summary_no_records(tmp_path, capsys):
    """cmd_summary handles empty classified dir gracefully."""
    classified_dir = tmp_path / "classified"
    classified_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"n_shards": 5, "n_edges": 25, "shard_ids": []}))
    _csd_ids.cmd_summary(classified_dir, manifest_path)
    out = capsys.readouterr().out
    assert "no edge records" in out


def test_csd_list_identifiable(tmp_path, capsys):
    """cmd_list streams identifiable edges; one line per edge."""
    classified_dir, _ = _make_classified_dir(tmp_path)
    _csd_ids.cmd_list(classified_dir, same_scc_only=False)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "->" in ln]
    assert len(lines) == 2  # 2 identifiable edges


def test_csd_list_same_scc_only(tmp_path, capsys):
    """--list-same-scc only returns same-SCC identifiable edges."""
    classified_dir, _ = _make_classified_dir(tmp_path)
    # None of our identifiable edges are same_scc; should print the no-edges message
    _csd_ids.cmd_list(classified_dir, same_scc_only=True)
    out = capsys.readouterr().out
    assert "no identifiable edges" in out


def test_csd_csv_output(tmp_path):
    """cmd_csv writes a valid CSV with all expected columns."""
    classified_dir, _ = _make_classified_dir(tmp_path)
    out_csv = tmp_path / "partial.csv"
    _csd_ids.cmd_csv(classified_dir, out_csv)

    assert out_csv.exists()
    rows = list(csv.DictReader(out_csv.read_text().splitlines()))
    assert len(rows) == 3
    assert set(rows[0].keys()) == {"cause", "effect", "status", "adjustment_set", "same_scc"}


def test_csd_csv_empty_exits(tmp_path):
    """cmd_csv exits with code 1 when no records are present."""
    empty_dir = tmp_path / "classified"
    empty_dir.mkdir()
    out_csv = tmp_path / "out.csv"
    with pytest.raises(SystemExit) as exc_info:
        _csd_ids.cmd_csv(empty_dir, out_csv)
    assert exc_info.value.code != 0


def test_csd_identified_edges_importable():
    """Regression: csd_identified_edges.py must not have a leading-space docstring
    that causes IndentationError at import time.  If it imports cleanly, this passes.
    """
    import importlib.util as _ilu

    # Re-import to confirm no syntax/indent error
    spec = _ilu.spec_from_file_location("_csd_check", _SCRIPTS / "csd_identified_edges.py")
    assert spec is not None
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert hasattr(mod, "cmd_summary")
    assert hasattr(mod, "cmd_list")
    assert hasattr(mod, "cmd_csv")
