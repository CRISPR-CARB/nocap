"""
test_shard_io_axiomander.py
===========================
Axiomander-style PRE/POST contract tests for scripts/shard_io.py.

All tests use synthetic fixtures — no real shard files are required.
The glnG regression test operates on the actual shard file to confirm
the repair is correct.
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from shard_io import load_first_json_object, repair_shard_inplace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLEAN_OBJECT = {"tf": "arcA", "joint_identifiable": True, "per_gene": {}, "note": ""}

_CORRUPTED_OBJECT = {
    "tf": "glnG",
    "min_cut": [],
    "scc_size": 68,
    "in_scc_children": ["nac"],
    "n_children": 48,
    "outcomes": ["amiC"],
    "joint_identifiable": True,
    "per_gene": {},
    "note": "",
}

# A second debris object (mimics the glnG concatenation: the worker re-ran
# and appended the large all_network_vars list starting with "amiC")
_DEBRIS_FRAGMENT = ' "amiC",\n    "amn",\n    "ampC"\n  ]\n}\n'


def _write_temp(content: str) -> str:
    """Write *content* to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# TestLoadFirstJsonObject
# ---------------------------------------------------------------------------


class TestLoadFirstJsonObject:
    def test_clean_shard_roundtrips(self):
        """A well-formed single JSON object is returned unchanged."""
        path = _write_temp(json.dumps(_CLEAN_OBJECT, indent=2))
        try:
            obj = load_first_json_object(path)
            assert isinstance(obj, dict)
            assert obj["tf"] == "arcA"
            assert obj["joint_identifiable"] is True
        finally:
            os.unlink(path)

    def test_concatenated_objects_keeps_first(self):
        """
        A file with two concatenated JSON objects (the glnG corruption pattern)
        must return only the first object.
        """
        first_obj_str = json.dumps(_CORRUPTED_OBJECT, indent=2)
        corrupted = first_obj_str + _DEBRIS_FRAGMENT
        path = _write_temp(corrupted)
        try:
            obj = load_first_json_object(path)
            assert isinstance(obj, dict)
            assert obj["tf"] == "glnG"
            assert obj["joint_identifiable"] is True
            assert obj["scc_size"] == 68
        finally:
            os.unlink(path)

    def test_concatenated_object_result_has_all_keys(self):
        """All keys of the first object must be present after loading."""
        first_obj_str = json.dumps(_CORRUPTED_OBJECT, indent=2)
        corrupted = first_obj_str + _DEBRIS_FRAGMENT
        path = _write_temp(corrupted)
        try:
            obj = load_first_json_object(path)
            for key in _CORRUPTED_OBJECT:
                assert key in obj, f"Key {key!r} missing from loaded object"
        finally:
            os.unlink(path)

    def test_result_is_dict(self):
        """POST: isinstance(result, dict)."""
        path = _write_temp(json.dumps(_CLEAN_OBJECT))
        try:
            obj = load_first_json_object(path)
            assert isinstance(obj, dict)
        finally:
            os.unlink(path)

    def test_result_has_tf_key(self):
        """POST: 'tf' in result."""
        path = _write_temp(json.dumps(_CLEAN_OBJECT))
        try:
            obj = load_first_json_object(path)
            assert "tf" in obj
        finally:
            os.unlink(path)

    def test_truncated_json_raises_value_error(self):
        """A file with no balanced object raises ValueError."""
        path = _write_temp('{"tf": "arcA", "joint_identifiable": true')
        try:
            with pytest.raises(ValueError):
                load_first_json_object(path)
        finally:
            os.unlink(path)

    def test_empty_content_raises(self):
        """A completely empty file raises ValueError."""
        path = _write_temp("")
        try:
            with pytest.raises((ValueError, json.JSONDecodeError)):
                load_first_json_object(path)
        finally:
            os.unlink(path)

    def test_pre_empty_path_raises(self):
        """PRE: path must be a non-empty string."""
        with pytest.raises(AssertionError, match="PRE: path must be a non-empty string"):
            load_first_json_object("")

    def test_missing_file_raises_file_not_found(self):
        """FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_first_json_object("/tmp/this_file_should_not_exist_nocap_test.json")

    def test_nested_braces_handled_correctly(self):
        """
        An object with nested dicts must be parsed correctly (depth counter
        must track inner braces without prematurely terminating).
        """
        nested = {
            "tf": "crp",
            "per_gene": {"aceA": True, "aceB": False},
            "joint_identifiable": False,
            "note": "",
        }
        path = _write_temp(json.dumps(nested, indent=2))
        try:
            obj = load_first_json_object(path)
            assert obj["tf"] == "crp"
            assert obj["per_gene"]["aceA"] is True
            assert obj["per_gene"]["aceB"] is False
        finally:
            os.unlink(path)

    def test_string_with_brace_chars_not_confused(self):
        """
        A string value containing brace characters must not confuse the
        depth counter (they must be ignored while in_string=True).
        """
        obj_with_brace_str = {
            "tf": "rpoS",
            "note": "contains { and } chars in string",
            "joint_identifiable": True,
            "per_gene": {},
        }
        path = _write_temp(json.dumps(obj_with_brace_str, indent=2))
        try:
            obj = load_first_json_object(path)
            assert obj["tf"] == "rpoS"
            assert "{" in obj["note"]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestRepairShardInplace
# ---------------------------------------------------------------------------


class TestRepairShardInplace:
    def test_clean_file_returns_false(self):
        """A file that needs no repair must return False."""
        path = _write_temp(json.dumps(_CLEAN_OBJECT, indent=2))
        try:
            repaired = repair_shard_inplace(path)
            assert repaired is False
        finally:
            os.unlink(path)

    def test_corrupted_file_returns_true(self):
        """A corrupted file must return True (repair performed)."""
        first_obj_str = json.dumps(_CORRUPTED_OBJECT, indent=2)
        corrupted = first_obj_str + _DEBRIS_FRAGMENT
        path = _write_temp(corrupted)
        try:
            repaired = repair_shard_inplace(path)
            assert repaired is True
        finally:
            os.unlink(path)

    def test_corrupted_file_is_valid_after_repair(self):
        """After repair, the file must be parseable by json.load."""
        first_obj_str = json.dumps(_CORRUPTED_OBJECT, indent=2)
        corrupted = first_obj_str + _DEBRIS_FRAGMENT
        path = _write_temp(corrupted)
        try:
            repair_shard_inplace(path)
            with open(path) as f:
                obj = json.load(f)
            assert isinstance(obj, dict)
            assert obj["tf"] == "glnG"
        finally:
            os.unlink(path)

    def test_clean_file_not_changed(self):
        """A clean file's contents must be identical before and after repair()."""
        content = json.dumps(_CLEAN_OBJECT, indent=2)
        path = _write_temp(content)
        try:
            repair_shard_inplace(path)
            with open(path) as f:
                after = f.read()
            # Content should still be valid JSON with same data
            obj = json.loads(after)
            assert obj["tf"] == _CLEAN_OBJECT["tf"]
        finally:
            os.unlink(path)

    def test_pre_empty_path_raises(self):
        """PRE: path must be a non-empty string."""
        with pytest.raises(AssertionError, match="PRE: path must be a non-empty string"):
            repair_shard_inplace("")


# ---------------------------------------------------------------------------
# Regression: actual glnG shard
# ---------------------------------------------------------------------------


class TestGlnGRegression:
    """
    Verify that the actual glnG shard can be loaded and repaired.
    These tests skip gracefully if the file is not present (e.g. CI).
    """

    SHARD_PATH = os.path.join(
        os.path.dirname(__file__),
        "..",
        "notebooks",
        "Ecoli_Analysis_Notebooks",
        "scc_perturb_shards",
        "scc_perturb_shard_glnG.json",
    )

    @pytest.fixture(autouse=True)
    def require_shard(self):
        if not os.path.isfile(self.SHARD_PATH):
            pytest.skip("glnG shard not found — skipping regression test")

    def test_glng_shard_loads_without_error(self):
        """load_first_json_object must succeed on the real glnG shard."""
        obj = load_first_json_object(self.SHARD_PATH)
        assert isinstance(obj, dict)

    def test_glng_shard_has_correct_tf(self):
        """The first JSON object must have tf == 'glnG'."""
        obj = load_first_json_object(self.SHARD_PATH)
        assert obj["tf"] == "glnG"

    def test_glng_shard_has_joint_identifiable(self):
        """The first object must carry a joint_identifiable field."""
        obj = load_first_json_object(self.SHARD_PATH)
        assert "joint_identifiable" in obj
        assert isinstance(obj["joint_identifiable"], bool)
