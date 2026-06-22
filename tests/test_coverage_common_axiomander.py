r"""
tests/test_coverage_common_axiomander.py
=========================================
Axiomander-style contract tests for scripts/coverage_common.py pure helpers:

  assign_work(n_queries, task_id, n_tasks) -> list[int]
  merge_shards(shard_list)                 -> list[list]
  rows_to_matrix(rows, query_labels)       -> tuple[list, list, dict]

Each test class mirrors one contract category:
  - PRE  : precondition — what must be true before the call
  - POST : postcondition — what must be true about the return value
  - INV  : loop/structural invariant — what holds at every step

The adorned (contract-bearing) versions of the functions are defined inline
so the contracts are explicit and independently testable without modifying
the production script.

To regenerate the property-based companion tests run:
  axiomander gen-tests scripts/coverage_common.py \\
      --output tests/test_coverage_common_generated.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from coverage_common import assign_work, merge_shards, rows_to_matrix

# ---------------------------------------------------------------------------
# Adorned versions with explicit inline assert contracts
# ---------------------------------------------------------------------------


def assign_work_adorned(n_queries: int, task_id: int, n_tasks: int) -> list:
    """assign_work with explicit assert-based contracts."""
    # --- PRECONDITIONS ---
    assert isinstance(n_queries, int), "PRE: n_queries must be int"
    assert isinstance(n_tasks, int), "PRE: n_tasks must be int"
    assert isinstance(task_id, int), "PRE: task_id must be int"
    assert n_queries >= 0, "PRE: n_queries >= 0"
    assert n_tasks >= 1, "PRE: n_tasks >= 1"
    assert task_id >= 0, "PRE: task_id >= 0"
    assert task_id < n_tasks, "PRE: task_id < n_tasks"

    # --- BODY ---
    indices = list(range(task_id, n_queries, n_tasks))

    # --- POSTCONDITIONS ---
    assert isinstance(indices, list), "POST: result must be a list"
    assert all(isinstance(i, int) for i in indices), "POST: all indices must be int"
    assert all(0 <= i < n_queries for i in indices), "POST: all indices in [0, n_queries)"

    return indices


def merge_shards_adorned(shard_list: list) -> list:
    """merge_shards with explicit assert-based contracts."""
    # --- PRECONDITIONS ---
    assert isinstance(shard_list, list), "PRE: shard_list must be a list"

    seen: set = set()
    merged: list = []
    for shard in shard_list:
        # LOOP INVARIANT: each shard is a list
        assert isinstance(shard, list), "INV: each shard must be a list"
        for row in shard:
            # LOOP INVARIANT: each row is a 4-element sequence
            assert isinstance(row, list | tuple) and len(row) == 4, (
                "INV: each row must be a 4-element list/tuple"
            )
            key = (row[0], row[1], row[2])
            if key not in seen:
                seen.add(key)
                merged.append(list(row))

    # --- POSTCONDITIONS ---
    assert isinstance(merged, list), "POST: result must be a list"
    assert all(len(r) == 4 for r in merged), "POST: every row has 4 elements"
    keys = [(r[0], r[1], r[2]) for r in merged]
    assert len(keys) == len(set(keys)), "POST: no duplicate (tf1, candidate, outcome) keys"

    return merged


def rows_to_matrix_adorned(rows: list, query_labels: list) -> tuple:
    """rows_to_matrix with explicit assert-based contracts."""
    # --- PRECONDITIONS ---
    assert isinstance(rows, list), "PRE: rows must be a list"
    assert isinstance(query_labels, list), "PRE: query_labels must be a list"

    lookup: dict = {}
    for row in rows:
        # LOOP INVARIANT: each row is a 4-element sequence
        assert isinstance(row, list | tuple) and len(row) == 4, (
            "INV: each row must be a 4-element list/tuple"
        )
        tf1, candidate, outcome, found = row
        label = f"{tf1}->{outcome}"
        lookup[(candidate, label)] = bool(found)

    candidate_set = sorted({r[1] for r in rows})

    # --- POSTCONDITIONS ---
    assert isinstance(candidate_set, list), "POST: candidate_set must be a list"
    assert isinstance(lookup, dict), "POST: lookup must be a dict"
    assert all(isinstance(c, str) for c in candidate_set), (
        "POST: all candidates must be str"
    )
    assert candidate_set == sorted(candidate_set), "POST: candidate_set is sorted"

    return candidate_set, query_labels, lookup


# ---------------------------------------------------------------------------
# PRE: assign_work preconditions
# ---------------------------------------------------------------------------


class TestAssignWorkPreconditions:
    """Verify that precondition violations raise AssertionError."""

    def test_pre_n_queries_must_be_int(self):
        with pytest.raises(AssertionError, match="PRE: n_queries must be int"):
            assign_work_adorned(5.0, 0, 1)

    def test_pre_n_tasks_must_be_int(self):
        with pytest.raises(AssertionError, match="PRE: n_tasks must be int"):
            assign_work_adorned(5, 0, 1.0)

    def test_pre_task_id_must_be_int(self):
        with pytest.raises(AssertionError, match="PRE: task_id must be int"):
            assign_work_adorned(5, 0.0, 1)

    def test_pre_n_queries_non_negative(self):
        with pytest.raises(AssertionError, match="PRE: n_queries >= 0"):
            assign_work_adorned(-1, 0, 1)

    def test_pre_n_tasks_at_least_one(self):
        with pytest.raises(AssertionError, match="PRE: n_tasks >= 1"):
            assign_work_adorned(5, 0, 0)

    def test_pre_task_id_non_negative(self):
        with pytest.raises(AssertionError, match="PRE: task_id >= 0"):
            assign_work_adorned(5, -1, 2)

    def test_pre_task_id_less_than_n_tasks(self):
        with pytest.raises(AssertionError, match="PRE: task_id < n_tasks"):
            assign_work_adorned(5, 3, 3)

    def test_pre_valid_call_succeeds(self):
        """Valid inputs do not raise."""
        result = assign_work_adorned(6, 1, 3)
        assert result == [1, 4]


# ---------------------------------------------------------------------------
# POST: assign_work postconditions
# ---------------------------------------------------------------------------


class TestAssignWorkPostconditions:
    """Verify postconditions hold on valid inputs."""

    def test_post_result_is_list(self):
        assert isinstance(assign_work_adorned(5, 0, 1), list)

    def test_post_all_indices_in_range(self):
        result = assign_work_adorned(10, 2, 4)
        assert all(0 <= i < 10 for i in result)

    def test_post_all_indices_are_int(self):
        result = assign_work_adorned(7, 0, 3)
        assert all(isinstance(i, int) for i in result)

    def test_post_agrees_with_production(self):
        """Adorned version returns same value as production version."""
        for t in range(4):
            assert assign_work_adorned(13, t, 4) == assign_work(13, t, 4)


# ---------------------------------------------------------------------------
# PRE: merge_shards preconditions
# ---------------------------------------------------------------------------


class TestMergeShardsPreconditions:
    """Verify that precondition violations raise AssertionError."""

    def test_pre_shard_list_must_be_list(self):
        with pytest.raises(AssertionError, match="PRE: shard_list must be a list"):
            merge_shards_adorned(None)

    def test_pre_shard_list_tuple_rejected(self):
        with pytest.raises(AssertionError, match="PRE: shard_list must be a list"):
            merge_shards_adorned(())

    def test_pre_valid_empty_list_succeeds(self):
        assert merge_shards_adorned([]) == []


# ---------------------------------------------------------------------------
# INV: merge_shards loop invariants
# ---------------------------------------------------------------------------


class TestMergeShardsInvariants:
    """Verify that loop invariant violations raise AssertionError."""

    def test_inv_shard_must_be_list(self):
        """A non-list shard triggers the loop invariant."""
        with pytest.raises(AssertionError, match="INV: each shard must be a list"):
            merge_shards_adorned([("tf1", "c1", "o1", True)])  # tuple shard

    def test_inv_row_must_be_4_elements(self):
        """A row with wrong length triggers the loop invariant."""
        with pytest.raises(AssertionError, match="INV: each row must be a 4-element"):
            merge_shards_adorned([[["tf1", "c1", "o1"]]])  # 3-element row

    def test_inv_row_must_be_list_or_tuple(self):
        """A row that is a string triggers the loop invariant."""
        with pytest.raises(AssertionError, match="INV: each row must be a 4-element"):
            merge_shards_adorned([["not_a_row"]])


# ---------------------------------------------------------------------------
# POST: merge_shards postconditions
# ---------------------------------------------------------------------------


class TestMergeShardsPostconditions:
    """Verify postconditions hold on valid inputs."""

    def test_post_result_is_list(self):
        assert isinstance(merge_shards_adorned([]), list)

    def test_post_all_rows_have_four_elements(self):
        shards = [[["tf1", "c1", "o1", True], ["tf1", "c2", "o1", False]]]
        result = merge_shards_adorned(shards)
        assert all(len(r) == 4 for r in result)

    def test_post_no_duplicate_keys(self):
        shards = [
            [["A", "B", "C", True]],
            [["A", "B", "C", False]],  # duplicate key
        ]
        result = merge_shards_adorned(shards)
        keys = [(r[0], r[1], r[2]) for r in result]
        assert len(keys) == len(set(keys))

    def test_post_agrees_with_production(self):
        shards = [
            [["tf1", "c1", "o1", True], ["tf1", "c2", "o1", False]],
            [["tf2", "c1", "o2", True]],
        ]
        assert merge_shards_adorned(shards) == merge_shards(shards)


# ---------------------------------------------------------------------------
# PRE: rows_to_matrix preconditions
# ---------------------------------------------------------------------------


class TestRowsToMatrixPreconditions:
    """Verify that precondition violations raise AssertionError."""

    def test_pre_rows_must_be_list(self):
        with pytest.raises(AssertionError, match="PRE: rows must be a list"):
            rows_to_matrix_adorned(None, [])

    def test_pre_query_labels_must_be_list(self):
        with pytest.raises(AssertionError, match="PRE: query_labels must be a list"):
            rows_to_matrix_adorned([], None)

    def test_pre_valid_empty_inputs_succeed(self):
        result = rows_to_matrix_adorned([], [])
        assert result == ([], [], {})


# ---------------------------------------------------------------------------
# INV: rows_to_matrix loop invariants
# ---------------------------------------------------------------------------


class TestRowsToMatrixInvariants:
    """Verify that loop invariant violations raise AssertionError."""

    def test_inv_row_must_be_4_elements(self):
        """A row with wrong length triggers the loop invariant."""
        with pytest.raises(AssertionError, match="INV: each row must be a 4-element"):
            rows_to_matrix_adorned([["tf1", "c1", "o1"]], [])  # 3-element row

    def test_inv_row_must_be_list_or_tuple(self):
        """A row that is a string triggers the loop invariant."""
        with pytest.raises(AssertionError, match="INV: each row must be a 4-element"):
            rows_to_matrix_adorned(["not_a_row"], [])


# ---------------------------------------------------------------------------
# POST: rows_to_matrix postconditions
# ---------------------------------------------------------------------------


class TestRowsToMatrixPostconditions:
    """Verify postconditions hold on valid inputs."""

    def test_post_result_is_tuple(self):
        result = rows_to_matrix_adorned([], [])
        assert isinstance(result, tuple) and len(result) == 3

    def test_post_candidate_set_is_sorted(self):
        rows = [
            ["tf1", "zebra", "out1", True],
            ["tf1", "alpha", "out1", False],
        ]
        candidate_set, _, _ = rows_to_matrix_adorned(rows, ["tf1->out1"])
        assert candidate_set == sorted(candidate_set)

    def test_post_all_candidates_are_str(self):
        rows = [["tf1", "cand1", "out1", True]]
        candidate_set, _, _ = rows_to_matrix_adorned(rows, ["tf1->out1"])
        assert all(isinstance(c, str) for c in candidate_set)

    def test_post_lookup_is_dict(self):
        _, _, lookup = rows_to_matrix_adorned([], [])
        assert isinstance(lookup, dict)

    def test_post_agrees_with_production(self):
        rows = [
            ["tf1", "cand1", "out1", True],
            ["tf1", "cand2", "out1", False],
            ["tf2", "cand1", "out2", True],
        ]
        labels = ["tf1->out1", "tf2->out2"]
        expected = rows_to_matrix(rows, labels)
        actual = rows_to_matrix_adorned(rows, labels)
        assert actual[0] == expected[0]  # candidate_set
        assert actual[2] == expected[2]  # lookup
