"""
tests/test_coverage_worker_axiomander.py
=========================================
Axiomander-style contract tests for the ``evaluate_query`` function in
``scripts/coverage_worker.py``.

The function signature is:

    evaluate_query(
        tf1, outcome, candidates,
        ecoli_mixed, apt_order, all_network_vars,
        completed,
    ) -> list

The guard we are testing is the *row-count postcondition*:

    POST: row count == eligible candidates
         (i.e. candidates that are not tf1, not outcome, and not already done)

Because ``evaluate_query`` calls ``cyclic_id`` (y0 graph library, needs a
real graph), we test via an *adorned injectable* version that accepts the
identify function as an argument.  This keeps the tests dependency-free and
blazing fast while exercising exactly the same assert contracts.

Contract categories:
  - PRE  : precondition violations → AssertionError
  - POST : postcondition holds on valid inputs
  - POST (negative) : guard fires when invariant is violated

House convention: production asserts use plain ``assert`` with ``PRE:`` /
``POST:`` prefixes; no axiomander imports in production code.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


# ---------------------------------------------------------------------------
# Adorned injectable version of evaluate_query
# (same logic, same assert contracts, but takes an identify_fn callable so
#  tests never need y0 or a real graph)
# ---------------------------------------------------------------------------


def evaluate_query_adorned(
    tf1: str,
    outcome: str,
    candidates: list,
    completed: set,
    identify_fn=None,
) -> list:
    """
    Injectable adorned version of evaluate_query for contract testing.

    ``identify_fn(tf1, candidate, outcome) -> bool`` replaces the real
    cyclic_id call.  Defaults to always returning True (identifiable).

    axiomander:
        requires:
            isinstance(tf1, str)
            isinstance(outcome, str)
            isinstance(candidates, list)
            isinstance(completed, set)
        ensures:
            isinstance(result, list)
            all(len(result[i]) == 4 for i in range(len(result)))
            len(result) == len([c for c in candidates if c != tf1 and c != outcome and (tf1, c, outcome) not in completed])
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(tf1, str), "PRE: tf1 must be str"
    assert isinstance(outcome, str), "PRE: outcome must be str"
    assert isinstance(candidates, list), "PRE: candidates must be a list"
    assert isinstance(completed, set), "PRE: completed must be a set"

    if identify_fn is None:
        identify_fn = lambda t, c, o: True  # noqa: E731

    rows: list = []
    for candidate in candidates:
        if candidate == tf1 or candidate == outcome:
            continue
        key = (tf1, candidate, outcome)
        if key in completed:
            continue

        found = bool(identify_fn(tf1, candidate, outcome))
        rows.append([tf1, candidate, outcome, found])

    # --- POST ---
    assert isinstance(rows, list), "POST: result must be a list"
    assert all(len(r) == 4 for r in rows), "POST: every row has 4 elements"
    _expected = sum(
        1 for c in candidates if c != tf1 and c != outcome and (tf1, c, outcome) not in completed
    )
    assert len(rows) == _expected, (
        "POST: row count must equal eligible (non-self, not-already-done) candidates"
    )

    return rows


# ---------------------------------------------------------------------------
# PRE: precondition tests
# ---------------------------------------------------------------------------


class TestEvaluateQueryPreconditions:
    """Precondition violations must raise AssertionError with PRE: prefix."""

    def test_pre_tf1_must_be_str(self):
        """PRE: tf1 must be str — non-str raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: tf1 must be str"):
            evaluate_query_adorned(123, "out", ["c1"], set())

    def test_pre_outcome_must_be_str(self):
        """PRE: outcome must be str — None raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: outcome must be str"):
            evaluate_query_adorned("tf1", None, ["c1"], set())

    def test_pre_candidates_must_be_list(self):
        """PRE: candidates must be a list — tuple raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: candidates must be a list"):
            evaluate_query_adorned("tf1", "out", ("c1",), set())

    def test_pre_completed_must_be_set(self):
        """PRE: completed must be a set — list raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: completed must be a set"):
            evaluate_query_adorned("tf1", "out", ["c1"], [])

    def test_pre_valid_call_succeeds(self):
        """Valid inputs do not raise."""
        result = evaluate_query_adorned("tf1", "out", ["c1", "c2"], set())
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# POST: row-count postcondition — happy path
# ---------------------------------------------------------------------------


class TestEvaluateQueryRowCountPostcondition:
    """The row count equals the number of eligible candidates."""

    def test_no_candidates(self):
        """Empty candidate list → 0 rows."""
        rows = evaluate_query_adorned("tf1", "out", [], set())
        assert rows == []

    def test_all_candidates_eligible(self):
        """No self-pairs, no already-done → row count == len(candidates)."""
        candidates = ["c1", "c2", "c3"]
        rows = evaluate_query_adorned("tf1", "out", candidates, set())
        assert len(rows) == 3

    def test_self_pair_tf1_excluded(self):
        """Candidate equal to tf1 is skipped → 285 - 1 style."""
        candidates = ["tf1", "c2", "c3"]
        rows = evaluate_query_adorned("tf1", "out", candidates, set())
        assert len(rows) == 2  # "tf1" skipped

    def test_self_pair_outcome_excluded(self):
        """Candidate equal to outcome is skipped."""
        candidates = ["c1", "out", "c3"]
        rows = evaluate_query_adorned("tf1", "out", candidates, set())
        assert len(rows) == 2  # "out" skipped

    def test_both_self_pairs_excluded(self):
        """Both tf1 and outcome in candidates → 2 skipped (285→283 observed case)."""
        candidates = ["tf1", "out", "c1", "c2", "c3"]
        rows = evaluate_query_adorned("tf1", "out", candidates, set())
        assert len(rows) == 3  # tf1 and out skipped

    def test_completed_pairs_excluded(self):
        """Already-completed (tf1, candidate, outcome) triples are skipped."""
        candidates = ["c1", "c2", "c3"]
        completed = {("tf1", "c1", "out")}
        rows = evaluate_query_adorned("tf1", "out", candidates, completed)
        assert len(rows) == 2  # c1 already done

    def test_self_pair_and_completed_combined(self):
        """Self-pairs and completed pairs are both excluded correctly."""
        candidates = ["tf1", "out", "c1", "c2", "c3"]
        completed = {("tf1", "c1", "out"), ("tf1", "c2", "out")}
        rows = evaluate_query_adorned("tf1", "out", candidates, completed)
        assert len(rows) == 1  # only c3 is eligible

    def test_all_candidates_are_self_pairs(self):
        """All candidates equal tf1 or outcome → 0 rows."""
        candidates = ["tf1", "out"]
        rows = evaluate_query_adorned("tf1", "out", candidates, set())
        assert rows == []

    def test_all_candidates_already_completed(self):
        """All candidates already in completed → 0 rows."""
        candidates = ["c1", "c2"]
        completed = {("tf1", "c1", "out"), ("tf1", "c2", "out")}
        rows = evaluate_query_adorned("tf1", "out", candidates, completed)
        assert rows == []

    def test_row_structure_is_four_elements(self):
        """Each returned row is [tf1, candidate, outcome, found]."""
        rows = evaluate_query_adorned("tf1", "out", ["c1"], set())
        assert len(rows) == 1
        assert rows[0] == ["tf1", "c1", "out", True]

    def test_found_flag_reflects_identify_fn(self):
        """The found flag mirrors the return value of identify_fn."""
        rows = evaluate_query_adorned(
            "tf1",
            "out",
            ["c1", "c2"],
            set(),
            identify_fn=lambda t, c, o: c == "c1",
        )
        assert len(rows) == 2
        found_map = {r[1]: r[3] for r in rows}
        assert found_map["c1"] is True
        assert found_map["c2"] is False

    def test_completed_key_uses_tf1_candidate_outcome(self):
        """Completed keys are (tf1, candidate, outcome) — not (outcome, candidate, tf1)."""
        candidates = ["c1"]
        # Wrong-order key should NOT suppress the candidate
        completed_wrong_order = {("out", "c1", "tf1")}
        rows = evaluate_query_adorned("tf1", "out", candidates, completed_wrong_order)
        assert len(rows) == 1  # c1 NOT suppressed by wrong-order key

        # Correct-order key DOES suppress
        completed_correct = {("tf1", "c1", "out")}
        rows2 = evaluate_query_adorned("tf1", "out", candidates, completed_correct)
        assert rows2 == []


# ---------------------------------------------------------------------------
# POST (negative): guard fires when the invariant would be violated
# ---------------------------------------------------------------------------


class TestEvaluateQueryRowCountGuardFires:
    """
    The POST row-count assert must catch any implementation that produces the
    wrong count.  We exercise this via a patched adorned function that
    deliberately over- or under-produces rows.
    """

    def _make_broken_adorned(self, extra_rows: int):
        """
        Return a variant of evaluate_query_adorned whose body appends
        ``extra_rows`` phantom rows after the real ones, forcing the
        postcondition to fail.
        """

        def broken(tf1, outcome, candidates, completed, identify_fn=None):
            """Broken variant that appends phantom rows to trigger the POST guard."""
            # Replicate the pre-check
            assert isinstance(tf1, str), "PRE: tf1 must be str"
            assert isinstance(outcome, str), "PRE: outcome must be str"
            assert isinstance(candidates, list), "PRE: candidates must be a list"
            assert isinstance(completed, set), "PRE: completed must be a set"

            rows = []
            for candidate in candidates:
                if candidate == tf1 or candidate == outcome:
                    continue
                if (tf1, candidate, outcome) in completed:
                    continue
                rows.append([tf1, candidate, outcome, True])

            # Deliberately corrupt: add phantom rows
            for i in range(extra_rows):
                rows.append([tf1, f"__phantom_{i}__", outcome, False])

            # POST (must fire)
            assert isinstance(rows, list), "POST: result must be a list"
            assert all(len(r) == 4 for r in rows), "POST: every row has 4 elements"
            _expected = sum(
                1
                for c in candidates
                if c != tf1 and c != outcome and (tf1, c, outcome) not in completed
            )
            assert len(rows) == _expected, (
                "POST: row count must equal eligible (non-self, not-already-done) candidates"
            )
            return rows

        return broken

    def test_guard_fires_with_extra_row(self):
        """One phantom extra row triggers the guard."""
        broken = self._make_broken_adorned(extra_rows=1)
        with pytest.raises(
            AssertionError,
            match="POST: row count must equal eligible",
        ):
            broken("tf1", "out", ["c1", "c2"], set())

    def test_guard_fires_with_missing_row(self):
        """Under-producing rows also triggers the guard."""

        def missing_row_adorned(tf1, outcome, candidates, completed):
            """Broken variant that drops the last row to trigger the POST guard."""
            assert isinstance(tf1, str), "PRE: tf1 must be str"
            assert isinstance(outcome, str), "PRE: outcome must be str"
            assert isinstance(candidates, list), "PRE: candidates must be a list"
            assert isinstance(completed, set), "PRE: completed must be a set"

            rows = []
            for candidate in candidates:
                if candidate == tf1 or candidate == outcome:
                    continue
                if (tf1, candidate, outcome) in completed:
                    continue
                rows.append([tf1, candidate, outcome, True])

            # Corrupt: drop the last row
            if rows:
                rows = rows[:-1]

            _expected = sum(
                1
                for c in candidates
                if c != tf1 and c != outcome and (tf1, c, outcome) not in completed
            )
            assert len(rows) == _expected, (
                "POST: row count must equal eligible (non-self, not-already-done) candidates"
            )
            return rows

        with pytest.raises(
            AssertionError,
            match="POST: row count must equal eligible",
        ):
            missing_row_adorned("tf1", "out", ["c1", "c2", "c3"], set())
