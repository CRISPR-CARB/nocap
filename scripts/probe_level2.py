"""Probe script: verify functions at Levels 1 and 2 (coq-hammer + SMT).

Run with:
    eval $(opam env)
    AXIOMANDER_ROOT=/Users/zuck016/Projects/CausalInference/Vericoding/axiomander \
    .tox/axiomander/bin/python scripts/probe_level2.py
"""

import sys

# Level 1 (Ltac/lia) — simple linear arithmetic
SOURCE_L1 = '''
def count_parents(parents: list) -> int:
    """Return the number of parents in a list.

    axiomander:
        ensures:
            result >= 0
    """
    result = len(parents)
    assert result >= 0  # noqa: S101
    return result


def bootstrap_iterations_positive(n_iterations: int) -> int:
    """Return n_iterations unchanged, asserting it is positive.

    axiomander:
        requires:
            n_iterations > 0
        ensures:
            result > 0
            result == n_iterations
    """
    assert n_iterations > 0  # noqa: S101
    result = n_iterations
    assert result > 0  # noqa: S101
    assert result == n_iterations  # noqa: S101
    return result
'''

# Level 2 (SMT/hammer) — non-linear: product of two non-negative ints is non-negative
SOURCE_L2 = '''
def scale_intercept(intercept: int, factor: int) -> int:
    """Scale an intercept by a factor; result has same sign as product.

    axiomander:
        requires:
            intercept >= 0
            factor >= 0
        ensures:
            result >= 0
    """
    assert intercept >= 0  # noqa: S101
    assert factor >= 0  # noqa: S101
    result = intercept * factor
    assert result >= 0  # noqa: S101
    return result
'''

CASES = [
    ("count_parents", SOURCE_L1, "hammer"),
    ("bootstrap_iterations_positive", SOURCE_L1, "hammer"),
    ("scale_intercept", SOURCE_L2, "hammer"),
]

if __name__ == "__main__":
    from axiomander.oracle.mcp_server import _verify_function

    all_proved = True
    for fn, src, hint in CASES:
        result = _verify_function(src, fn, hint=hint)
        proved = result.is_proved() if result is not None else False
        level = getattr(result, "level", None)
        detail = getattr(result, "error_detail", "") or ""
        status = "PROVED  " if proved else "UNPROVED"
        print(f"  {status}  {fn}  [{level}]  {detail[:80]}")
        if not proved:
            all_proved = False

    sys.exit(0 if all_proved else 1)
