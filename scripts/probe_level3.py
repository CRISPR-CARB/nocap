"""Probe Level 1 (bootstrap_ATE) and Level 3 (oracle) verification."""
import os

SOURCE_BOOTSTRAP = '''
def bootstrap_ATE(n_iterations: int) -> int:
    """Bootstrap ATE stub for contract verification.

    axiomander:
        requires:
            n_iterations > 0
    """
    assert n_iterations > 0
    return n_iterations
'''

SOURCE_NONLINEAR = '''
def scale_intercept(intercept: int, factor: int) -> int:
    """Scale an intercept by a factor.

    axiomander:
        requires:
            intercept >= 0
            factor >= 0
        ensures:
            result >= 0
    """
    assert intercept >= 0
    assert factor >= 0
    result = intercept * factor
    assert result >= 0
    return result
'''

if __name__ == "__main__":
    from axiomander.oracle.mcp_server import _verify_function

    print("=== Level 1: bootstrap_ATE (n_iterations > 0) ===")
    r1 = _verify_function(SOURCE_BOOTSTRAP, "bootstrap_ATE")
    print(f"  proved={r1.is_proved()}  level={r1.level}")

    print()
    print("=== Level 1 (nlinarith): scale_intercept (non-linear) ===")
    r2 = _verify_function(SOURCE_NONLINEAR, "scale_intercept", hint="nlinarith")
    print(f"  proved={r2.is_proved()}  level={r2.level}")

    print()
    print("=== Level 3: oracle test (scale_intercept without hint) ===")
    oracle_key = os.environ.get("ORACLE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not oracle_key:
        print("  SKIPPED: no ORACLE_API_KEY / OPENAI_API_KEY set")
    else:
        r3 = _verify_function(SOURCE_NONLINEAR, "scale_intercept")
        print(f"  proved={r3.is_proved()}  level={r3.level}")
        if not r3.is_proved():
            print(f"  detail: {(r3.error_detail or '')[:200]}")
