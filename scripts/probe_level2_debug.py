"""Debug probe: print full result for scale_intercept with different hints."""
import sys

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

if __name__ == "__main__":
    from axiomander.oracle.mcp_server import _verify_function

    for hint in ["nlinarith", "nia", "hammer", None]:
        result = _verify_function(SOURCE_L2, "scale_intercept", hint=hint)
        proved = result.is_proved() if result is not None else False
        level = getattr(result, "level", None)
        detail = (getattr(result, "error_detail", "") or "")[:120]
        status = "PROVED  " if proved else "UNPROVED"
        print(f"  hint={str(hint):12s}  {status}  [{level}]  {detail}")
        if proved:
            break
