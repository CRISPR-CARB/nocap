"""Expectation and treatment-effect estimates from evaluated distributions using g-computation, IPW, and AIPW."""

from .distribution import estimate_ate, estimate_expectation

__all__ = ["estimate_ate", "estimate_expectation"]

# TODO: Add the expectation estimation stuff that isn't derived from KDE
