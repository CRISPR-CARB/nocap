"""Protocols and shared errors for estimation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

__all__ = ["ContinuousBackend", "DistributionEstimationError", "JointKDE", "MarginalKDE"]


class DistributionEstimationError(ValueError):
    """Raised when an expression or observed distribution cannot be estimated."""


@runtime_checkable
class JointKDE(Protocol):
    """Protocol for a fitted joint density and analytic projection."""

    variables: tuple[str, ...]

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the fitted density at point rows."""
        ...

    def project(self, variables: Sequence[str]) -> JointKDE:
        """Return the analytic marginal over ``variables``."""
        ...


@runtime_checkable
class MarginalKDE(JointKDE, Protocol):
    """Protocol for a marginal KDE."""


@runtime_checkable
class ContinuousBackend(Protocol):
    """Protocol used by the continuous recursive evaluator."""

    def fit(self, data: pd.DataFrame, variables: Sequence[str]) -> JointKDE:
        """Fit and return a joint KDE for the requested columns."""
        ...
