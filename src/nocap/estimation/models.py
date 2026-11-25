"""Protocols and shared errors for estimation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pgmpy.factors.discrete import DiscreteFactor

__all__ = [
    "ContinuousBackend",
    "DiscreteBackend",
    "DiscreteNetworkBackend",
    "DistributionEstimationError",
    "JointKDE",
    "MarginalKDE",
]


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

    def fit(
        self,
        data: pd.DataFrame,
        variables: Sequence[str],
        *,
        bandwidth=None,
    ) -> JointKDE:
        """Fit a KDE for the requested columns, optionally reusing a bandwidth."""
        ...

@runtime_checkable
class DiscreteBackend(Protocol):
    """Protocol used by the discrete recursive evaluator."""

    def fit(
        self,
        data: pd.DataFrame,
        variables: Sequence[str],
    ) -> DiscreteFactor:
        """Return a fitted joint discrete factor for the requested variables."""
        ...


@runtime_checkable
class DiscreteNetworkBackend(DiscreteBackend, Protocol):
    """Discrete backend that fits a full network once before leaf queries."""

    def fit_network(self, data: pd.DataFrame) -> None:
        """Fit and retain a full network from observed data."""
        ...
