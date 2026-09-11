"""Continuous KDE backends and fitted density projections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from .models import JointKDE

__all__ = ["KDEpyKDE", "SciPyGaussianKDE", "SklearnKDE", "StatsmodelsKDE"]


def _set_bandwidth(
    bw: float | str | None, n: int | None = None, d: int | None = None, sigma: float = 1
) -> float | None:

    match bw:
        case "scott":
            if n is None or d is None:
                raise ValueError(f"n and d must not be None when using bandwidth method {bw}")
            return sigma * (n ** (-1.0 / (d + 4)))
        case "silverman":
            if n is None or d is None:
                raise ValueError(f"n and d must not be None when using bandwidth method {bw}")
            return sigma * (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))
        case float():
            return bw
        case int():
            return float(bw)
        case None:
            return bw
        case _:
            raise ValueError(f"Bandwidth value {bw} is not valid.")


class KDEpyKDE:
    """KDEpy FFTKDE backend with pointwise evaluation and projection."""

    def fit(
        self,
        data: pd.DataFrame,
        variables: Sequence[str],
        *,
        bandwidth: float | str | None = "scott",
    ) -> JointKDE:
        """Fit PDF using KDE to data."""
        bandwidth = _set_bandwidth(
            bandwidth,
            data.shape[0],
            len(variables),
            min(
                np.std(data, ddof=1),
                (np.percentile(data, q=75) - np.percentile(data, q=25)) / 1.3489795003921634,
            ),
        )

        """Fit a KDEpy FFTKDE for continuous columns."""
        values = data.loc[:, list(variables)].to_numpy(dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("continuous KDE data must be finite and contain at least two rows")

        return _KDEpyFitted(tuple(variables), values, bandwidth)


class _KDEpyFitted:
    def __init__(self, variables, values, bandwidth: float | str | None = "scott"):
        """Initialize a KDEpy fit and defer grid construction until evaluation."""
        self.variables = variables
        self.values = values
        self.bandwidth = _set_bandwidth(
            bandwidth,
            values.shape[0],
            len(variables),
            min(
                np.std(values, ddof=1),
                (np.percentile(values, q=75) - np.percentile(values, q=25)) / 1.3489795003921634,
            ),
        )

        self._grid = None

    def _interpolator(self):
        if self._grid is None:
            from KDEpy import TreeKDE

            estimator = TreeKDE() if self.bandwidth is None else TreeKDE(bw=self.bandwidth)  # type: ignore
            grid, density = estimator.fit(self.values).evaluate()
            self.bandwidth = estimator.bw  # type: ignore
            dimensions = len(self.variables)
            if dimensions == 1:
                self._grid = (grid[:, 0] if grid.ndim == 2 else grid, density)
            else:
                axes = tuple(np.unique(grid[:, index]) for index in range(dimensions))
                from scipy.interpolate import RegularGridInterpolator

                self._grid = RegularGridInterpolator(
                    axes,
                    density.reshape(tuple(len(axis) for axis in axes)),
                    bounds_error=False,
                    fill_value=0.0,
                )
        return self._grid

    def evaluate(self, points):
        """Evaluate the FFTKDE density at arbitrary point rows."""
        points = np.asarray(points, dtype=float)
        interpolator = self._interpolator()
        if len(self.variables) == 1:
            grid, density = interpolator
            return np.interp(points[:, 0], grid, density, left=0.0, right=0.0)
        return np.asarray(interpolator(points))

    def project(self, variables):
        """Return a marginal by refitting FFTKDE on the selected coordinates."""
        indexes = [self.variables.index(name) for name in variables]
        if not indexes:
            return _ConstantFitted((), 1.0)
        return _KDEpyFitted(tuple(variables), self.values[:, indexes], self.bandwidth)  # type: ignore


class SciPyGaussianKDE:
    """SciPy Gaussian KDE backend with exact Gaussian-mixture projection."""

    def fit(self, data: pd.DataFrame, variables: Sequence[str], *, bandwidth=None) -> JointKDE:
        """Fit a SciPy Gaussian KDE while retaining its original coordinates."""
        from scipy.stats import gaussian_kde

        values = data.loc[:, list(variables)].to_numpy(dtype=float)
        if not np.isfinite(values).all() or len(values) < 2:
            raise ValueError("continuous KDE data must be finite and contain at least two rows")
        if values.shape[1] == 1:
            values = values[:, 0]
        kde = gaussian_kde(values.T if values.ndim > 1 else values, bw_method=bandwidth)
        return _SciPyFitted(
            tuple(variables), kde, values if values.ndim > 1 else values[:, None], kde.factor
        )


class _SciPyFitted:
    def __init__(self, variables, kde, samples, bandwidth):
        """Store a fitted SciPy KDE and its original sample coordinates."""
        self.variables, self.kde, self.samples, self.bandwidth = variables, kde, samples, bandwidth

    def evaluate(self, points):
        """Evaluate the fitted SciPy density at point rows."""
        points = np.asarray(points, dtype=float)
        if len(self.variables) == 1:
            return np.asarray(self.kde(points[:, 0]))
        return np.asarray(self.kde(points.T))

    def project(self, variables):
        """Project the Gaussian mixture onto a subset of variables."""
        indexes = [self.variables.index(name) for name in variables]
        if not indexes:
            return _ConstantFitted((), 1.0)
        return _SciPyProjected(
            tuple(variables),
            self.samples[:, indexes],
            self.kde.weights,
            self.kde.covariance[np.ix_(indexes, indexes)],
        )


class _SciPyProjected:
    def __init__(self, variables, samples, weights, covariance):
        """Initialize an analytically projected Gaussian mixture."""
        from scipy.stats import multivariate_normal

        self.variables, self.samples, self.weights, self.covariance = (
            variables,
            samples,
            weights,
            covariance,
        )
        self._normal = multivariate_normal

    def evaluate(self, points):
        """Evaluate the projected Gaussian mixture at point rows."""
        points = np.asarray(points, dtype=float)
        if len(self.variables) == 1:
            scale = float(np.sqrt(self.covariance[0, 0]))
            return np.sum(
                self.weights[None, :]
                * np.exp(-0.5 * ((points[:, 0, None] - self.samples[:, 0]) / scale) ** 2)
                / (np.sqrt(2 * np.pi) * scale),
                axis=1,
            )
        return np.sum(
            [
                weight * self._normal.pdf(points, mean=sample, cov=self.covariance)
                for weight, sample in zip(self.weights, self.samples, strict=True)
            ],
            axis=0,
        )

    def project(self, variables):
        """Project the Gaussian mixture again onto selected variables."""
        indexes = [self.variables.index(name) for name in variables]
        return _SciPyProjected(
            tuple(variables),
            self.samples[:, indexes],
            self.weights,
            self.covariance[np.ix_(indexes, indexes)],
        )


class SklearnKDE:
    """scikit-learn ``KernelDensity`` backend with refit-based projection."""

    def __init__(
        self,
        *,
        bandwidth: float | Literal["scott", "silverman"] = "scott",
        kernel: Literal[
            "gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"
        ] = "gaussian",
        **kwargs,
    ) -> None:
        """Configure a scikit-learn ``KernelDensity`` estimator."""
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kwargs = kwargs

    def fit(
        self,
        data: pd.DataFrame,
        variables: Sequence[str],
        *,
        bandwidth: float | Literal["scott", "silverman"] | None = None,
    ) -> JointKDE:
        """Fit ``KernelDensity`` to the selected continuous columns."""
        from sklearn.neighbors import KernelDensity

        values = data.loc[:, list(variables)].to_numpy(dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("continuous KDE data must be finite and contain at least two rows")
        selected_bandwidth = self.bandwidth if bandwidth is None else bandwidth
        estimator = KernelDensity(
            bandwidth=selected_bandwidth,
            kernel=self.kernel,
            **self.kwargs,
        ).fit(values)
        effective_bandwidth = getattr(estimator, "bandwidth_", estimator.bandwidth_)
        return _SklearnFitted(
            tuple(variables),
            values,
            estimator,
            self.kernel,
            self.kwargs,
            effective_bandwidth,
        )


class _SklearnFitted:
    def __init__(self, variables, values, estimator, kernel, kwargs, bandwidth):
        """Store a fitted scikit-learn estimator and projection metadata."""
        self.variables = variables
        self.values = values
        self.estimator = estimator
        self.kernel = kernel
        self.kwargs = kwargs
        self.bandwidth = bandwidth

    def evaluate(self, points):
        """Evaluate the fitted log-density after converting it to density."""
        points = np.asarray(points, dtype=float)
        return np.exp(self.estimator.score_samples(points))

    def project(self, variables):
        """Refit a scikit-learn KDE using only the selected coordinates."""
        indexes = [self.variables.index(name) for name in variables]
        if not indexes:
            return _ConstantFitted((), 1.0)
        estimator = SklearnKDE(
            bandwidth=self.bandwidth,
            kernel=self.kernel,
            **self.kwargs,
        )
        return estimator.fit(
            pd.DataFrame(self.values[:, indexes], columns=list(variables)),
            tuple(variables),
        )


class _ConstantFitted:
    def __init__(self, variables, value):
        """Initialize a constant density used for zero-dimensional projections."""
        self.variables, self.value = variables, value

    def evaluate(self, points):
        """Return the constant density value for every point row."""
        return np.full(len(points), self.value)

    def project(self, variables):
        """Return the constant density unchanged for any projection."""
        return self


class StatsmodelsKDE:
    """Statsmodels product-kernel KDE backend with analytic projection."""

    def fit(self, data: pd.DataFrame, variables: Sequence[str], *, bandwidth=None) -> JointKDE:
        """Fit a statsmodels product-kernel KDE for continuous columns."""
        from statsmodels.nonparametric.kernel_density import KDEMultivariate

        values = data.loc[:, list(variables)].to_numpy(dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("continuous KDE data must be finite and contain at least two rows")
        selected_bandwidth = "normal_reference" if bandwidth is None else bandwidth
        kde = KDEMultivariate(data=values, var_type="c" * len(variables), bw=selected_bandwidth)
        return _StatsmodelsFitted(tuple(variables), values, kde, np.asarray(kde.bw))


class _StatsmodelsFitted:
    def __init__(self, variables, values, kde, bandwidth):
        """Store a fitted statsmodels KDE and its sample representation."""
        self.variables, self.values, self.kde, self.bandwidth = variables, values, kde, bandwidth

    def evaluate(self, points):
        """Evaluate the fitted statsmodels density at point rows."""
        return np.asarray(self.kde.pdf(np.asarray(points)))

    def project(self, variables):
        """Project the product-kernel density onto selected variables."""
        indexes = [self.variables.index(name) for name in variables]
        if not indexes:
            return _ConstantFitted((), 1.0)
        return _StatsmodelsProjected(
            tuple(variables), self.values[:, indexes], np.asarray(self.kde.bw)[indexes]
        )


class _StatsmodelsProjected:
    def __init__(self, variables, values, bandwidth):
        """Initialize a projected statsmodels product-kernel density."""
        self.variables, self.values, self.bandwidth = variables, values, bandwidth

    def evaluate(self, points):
        """Evaluate the projected product-kernel density at point rows."""
        points = np.asarray(points)
        result = np.ones(len(points))
        for index, bandwidth in enumerate(self.bandwidth):
            u = (points[:, index, None] - self.values[None, :, index]) / bandwidth
            result *= np.exp(-0.5 * u * u).mean(axis=1) / (np.sqrt(2 * np.pi) * bandwidth)
        return result

    def project(self, variables):
        """Project the product-kernel density onto another variable subset."""
        indexes = [self.variables.index(name) for name in variables]
        return _StatsmodelsProjected(
            tuple(variables), self.values[:, indexes], self.bandwidth[indexes]
        )
