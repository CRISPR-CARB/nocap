"""Continuous KDE backends and fitted density projections."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .models import JointKDE

__all__ = ["SciPyGaussianKDE", "StatsmodelsKDE"]


class SciPyGaussianKDE:
    """SciPy Gaussian KDE backend with exact Gaussian-mixture projection."""

    def fit(self, data: pd.DataFrame, variables: Sequence[str]) -> JointKDE:
        """Fit a SciPy Gaussian KDE while retaining its original coordinates."""
        from scipy.stats import gaussian_kde

        values = data.loc[:, list(variables)].to_numpy(dtype=float)
        if not np.isfinite(values).all() or len(values) < 2:
            raise ValueError("continuous KDE data must be finite and contain at least two rows")
        if values.shape[1] == 1:
            values = values[:, 0]
        kde = gaussian_kde(values.T if values.ndim > 1 else values)
        return _SciPyFitted(tuple(variables), kde, values if values.ndim > 1 else values[:, None])


class _SciPyFitted:
    def __init__(self, variables, kde, samples):
        self.variables, self.kde, self.samples = variables, kde, samples

    def evaluate(self, points):
        points = np.asarray(points, dtype=float)
        if len(self.variables) == 1:
            return np.asarray(self.kde(points[:, 0]))
        return np.asarray(self.kde(points.T))

    def project(self, variables):
        indexes = [self.variables.index(name) for name in variables]
        if not indexes:
            return _ConstantFitted((), 1.0)
        return _SciPyProjected(tuple(variables), self.samples[:, indexes], self.kde.weights, self.kde.covariance[np.ix_(indexes, indexes)])


class _SciPyProjected:
    def __init__(self, variables, samples, weights, covariance):
        from scipy.stats import multivariate_normal

        self.variables, self.samples, self.weights, self.covariance = variables, samples, weights, covariance
        self._normal = multivariate_normal

    def evaluate(self, points):
        points = np.asarray(points, dtype=float)
        if len(self.variables) == 1:
            scale = float(np.sqrt(self.covariance[0, 0]))
            return np.sum(self.weights[None, :] * np.exp(-0.5 * ((points[:, 0, None] - self.samples[:, 0]) / scale) ** 2) / (np.sqrt(2 * np.pi) * scale), axis=1)
        return np.sum([weight * self._normal.pdf(points, mean=sample, cov=self.covariance) for weight, sample in zip(self.weights, self.samples, strict=True)], axis=0)

    def project(self, variables):
        indexes = [self.variables.index(name) for name in variables]
        return _SciPyProjected(tuple(variables), self.samples[:, indexes], self.weights, self.covariance[np.ix_(indexes, indexes)])


class _ConstantFitted:
    def __init__(self, variables, value):
        self.variables, self.value = variables, value

    def evaluate(self, points):
        return np.full(len(points), self.value)

    def project(self, variables):
        return self


class StatsmodelsKDE:
    """Statsmodels product-kernel KDE backend with analytic projection."""

    def fit(self, data: pd.DataFrame, variables: Sequence[str]) -> JointKDE:
        """Fit a statsmodels product-kernel KDE for continuous columns."""
        from statsmodels.nonparametric.kernel_density import KDEMultivariate

        values = data.loc[:, list(variables)].to_numpy(dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("continuous KDE data must be finite and contain at least two rows")
        kde = KDEMultivariate(data=values, var_type="c" * len(variables), bw="normal_reference")
        return _StatsmodelsFitted(tuple(variables), values, kde)


class _StatsmodelsFitted:
    def __init__(self, variables, values, kde):
        self.variables, self.values, self.kde = variables, values, kde

    def evaluate(self, points):
        return np.asarray(self.kde.pdf(np.asarray(points)))

    def project(self, variables):
        indexes = [self.variables.index(name) for name in variables]
        if not indexes:
            return _ConstantFitted((), 1.0)
        return _StatsmodelsProjected(tuple(variables), self.values[:, indexes], np.asarray(self.kde.bw)[indexes])


class _StatsmodelsProjected:
    def __init__(self, variables, values, bandwidth):
        self.variables, self.values, self.bandwidth = variables, values, bandwidth

    def evaluate(self, points):
        points = np.asarray(points)
        result = np.ones(len(points))
        for index, bandwidth in enumerate(self.bandwidth):
            u = (points[:, index, None] - self.values[None, :, index]) / bandwidth
            result *= np.exp(-0.5 * u * u).mean(axis=1) / (np.sqrt(2 * np.pi) * bandwidth)
        return result

    def project(self, variables):
        indexes = [self.variables.index(name) for name in variables]
        return _StatsmodelsProjected(tuple(variables), self.values[:, indexes], self.bandwidth[indexes])
