"""Evaluate y0 probability expressions from discrete data or KDE densities.

The recursive evaluator deliberately depends on small backend protocols.  This
keeps symbolic evaluation testable independently of any particular KDE
implementation and makes analytic projection explicit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd
from y0.dsl import (
    Distribution,
    Expression,
    Fraction,
    One,
    Probability,
    Product,
    Sum,
    Variable,
    Zero,
)
from y0.mutate import fraction_expand

from .kde import KDEpyKDE, SciPyGaussianKDE, SklearnKDE, StatsmodelsKDE
from .models import ContinuousBackend, DistributionEstimationError, JointKDE, MarginalKDE
from .profiler import EstimationProfiler

__all__ = [
    "ContinuousBackend",
    "DistributionEstimationError",
    "DistributionEstimator",
    "EstimationProfiler",
    "JointKDE",
    "KDEpyKDE",
    "MarginalKDE",
    "SciPyGaussianKDE",
    "SklearnKDE",
    "StatsmodelsKDE",
    "estimate_ate",
    "estimate_expectation",
    "evaluate_probability_expression",
    "normalize_expression",
]


def _measure(profiler: EstimationProfiler | None, name: str, **metadata: object):
    """Return a no-op or profiling context for an estimation operation."""
    return profiler.measure(name, **metadata) if profiler is not None else nullcontext()


class _CallableDensity:
    """Callable wrapper that preserves deterministic density variable order."""

    def __init__(
        self,
        variables: tuple[str, ...],
        function: Callable[[np.ndarray], np.ndarray],
        bandwidth: object = None,
    ):
        """Initialize a density over ``variables`` backed by ``function``."""
        self.variables = variables
        self._function = function
        self.bandwidth = bandwidth

    def __call__(self, values: object = None, **kwargs: float) -> float | np.ndarray:
        """Evaluate the density from positional values or variable keywords."""
        if kwargs:
            if set(kwargs) != set(self.variables):
                missing = set(self.variables) - set(kwargs)
                extra = set(kwargs) - set(self.variables)
                raise ValueError(
                    f"density keyword variables do not match; missing={sorted(missing)}, "
                    f"extra={sorted(extra)}"
                )
            arrays = np.broadcast_arrays(
                *(np.asarray(kwargs[name], dtype=float) for name in self.variables)
            )
            scalar = all(array.ndim == 0 for array in arrays)
            values = np.column_stack([array.reshape(-1) for array in arrays])
        else:
            scalar = False
        if values is None:
            raise TypeError("density requires values or named arguments")
        array = np.asarray(values, dtype=float)
        if not kwargs and array.ndim == 0:
            scalar = True
            array = array.reshape(1)
        if array.ndim == 1:
            points = array.reshape(1, -1) if len(self.variables) > 1 else array.reshape(-1, 1)
        else:
            points = array
        result = np.asarray(self._function(points), dtype=float)
        return float(result[0]) if scalar and result.size else result


class _DensityWithVariables(Protocol):
    variables: tuple[str, ...]

    def __call__(self, **kwargs: object) -> float | np.ndarray: ...


def _names(variables: Iterable[Variable]) -> tuple[str, ...]:
    """Return unique y0 variable names in deterministic order."""
    return tuple(sorted({variable.name for variable in variables}))


def _density_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide densities while treating jointly underflowed tails as zero."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    zero_denominator = denominator == 0
    if np.any(zero_denominator & (np.abs(numerator) > 1e-14)):
        raise DistributionEstimationError("density fraction has a zero denominator")
    result = np.zeros(np.broadcast_shapes(numerator.shape, denominator.shape), dtype=float)
    np.divide(numerator, denominator, out=result, where=~zero_denominator)
    return result


def _distribution_density(
    backend: ContinuousBackend,
    data: pd.DataFrame,
    distribution: Distribution,
    profiler: EstimationProfiler | None = None,
):
    """Build a callable joint or conditional density for a y0 distribution."""
    children = _names(distribution.children)
    parents = _names(distribution.parents)
    all_names = children + tuple(name for name in parents if name not in children)
    with _measure(profiler, "continuous.fit", variables=all_names, rows=len(data)):
        fitted = backend.fit(data, all_names)
    numerator = fitted.project(all_names)
    if not parents:
        return _CallableDensity(all_names, numerator.evaluate, getattr(fitted, "bandwidth", None))
    denominator = fitted.project(parents)
    free = children + tuple(name for name in parents if name not in children)

    def conditional(points: np.ndarray) -> np.ndarray:
        """Evaluate the conditional density as a joint-to-marginal ratio."""
        num = numerator.evaluate(points)
        den = denominator.evaluate(points[:, [all_names.index(name) for name in parents]])
        try:
            return _density_ratio(num, den)
        except DistributionEstimationError as error:
            raise DistributionEstimationError(
                "conditional density has a zero denominator"
            ) from error

    return _CallableDensity(free, conditional, getattr(fitted, "bandwidth", None))


def _continuous_eval(
    expression: Expression,
    data: pd.DataFrame,
    backend: ContinuousBackend,
    bounds: Mapping[str, tuple[float, float]] | tuple[float, float] | None = None,
    profiler: EstimationProfiler | None = None,
    marginalization: Literal["quadrature", "kde"] = "quadrature",
):
    """Recursively evaluate an expression using continuous density callables."""
    if isinstance(expression, Probability):
        with _measure(profiler, "continuous.distribution"):
            return _distribution_density(backend, data, expression.distribution, profiler)
    if isinstance(expression, One):
        return _CallableDensity((), lambda points: np.ones(len(points)))
    if isinstance(expression, Zero):
        return _CallableDensity((), lambda points: np.zeros(len(points)))
    if isinstance(expression, Product):
        densities = [
            _continuous_eval(item, data, backend, bounds, profiler, marginalization)
            for item in expression.expressions
        ]
        variables = tuple(
            dict.fromkeys(name for density in densities for name in density.variables)
        )

        def product(points: np.ndarray) -> np.ndarray:
            """Evaluate and multiply all factor densities on shared points."""
            result = np.ones(len(points))
            for density in densities:
                indexes = [variables.index(name) for name in density.variables]
                result *= np.asarray(
                    density(points[:, indexes] if indexes else np.empty((len(points), 0)))
                )
            return result

        return _CallableDensity(variables, product)
    if isinstance(expression, Fraction):
        numerator = _continuous_eval(
            expression.numerator, data, backend, bounds, profiler, marginalization
        )
        denominator = _continuous_eval(
            expression.denominator, data, backend, bounds, profiler, marginalization
        )
        variables = tuple(dict.fromkeys((*numerator.variables, *denominator.variables)))

        def fraction(points: np.ndarray) -> np.ndarray:
            """Evaluate a density ratio while rejecting tiny denominators."""
            n = numerator(
                points[:, [variables.index(x) for x in numerator.variables]]
                if numerator.variables
                else np.empty((len(points), 0))
            )
            d = denominator(
                points[:, [variables.index(x) for x in denominator.variables]]
                if denominator.variables
                else np.empty((len(points), 0))
            )
            return _density_ratio(n, d)

        return _CallableDensity(variables, fraction)
    if isinstance(expression, Sum):
        if isinstance(expression.expression, Probability) and not expression.expression.parents:
            inner_variables = _names(expression.expression.children)
            ranges = {variable.name for variable in expression.ranges}
            kept = tuple(name for name in inner_variables if name not in ranges)
            with _measure(profiler, "continuous.fit", variables=inner_variables, rows=len(data)):
                fitted = backend.fit(data, inner_variables)
            projected = fitted.project(kept)
            return _CallableDensity(kept, projected.evaluate, getattr(fitted, "bandwidth", None))
        inner = _continuous_eval(
            expression.expression, data, backend, bounds, profiler, marginalization
        )
        ranges = {variable.name for variable in expression.ranges}
        if not ranges.issubset(inner.variables):
            return inner
        integrated = tuple(name for name in inner.variables if name not in ranges)
        if not ranges:
            return inner
        if bounds is None:
            raise DistributionEstimationError(
                "integration bounds are required for summed variables"
            )
        limits = {}
        for name in sorted(ranges):
            if isinstance(bounds, Mapping):
                if name not in bounds:
                    raise DistributionEstimationError(
                        f"integration bounds are missing for summed variable {name!r}"
                    )
                lower, upper = bounds[name]
            else:
                lower, upper = bounds
            if np.any(np.isnan((lower, upper))) or not float(lower) < float(upper):
                raise DistributionEstimationError(
                    f"integration bounds for {name!r} must be increasing and non-NaN"
                )
            lower, upper = float(lower), float(upper)
            limits[name] = (lower, upper)

        if marginalization == "kde":
            from numpy.polynomial.legendre import leggauss

            integrated_names = tuple(sorted(ranges))
            nodes, node_weights = leggauss(64)
            axes = []
            axis_weights = []
            for name in integrated_names:
                lower, upper = limits[name]
                axes.append((lower + upper) / 2 + (upper - lower) / 2 * nodes)
                axis_weights.append((upper - lower) / 2 * node_weights)
            meshes = np.meshgrid(*axes, indexing="ij")
            integration_points = np.column_stack([mesh.reshape(-1) for mesh in meshes])
            weight_meshes = np.meshgrid(*axis_weights, indexing="ij")
            integration_weights = np.prod(weight_meshes, axis=0).reshape(-1)

            def fixed_grid_marginal(points: np.ndarray) -> np.ndarray:
                """Integrate the evaluated inner density on a fixed quadrature grid."""
                result = np.empty(len(points), dtype=float)
                for row_index, point in enumerate(points):
                    fixed = dict(zip(integrated, point, strict=True))
                    values = np.empty((len(integration_points), len(inner.variables)))
                    for column, name in enumerate(inner.variables):
                        if name in fixed:
                            values[:, column] = fixed[name]
                        else:
                            values[:, column] = integration_points[:, integrated_names.index(name)]
                    result[row_index] = np.dot(
                        integration_weights, np.asarray(inner(values), dtype=float)
                    )
                return result

            return _CallableDensity(integrated, fixed_grid_marginal)

        from scipy.integrate import nquad

        def marginal(points: np.ndarray) -> np.ndarray:
            """Integrate composed densities over y0 summation variables."""
            result = np.empty(len(points), dtype=float)
            integrated_names = tuple(sorted(ranges))
            integration_ranges = [limits[name] for name in integrated_names]
            with _measure(
                profiler, "continuous.integration", rows=len(points), variables=integrated_names
            ):
                for row_index, point in enumerate(points):
                    fixed = dict(zip(integrated, point, strict=True))

                    def integrand(*values: float) -> float:
                        """Evaluate the inner density at one integration point."""
                        current = {**fixed, **dict(zip(integrated_names, values, strict=True))}
                        ordered = np.asarray(
                            [[current[name] for name in inner.variables]], dtype=float
                        )
                        value = float(np.asarray(inner(ordered)).reshape(-1)[0])
                        if not np.isfinite(value) or value < 0:
                            raise DistributionEstimationError(
                                "density returned a non-finite or negative value during integration"
                            )
                        return value

                    result[row_index] = float(nquad(integrand, integration_ranges)[0])
            return result

        return _CallableDensity(integrated, marginal)
    raise TypeError(f"unsupported y0 expression: {type(expression).__name__}")


def normalize_expression(
    expression: Expression, *, profiler: EstimationProfiler | None = None
) -> Expression:
    """Normalize a y0 expression using its available simplifier.

    axiomander:
        requires: isinstance(expression, Expression)
        ensures: isinstance(result, Expression)
        modifies: none
    """
    if not isinstance(expression, Expression):
        raise TypeError("expression must be a y0 Expression")
    previous = expression
    started = perf_counter() if profiler is not None else None
    if isinstance(previous, Sum):
        result = Sum.safe(
            normalize_expression(previous.expression, profiler=profiler), previous.ranges
        )
        if profiler is not None and started is not None:
            profiler._record("normalize", perf_counter() - started, iterations=1)
        return result
    if isinstance(previous, Probability):
        result = fraction_expand(previous)

        if profiler is not None and started is not None:
            profiler._record("normalize", perf_counter() - started, iterations=0)

        return result
    for _ in range(32):
        current = previous.simplify()
        if current == previous:
            if profiler is not None and started is not None:
                profiler._record("normalize", perf_counter() - started, iterations=_ + 1)
            return current
        previous = current

    raise DistributionEstimationError("expression normalization did not converge")


def _table(data: pd.DataFrame, variables: tuple[str, ...]) -> pd.DataFrame:
    """Prepare normalized empirical or explicitly weighted probability rows."""
    missing = set(variables) - set(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if "prob" in data.columns:
        if data["prob"].isna().any() or (data["prob"] < 0).any() or data["prob"].sum() <= 0:
            raise ValueError("prob must be finite, nonnegative, and have positive mass")
        result = data.loc[:, [*variables, "prob"]].copy()
        result["prob"] = result["prob"] / result["prob"].sum()
        return result.groupby(list(variables), dropna=False, as_index=False)["prob"].sum()
    if data.empty:
        raise ValueError("observed data must not be empty")
    result = data.loc[:, list(variables)].copy()
    return result.value_counts(dropna=False, normalize=True).rename("prob").reset_index()


def _discrete_eval(expression: Expression, data: pd.DataFrame):
    """Recursively evaluate an expression as a discrete probability table."""
    if isinstance(expression, Probability):
        children = _names(expression.children)
        parents = _names(expression.parents)
        keep = [*children, *parents]
        result = _table(data, tuple(keep))
        if not parents:
            return result
        parent_mass = result.groupby(list(parents), dropna=False)["prob"].transform("sum")
        result["prob"] = result["prob"] / parent_mass
        return result
    if isinstance(expression, One):
        return pd.DataFrame({"prob": [1.0]})
    if isinstance(expression, Zero):
        return pd.DataFrame({"prob": [0.0]})
    if isinstance(expression, Product):
        result = pd.DataFrame({"prob": [1.0]})
        for item in expression.expressions:
            factor = _discrete_eval(item, data)
            shared = [name for name in result.columns if name != "prob" and name in factor.columns]
            if shared:
                result = result.merge(factor, on=shared, how="outer", suffixes=("_left", "_right"))
            else:
                result["_join_key"] = 1
                factor["_join_key"] = 1
                result = result.merge(
                    factor, on="_join_key", how="outer", suffixes=("_left", "_right")
                ).drop(columns="_join_key")
            result["prob"] = result["prob_left"].fillna(0) * result["prob_right"].fillna(0)
            result = result.drop(columns=["prob_left", "prob_right"])
        return result
    if isinstance(expression, Fraction):
        numerator = _discrete_eval(expression.numerator, data)
        denominator = _discrete_eval(expression.denominator, data)
        shared = [
            name for name in numerator.columns if name != "prob" and name in denominator.columns
        ]
        if shared:
            result = numerator.merge(denominator, on=shared, how="left", suffixes=("_n", "_d"))
        else:
            numerator["_join_key"] = 1
            denominator["_join_key"] = 1
            result = numerator.merge(
                denominator, on="_join_key", how="left", suffixes=("_n", "_d")
            ).drop(columns="_join_key")
        if result["prob_d"].isna().any() or (result["prob_d"] <= 1e-14).any():
            raise DistributionEstimationError("probability fraction has a zero denominator")
        result["prob"] = result["prob_n"] / result["prob_d"]
        return result.drop(columns=["prob_n", "prob_d"])
    if isinstance(expression, Sum):
        result = _discrete_eval(expression.expression, data)
        ranges = [
            variable.name for variable in expression.ranges if variable.name in result.columns
        ]
        if ranges:
            remaining = [c for c in result.columns if c not in [*ranges, "prob"]]
            if remaining:
                result = result.groupby(remaining, dropna=False, as_index=False)["prob"].sum()
            else:
                result = pd.DataFrame({"prob": [result["prob"].sum()]})
        return result
    raise TypeError(f"unsupported y0 expression: {type(expression).__name__}")


@dataclass
class DistributionEstimator:
    """Configuration and entry point for homogeneous distribution estimation.

    axiomander:
        ensures: isinstance(result, DistributionEstimator)
        modifies: none
    """

    data: pd.DataFrame
    mode: str = "discrete"
    backend: ContinuousBackend | None = None
    bounds: Mapping[str, tuple[float, float]] | None = None
    profiler: EstimationProfiler | None = None
    marginalization: Literal["quadrature", "kde"] = "quadrature"

    def evaluate(self, expression: Expression):
        """Evaluate one expression using this estimator's configured mode.

        axiomander:
            requires: isinstance(expression, Expression)
            ensures: result is a pandas.DataFrame if self.mode == 'discrete' else callable(result)
            modifies: none
        """
        """Evaluate one expression using this estimator's configured mode."""
        return evaluate_probability_expression(
            expression,
            self.data,
            mode=self.mode,
            backend=self.backend,
            bounds=self.bounds,
            profiler=self.profiler,
            marginalization=self.marginalization,
        )


def evaluate_probability_expression(
    expression: Expression,
    data: pd.DataFrame,
    *,
    mode: str = "discrete",
    backend: ContinuousBackend | None = None,
    bounds: Mapping[str, tuple[float, float]] | tuple[float, float] | None = None,
    profiler: EstimationProfiler | None = None,
    marginalization: Literal["quadrature", "kde"] = "quadrature",
):
    """Evaluate a normalized y0 expression as a table or density callable.

    axiomander:
        requires: isinstance(expression, Expression); isinstance(data, pd.DataFrame); mode in {'discrete', 'continuous'}
        ensures: result is a pandas.DataFrame if mode == 'discrete' else callable(result)
        modifies: none
    """
    expression = normalize_expression(expression, profiler=profiler)
    if mode == "discrete":
        with _measure(profiler, "discrete.evaluate", rows=len(data)):
            return _discrete_eval(expression, data)
    if mode == "continuous":
        if marginalization not in {"quadrature", "kde"}:
            raise ValueError("marginalization must be 'quadrature' or 'kde'")
        if backend is None:
            backend = KDEpyKDE()
        with _measure(profiler, "continuous.evaluate", rows=len(data)):
            return _continuous_eval(expression, data, backend, bounds, profiler, marginalization)
    raise ValueError("mode must be 'discrete' or 'continuous'")


def _variable_name(variable: str | Variable) -> str:
    """Return a string name from a string or y0 variable."""
    return variable.name if isinstance(variable, Variable) else variable


def _validate_bounds(bounds: tuple[float, float] | None) -> tuple[float, float]:
    """Validate ordered integration bounds, allowing either infinite endpoint."""
    if bounds is None:
        raise ValueError("outcome bounds are required for continuous expectation")
    if len(bounds) != 2 or np.any(np.isnan(bounds)):
        raise ValueError("outcome bounds must contain two non-NaN values")
    lower, upper = map(float, bounds)
    if lower >= upper:
        raise ValueError("outcome bounds must be increasing")
    return lower, upper


def estimate_expectation(
    evaluated: pd.DataFrame | Callable[..., float | np.ndarray],
    outcome: str | Variable,
    *,
    bounds: tuple[float, float] | None = None,
    evaluation_values: Mapping[str, float] | None = None,
    profiler: EstimationProfiler | None = None,
) -> float:
    """Estimate an outcome expectation from an evaluated distribution.

    ``evaluated`` must be a probability table or callable density returned by
    :func:`evaluate_probability_expression`. Continuous expectations use bounded
    numerical quadrature. Any free density variables other than ``outcome`` must
    be supplied in ``evaluation_values``.
    """
    outcome_name = _variable_name(outcome)
    if isinstance(evaluated, pd.DataFrame):
        if bounds is not None:
            raise ValueError("bounds are only valid for continuous densities")
        with _measure(profiler, "discrete.expectation", rows=len(evaluated)):
            return _discrete_expectation(evaluated, outcome_name)
    if not callable(evaluated) or not hasattr(evaluated, "variables"):
        raise TypeError("evaluated must be a probability table or density callable")
    lower, upper = _validate_bounds(bounds)
    return _continuous_expectation(
        evaluated,
        outcome_name,
        bounds=(lower, upper),
        evaluation_values=evaluation_values,
        profiler=profiler,
    )


def _discrete_expectation(table: pd.DataFrame, outcome_name: str) -> float:
    """Calculate an expectation from an already evaluated probability table."""
    if outcome_name not in table:
        raise ValueError("identified expression does not contain the outcome")
    return float((table[outcome_name] * table["prob"]).sum())


def _continuous_expectation(
    density: Callable[..., float | np.ndarray],
    outcome_name: str,
    *,
    bounds: tuple[float, float],
    evaluation_values: Mapping[str, float] | None = None,
    profiler: EstimationProfiler | None = None,
) -> float:
    """Calculate an expectation with vectorized Gauss-Legendre quadrature."""
    lower, upper = bounds
    variables = tuple(cast(_DensityWithVariables, density).variables)
    if outcome_name not in variables:
        raise ValueError("identified expression does not contain the outcome")
    supplied = {} if evaluation_values is None else dict(evaluation_values)
    unexpected = set(supplied) - set(variables)
    if unexpected:
        raise ValueError(f"evaluation values contain unknown variables: {sorted(unexpected)}")
    remaining = set(variables) - {outcome_name}
    free_variables = remaining - set(supplied)

    from numpy.polynomial.legendre import leggauss

    nodes, node_weights = leggauss(64)
    if np.isfinite(lower) and np.isfinite(upper):
        midpoint = (lower + upper) / 2.0
        half_width = (upper - lower) / 2.0
        values = midpoint + half_width * nodes
        weights = half_width * node_weights
    elif not np.isfinite(lower) and not np.isfinite(upper):
        # Map (-1, 1) to the real line with x = tan(pi * u), u in (-1/2, 1/2).
        u = nodes / 2.0
        values = np.tan(np.pi * u)
        weights = node_weights * (np.pi / 2.0) / np.cos(np.pi * u) ** 2
    elif np.isfinite(lower):
        # Map (-1, 1) to [lower, inf) with x = lower + u / (1 - u).
        u = (nodes + 1.0) / 2.0
        values = lower + u / (1.0 - u)
        weights = node_weights / (2.0 * (1.0 - u) ** 2)
    else:
        # Map (-1, 1) to (-inf, upper] by reflecting the half-line mapping.
        u = (nodes + 1.0) / 2.0
        values = upper - (1.0 - u) / u
        weights = node_weights / (2.0 * u**2)
    with _measure(
        profiler,
        "continuous.expectation.quadrature",
        bounds=bounds,
        nodes=len(values),
    ):
        point = {outcome_name: values, **supplied}
        for free_variable in free_variables:
            point[free_variable] = values
        try:
            density_values = np.asarray(density(**point), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            density_values = np.asarray(
                [
                    density(
                        **{
                            **supplied,
                            outcome_name: value,
                            **dict.fromkeys(free_variables, value),
                        }
                    )
                    for value in values
                ],
                dtype=float,
            )
    if density_values.shape != values.shape:
        raise DistributionEstimationError("density returned an unexpected number of values")
    if not np.isfinite(density_values).all() or (density_values < 0).any():
        raise DistributionEstimationError("density returned a non-finite or negative value")
    mass = float(np.sum(weights * density_values))
    result = float(np.sum(weights * values * density_values))
    if not np.isfinite(mass) or mass <= 0:
        raise DistributionEstimationError(
            "outcome density has no positive mass in the supplied bounds"
        )
    if not np.isfinite(result):
        raise DistributionEstimationError("outcome expectation is not finite")
    return float(result / mass)


def estimate_ate(
    expression: Expression,
    data: pd.DataFrame,
    treatment: str | Variable,
    outcome: str | Variable,
    treatment_levels: tuple[object, object],
    *,
    mode: str = "discrete",
    backend: ContinuousBackend | None = None,
    outcome_bounds: tuple[float, float] | None = None,
    profiler: EstimationProfiler | None = None,
    marginalization: Literal["quadrature", "kde"] = "quadrature",
) -> float:
    """Estimate an ATE as the difference between two interventional expectations.

    axiomander:
        requires: isinstance(expression, Expression); isinstance(data, pd.DataFrame); len(treatment_levels) == 2
        ensures: isinstance(result, float)
        modifies: none
    """
    if len(treatment_levels) != 2:
        raise ValueError("treatment_levels must contain exactly two values")
    treatment_name = _variable_name(treatment)
    outcome_name = _variable_name(outcome)
    if treatment_name not in data or outcome_name not in data:
        raise ValueError("treatment and outcome must be data columns")
    if mode == "discrete":
        values = []
        for level in treatment_levels:
            subset = data[data[treatment_name] == level]
            if subset.empty:
                raise ValueError(f"no observations at treatment level {level!r}")
            values.append(
                estimate_expectation(
                    evaluate_probability_expression(
                        expression,
                        subset,
                        mode=mode,
                        backend=backend,
                        profiler=profiler,
                        marginalization=marginalization,
                    ),
                    outcome,
                    profiler=profiler,
                )
            )
        return float(values[1] - values[0])
    if mode == "continuous":
        expression_bounds: dict[str, tuple[float, float]] = {
            outcome_name: _validate_bounds(outcome_bounds),
        }
        for name in _names(expression.get_variables()):
            if name == outcome_name:
                continue
            if name not in data:
                raise ValueError(f"missing required columns: {[name]}")
            values = data[name].to_numpy(dtype=float)
            if not np.isfinite(values).all() or values.min() >= values.max():
                raise ValueError(f"continuous variable {name!r} must have finite variation")
            expression_bounds[name] = (float(values.min()), float(values.max()))
        evaluated = evaluate_probability_expression(
            expression,
            data,
            mode=mode,
            backend=backend,
            bounds=expression_bounds,
            profiler=profiler,
            marginalization=marginalization,
        )
        bounds = _validate_bounds(outcome_bounds)
        extra_variables = tuple(
            name for name in evaluated.variables if name not in {outcome_name, treatment_name}
        )
        if extra_variables:
            raise ValueError(
                "continuous ATE requires expression marginalization to be one-dimensional "
                f"over {outcome_name!r} and {treatment_name!r}; "
                f"found extra variables {extra_variables!r}"
            )
        has_treatment = treatment_name in evaluated.variables
        values = []
        for level in treatment_levels:
            level_float = float(cast(Any, level))
            values.append(
                estimate_expectation(
                    evaluated,
                    outcome,
                    bounds=bounds,
                    evaluation_values=({treatment_name: level_float} if has_treatment else None),
                    profiler=profiler,
                )
            )
        return float(values[1] - values[0])
    raise ValueError("mode must be 'discrete' or 'continuous'")
