"""Evaluate y0 probability expressions from discrete data or KDE densities.

The recursive evaluator deliberately depends on small backend protocols.  This
keeps symbolic evaluation testable independently of any particular KDE
implementation and makes analytic projection explicit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

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

from .kde import SciPyGaussianKDE, StatsmodelsKDE
from .models import ContinuousBackend, DistributionEstimationError, JointKDE, MarginalKDE

__all__ = [
    "ContinuousBackend",
    "DistributionEstimationError",
    "DistributionEstimator",
    "JointKDE",
    "MarginalKDE",
    "SciPyGaussianKDE",
    "StatsmodelsKDE",
    "estimate_ate",
    "estimate_expectation",
    "evaluate_probability_expression",
    "normalize_expression",
]


class _CallableDensity:
    """Callable wrapper that preserves deterministic density variable order."""

    def __init__(self, variables: tuple[str, ...], function: Callable[[np.ndarray], np.ndarray]):
        """Initialize a density over ``variables`` backed by ``function``."""
        self.variables = variables
        self._function = function

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


def _names(variables: Iterable[Variable]) -> tuple[str, ...]:
    """Return unique y0 variable names in deterministic order."""
    return tuple(sorted({variable.name for variable in variables}))


def _density_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide densities while treating jointly underflowed tails as zero."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    tiny_denominator = np.abs(denominator) <= 1e-14
    if np.any(tiny_denominator & (np.abs(numerator) > 1e-14)):
        raise DistributionEstimationError("density fraction has a zero denominator")
    result = np.zeros(np.broadcast_shapes(numerator.shape, denominator.shape), dtype=float)
    np.divide(numerator, denominator, out=result, where=~tiny_denominator)
    return result


def _distribution_density(backend: ContinuousBackend, data: pd.DataFrame, distribution: Distribution):
    """Build a callable joint or conditional density for a y0 distribution."""
    children = _names(distribution.children)
    parents = _names(distribution.parents)
    all_names = children + tuple(name for name in parents if name not in children)
    fitted = backend.fit(data, all_names)
    numerator = fitted.project(all_names)
    if not parents:
        return _CallableDensity(all_names, numerator.evaluate)
    denominator = fitted.project(parents)
    free = children + tuple(name for name in parents if name not in children)

    def conditional(points: np.ndarray) -> np.ndarray:
        """Evaluate the conditional density as a joint-to-marginal ratio."""
        num = numerator.evaluate(points)
        den = denominator.evaluate(points[:, [all_names.index(name) for name in parents]])
        try:
            return _density_ratio(num, den)
        except DistributionEstimationError as error:
            raise DistributionEstimationError("conditional density has a zero denominator") from error

    return _CallableDensity(free, conditional)


def _continuous_eval(
    expression: Expression,
    data: pd.DataFrame,
    backend: ContinuousBackend,
    bounds: Mapping[str, tuple[float, float]] | tuple[float, float] | None = None,
):
    """Recursively evaluate an expression using continuous density callables."""
    if isinstance(expression, Probability):
        return _distribution_density(backend, data, expression.distribution)
    if isinstance(expression, One):
        return _CallableDensity((), lambda points: np.ones(len(points)))
    if isinstance(expression, Zero):
        return _CallableDensity((), lambda points: np.zeros(len(points)))
    if isinstance(expression, Product):
        densities = [
            _continuous_eval(item, data, backend, bounds) for item in expression.expressions
        ]
        variables = tuple(dict.fromkeys(name for density in densities for name in density.variables))

        def product(points: np.ndarray) -> np.ndarray:
            """Evaluate and multiply all factor densities on shared points."""
            result = np.ones(len(points))
            for density in densities:
                indexes = [variables.index(name) for name in density.variables]
                result *= np.asarray(density(points[:, indexes] if indexes else np.empty((len(points), 0))))
            return result

        return _CallableDensity(variables, product)
    if isinstance(expression, Fraction):
        numerator = _continuous_eval(expression.numerator, data, backend, bounds)
        denominator = _continuous_eval(expression.denominator, data, backend, bounds)
        variables = tuple(dict.fromkeys((*numerator.variables, *denominator.variables)))

        def fraction(points: np.ndarray) -> np.ndarray:
            """Evaluate a density ratio while rejecting tiny denominators."""
            n = numerator(points[:, [variables.index(x) for x in numerator.variables]] if numerator.variables else np.empty((len(points), 0)))
            d = denominator(points[:, [variables.index(x) for x in denominator.variables]] if denominator.variables else np.empty((len(points), 0)))
            return _density_ratio(n, d)

        return _CallableDensity(variables, fraction)
    if isinstance(expression, Sum):
        if isinstance(expression.expression, Probability) and not expression.expression.parents:
            inner_variables = _names(expression.expression.children)
            ranges = {variable.name for variable in expression.ranges}
            kept = tuple(name for name in inner_variables if name not in ranges)
            fitted = backend.fit(data, inner_variables)
            projected = fitted.project(kept)
            return _CallableDensity(kept, projected.evaluate)
        inner = _continuous_eval(expression.expression, data, backend, bounds)
        ranges = {variable.name for variable in expression.ranges}
        if not ranges.issubset(inner.variables):
            return inner
        integrated = tuple(name for name in inner.variables if name not in ranges)
        limits = {}
        for name in ranges:
            if bounds is None:
                raise DistributionEstimationError(
                    f"integration bounds are required for summed variable {name!r}"
                )
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

        from scipy.integrate import nquad

        def marginal(points: np.ndarray) -> np.ndarray:
            """Integrate composed densities over y0 summation variables."""
            result = np.empty(len(points), dtype=float)
            integrated_names = tuple(sorted(ranges))
            integration_ranges = [limits[name] for name in integrated_names]
            for row_index, point in enumerate(points):
                fixed = dict(zip(integrated, point, strict=True))

                def integrand(*values: float) -> float:
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


def normalize_expression(expression: Expression) -> Expression:
    """Normalize a y0 expression using its available simplifier.

    axiomander:
        requires: isinstance(expression, Expression)
        ensures: isinstance(result, Expression)
        modifies: none
    """
    if not isinstance(expression, Expression):
        raise TypeError("expression must be a y0 Expression")
    previous = expression
    if isinstance(previous, Sum):
        return Sum.safe(normalize_expression(previous.expression), previous.ranges)
    if isinstance(previous, Probability):
        return previous
    for _ in range(32):
        current = previous.simplify()
        if current == previous:
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
    variables = _names(expression.get_variables())
    base = _table(data, variables)
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
                result = result.merge(factor, on="_join_key", how="outer", suffixes=("_left", "_right")).drop(columns="_join_key")
            result["prob"] = result["prob_left"].fillna(0) * result["prob_right"].fillna(0)
            result = result.drop(columns=["prob_left", "prob_right"])
        return result
    if isinstance(expression, Fraction):
        numerator = _discrete_eval(expression.numerator, data)
        denominator = _discrete_eval(expression.denominator, data)
        shared = [name for name in numerator.columns if name != "prob" and name in denominator.columns]
        if shared:
            result = numerator.merge(denominator, on=shared, how="left", suffixes=("_n", "_d"))
        else:
            numerator["_join_key"] = 1
            denominator["_join_key"] = 1
            result = numerator.merge(denominator, on="_join_key", how="left", suffixes=("_n", "_d")).drop(columns="_join_key")
        if result["prob_d"].isna().any() or (result["prob_d"] <= 1e-14).any():
            raise DistributionEstimationError("probability fraction has a zero denominator")
        result["prob"] = result["prob_n"] / result["prob_d"]
        return result.drop(columns=["prob_n", "prob_d"])
    if isinstance(expression, Sum):
        result = _discrete_eval(expression.expression, data)
        ranges = [variable.name for variable in expression.ranges if variable.name in result.columns]
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
        )


def evaluate_probability_expression(
    expression: Expression,
    data: pd.DataFrame,
    *,
    mode: str = "discrete",
    backend: ContinuousBackend | None = None,
    bounds: Mapping[str, tuple[float, float]] | tuple[float, float] | None = None,
):
    """Evaluate a normalized y0 expression as a table or density callable.

    axiomander:
        requires: isinstance(expression, Expression); isinstance(data, pd.DataFrame); mode in {'discrete', 'continuous'}
        ensures: result is a pandas.DataFrame if mode == 'discrete' else callable(result)
        modifies: none
    """
    expression = normalize_expression(expression)
    if mode == "discrete":
        return _discrete_eval(expression, data)
    if mode == "continuous":
        if backend is None:
            backend = SciPyGaussianKDE()
        return _continuous_eval(expression, data, backend, bounds)
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
        return _discrete_expectation(evaluated, outcome_name)
    if not callable(evaluated) or not hasattr(evaluated, "variables"):
        raise TypeError("evaluated must be a probability table or density callable")
    lower, upper = _validate_bounds(bounds)
    return _continuous_expectation(
        evaluated,
        outcome_name,
        bounds=(lower, upper),
        evaluation_values=evaluation_values,
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
) -> float:
    """Calculate an expectation from an already evaluated continuous density."""
    lower, upper = bounds
    variables = tuple(density.variables)
    if outcome_name not in variables:
        raise ValueError("identified expression does not contain the outcome")
    supplied = {} if evaluation_values is None else dict(evaluation_values)
    unexpected = set(supplied) - set(variables)
    if unexpected:
        raise ValueError(f"evaluation values contain unknown variables: {sorted(unexpected)}")
    remaining = set(variables) - {outcome_name}
    if remaining != set(supplied):
        raise ValueError(
            "continuous outcome density must be one-dimensional or have explicit "
            "evaluation values for every other free variable"
        )

    from scipy.integrate import quad

    def integrand(value: float) -> float:
        point = {outcome_name: value, **supplied}
        result = float(np.asarray(density(**point), dtype=float).reshape(-1)[0])
        if not np.isfinite(result) or result < 0:
            raise DistributionEstimationError("density returned a non-finite or negative value")
        return value * result

    def density_integrand(value: float) -> float:
        point = {outcome_name: value, **supplied}
        density_value = float(np.asarray(density(**point), dtype=float).reshape(-1)[0])
        if not np.isfinite(density_value) or density_value < 0:
            raise DistributionEstimationError("density returned a non-finite or negative value")
        return density_value

    mass, _ = quad(density_integrand, lower, upper)
    result, _ = quad(integrand, lower, upper)
    if not np.isfinite(mass) or mass <= 0:
        raise DistributionEstimationError("outcome density has no positive mass in the supplied bounds")
    if not np.isfinite(result):
        raise DistributionEstimationError("outcome expectation is not finite")
    return float(result)


def estimate_ate(expression: Expression, data: pd.DataFrame, treatment: str | Variable, outcome: str | Variable, treatment_levels: tuple[object, object], *, mode: str = "discrete", backend: ContinuousBackend | None = None, outcome_bounds: tuple[float, float] | None = None) -> float:
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
                        expression, subset, mode=mode, backend=backend
                    ),
                    outcome,
                )
            )
        return float(values[1] - values[0])
    if mode == "continuous":
        evaluated = evaluate_probability_expression(
            expression,
            data,
            mode=mode,
            backend=backend,
            bounds=outcome_bounds,
        )
        bounds = _validate_bounds(outcome_bounds)
        has_treatment = treatment_name in evaluated.variables
        values = []
        for level in treatment_levels:
            values.append(
                estimate_expectation(
                    evaluated,
                    outcome,
                    bounds=bounds,
                    evaluation_values=(
                        {treatment_name: float(level)} if has_treatment else None
                    ),
                )
            )
        return float(values[1] - values[0])
    raise ValueError("mode must be 'discrete' or 'continuous'")
