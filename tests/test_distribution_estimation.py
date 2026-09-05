"""Tests for y0 distribution expression estimation."""

import numpy as np
import pandas as pd
import pytest
from y0.dsl import Fraction, P, Product, Sum, Variable

from nocap.estimation import (
    DistributionEstimationError,
    DistributionEstimator,
    estimate_ate,
    estimate_expectation,
    evaluate_probability_expression,
    normalize_expression,
)
from nocap.estimation.distribution import _density_ratio


def _data() -> pd.DataFrame:
    return pd.DataFrame({"X": [0, 0, 1, 1], "Y": [0, 1, 1, 1]})


def test_discrete_marginal_conditional_product_fraction_and_sum() -> None:
    data = _data()
    marginal = evaluate_probability_expression(P("X"), data)
    assert marginal.set_index("X").loc[0, "prob"] == pytest.approx(0.5)
    conditional = evaluate_probability_expression(P(Variable("X") | Variable("Y")), data)
    assert conditional.groupby("Y")["prob"].sum().to_numpy() == pytest.approx([1, 1])
    product = evaluate_probability_expression(Product.safe([P("X"), P("Y")]), data)
    assert product["prob"].sum() == pytest.approx(1)
    fraction = evaluate_probability_expression(Fraction(P("X", "Y"), P("Y")), data)
    assert fraction.groupby("Y")["prob"].sum().to_numpy() == pytest.approx([1, 1])
    summed = evaluate_probability_expression(Sum.safe(P("X", "Y"), "X"), data)
    assert summed["prob"].sum() == pytest.approx(1)


def test_raw_samples_and_probability_table_are_equivalent() -> None:
    raw = _data()
    table = raw.value_counts().rename("prob").reset_index()
    left = evaluate_probability_expression(P("X", "Y"), raw).sort_values(["X", "Y"]).reset_index(drop=True)
    right = evaluate_probability_expression(P("X", "Y"), table).sort_values(["X", "Y"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_expression_normalization_uses_y0_simplifier() -> None:
    expression = (P("X") / P("Y")) * P("Y")
    assert normalize_expression(expression) == P("X")


class _FakeKDE:
    def __init__(self, variables: tuple[str, ...], calls: list[tuple[str, ...]]):
        self.variables = variables
        self.calls = calls

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        return np.ones(len(points))

    def project(self, variables: tuple[str, ...]) -> "_FakeKDE":
        self.calls.append(tuple(variables))
        return _FakeKDE(tuple(variables), self.calls)


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def fit(self, data: pd.DataFrame, variables: tuple[str, ...]) -> _FakeKDE:
        self.calls.append(tuple(variables))
        return _FakeKDE(tuple(variables), self.calls)


def test_continuous_sum_projects_without_refitting() -> None:
    backend = _FakeBackend()
    density = evaluate_probability_expression(
        Sum.safe(P("X", "Y"), "X"),
        pd.DataFrame({"X": [0.0, 1.0], "Y": [1.0, 2.0]}),
        mode="continuous",
        backend=backend,
    )
    assert density.variables == ("Y",)
    assert backend.calls == [("X", "Y"), ("Y",)]
    assert density([2.0]) == pytest.approx(1.0)


def test_continuous_fraction_allows_joint_tail_underflow() -> None:
    assert _density_ratio(np.array([0.0]), np.array([0.0])) == pytest.approx([0.0])


def test_continuous_fraction_rejects_material_numerator_over_zero_denominator() -> None:
    with pytest.raises(DistributionEstimationError, match="zero denominator"):
        _density_ratio(np.array([1.0]), np.array([0.0]))


def test_validation_and_estimator_contract() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_probability_expression(P("Z"), _data())
    with pytest.raises(ValueError, match="positive mass"):
        evaluate_probability_expression(P("X"), pd.DataFrame({"X": [1], "prob": [0]}))
    assert DistributionEstimator(_data()).evaluate(P(Variable("X")))["prob"].sum() == pytest.approx(1)


class _KnownDensity:
    variables = ("Y",)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        return np.ones(len(points))

    def project(self, variables: tuple[str, ...]) -> "_KnownDensity":
        return self


class _KnownBackend:
    def fit(self, data: pd.DataFrame, variables: tuple[str, ...]) -> _KnownDensity:
        return _KnownDensity()


def test_continuous_expectation_uses_explicit_bounds() -> None:
    density = evaluate_probability_expression(
        P("Y"),
        pd.DataFrame({"Y": [0.0, 0.5, 1.0]}),
        mode="continuous",
        backend=_KnownBackend(),
    )
    result = estimate_expectation(
        density,
        "Y",
        bounds=(0.0, 1.0),
    )
    assert result == pytest.approx(0.5)


def test_continuous_expectation_rejects_missing_bounds_and_extra_variables() -> None:
    density = evaluate_probability_expression(
        P("Y"), pd.DataFrame({"Y": [0.0, 1.0]}), mode="continuous", backend=_KnownBackend()
    )
    with pytest.raises(ValueError, match="bounds"):
        estimate_expectation(density, "Y")
    with pytest.raises(ValueError, match="does not contain"):
        estimate_expectation(density, "X", bounds=(0.0, 1.0))


def test_continuous_expectation_accepts_infinite_bounds() -> None:
    class NormalDensity:
        variables = ("Y",)

        def __call__(self, **values: float) -> float:
            value = float(values["Y"])
            return float(np.exp(-0.5 * value**2) / np.sqrt(2 * np.pi))

        def evaluate(self, points: np.ndarray) -> np.ndarray:
            return np.exp(-0.5 * points[:, 0] ** 2) / np.sqrt(2 * np.pi)

        def project(self, variables: tuple[str, ...]) -> "NormalDensity":
            return self

    class NormalBackend:
        def fit(self, data: pd.DataFrame, variables: tuple[str, ...]) -> NormalDensity:
            return NormalDensity()

    result = estimate_expectation(
        NormalDensity(),
        "Y",
        bounds=(-np.inf, np.inf),
    )
    assert result == pytest.approx(0.0, abs=1e-8)


def test_continuous_ate_evaluates_treatment_levels_without_filtering() -> None:
    class ConditionalDensity(_KnownDensity):
        variables = ("Y", "T")

        def __call__(self, **values: float) -> float:
            return 1.0 if 0.0 <= values["Y"] <= 1.0 else 0.0

    class ConditionalBackend:
        def fit(self, data: pd.DataFrame, variables: tuple[str, ...]) -> ConditionalDensity:
            return ConditionalDensity()

    result = estimate_ate(
        P(Variable("Y") | Variable("T")),
        pd.DataFrame({"T": np.linspace(-2.0, 2.0, 20), "Y": np.linspace(0.0, 1.0, 20)}),
        "T",
        "Y",
        (-1.0, 1.0),
        mode="continuous",
        backend=ConditionalBackend(),
        outcome_bounds=(0.0, 1.0),
    )
    assert result == pytest.approx(0.0)
