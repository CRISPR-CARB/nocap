"""Tests for y0 distribution expression estimation."""

import numpy as np
import pandas as pd
import pytest
from y0.dsl import Fraction, P, Product, Sum, Variable

from nocap.estimation import (
    DistributionEstimationError,
    DistributionEstimator,
    EstimationProfiler,
    KDEpyKDE,
    SklearnKDE,
    estimate_ate,
    estimate_expectation,
    evaluate_probability_expression,
    normalize_expression,
)
from nocap.estimation.distribution import _density_ratio
from nocap.estimation.kde import _KDEpyFitted


def _data() -> pd.DataFrame:
    return pd.DataFrame({"X": [0, 0, 1, 1], "Y": [0, 1, 1, 1]})


def test_discrete_marginal_conditional_product_fraction_and_sum() -> None:
    """Evaluate common discrete probability-expression constructions."""
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
    """Treat raw observations and normalized probability tables equivalently."""
    raw = _data()
    table = raw.value_counts().rename("prob").reset_index()
    left = (
        evaluate_probability_expression(P("X", "Y"), raw)
        .sort_values(["X", "Y"])
        .reset_index(drop=True)
    )
    right = (
        evaluate_probability_expression(P("X", "Y"), table)
        .sort_values(["X", "Y"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(left, right)


def test_expression_normalization_uses_y0_simplifier() -> None:
    """Use y0 simplification when normalizing an expression."""
    expression = (P("X") / P("Y")) * P("Y")
    assert normalize_expression(expression) == P("X")


class _FakeKDE:
    def __init__(self, variables: tuple[str, ...], calls: list[tuple[str, ...]]):
        """Create a fake fitted density that records projections."""
        self.variables = variables
        self.calls = calls

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Return unit density values for all point rows."""
        return np.ones(len(points))

    def project(self, variables: tuple[str, ...]) -> "_FakeKDE":
        """Record and return a fake projection."""
        self.calls.append(tuple(variables))
        return _FakeKDE(tuple(variables), self.calls)


class _FakeBackend:
    def __init__(self) -> None:
        """Create a fake backend that records fit requests."""
        self.calls: list[tuple[str, ...]] = []

    def fit(self, data: pd.DataFrame, variables: tuple[str, ...]) -> _FakeKDE:
        """Fit and record a fake density for selected columns."""
        self.calls.append(tuple(variables))
        return _FakeKDE(tuple(variables), self.calls)


def test_continuous_sum_projects_without_refitting() -> None:
    """Project a continuous sum without fitting the inner density twice."""
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


def test_continuous_sum_kde_evaluates_inner_density_on_grid() -> None:
    """Evaluate KDE marginalization on a fixed quadrature grid."""

    class BandwidthKDE:
        """Fake fitted density used to inspect quadrature bandwidth behavior."""

        def __init__(self, variables: tuple[str, ...], bandwidth: float | None):
            """Create a fake density carrying its bandwidth."""
            self.variables = variables
            self.bandwidth = bandwidth

        def evaluate(self, points: np.ndarray) -> np.ndarray:
            """Return unit density values for all point rows."""
            return np.ones(len(points))

        def project(self, variables: tuple[str, ...]) -> "BandwidthKDE":
            """Project while retaining the fitted bandwidth."""
            return BandwidthKDE(tuple(variables), self.bandwidth)

    class BandwidthBackend:
        """Fake backend that records the bandwidth passed to each fit."""

        def __init__(self) -> None:
            """Create a fake backend that records bandwidth requests."""
            self.calls: list[tuple[tuple[str, ...], object]] = []

        def fit(
            self,
            data: pd.DataFrame,
            variables: tuple[str, ...],
            *,
            bandwidth=None,
        ) -> BandwidthKDE:
            """Fit a fake bandwidth-aware density."""
            self.calls.append((tuple(variables), bandwidth))
            return BandwidthKDE(tuple(variables), 0.25 if bandwidth is None else bandwidth)

    backend = BandwidthBackend()
    density = evaluate_probability_expression(
        Sum.safe(P("X", "Y"), "X"),
        pd.DataFrame({"X": [0.0, 1.0], "Y": [1.0, 2.0]}),
        mode="continuous",
        backend=backend,
        marginalization="kde",
    )
    assert density.variables == ("Y",)
    assert backend.calls == [(("X", "Y"), None)]


def test_kdepy_backend_evaluates_and_projects() -> None:
    """Evaluate and project a KDEpy fitted density."""
    data = pd.DataFrame(
        {
            "X": np.linspace(-2.0, 2.0, 32),
            "Y": np.linspace(0.0, 1.0, 32),
        }
    )
    fitted = KDEpyKDE().fit(data, ("X", "Y"))
    values = fitted.evaluate(np.array([[0.0, 0.5], [1.0, 0.75]]))
    projected = fitted.project(("Y",))
    assert values.shape == (2,)
    assert np.isfinite(values).all()
    assert projected.evaluate(np.array([[0.5], [0.75]])).shape == (2,)


def test_sklearn_backend_evaluates_and_projects() -> None:
    """Evaluate and project a scikit-learn fitted density."""
    data = pd.DataFrame(
        {
            "X": np.linspace(-2.0, 2.0, 32),
            "Y": np.linspace(0.0, 1.0, 32),
        }
    )
    fitted = SklearnKDE(bandwidth=0.25).fit(data, ("X", "Y"))
    values = fitted.evaluate(np.array([[0.0, 0.5], [1.0, 0.75]]))
    projected = fitted.project(("Y",))
    assert values.shape == (2,)
    assert np.isfinite(values).all()
    assert projected.evaluate(np.array([[0.5], [0.75]])).shape == (2,)
    assert projected.bandwidth == pytest.approx(fitted.bandwidth)


def test_sklearn_backend_accepts_kernel_and_fit_bandwidth_override() -> None:
    """Honor the configured kernel and per-fit bandwidth override."""
    data = pd.DataFrame({"X": np.linspace(-1.0, 1.0, 8)})
    fitted = SklearnKDE(bandwidth=0.5, kernel="tophat").fit(data, ("X",), bandwidth=0.2)
    assert fitted.estimator.kernel == "tophat"
    assert fitted.bandwidth == pytest.approx(0.2)


def test_sklearn_backend_rejects_invalid_data() -> None:
    """Reject non-finite or undersized continuous KDE data."""
    with pytest.raises(ValueError, match="finite and contain at least two rows"):
        SklearnKDE().fit(pd.DataFrame({"X": [1.0, np.nan]}), ("X",))


def test_continuous_default_uses_kdepy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use KDEpy as the default continuous estimation backend."""
    calls = []

    class FakeKDEpy:
        """Fake KDEpy backend used to verify the default backend selection."""

        def fit(self, data: pd.DataFrame, variables: tuple[str, ...]) -> _KDEpyFitted:
            """Record a fit and return a KDEpy-compatible fake fit."""
            calls.append(tuple(variables))
            return _KDEpyFitted(tuple(variables), data.loc[:, list(variables)].to_numpy())

    monkeypatch.setattr("nocap.estimation.distribution.KDEpyKDE", FakeKDEpy)
    evaluate_probability_expression(
        P("X"),
        pd.DataFrame({"X": np.linspace(-1.0, 1.0, 8)}),
        mode="continuous",
    )
    assert calls == [("X",)]


def test_continuous_fraction_allows_joint_tail_underflow() -> None:
    """Treat simultaneous numerator and denominator underflow as zero."""
    assert _density_ratio(np.array([0.0]), np.array([0.0])) == pytest.approx([0.0])


def test_continuous_fraction_rejects_material_numerator_over_zero_denominator() -> None:
    """Reject a nonzero numerator divided by a zero density."""
    with pytest.raises(DistributionEstimationError, match="zero denominator"):
        _density_ratio(np.array([1.0]), np.array([0.0]))


def test_validation_and_estimator_contract() -> None:
    """Validate input columns, probability mass, and estimator delegation."""
    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_probability_expression(P("Z"), _data())
    with pytest.raises(ValueError, match="positive mass"):
        evaluate_probability_expression(P("X"), pd.DataFrame({"X": [1], "prob": [0]}))
    assert DistributionEstimator(_data()).evaluate(P(Variable("X")))["prob"].sum() == pytest.approx(
        1
    )


def test_profiler_collects_discrete_estimation_statistics() -> None:
    """Collect operation counts and metadata during discrete estimation."""
    profiler = EstimationProfiler()
    result = evaluate_probability_expression(P("X"), _data(), profiler=profiler)
    assert result["prob"].sum() == pytest.approx(1)
    rows = {row["operation"]: row for row in profiler.summary()}
    assert rows["normalize"]["calls"] == 1
    assert rows["discrete.evaluate"]["calls"] == 1
    assert rows["discrete.evaluate"]["last_metadata"] == {"rows": 4}
    assert rows["discrete.evaluate"]["total_seconds"] >= 0


class _KnownDensity:
    variables = ("Y",)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Return unit density values for all point rows."""
        return np.ones(len(points))

    def project(self, variables: tuple[str, ...]) -> "_KnownDensity":
        """Return the known density for any requested projection."""
        return self


class _KnownBackend:
    def fit(self, data: pd.DataFrame, variables: tuple[str, ...]) -> _KnownDensity:
        """Return a known density without fitting data."""
        return _KnownDensity()


def test_continuous_expectation_uses_explicit_bounds() -> None:
    """Integrate a continuous density over caller-provided bounds."""
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


def test_continuous_expectation_vectorizes_density_evaluation() -> None:
    """Use one vectorized density call for continuous expectation quadrature."""
    calls: list[np.ndarray] = []

    class VectorizedDensity:
        """Density whose call history demonstrates vectorized evaluation."""

        variables = ("Y",)

        def __call__(self, **values: np.ndarray) -> np.ndarray:
            """Evaluate a simple vectorized quadratic density."""
            points = np.asarray(values["Y"])
            calls.append(points)
            return np.where((points >= -1.0) & (points <= 1.0), 1.0 + points**2, 0.0)

    result = estimate_expectation(VectorizedDensity(), "Y", bounds=(-1.0, 1.0))

    assert result == pytest.approx(0.0, abs=1e-12)
    assert len(calls) == 1
    assert calls[0].shape == (64,)


def test_continuous_expectation_normalizes_truncated_mass() -> None:
    """Normalize an expectation when the integration bounds truncate mass."""

    class LinearDensity:
        """Simple non-normalized density used to test mass correction."""

        variables = ("Y",)

        def __call__(self, **values: np.ndarray) -> np.ndarray:
            """Evaluate a linear density over the requested points."""
            return np.asarray(values["Y"])

    result = estimate_expectation(LinearDensity(), "Y", bounds=(1.0, 3.0))

    assert result == pytest.approx(13.0 / 6.0)


def test_continuous_expectation_rejects_missing_bounds_and_extra_variables() -> None:
    """Reject missing outcome bounds and outcomes absent from a density."""
    density = evaluate_probability_expression(
        P("Y"), pd.DataFrame({"Y": [0.0, 1.0]}), mode="continuous", backend=_KnownBackend()
    )
    with pytest.raises(ValueError, match="bounds"):
        estimate_expectation(density, "Y")
    with pytest.raises(ValueError, match="does not contain"):
        estimate_expectation(density, "X", bounds=(0.0, 1.0))


def test_continuous_expectation_accepts_infinite_bounds() -> None:
    """Integrate a normal density over infinite bounds."""

    class NormalDensity:
        """Analytic standard normal density used for infinite-bound integration."""

        variables = ("Y",)

        def __call__(self, **values: float) -> float:
            """Evaluate the standard normal density at one point."""
            value = float(values["Y"])
            return float(np.exp(-0.5 * value**2) / np.sqrt(2 * np.pi))

        def evaluate(self, points: np.ndarray) -> np.ndarray:
            """Evaluate the standard normal density for point rows."""
            return np.exp(-0.5 * points[:, 0] ** 2) / np.sqrt(2 * np.pi)

        def project(self, variables: tuple[str, ...]) -> "NormalDensity":
            """Return the one-dimensional normal density unchanged."""
            return self

    class NormalBackend:
        """Backend returning the analytic normal density."""

        def fit(self, data: pd.DataFrame, variables: tuple[str, ...]) -> NormalDensity:
            """Return the analytic normal density."""
            return NormalDensity()

    result = estimate_expectation(
        NormalDensity(),
        "Y",
        bounds=(-np.inf, np.inf),
    )
    assert result == pytest.approx(0.0, abs=1e-8)


def test_continuous_ate_evaluates_treatment_levels_without_filtering() -> None:
    """Evaluate both treatment levels using one conditional density."""

    class ConditionalDensity(_KnownDensity):
        """Density independent of treatment for the continuous ATE test."""

        variables = ("Y", "T")

        def __call__(self, **values: float) -> float:
            """Evaluate a uniform outcome density independent of treatment."""
            return 1.0 if 0.0 <= values["Y"] <= 1.0 else 0.0

    class ConditionalBackend:
        """Backend returning the treatment-independent conditional density."""

        def fit(self, data: pd.DataFrame, variables: tuple[str, ...]) -> ConditionalDensity:
            """Return the conditional test density."""
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


def test_continuous_ate_requires_expression_marginalization() -> None:
    """Reject a continuous ATE with an un-marginalized extra variable."""

    class JointDensity(_KnownDensity):
        """Joint density retaining an extra covariate for rejection testing."""

        variables = ("Y", "T", "X")

        def __call__(self, **values: float) -> float:
            """Evaluate a joint density with an extra covariate."""
            return 1.0 if 0.0 <= values["Y"] <= 1.0 else 0.0

    class JointBackend:
        """Backend returning the joint density with an extra covariate."""

        def fit(self, data: pd.DataFrame, variables: tuple[str, ...]) -> JointDensity:
            """Return the joint test density."""
            return JointDensity()

    with pytest.raises(ValueError, match="one-dimensional"):
        estimate_ate(
            P("Y", "T", "X"),
            pd.DataFrame(
                {
                    "T": np.linspace(-1.0, 1.0, 20),
                    "X": np.linspace(0.0, 1.0, 20),
                    "Y": np.linspace(0.0, 1.0, 20),
                }
            ),
            "T",
            "Y",
            (-1.0, 1.0),
            mode="continuous",
            backend=JointBackend(),
            outcome_bounds=(0.0, 1.0),
        )
