"""Tests for discrete probability backend implementations."""

import sys
import types
from typing import cast

import numpy as np
import pandas as pd
import pytest
from y0.dsl import P, Product, Variable
from y0.graph import NxMixedGraph

from nocap.estimation import (
    PgmpyDiscreteBackend,
    PluginDiscreteBackend,
    evaluate_probability_expression,
)


def test_plugin_backend_preserves_state_order_and_requested_variable_order() -> None:
    """Construct a factor with states and axes matching the requested order."""
    data = pd.DataFrame({"X": [1, 0, 1], "Y": ["b", "a", "a"]})

    factor = PluginDiscreteBackend().fit(data, ("Y", "X"))

    assert factor.variables == ["Y", "X"]
    assert factor.state_names == {"Y": ["b", "a"], "X": [1, 0]}


def test_plugin_backend_keeps_weighted_probability_semantics() -> None:
    """Aggregate repeated weighted rows into a normalized joint factor."""
    data = pd.DataFrame({"X": [0, 0, 1], "prob": [1.0, 2.0, 1.0]})

    table = cast(pd.DataFrame, evaluate_probability_expression(P("X"), data)).set_index("X")

    assert table.loc[0, "prob"] == pytest.approx(0.75)
    assert table.loc[1, "prob"] == pytest.approx(0.25)


def test_pgmpy_backend_fits_once_and_queries_expression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit a DAG once for an expression with multiple probability leaves."""
    module = types.ModuleType("nocap.acyclification")
    module.acyclification = lambda graph, acyclification_type="dag": graph
    monkeypatch.setitem(sys.modules, "nocap.acyclification", module)
    x, y, z = Variable("X"), Variable("Y"), Variable("Z")
    graph = NxMixedGraph.from_edges(nodes=[x, y, z], directed=[(x, y)])
    backend = PgmpyDiscreteBackend(graph)
    fit_calls = 0
    fit_network = backend.fit_network

    def counted_fit(data: pd.DataFrame) -> None:
        """Count network fits while invoking the original fitter."""
        nonlocal fit_calls
        fit_calls += 1
        fit_network(data)

    backend.fit_network = counted_fit
    data = pd.DataFrame({"X": [0, 0, 1, 1], "Y": [0, 1, 1, 1], "Z": [1, 0, 1, 0]})

    result = cast(
        pd.DataFrame,
        evaluate_probability_expression(Product.safe([P("X"), P("Y")]), data, backend=backend),
    )

    assert fit_calls == 1
    assert backend.model is not None
    assert backend.model.check_model()
    assert "Z" in backend.model.nodes()
    assert set(result.columns) == {"X", "Y", "prob"}
    assert result["prob"].sum() == pytest.approx(1.0)


def test_pgmpy_backend_supports_belief_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use pgmpy's belief-propagation inference with the common query contract."""
    module = types.ModuleType("nocap.acyclification")
    module.acyclification = lambda graph, acyclification_type="dag": graph
    monkeypatch.setitem(sys.modules, "nocap.acyclification", module)
    x, y = Variable("X"), Variable("Y")
    graph = NxMixedGraph.from_edges(nodes=[x, y], directed=[(x, y)])
    data = pd.DataFrame({"X": [0, 0, 1, 1], "Y": [0, 1, 1, 1]})

    result = cast(
        pd.DataFrame,
        evaluate_probability_expression(
            P("Y"),
            data,
            backend=PgmpyDiscreteBackend(graph, inference_backend="belief_propagation"),
        ),
    )

    probabilities = result.set_index("Y")["prob"]
    assert probabilities.loc[0] == pytest.approx(0.25)
    assert probabilities.loc[1] == pytest.approx(0.75)


def test_pgmpy_backend_rejects_unknown_inference_backend() -> None:
    """Require inference backend names to be explicit and supported."""
    graph = NxMixedGraph.from_edges(nodes=[Variable("X")], directed=[])

    with pytest.raises(ValueError, match="inference_backend"):
        PgmpyDiscreteBackend(graph, inference_backend="unknown")  # type: ignore[arg-type]


def test_pgmpy_backend_accepts_numpy_boolean_states(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept NumPy boolean states from latent binary simulation data."""
    module = types.ModuleType("nocap.acyclification")
    module.acyclification = lambda graph, acyclification_type="dag": graph
    monkeypatch.setitem(sys.modules, "nocap.acyclification", module)
    x, y = Variable("X"), Variable("Y")
    graph = NxMixedGraph.from_edges(nodes=[x, y], directed=[(x, y)])
    data = pd.DataFrame({"X": [False, False, True, True], "Y": [False, True, True, True]})

    result = cast(pd.DataFrame, evaluate_probability_expression(P("Y"), data, backend=PgmpyDiscreteBackend(graph)))

    assert set(result["Y"]) == {False, True}
    assert result["prob"].sum() == pytest.approx(1.0)


def test_pgmpy_backend_rejects_weighted_data_and_non_em_latents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject inputs and estimator choices incompatible with full-network fitting."""
    module = types.ModuleType("nocap.acyclification")
    module.acyclification = lambda graph, acyclification_type="dag": graph
    monkeypatch.setitem(sys.modules, "nocap.acyclification", module)
    x, y = Variable("X"), Variable("Y")
    graph = NxMixedGraph.from_edges(nodes=[x, y], undirected=[(x, y)])
    data = pd.DataFrame({"X": [0, 1], "Y": [1, 0]})

    with pytest.raises(ValueError, match="weighted 'prob'"):
        PgmpyDiscreteBackend(graph).fit_network(data.assign(prob=1.0))
    with pytest.raises(ValueError, match="require method='EM'"):
        PgmpyDiscreteBackend(graph, method="MLE").fit_network(data)


def test_pgmpy_backend_fits_bidirected_latent_parents_with_em(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expand a bidirected edge to a latent parent and fit it using EM."""
    module = types.ModuleType("nocap.acyclification")
    module.acyclification = lambda graph, acyclification_type="dag": graph
    monkeypatch.setitem(sys.modules, "nocap.acyclification", module)
    x, y = Variable("X"), Variable("Y")
    graph = NxMixedGraph.from_edges(nodes=[x, y], undirected=[(x, y)])
    data = pd.DataFrame({"X": [0, 0, 1, 1, 0, 1], "Y": [0, 1, 1, 0, 0, 1]})
    data.attrs["metadata"] = np.array([1, 2, 3])
    backend = PgmpyDiscreteBackend(
        graph,
        method="EM",
        estimator_kwargs={"max_iter": 2, "show_progress": False},
    )

    result = cast(pd.DataFrame, evaluate_probability_expression(P("X", "Y"), data, backend=backend))

    assert backend.model is not None
    assert len(backend.model.latents) == 1
    assert backend.model.check_model()
    assert result["prob"].sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(data.attrs["metadata"], [1, 2, 3])
