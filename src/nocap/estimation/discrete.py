"""Discrete probability backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from nocap.acyclification import acyclification

from .models import DistributionEstimationError

if TYPE_CHECKING:
    from pgmpy.factors.discrete import DiscreteFactor

__all__ = ["PgmpyDiscreteBackend", "PluginDiscreteBackend"]

_PGMPY_INFERENCE_BACKENDS = {
    "variable_elimination": "VariableElimination",
    "belief_propagation": "BeliefPropagation",
}


def _factor_type():
    """Import pgmpy's factor class only when a discrete backend is used."""
    try:
        from pgmpy.factors.discrete import DiscreteFactor
    except ImportError as error:
        raise ImportError("pgmpy is required for discrete probability estimation") from error
    return DiscreteFactor


def _state_index(states: Sequence[object], value: object) -> int:
    """Locate a state while treating missing values as one discrete level."""
    for index, state in enumerate(states):
        if bool(np.asarray(pd.isna(state)).all()) and bool(np.asarray(pd.isna(value)).all()):
            return index
        try:
            if state == value:
                return index
        except (TypeError, ValueError):
            continue
    raise KeyError(value)


class PluginDiscreteBackend:
    """Empirical backend supporting samples and explicitly weighted rows."""

    def fit(self, data: pd.DataFrame, variables: Sequence[str]) -> DiscreteFactor:
        """Construct the normalized empirical joint factor for requested variables."""
        from pgmpy.factors.discrete import DiscreteFactor

        names = tuple(variables)
        missing = set(names) - set(data.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        if data.empty:
            raise ValueError("observed data must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("requested variables must be unique")

        values, state_names, cardinality = (
            _empirical_joint(data, names) if names else _empty_joint(data)
        )

        return DiscreteFactor(
            variables=list(names),
            cardinality=cardinality,
            values=values,
            state_names=dict(zip(names, state_names, strict=True)),
        )


def _empirical_joint(data: pd.DataFrame, names: tuple[str, ...]):
    """Build normalized values and state domains for non-empty variable sets."""
    if "prob" in data.columns:
        probabilities = data["prob"].to_numpy(dtype=float)
        total = float(probabilities.sum())
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.isfinite(total)
            or total <= 0
        ):
            raise ValueError("prob must be finite, nonnegative, and have positive mass")
        table = data.loc[:, list(names)].copy()
        table["_weight"] = probabilities / total
        table = table.groupby(list(names), dropna=False, sort=True)["_weight"].sum()
    else:
        table = data.value_counts(subset=list(names), dropna=False, sort=False)
        table = table / float(table.sum())
    state_names = [list(pd.unique(cast(pd.Series, data[name]))) for name in names]
    cardinality = [len(states) for states in state_names]
    values = np.zeros(cardinality, dtype=float)
    for key, mass in table.items():
        key = key if isinstance(key, tuple) else (key,)
        try:
            indices = tuple(
                _state_index(states, value)
                for states, value in zip(state_names, key, strict=True)
            )
        except (KeyError, TypeError) as error:
            raise ValueError("discrete variables must contain hashable state values") from error
        values[indices] = float(mass)
    return values, state_names, cardinality


def _empty_joint(data: pd.DataFrame):
    """Validate optional weights for a joint factor with no variables."""
    if "prob" in data.columns:
        probabilities = data["prob"].to_numpy(dtype=float)
        total = float(probabilities.sum())
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.isfinite(total)
            or total <= 0
        ):
            raise ValueError("prob must be finite, nonnegative, and have positive mass")
    return np.asarray(1.0), [], []


class PgmpyDiscreteBackend:
    """Fit a pgmpy Bayesian network from an acyclified mixed graph."""

    def __init__(
        self,
        graph,
        *,
        acyclification_type: Literal["dag", "bidirected"] = "dag",
        method: Literal["MLE", "Bayesian", "EM"] = "MLE",
        inference_backend: Literal[
            "variable_elimination", "belief_propagation"
        ] = "variable_elimination",
        estimator_kwargs: Mapping[str, Any] | None = None,
        acyclification_kwargs: Mapping[str, Any] | None = None,
        latent_cardinality: int = 2,
    ):
        """Configure graph fitting and inference methods."""
        if method not in {"MLE", "Bayesian", "EM"}:
            raise ValueError("method must be 'MLE', 'Bayesian', or 'EM'")
        if inference_backend not in _PGMPY_INFERENCE_BACKENDS:
            choices = ", ".join(repr(name) for name in _PGMPY_INFERENCE_BACKENDS)
            raise ValueError(f"inference_backend must be one of {choices}")
        if isinstance(latent_cardinality, bool) or not isinstance(latent_cardinality, int) or latent_cardinality < 1:
            raise ValueError("latent_cardinality must be a positive integer")
        self.graph = graph
        self.acyclification_type = acyclification_type
        self.method = method
        self.inference_backend = inference_backend
        self.estimator_kwargs = dict(estimator_kwargs or {})
        self.acyclification_kwargs = dict(acyclification_kwargs or {})
        self.latent_cardinality = latent_cardinality
        self.model: Any = None
        self._inference: Any = None

    def fit_network(self, data: pd.DataFrame) -> None:
        """Acyclify the configured graph and fit its full pgmpy network once."""
        if "prob" in data.columns:
            raise ValueError("pgmpy backend does not accept weighted 'prob' input")
        if data.empty:
            raise ValueError("observed data must not be empty")

        nx, pgmpy_inference, network_class, estimators = self._load_pgmpy()
        name_values, observed_data, state_names = self._validate_observed_data(data)

        acyclic = self._acyclify(nx, name_values)
        model, latent_cards = self._build_model(acyclic, name_values, network_class)
        estimator = self._get_estimator(estimators)
        self._fit_model(model, observed_data, state_names, latent_cards, estimator)

        self.model = model
        inference_class = getattr(
            pgmpy_inference, _PGMPY_INFERENCE_BACKENDS[self.inference_backend]
        )
        self._inference = inference_class(model)

    @staticmethod
    def _load_acyclification():
        """Load the optional acyclification dependency on demand."""
        try:
            acyclification = import_module("nocap.acyclification").acyclification
        except ImportError as error:
            raise ImportError(
                "PgmpyDiscreteBackend requires nocap.acyclification.acyclification"
            ) from error
        return acyclification

    @staticmethod
    def _load_pgmpy():
        """Load pgmpy components used for network fitting and inference."""
        try:
            import networkx as nx
            from pgmpy import inference as pgmpy_inference
            from pgmpy.models import DiscreteBayesianNetwork
            from pgmpy.parameter_estimator import DiscreteBayesianEstimator, DiscreteEM, DiscreteMLE
        except ImportError as error:
            raise ImportError("pgmpy is required for PgmpyDiscreteBackend") from error
        estimators = {
            "MLE": DiscreteMLE,
            "Bayesian": DiscreteBayesianEstimator,
            "EM": DiscreteEM,
        }
        return nx, pgmpy_inference, DiscreteBayesianNetwork, estimators

    def _validate_observed_data(self, data: pd.DataFrame) -> tuple[list[str], pd.DataFrame, dict]:
        """Validate observed columns and derive their names and state domains."""
        observed_nodes = tuple(self.graph.nodes())
        names = {node: getattr(node, "name", str(node)) for node in observed_nodes}
        name_values = list(names.values())
        if len(set(name_values)) != len(name_values):
            raise ValueError("graph node names must be unique")
        missing = set(name_values) - set(data.columns)
        if missing:
            raise ValueError(f"missing observed graph columns: {sorted(missing)}")
        for name in name_values:
            column = cast(pd.Series, data[name])
            if bool(column.isna().any()):
                raise ValueError(f"observed graph column {name!r} contains missing values")
            states = pd.unique(column)
            if any(not isinstance(state, (str, int, float, bool, np.bool_, np.number)) for state in states):
                raise ValueError(f"observed graph column {name!r} must contain scalar state names")
        observed_data = data.loc[:, name_values].copy()
        if self.method == "EM":
            observed_data.attrs = {}
        state_names = {
            name: list(pd.unique(cast(pd.Series, observed_data[name]))) for name in name_values
        }
        return name_values, observed_data, state_names

    def _acyclify(self, nx, observed_names: list[str]):
        """Acyclify the configured graph and verify its structure."""
        try:
            acyclic = acyclification(self.graph, acyclification_type=self.acyclification_type, **self.acyclification_kwargs)
        except Exception as error:
            raise ValueError("failed to acyclify the configured mixed graph") from error
        directed = acyclic.directed
        if not nx.is_directed_acyclic_graph(directed):
            raise ValueError("acyclification output must be a directed acyclic graph")
        acyclic_nodes = {getattr(node, "name", str(node)) for node in acyclic.nodes()}
        if not set(observed_names).issubset(acyclic_nodes):
            raise ValueError("acyclification output must retain all configured observed nodes")
        return acyclic

    def _build_model(self, acyclic, observed_names: list[str], model_class):
        """Convert the acyclified mixed graph into a pgmpy Bayesian network."""
        directed = acyclic.directed
        edges = [
            (getattr(source, "name", str(source)), getattr(target, "name", str(target)))
            for source, target in directed.edges()
        ]
        model = model_class(edges)
        model.add_nodes_from(observed_names)
        undirected_edges = list(acyclic.undirected.edges())
        if undirected_edges and self.method != "EM":
            raise ValueError("bidirected edges introduce latent parents and require method='EM'")
        latent_cards = {}
        for index, (left, right) in enumerate(undirected_edges):
            latent = f"__nocap_latent_{index}"
            while latent in model:
                latent += "_"
            model.add_edge(latent, getattr(left, "name", str(left)))
            model.add_edge(latent, getattr(right, "name", str(right)))
            latent_cards[latent] = self.latent_cardinality
        if latent_cards:
            model.latents = set(latent_cards)
        return model, latent_cards

    def _get_estimator(self, estimators: Mapping[str, Any]) -> Any:
        """Select the estimator and reject unsupported method-specific options."""
        allowed_kwargs = {
            "MLE": {"n_jobs"},
            "Bayesian": {"prior_type", "equivalent_sample_size", "pseudo_counts", "n_jobs"},
            "EM": {
                "max_iter",
                "atol",
                "n_jobs",
                "batch_size",
                "seed",
                "m_step_estimator",
                "init_cpds",
                "show_progress",
            },
        }
        allowed = allowed_kwargs[self.method]
        unsupported = set(self.estimator_kwargs) - allowed
        if unsupported:
            raise ValueError(f"unsupported {self.method} estimator kwargs: {sorted(unsupported)}")
        return estimators[self.method]

    def _fit_model(
        self,
        model,
        observed_data: pd.DataFrame,
        state_names: dict,
        latent_cards: dict[str, int],
        estimator: Any,
    ) -> None:
        """Fit and validate the pgmpy model with the selected estimator."""
        try:
            if self.method == "EM":
                estimator_instance = estimator(
                    state_names=state_names,
                    latent_card=latent_cards or None,
                    **self.estimator_kwargs,
                )
            else:
                estimator_instance = estimator(state_names=state_names, **self.estimator_kwargs)
            model.fit(observed_data, estimator=estimator_instance)
            if not model.check_model():
                raise ValueError("pgmpy model validation failed")
        except Exception as error:
            raise ValueError("failed to fit or validate pgmpy discrete Bayesian network") from error

    def fit(self, data: pd.DataFrame, variables: Sequence[str]) -> DiscreteFactor:
        """Query a joint factor from the fitted full Bayesian network."""
        if self._inference is None:
            self.fit_network(data)
        names = tuple(variables)
        if not names:
            return _factor_type()([], [], np.asarray(1.0))
        missing = set(names) - set(self.model.nodes())
        if missing:
            raise ValueError(f"requested variables are missing from pgmpy model: {sorted(missing)}")
        try:
            inference = self._inference
            if inference is None:
                raise DistributionEstimationError("pgmpy inference model was not fitted")
            result = inference.query(variables=list(names), show_progress=False, joint=True)
            return cast(Any, result)
        except Exception as error:
            raise DistributionEstimationError(f"failed pgmpy query for variables {names!r}") from error
