"""Reusable helpers for cyclic single-door estimation and query validation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from y0.algorithm.identify.utils import Unidentifiable
from y0.dsl import Expression

from ..cyclic_id import identify_causal_query
from ..cyclic_single_door import classify_edge, estimate_path_coefficient_for_edge
from ..scm_model import DirectedScm
from ..simulation import true_scm_ate

__all__ = [
    "EdgeEstimate",
    "QueryResult",
    "build_estimated_scm",
    "estimate_scm_edges",
    "estimated_scm_ate",
    "identify_query_status",
]


@dataclass(frozen=True)
class QueryResult:
    """Structured result for an identified or unidentifiable query."""

    identifiable: bool
    expression: Expression | None
    error: str | None = None


@dataclass(frozen=True)
class EdgeEstimate:
    """CSD estimate and metadata for one graph edge."""

    cause: str
    effect: str
    status: str
    adjustment_set: frozenset[str] | None
    coefficient: float | None
    stderr: float | None
    residual_variance: float | None
    t_value: float | None
    n_rows: int
    error: str | None = None


def identify_query_status(
    graph: nx.DiGraph,
    treatment: str,
    outcome: str,
    unobserved: set[str] | frozenset[str] | None = None,
) -> QueryResult:
    """Identify one ordered treatment/outcome query without raising y0 errors."""
    if treatment not in graph or outcome not in graph:
        raise ValueError("treatment and outcome must be graph nodes")
    try:
        expression = identify_causal_query(graph, {treatment}, {outcome}, unobserved)
    except Unidentifiable as error:
        return QueryResult(False, None, str(error) or "unidentifiable")
    return QueryResult(True, expression)


def estimated_scm_ate(
    graph: nx.DiGraph,
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    treatment_levels: tuple[float, float],
    *,
    min_rows: int = 30,
    n_samples: int = 100_000,
    seed: int = 0,
    allow_partial: bool = False,
) -> tuple[float, list[EdgeEstimate], DirectedScm]:
    """Estimate an SCM with cyclic single-door coefficients and compute its ATE.

    The returned coefficient table preserves CSD status, adjustment sets, and
    row counts. Failed or unidentifiable edges raise through
    :func:`build_estimated_scm` unless ``allow_partial`` is explicitly enabled.
    """
    estimates = estimate_scm_edges(graph, data, min_rows=min_rows)
    estimated = build_estimated_scm(graph, estimates, allow_partial=allow_partial)
    ate = true_scm_ate(
        estimated,
        treatment,
        outcome,
        treatment_levels,
        n_samples=n_samples,
        seed=seed,
    )
    return ate, estimates, estimated


def estimate_scm_edges(
    graph: nx.DiGraph,
    data: pd.DataFrame,
    *,
    min_rows: int = 30,
) -> list[EdgeEstimate]:
    """Estimate every graph edge using CSD and complete-case rows."""
    estimates = []
    for cause, effect in sorted(graph.edges()):
        classification = classify_edge(graph, cause, effect)
        adjustment = classification["adjustment_set"]
        columns = [cause, effect, *(sorted(adjustment or ()))]
        complete = data.loc[:, columns].dropna()
        if adjustment is None:
            estimates.append(
                EdgeEstimate(
                    cause, effect, "unidentifiable", None, None, None, None, None, len(complete)
                )
            )
            continue
        if len(complete) < min_rows:
            estimates.append(
                EdgeEstimate(
                    cause,
                    effect,
                    "insufficient_rows",
                    adjustment,
                    None,
                    None,
                    None,
                    None,
                    len(complete),
                )
            )
            continue
        try:
            result = estimate_path_coefficient_for_edge(
                graph, cause, effect, complete, adj_set=adjustment
            )
        except (KeyError, ValueError, np.linalg.LinAlgError) as error:
            estimates.append(
                EdgeEstimate(
                    cause,
                    effect,
                    "failed",
                    adjustment,
                    None,
                    None,
                    None,
                    None,
                    len(complete),
                    str(error),
                )
            )
            continue
        assert result is not None
        coefficient, stderr, residual, t_value = result
        estimates.append(
            EdgeEstimate(
                cause,
                effect,
                "estimated",
                adjustment,
                coefficient,
                stderr,
                residual,
                t_value,
                len(complete),
            )
        )
    return estimates


def build_estimated_scm(
    graph: nx.DiGraph, estimates: Iterable[EdgeEstimate], *, allow_partial: bool = False
) -> DirectedScm:
    """Construct an SCM from CSD coefficients without hiding failed edges."""
    by_edge = {(item.cause, item.effect): item for item in estimates}
    missing = [
        edge
        for edge in graph.edges()
        if by_edge.get(edge) is None or by_edge[edge].coefficient is None
    ]
    if missing and not allow_partial:
        raise ValueError(f"estimated SCM has incomplete coefficients: {sorted(missing)}")
    betas = {
        edge: float(item.coefficient)
        for edge, item in by_edge.items()
        if graph.has_edge(*edge) and item.coefficient is not None
    }
    estimated_graph = nx.DiGraph()
    estimated_graph.add_nodes_from(str(node) for node in graph.nodes())
    estimated_graph.add_edges_from(betas)
    return DirectedScm(tuple(sorted(estimated_graph.nodes())), estimated_graph, betas)
