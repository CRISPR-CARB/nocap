"""Neutral numeric directed SCM representation and synthetic model builder."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .scc_perturb import build_intervened_graph

COND_NUMBER_THRESHOLD = 1000  # https://en.wikipedia.org/wiki/Condition_number


@dataclass(frozen=True)
class DirectedScm:
    """A directed linear SCM independent of noise and observation models."""

    nodes: tuple[str, ...]
    graph: nx.DiGraph
    betas: dict[tuple[str, str], float]

    def __post_init__(self) -> None:
        nodes = tuple(str(n) for n in self.nodes)
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from((str(u), str(v)) for u, v in self.graph.edges())
        if set(self.betas) != set(graph.edges()):
            raise ValueError("SCM coefficients must correspond exactly to graph edges.")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "graph", nx.freeze(graph))
        object.__setattr__(
            self,
            "betas",
            {edge: float(value) for edge, value in self.betas.items()},
        )

    @property
    def beta_matrix(self) -> np.ndarray:
        """Return edge-oriented ``B`` with ``B[source, target]`` for ``source -> target``.

        The mathematical target-parent matrix is ``B.T``. The numerical solver
        therefore uses ``(I - beta_matrix.T)`` to implement ``X = B X + eps``.
        """
        index = {node: i for i, node in enumerate(self.nodes)}
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for (u, v), value in self.betas.items():
            matrix[index[u], index[v]] = value
        return matrix


@dataclass(frozen=True)
class SyntheticScmBuild:
    """Result of constructing a synthetic SCM."""

    scm: DirectedScm
    added_edges: tuple[tuple[str, str], ...]
    condition_number: float


def build_intervened_scm(scm: DirectedScm, perturb_set: list[str]) -> DirectedScm:
    """Return the hard-``do(perturb_set)`` SCM without changing ``scm``."""
    nodes = list(scm.nodes)
    unknown = set(perturb_set) - set(nodes)
    if unknown:
        raise ValueError(f"Intervention contains unknown SCM nodes: {sorted(unknown)}")
    graph = build_intervened_graph(scm.graph, sorted(set(perturb_set)))
    betas = {edge: value for edge, value in scm.betas.items() if graph.has_edge(*edge)}
    return DirectedScm(scm.nodes, graph, betas)


def _select_true_edges(
    estimation_graph: nx.DiGraph,
    nodes: tuple[str, ...],
    *,
    missing_edge_rate: float,
    rng: np.random.Generator,
    forbidden_edges: Iterable[tuple[str, str]] = (),
) -> tuple[list[tuple[str, str]], tuple[tuple[str, str], ...]]:
    """Select the true SCM edges and the edges hidden from estimation.

    ``missing_edge_rate`` means how many edges are present in the true SCM,
    but missing from the provided or constructed estimation graph. Existing
    estimation edges are never removed.
    """
    original_edges = list(estimation_graph.edges())
    original_edge_set = set(original_edges)
    forbidden = {(str(u), str(v)) for u, v in forbidden_edges}
    true_edges = list(original_edges)

    # Choose the number of extra edges to add (expected:
    # missing_edge_rate * |E_orig|).
    n_to_add = int(
        rng.binomial(
            len(original_edges),
            np.clip(float(missing_edge_rate), 0.0, 1.0),
        )
    )

    candidates = [
        (u, v)
        for u in nodes
        for v in nodes
        if u != v and (u, v) not in original_edge_set and (u, v) not in forbidden
    ]

    if candidates and n_to_add > 0:
        n_to_add = min(n_to_add, len(candidates))
        selected = rng.choice(
            len(candidates),
            size=n_to_add,
            replace=False,
        )
        true_edges.extend(candidates[int(i)] for i in selected)

    added_edges = tuple(edge for edge in true_edges if edge not in original_edge_set)
    return true_edges, added_edges


def _beta_matrix_from_edges(
    nodes: tuple[str, ...],
    betas: dict[tuple[str, str], float],
) -> np.ndarray:
    """Build edge-oriented ``B`` where ``B[source, target]`` is beta."""
    node_index = {node: i for i, node in enumerate(nodes)}
    beta_matrix = np.zeros((len(nodes), len(nodes)), dtype=float)

    for (u, v), beta in betas.items():
        beta_matrix[node_index[u], node_index[v]] = beta

    return beta_matrix


def stabilize_cyclic_beta(
    beta_matrix: np.ndarray,
    target_rho: float = 0.8,
) -> np.ndarray:
    """Rescale ``B`` so its spectral radius is at most ``target_rho``.

    This guarantees that ``(I - B.T)`` is stable for the fixed-point solution
    used by the numerical SCM solver.
    """
    eigs = np.linalg.eigvals(beta_matrix)
    rho = np.max(np.abs(eigs), initial=0.0)

    if rho >= target_rho and rho > 0:
        beta_matrix = beta_matrix * (target_rho / rho)

    return beta_matrix


def _sync_betas_from_matrix(
    beta_matrix: np.ndarray,
    betas: dict[tuple[str, str], float],
    nodes: tuple[str, ...],
) -> dict[tuple[str, str], float]:
    """Synchronize edge coefficients after matrix stabilization."""
    node_index = {node: i for i, node in enumerate(nodes)}
    return {(u, v): float(beta_matrix[node_index[u], node_index[v]]) for u, v in betas}


def _matrix_condition_number(beta_matrix: np.ndarray) -> float:
    """Return the condition number of the linear SCM coefficient matrix."""
    coefficient_matrix = np.eye(beta_matrix.shape[0]) - beta_matrix.T
    return float(np.linalg.cond(coefficient_matrix))


def _sample_betas(
    edges: Iterable[tuple[str, str]],
    rng: np.random.Generator,
    beta_med: float,
    beta_log_sd: float,
    beta_p: float,
    beta_abs_max: float,
    signs: dict[tuple[str, str], float] | None = None,
) -> dict[tuple[str, str], float]:
    """Sample nonzero coefficients for regulatory edges.

    Parameters
    ----------
    edges:
        Iterable of (source_gene, target_gene) edges.
    beta_med:
        Median absolute change to the log2fc of a gene
        for a one-unit increase in the log2fc of a regulator.
    beta_log_sd:
        Standard deviation of log(abs(beta)).
    beta_p:
        Probability that an edge is activating.
    beta_abs_max:
        Maximum allowed absolute value of beta, on the log scale.
    rng:
        NumPy random number generator.
    signs:
        Optional dict mapping a pair of edges to an edge polarity (-1 or 1).

    Returns
    -------
    dict[tuple[str, str], float]
        Mapping from edge to a nonzero signed log fold change.

    Notes
    -----
    The sampled magnitude follows approximately
    ``log(abs(beta)) ~ Normal(log(beta_med), beta_log_sd**2)``.
    """
    if beta_med <= 0:
        raise ValueError("beta_med must be greater than 0.")
    if beta_log_sd <= 0:
        raise ValueError("beta_log_sd must be positive.")
    if beta_abs_max <= beta_med:
        raise ValueError("beta_abs_max must be greater than beta_med.")
    if not 0 <= beta_p <= 1:
        raise ValueError("beta_p must be between 0 and 1.")
    edges = list(edges)
    magnitudes = np.empty(len(edges))
    remaining = np.arange(len(edges))

    while remaining.size:
        proposed = rng.lognormal(np.log(beta_med), beta_log_sd, remaining.size)
        accepted = proposed < beta_abs_max
        magnitudes[remaining[accepted]] = proposed[accepted]
        remaining = remaining[~accepted]
    sampled_signs = np.where(rng.random(len(edges)) < beta_p, 1.0, -1.0)
    signs = {} if signs is None else signs
    return {
        edge: float(signs.get(edge, sampled_signs[i]) * magnitude)
        for i, (edge, magnitude) in enumerate(zip(edges, magnitudes, strict=True))
    }


def build_synthetic_scm(
    estimation_graph: nx.DiGraph,
    nodes: Iterable[str],
    *,
    missing_edge_rate: float,
    beta_med: float,
    beta_log_sd: float,
    beta_p: float,
    beta_abs_max: float,
    rng: np.random.Generator,
    forbidden_edges: Iterable[tuple[str, str]] = (),
    signs: dict[tuple[str, str], float] | None = None,
) -> SyntheticScmBuild:
    """Build a stable SCM, adding true edges according to ``missing_edge_rate``.

    ``B[u, v]`` represents the coefficient for ``u -> v``. The construction
    is independent of observations, missingness, noise realizations, and the
    solver. A fresh coefficient matrix is built on each retry.
    """
    nodes = tuple(str(node) for node in nodes)
    graph_signs = {
        (str(u), str(v)): data.get(
            "polarity", data.get("d0", data.get("sign"))
        )
        for u, v, data in estimation_graph.edges(data=True)
    }
    normalized_signs = {
        (str(u), str(v)): float(np.sign(sign))
        for (u, v), sign in (graph_signs if signs is None else signs).items()
        if sign in ("+", "-", "1", "-1", 1, -1, 1.0, -1.0)
    }

    edges, added_edges = _select_true_edges(
        estimation_graph,
        nodes,
        missing_edge_rate=missing_edge_rate,
        rng=rng,
        forbidden_edges=forbidden_edges,
    )

    for _ in range(100):
        betas = _sample_betas(
            edges,
            rng,
            beta_med,
            beta_log_sd,
            beta_p,
            beta_abs_max,
            normalized_signs
        )

        matrix = _beta_matrix_from_edges(nodes, betas)
        matrix = stabilize_cyclic_beta(matrix)
        betas = _sync_betas_from_matrix(matrix, betas, nodes)
        condition = _matrix_condition_number(matrix)

        if condition < COND_NUMBER_THRESHOLD:
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(edges)
            return SyntheticScmBuild(
                DirectedScm(nodes, graph, betas),
                added_edges,
                condition,
            )
    raise RuntimeError("Could not construct a well-conditioned synthetic SCM.")
