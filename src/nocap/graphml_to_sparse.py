"""Project GraphML regulatory networks into aligned sparse tensors."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch
from torch import Tensor

DEFAULT_POSITIVE_POLARITIES = frozenset({"+", "activation", "positive", "1"})
DEFAULT_NEGATIVE_POLARITIES = frozenset({"-", "repression", "negative", "-1"})


@dataclass(frozen=True)
class GraphAlignmentReport:
    """Describe information retained and dropped during graph alignment."""

    measured_genes: int
    graph_genes: int
    retained_edges: int
    dropped_edges: int
    dropped_self_loops: tuple[tuple[str, str], ...]
    graph_only_genes: tuple[str, ...]
    isolated_measured_genes: tuple[str, ...]


@dataclass(frozen=True)
class SparseRegulatoryGraph:
    """Sparse graph tensors aligned to an externally supplied gene order."""

    gene_order: tuple[str, ...]
    edge_index: Tensor
    edge_signs: Tensor
    report: GraphAlignmentReport


def graphml_to_sparse(
    path_or_graph: str | Path | nx.DiGraph,
    gene_order: Sequence[str],
    *,
    polarity_attr: str = "polarity",
    positive_polarities: Iterable[str] = DEFAULT_POSITIVE_POLARITIES,
    negative_polarities: Iterable[str] = DEFAULT_NEGATIVE_POLARITIES,
) -> SparseRegulatoryGraph:
    """Align a directed GraphML network to authoritative expression columns."""
    genes = _validate_gene_order(gene_order)
    graph = _load_directed_graph(path_or_graph)
    _validate_graph_nodes(graph)
    positive = frozenset(str(value).strip().lower() for value in positive_polarities)
    negative = frozenset(str(value).strip().lower() for value in negative_polarities)
    if positive & negative:
        raise ValueError("positive and negative polarity vocabularies must be disjoint")

    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    graph_only_genes = tuple(sorted(set(graph.nodes) - set(genes)))
    retained: list[tuple[int, int, int]] = []
    dropped_edges = 0
    dropped_self_loops: list[tuple[str, str]] = []

    for regulator, target, data in graph.edges(data=True):
        if regulator == target:
            dropped_self_loops.append((regulator, target))
            dropped_edges += 1
            continue
        if regulator not in gene_to_index or target not in gene_to_index:
            dropped_edges += 1
            continue
        sign = _polarity_sign(data.get(polarity_attr), positive, negative)
        retained.append((gene_to_index[target], gene_to_index[regulator], sign))

    retained.sort()
    if retained:
        targets, regulators, signs = zip(*retained, strict=True)
        edge_index = torch.tensor([regulators, targets], dtype=torch.long)
        edge_signs = torch.tensor(signs, dtype=torch.int8)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_signs = torch.empty(0, dtype=torch.int8)

    connected_indices = set(edge_index.flatten().tolist())
    isolated_genes = tuple(
        gene for index, gene in enumerate(genes) if index not in connected_indices
    )
    report = GraphAlignmentReport(
        measured_genes=len(genes),
        graph_genes=graph.number_of_nodes(),
        retained_edges=len(retained),
        dropped_edges=dropped_edges,
        dropped_self_loops=tuple(sorted(dropped_self_loops)),
        graph_only_genes=graph_only_genes,
        isolated_measured_genes=isolated_genes,
    )
    return SparseRegulatoryGraph(genes, edge_index, edge_signs, report)


def _validate_gene_order(gene_order: Sequence[str]) -> tuple[str, ...]:
    if isinstance(gene_order, str):
        raise TypeError("gene_order must be a sequence of gene names, not a string")
    genes = tuple(gene_order)
    if not genes:
        raise ValueError("gene_order must contain at least one gene")
    if any(not isinstance(gene, str) or not gene for gene in genes):
        raise ValueError("gene_order entries must be nonempty strings")
    if len(set(genes)) != len(genes):
        raise ValueError("gene_order must not contain duplicate genes")
    return genes


def _load_directed_graph(path_or_graph: str | Path | nx.DiGraph) -> nx.DiGraph:
    if isinstance(path_or_graph, str | Path):
        graph = nx.read_graphml(path_or_graph)
    elif isinstance(path_or_graph, nx.Graph):
        graph = path_or_graph
    else:
        raise TypeError("path_or_graph must be a path or NetworkX graph")
    if not graph.is_directed():
        raise ValueError("regulatory graph must be directed")
    if graph.is_multigraph():
        raise ValueError("parallel regulatory edges are not supported")
    return graph


def _validate_graph_nodes(graph: nx.DiGraph) -> None:
    if any(not isinstance(node, str) for node in graph.nodes):
        raise ValueError("regulatory graph nodes must be strings")


def _polarity_sign(
    raw_polarity: object,
    positive_polarities: frozenset[str],
    negative_polarities: frozenset[str],
) -> int:
    if raw_polarity is None:
        return 0
    polarity = str(raw_polarity).strip().lower()
    if polarity in positive_polarities:
        return 1
    if polarity in negative_polarities:
        return -1
    return 0
