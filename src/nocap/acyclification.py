"""Acyclification helpers for directed mixed graphs."""

from collections.abc import Collection, Sequence
from typing import Literal

import networkx as nx
from y0.algorithm.ioscm.utils import get_apt_order
from y0.dsl import Variable
from y0.graph import NxMixedGraph


def acyclification(
    graph: NxMixedGraph,
    acyclification_type: Literal["bidirected", "dag"] = "bidirected",
    *,
    node_order: Sequence[Variable] | None = None,
    bidirected_edges: Collection[tuple[Variable, Variable]] = (),
) -> NxMixedGraph:
    """Acyclify a mixed graph using one of the constructions in Definition 3.

    ``bidirected`` uses the standard ``G^acy`` construction:
    each non-trivial strongly connected component (SCC) becomes a complete
    bidirected component with no directed edges inside it.

    ``dag`` replaces each non-trivial SCC by a complete DAG
    ordered by ``node_order``. If ``bidirected_edges`` are provided, those
    additional bidirected edges are added. Nodes omitted from
    ``node_order`` are ordered after listed nodes, with ties resolved by their
    order in the input graph.

    Edges between SCCs are retained. Edges entering an SCC are expanded to
    connect to every node in that SCC, as required by the acyclification
    definition. The input graph is not modified.

    Parameters
    ----------
    graph:
        A directed mixed graph represented as :class:`y0.graph.NxMixedGraph`.
    acyclification_type:
        ``"bidirected"`` or ``"dag"``.
    node_order:
        Optional complete ordering used to orient complete DAGs inside SCCs.
        Defaults to the assembling pseudo-topological order from
        :func:`y0.algorithm.ioscm.utils.get_apt_order`.
    bidirected_edges:
        Optional additional bidirected edges to add in DAG mode. Each edge
        must connect distinct :class:`y0.dsl.Variable` nodes in the graph.

    Returns
    -------
    NxMixedGraph
        An acyclic directed mixed graph that is sigma-separation-equivalent
        to the input graph.
    """
    match acyclification_type:

        case "bidirected":
            return _bidirected(graph)

        case "dag":
            return _dag(graph, node_order=node_order, bidirected_edges=bidirected_edges)

        case _:
            raise ValueError(f"Unsupported acyclification_type: {acyclification_type!r}")



def _bidirected(graph: NxMixedGraph) -> NxMixedGraph:
    """Replace directed SCCs with complete bidirected components.

    Also ensures every edge pointing into the SCC is fully connected with
    all the nodes of the SCC.

    Adapted from pywhy-graphs' acyclification implementation:
    https://github.com/py-why/pywhy-graphs
    """
    directed = graph.directed.copy()
    undirected = graph.undirected.copy()
    nodes = list(graph.nodes())
    directed.add_nodes_from(nodes)
    undirected.add_nodes_from(nodes)

    for component in nx.strongly_connected_components(graph.directed):
        if len(component) < 2:
            continue

        # Collect directed parents, bidirected neighbors, and outgoing edges
        # before removing the SCC nodes from either edge-type graph.
        parents = {
            parent
            for node in component
            for parent in graph.directed.predecessors(node)
            if parent not in component
        }
        bidirected_neighbors = {
            neighbor
            for node in component
            for neighbor in graph.undirected.neighbors(node)
            if neighbor not in component
        }
        children = [
            (node, child)
            for node in component
            for child in graph.directed.successors(node)
            if child not in component
        ]

        # Replace the directed cycle with a complete bidirected component.
        directed.remove_nodes_from(component)
        undirected.remove_nodes_from(component)
        directed.add_nodes_from(component)
        undirected.add_nodes_from(component)

        undirected.add_edges_from(nx.complete_graph(component).edges)

        # Restore edges leaving the SCC and expand incoming edges to each node.
        directed.add_edges_from(children)
        directed.add_edges_from((parent, node) for parent in parents for node in component)
        undirected.add_edges_from(
            (neighbor, node) for neighbor in bidirected_neighbors for node in component
        )

    return NxMixedGraph(
        directed=directed,
        undirected=undirected,
    )


def _dag(
    graph: NxMixedGraph,
    node_order: Sequence[Variable] | None,
    bidirected_edges: Collection[tuple[Variable, Variable]],
) -> NxMixedGraph:
    """Build a fully connected DAG within each SCC, plus optional bidirected edges.

    Per Mooji & Claassen (2020), a given DMG can have multiple acyclifications. While
    most implementations will add a bidirected edges between all nodes in an SCC to make
    them fully-connected, producing Gacy, they show that this isn't necessarily required,
    and any fully-connected SCC structure is sigma-separation-equivalent.

    The following structures must hold in the acyclified DAG:
        1. If a node is not an ancestor of another in the G, then it must not be an ancestor in the acyclified graph G'.
        2. There is an inducing path between two nodes in G if and only if there is an inducing path between the two nodes for any G'.

    We induce a fully-connected DAG in an SCC by either taking in an existing node order and building a
    fully-connected DAG from that, or via apt-order (Forré & Mooji, 2019), a pseudo-ordering of nodes in
    a directed mixed graph.
    """
    nodes = list(graph.nodes())
    node_set = set(nodes)

    if node_order is None:
        order = get_apt_order(graph)
    else:
        order = list(node_order)
        if len(order) != len(set(order)) or any(node not in node_set for node in order):
            raise ValueError("node_order must contain distinct graph variables")
    ordered_nodes = set(order)
    order.extend(node for node in nodes if node not in ordered_nodes)
    rank = {node: index for index, node in enumerate(order)}

    for left, right in bidirected_edges:
        if left not in node_set or right not in node_set or left == right:
            raise ValueError("bidirected_edges must connect distinct variables in the graph")

    components = list(nx.strongly_connected_components(graph.directed))
    component_of = {node: component for component in components for node in component}
    cyclic_components = [
        component
        for component in components
        if len(component) > 1 or any(graph.directed.has_edge(node, node) for node in component)
    ]
    cyclic_nodes = {node for component in cyclic_components for node in component}

    directed = nx.DiGraph()
    undirected = nx.Graph()
    directed.add_nodes_from(nodes)
    undirected.add_nodes_from(nodes)

    # Preserve edges between SCCs, expanding each edge entering an SCC to all
    # nodes in its target component.
    for source, target in graph.directed.edges():
        source_component = component_of[source]
        target_component = component_of[target]
        if source_component == target_component:
            continue
        targets = target_component if target in cyclic_nodes else (target,)
        directed.add_edges_from((source, expanded_target) for expanded_target in targets)

    # Replace each cyclic SCC with a complete DAG in the selected node order.
    for component in cyclic_components:
        ordered_component = sorted(component, key=rank.__getitem__)
        directed.add_edges_from(
            (source, target)
            for index, source in enumerate(ordered_component)
            for target in ordered_component[index + 1 :]
        )

    # Preserve bidirected adjacencies between SCCs and expand those touching a
    # cyclic SCC to every member of that component. Internal bidirected edges
    # are omitted unless explicitly requested below.
    for left, right in graph.undirected.edges():
        left_component = component_of[left]
        right_component = component_of[right]
        if left_component == right_component:
            continue
        left_nodes = left_component if left in cyclic_nodes else (left,)
        right_nodes = right_component if right in cyclic_nodes else (right,)
        undirected.add_edges_from((a, b) for a in left_nodes for b in right_nodes)

    undirected.add_edges_from(bidirected_edges)
    return NxMixedGraph(directed=directed, undirected=undirected)
