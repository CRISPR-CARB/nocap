r"""cyclic_single_door.py — σ-separation single-door criterion for cyclic causal graphs.

Classifies every directed edge in a cyclic causal graph as **identifiable** or
**unidentifiable** under the σ-separation single-door criterion (Forré & Mooij 2018;
Rantanen et al. 2020), and provides a greedy intervention-rescue optimizer that
maximises the number of identifiable edges subject to a budget of *k* hard
interventions.

Algorithm overview
------------------
**Phase 1 — σ-extension (once per graph).**
Build the σ-extension G^σ by adding a bidirected edge X↔Y for every directed
edge X→Y where X and Y share a strongly connected component (SCC).  By
Rantanen et al. (2020), σ-separation in G is *equivalent* to d-separation in
G^σ, so all downstream work uses the fast d-separation oracle.

**Phase 2 — per-edge classification.**
For each directed edge X→Y call ``find_sigma_single_door_set`` (y0), which:
  1. Removes X→Y from G^σ.
  2. Builds the proper back-door graph G^pbd.
  3. Computes the O-set ``pa(cn(X→Y)) \ forb(X→Y)`` (Henckel et al. 2022).
  4. Verifies via d-separation in G^pbd.
Returns a frozenset (identifiable, with that adjustment set) or None
(unidentifiable).

**Completeness of the O-set test.**
The O-set test is *complete* for the existence of any valid adjustment set:
Perković et al. (2018) prove that a valid adjustment set exists iff the
canonical set ``pa(cn) \ forb`` is itself valid.  Enumerating subsets cannot
find a set the O-set test misses; it would only run through a weaker oracle.
Therefore ``classify_edge`` returns a definitive verdict without enumeration.

**Phase 3 — greedy rescue (optional).**
``maximize_identifiable_edges`` greedily picks up to *k* intervention nodes
(candidates drawn from ``compute_min_cut_b``'s cut pool) to maximise the count
of edges flipping unidentifiable→identifiable.  Intervention semantics are
``do(S)`` = remove in-edges to each node in S (via ``build_intervened_graph``);
nodes and their out-edges are preserved.

Completeness caveat
-------------------
Completeness holds relative to (a) the adjustment-criterion theory of Perković
et al. (2018), which assumes the σ-extension correctly captures σ-separation
(Rantanen's equivalence holds for the simple/acyclification-compatible SCM
class), and (b) the standard O-set non-existence condition (O-set contains a
descendant of the effect ⇒ no valid set exists).

References
----------
P. Forré, J. M. Mooij (2018). Constraint-based Causal Discovery for
Non-linear Structural Causal Models with Cycles and Latent Confounders.

K. Rantanen, A. Hyttinen, M. Järvisalo (2020). Learning Optimal Cyclic Causal
Graphs from Interventional Data. *Probabilistic Graphical Models*.

E. Perković, J. Textor, M. Kalisch, M. H. Maathuis (2018). Complete Graphical
Characterization and Construction of Adjustment Sets in Markov Equivalence
Classes of Ancestral Graphs. *JMLR*, 19(1).

L. Henckel, E. Perković, M. H. Maathuis (2022). Graphical criteria for
efficient total effect estimation via adjustment in causal linear models.
*JRSS-B*, 84(2).
"""

from __future__ import annotations

import signal
from contextlib import contextmanager

import networkx as nx
from y0.algorithm.separation.sigma_extension import sigma_extension
from y0.algorithm.separation.sigma_single_door import find_sigma_single_door_set
from y0.dsl import Variable
from y0.graph import NxMixedGraph

from nocap.scc_perturb import build_intervened_graph, compute_min_cut_b, find_in_scc_children

__all__ = [
    "classify_edge",
    "evaluate_all_edges",
    "maximize_identifiable_edges",
    "nx_digraph_to_y0",
    "same_scc",
]

# ---------------------------------------------------------------------------
# Per-edge timeout helper (POSIX signal.alarm; no-op on non-POSIX)
# ---------------------------------------------------------------------------


class _EdgeTimeoutError(Exception):
    """Raised when a per-edge classification exceeds the timeout budget."""


# Backwards-compatible alias used by tests
_EdgeTimeout = _EdgeTimeoutError


@contextmanager
def _timeout_context(seconds: int):
    """Context manager that raises _EdgeTimeout after *seconds* seconds.

    Uses POSIX ``signal.SIGALRM``.  On Windows or in a non-main thread the
    context manager is a no-op (the alarm call is silently skipped).

    PRE: seconds > 0
    """
    assert seconds > 0, "PRE: timeout must be a positive number of seconds"

    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None

    if have_alarm:

        def _handler(signum, frame):
            raise _EdgeTimeoutError

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)

    try:
        yield
    finally:
        if have_alarm:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# nx_digraph_to_y0
# ---------------------------------------------------------------------------


def nx_digraph_to_y0(graph: nx.DiGraph) -> NxMixedGraph:
    """Convert a plain ``nx.DiGraph`` to a y0 ``NxMixedGraph``.

    Each node name is wrapped in a :class:`~y0.dsl.Variable`.  No bidirected
    edges are added — the σ-extension step adds those later.

    Parameters
    ----------
    graph:
        A directed graph whose node names are strings (or any hashable that
        can be coerced to a string for the Variable name).

    Returns
    -------
    NxMixedGraph
        A y0 mixed graph with the same directed edges and no bidirected edges.

    axiomander:
        ensures:
            result.directed.number_of_nodes() == graph.number_of_nodes()
            result.directed.number_of_edges() == graph.number_of_edges()
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(graph, nx.DiGraph), "PRE: graph must be an nx.DiGraph"

    node_map: dict = {n: Variable(str(n)) for n in graph.nodes()}
    directed_edges = [(node_map[u], node_map[v]) for u, v in graph.edges()]
    result = NxMixedGraph.from_edges(directed=directed_edges)

    # NxMixedGraph.from_edges only adds nodes that appear in at least one edge.
    # Isolated nodes (e.g. after do-intervention removes all in-edges) must be
    # added explicitly so the node set is preserved.
    for var in node_map.values():
        if var not in result.directed:
            result.directed.add_node(var)

    # --- POST ---
    assert result.directed.number_of_nodes() == graph.number_of_nodes(), (
        "POST: node count must be preserved"
    )
    assert result.directed.number_of_edges() == graph.number_of_edges(), (
        "POST: edge count must be preserved"
    )
    return result


# ---------------------------------------------------------------------------
# same_scc
# ---------------------------------------------------------------------------


def same_scc(graph: nx.DiGraph, u: object, v: object) -> bool:
    """Return True iff nodes *u* and *v* belong to the same strongly connected component.

    A node is always in the same SCC as itself (trivially).  Two distinct nodes
    are in the same SCC iff there is a directed path from *u* to *v* **and**
    from *v* to *u*.

    Parameters
    ----------
    graph:
        A directed graph.
    u:
        First node.
    v:
        Second node.

    Returns
    -------
    bool

    axiomander:
        ensures:
            isinstance(result, bool)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(graph, nx.DiGraph), "PRE: graph must be an nx.DiGraph"

    if u not in graph or v not in graph:
        return False

    if u == v:
        return True

    # Use condensation-based SCC lookup for O(V+E) amortised cost.
    # For a single call, nx.has_path in both directions is simpler.
    result = bool(nx.has_path(graph, u, v) and nx.has_path(graph, v, u))

    # --- POST ---
    assert isinstance(result, bool), "POST: result must be bool"
    return result


# ---------------------------------------------------------------------------
# classify_edge
# ---------------------------------------------------------------------------


def classify_edge(
    graph: nx.DiGraph,
    cause: str,
    effect: str,
    *,
    precomputed_extension: NxMixedGraph | None = None,
    precomputed_y0: NxMixedGraph | None = None,
) -> dict:
    r"""Classify a single directed edge as identifiable or unidentifiable.

    Uses the σ-separation single-door criterion (Forré & Mooij 2018) via the
    O-set construction (Henckel et al. 2022).  The O-set test is **complete**:
    if no valid adjustment set exists, the O-set is None and no enumeration
    can find one (Perković et al. 2018).

    Parameters
    ----------
    graph:
        The original directed graph (``nx.DiGraph``).
    cause:
        Source node name of the edge to classify.
    effect:
        Target node name of the edge to classify.
    precomputed_extension:
        Pre-built σ-extension of the y0 representation of *graph*.  Pass this
        when classifying many edges of the same graph to avoid recomputing.
    precomputed_y0:
        Pre-built y0 ``NxMixedGraph`` of *graph*.  Pass together with
        *precomputed_extension* for maximum reuse.

    Returns
    -------
    dict with keys:
        ``cause``           -- str: source node name
        ``effect``          -- str: target node name
        ``status``          -- ``"identifiable"`` or ``"unidentifiable"``
        ``adjustment_set``  -- frozenset[str] | None: the O-set (None if unidentifiable)
        ``same_scc``        -- bool: True iff cause and effect share an SCC

    axiomander:
        ensures:
            result["status"] in ("identifiable", "unidentifiable")
            isinstance(result["same_scc"], bool)
            implies(result["status"] == "identifiable", result["adjustment_set"] is not None)
            implies(result["status"] == "unidentifiable", result["adjustment_set"] is None)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(graph, nx.DiGraph), "PRE: graph must be an nx.DiGraph"
    assert isinstance(cause, str), "PRE: cause must be a str"
    assert isinstance(effect, str), "PRE: effect must be a str"
    assert graph.has_edge(cause, effect), f"PRE: edge {cause!r} -> {effect!r} must exist in graph"

    # Build y0 representation if not provided
    g_y0: NxMixedGraph = precomputed_y0 if precomputed_y0 is not None else nx_digraph_to_y0(graph)

    cause_var = Variable(cause)
    effect_var = Variable(effect)

    # Build sigma-extension if not provided
    g_sigma: NxMixedGraph = (
        precomputed_extension if precomputed_extension is not None else sigma_extension(g_y0)
    )

    # Classify via O-set (complete by Perkovic et al. 2018)
    adj_set_vars = find_sigma_single_door_set(
        g_y0,
        cause_var,
        effect_var,
        precomputed_extension=g_sigma,
    )

    is_same_scc = same_scc(graph, cause, effect)

    if adj_set_vars is not None:
        # Convert Variable objects back to plain strings
        adj_set_str: frozenset[str] | None = frozenset(v.name for v in adj_set_vars)
        status = "identifiable"
    else:
        adj_set_str = None
        status = "unidentifiable"

    result = {
        "cause": cause,
        "effect": effect,
        "status": status,
        "adjustment_set": adj_set_str,
        "same_scc": is_same_scc,
    }

    # --- POST ---
    assert result["status"] in ("identifiable", "unidentifiable"), (
        "POST: status must be identifiable or unidentifiable"
    )
    assert isinstance(result["same_scc"], bool), "POST: same_scc must be bool"
    assert (result["status"] == "identifiable") == (result["adjustment_set"] is not None), (
        "POST: adjustment_set must be non-None iff identifiable"
    )
    return result


# ---------------------------------------------------------------------------
# evaluate_all_edges
# ---------------------------------------------------------------------------


def evaluate_all_edges(
    graph: nx.DiGraph,
    *,
    restrict_edges: list[tuple[str, str]] | None = None,
    timeout_seconds: int | None = None,
) -> list[dict]:
    r"""Classify every directed edge in *graph* under the σ-single-door criterion.

    Builds the σ-extension **once** and reuses it for all edges, giving
    O(E × (V+E)) total complexity.

    Parameters
    ----------
    graph:
        The directed graph to evaluate.
    restrict_edges:
        If provided, only classify these (cause, effect) pairs.  Each pair
        must be a directed edge in *graph*.
    timeout_seconds:
        If provided, each individual edge classification is capped at this
        many seconds (POSIX ``signal.SIGALRM``).  Edges that exceed the
        timeout are recorded with ``status="timeout"`` and
        ``adjustment_set=None``.  Pass ``None`` (default) to disable.

    Returns
    -------
    list[dict]
        One dict per edge (see :func:`classify_edge` for the dict schema),
        in the order the edges are iterated (or in *restrict_edges* order).
        When *timeout_seconds* is set, timed-out edges additionally carry
        ``"timed_out": True``.

    axiomander:
        ensures:
            len(result) == len(restrict_edges) if restrict_edges is not None else graph.number_of_edges()
            all(r["status"] in ("identifiable", "unidentifiable", "timeout") for r in result)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(graph, nx.DiGraph), "PRE: graph must be an nx.DiGraph"
    assert timeout_seconds is None or (isinstance(timeout_seconds, int) and timeout_seconds > 0), (
        "PRE: timeout_seconds must be None or a positive int"
    )

    edges: list[tuple[str, str]] = (
        list(restrict_edges) if restrict_edges is not None else list(graph.edges())
    )

    if not edges:
        return []

    # Build y0 representation and sigma-extension once
    g_y0 = nx_digraph_to_y0(graph)
    g_sigma = sigma_extension(g_y0)

    results = []
    for cause, effect in edges:
        if timeout_seconds is not None:
            try:
                with _timeout_context(timeout_seconds):
                    row = classify_edge(
                        graph,
                        cause,
                        effect,
                        precomputed_extension=g_sigma,
                        precomputed_y0=g_y0,
                    )
            except _EdgeTimeoutError:
                row = {
                    "cause": cause,
                    "effect": effect,
                    "status": "timeout",
                    "adjustment_set": None,
                    "same_scc": same_scc(graph, cause, effect),
                    "timed_out": True,
                }
        else:
            row = classify_edge(
                graph,
                cause,
                effect,
                precomputed_extension=g_sigma,
                precomputed_y0=g_y0,
            )
        results.append(row)

    # --- POST ---
    assert isinstance(results, list), "POST: result must be a list"
    assert all(r["status"] in ("identifiable", "unidentifiable", "timeout") for r in results), (
        "POST: all statuses must be valid"
    )
    return results


# ---------------------------------------------------------------------------
# _candidate_pool (module-level helper for maximize_identifiable_edges)
# ---------------------------------------------------------------------------


def _candidate_pool(g: nx.DiGraph) -> set:
    """Return all nodes that appear in any SCC's min-cut B(t)."""
    candidates: set = set()
    sccs = list(nx.strongly_connected_components(g))
    for scc in sccs:
        if len(scc) <= 1:
            continue
        scc_frozen = frozenset(scc)
        for tf in scc:
            in_scc_ch = find_in_scc_children(tf, scc_frozen, g)
            if not in_scc_ch:
                continue
            cut = compute_min_cut_b(tf, scc_frozen, in_scc_ch, g)
            candidates.update(cut)
    return candidates


# ---------------------------------------------------------------------------
# maximize_identifiable_edges
# ---------------------------------------------------------------------------


def maximize_identifiable_edges(
    graph: nx.DiGraph,
    k: int,
) -> dict:
    r"""Greedily maximise identifiable edges via up to *k* hard interventions.

    At each step, pick the candidate intervention node (drawn from the
    ``compute_min_cut_b`` cut pool across all non-trivial SCCs) that flips the
    most edges from unidentifiable to identifiable when ``do(node)`` is applied
    via ``build_intervened_graph``.  Re-score with ``evaluate_all_edges`` after
    each pick.

    Intervention semantics: ``do(S)`` removes **in-edges** to each node in S;
    the nodes themselves and their out-edges are preserved.  This is the
    node-split-cut / do-calculus semantics, not vertex deletion.

    Parameters
    ----------
    graph:
        The original directed graph.
    k:
        Maximum number of intervention nodes to select (budget).

    Returns
    -------
    dict with keys:
        ``curve``           -- list[(int, int)]: (n_interventions, n_identifiable) for 0..len(chosen)
        ``chosen_nodes``    -- list[str]: intervention nodes selected in order
        ``final_graph``     -- nx.DiGraph: post-intervention graph after all chosen interventions
        ``final_results``   -- list[dict]: per-edge classification on *final_graph*
        ``n_identifiable_baseline`` -- int: identifiable count before any intervention
        ``n_identifiable_final``    -- int: identifiable count after all interventions

    axiomander:
        ensures:
            len(result["curve"]) == len(result["chosen_nodes"]) + 1
            result["curve"][0][0] == 0
            result["n_identifiable_baseline"] == result["curve"][0][1]
            result["n_identifiable_final"] == result["curve"][-1][1]
            len(result["chosen_nodes"]) <= k
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(graph, nx.DiGraph), "PRE: graph must be an nx.DiGraph"
    assert isinstance(k, int) and k >= 0, "PRE: k must be a non-negative int"

    current_graph = graph.copy()
    chosen_nodes: list[str] = []
    intervened_so_far: list[str] = []

    # Baseline evaluation
    baseline_results = evaluate_all_edges(current_graph)
    n_baseline = sum(1 for r in baseline_results if r["status"] == "identifiable")
    curve: list[tuple[int, int]] = [(0, n_baseline)]

    for step in range(k):
        candidates = _candidate_pool(current_graph)
        # Exclude already-intervened nodes
        candidates -= set(intervened_so_far)

        if not candidates:
            break

        best_node: str | None = None
        best_gain = -1
        best_results: list[dict] = []

        for node in sorted(candidates):  # sorted for determinism
            trial_graph = build_intervened_graph(current_graph, [node])
            trial_results = evaluate_all_edges(trial_graph)
            n_ident = sum(1 for r in trial_results if r["status"] == "identifiable")
            gain = n_ident - curve[-1][1]
            if gain > best_gain:
                best_gain = gain
                best_node = node
                best_results = trial_results

        if best_node is None or best_gain <= 0:
            # No further improvement possible
            break

        # Apply the best intervention permanently
        current_graph = build_intervened_graph(current_graph, [best_node])
        chosen_nodes.append(best_node)
        intervened_so_far.append(best_node)
        n_new = sum(1 for r in best_results if r["status"] == "identifiable")
        curve.append((step + 1, n_new))

    # Final evaluation on the fully-intervened graph
    final_results = evaluate_all_edges(current_graph)
    n_final = sum(1 for r in final_results if r["status"] == "identifiable")

    result = {
        "curve": curve,
        "chosen_nodes": chosen_nodes,
        "final_graph": current_graph,
        "final_results": final_results,
        "n_identifiable_baseline": n_baseline,
        "n_identifiable_final": n_final,
    }

    # --- POST ---
    assert len(result["curve"]) == len(result["chosen_nodes"]) + 1, (
        "POST: curve length must be chosen_nodes + 1"
    )
    assert result["curve"][0][0] == 0, "POST: curve must start at step 0"
    assert result["n_identifiable_baseline"] == result["curve"][0][1], (
        "POST: baseline must match curve[0]"
    )
    assert result["n_identifiable_final"] == result["curve"][-1][1], (
        "POST: final must match curve[-1]"
    )
    assert len(result["chosen_nodes"]) <= k, "POST: chosen_nodes must not exceed budget k"
    return result
