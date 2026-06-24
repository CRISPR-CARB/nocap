"""
scc_perturb.py — pure graph-manipulation helpers for the SCC-perturbation pipeline.

These functions contain no I/O, no SLURM logic, and no y0 imports at the
module level.  They are the reusable "library" layer consumed by:

  - scripts/scc_perturb_prepare.py  (compute B(t))
  - scripts/scc_perturb_worker.py   (apply do(B(t)), query children)
  - notebooks                        (residual-SCC analysis)

Perturbation semantics
----------------------
A hard intervention ``do(S)`` on a node set *S* **removes the incoming edges**
to every node in *S*.  The nodes themselves and their out-edges are retained.
This is distinct from node deletion.

B(t) — minimum background perturbation
---------------------------------------
For a TF *t* inside a non-trivial SCC, ``compute_min_cut_b`` finds the
smallest set of *intermediate* SCC nodes (excluding *t* and its direct
children) such that ``do(B(t))`` severs every return path ``child(t) -> t``
within the SCC.  The cut is computed as a minimum vertex cut (node-split
max-flow, ``networkx.minimum_node_cut``) from a super-source over the
in-SCC children of *t* to sink *t*.
"""

from __future__ import annotations

import networkx as nx

# ---------------------------------------------------------------------------
# build_intervened_graph
# ---------------------------------------------------------------------------


def build_intervened_graph(graph: nx.DiGraph, perturb_set: list) -> nx.DiGraph:
    """
    Return a copy of *graph* with all in-edges to every node in *perturb_set* removed.

    Hard intervention do(perturb_set): the intervened nodes themselves and all
    their out-edges are preserved.  Only their *incoming* edges are deleted,
    reflecting the ``do(.)`` semantics of a hard perturbation (knock-out or
    knock-in that blocks upstream signal).

    Parameters
    ----------
    graph:
        The original directed graph (not mutated).
    perturb_set:
        List of node names to intervene on.  May be empty.

    Returns
    -------
    nx.DiGraph
        A new graph (deep copy of *graph*) with in-edges to *perturb_set*
        nodes removed.

    axiomander:
        ensures:
            all(result.in_degree(n) == 0 for n in perturb_set if n in result.nodes())
            result.number_of_nodes() == graph.number_of_nodes()
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(perturb_set, list), "PRE: perturb_set must be a list"

    intervened = graph.copy()
    for node in perturb_set:
        if node in intervened:
            in_edges = list(intervened.in_edges(node))
            intervened.remove_edges_from(in_edges)

    # --- POST ---
    for node in perturb_set:
        if node in intervened:
            assert intervened.in_degree(node) == 0, (
                f"POST: node {node!r} must have in-degree 0 after intervention"
            )
    return intervened


# ---------------------------------------------------------------------------
# get_direct_children
# ---------------------------------------------------------------------------


def get_direct_children(tf: str, graph: nx.DiGraph) -> list:
    """
    Return the sorted list of direct out-neighbours of *tf* in *graph*.

    Excludes *tf* itself (no self-loops).  This is the set O(t) used as the
    outcome variables for the ``cyclic_id`` query.  When called on the
    post-``do(B(t))`` graph, in-SCC children whose return paths have been
    severed are included; the cut nodes are excluded from the children because
    they were not targeted in the intervention.

    Parameters
    ----------
    tf:
        The TF gene name.
    graph:
        The (possibly intervened) directed graph.

    Returns
    -------
    list[str]
        Sorted list of direct children.

    axiomander:
        ensures:
            tf not in result
            result == sorted(result)
            all(isinstance(g, str) for g in result)
            all(graph.has_edge(tf, g) for g in result)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(tf, str), "PRE: tf must be a str"

    if tf not in graph:
        return []

    children = sorted(c for c in graph.successors(tf) if c != tf)

    # --- POST ---
    assert isinstance(children, list), "POST: result must be a list"
    assert tf not in children, "POST: tf must not be in its own children"
    return children


# ---------------------------------------------------------------------------
# find_in_scc_children
# ---------------------------------------------------------------------------


def find_in_scc_children(tf: str, scc_nodes: frozenset | set, graph: nx.DiGraph) -> list:
    """
    Return the direct children of *tf* that are members of *scc_nodes*.

    These are the nodes ``c`` such that there is a directed edge ``tf -> c``
    and ``c`` is in the same SCC as *tf*.  They are the nodes whose return
    paths ``c -> ... -> tf`` need to be severed by ``do(B(t))``.

    Parameters
    ----------
    tf:
        The TF gene name.
    scc_nodes:
        The frozenset (or set) of nodes in the SCC containing *tf*.
    graph:
        The full directed graph.

    Returns
    -------
    list[str]
        Unsorted list of in-SCC direct children.

    axiomander:
        ensures:
            len(result) <= len(list(graph.successors(tf)))
            all(c != tf for c in result)
            all(c in scc_nodes for c in result)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(tf, str), "PRE: tf must be a str"
    assert isinstance(scc_nodes, set | frozenset), "PRE: scc_nodes must be a set or frozenset"

    children = [c for c in graph.successors(tf) if c in scc_nodes and c != tf]

    # --- POST ---
    assert isinstance(children, list), "POST: result must be a list"
    assert all(c != tf for c in children), "POST: no self-loops in result"
    assert all(c in scc_nodes for c in children), "POST: all children must be in scc_nodes"
    return children


# ---------------------------------------------------------------------------
# compute_min_cut_b
# ---------------------------------------------------------------------------


def compute_min_cut_b(
    tf: str,
    scc_nodes: frozenset | set,
    in_scc_children: list,
    graph: nx.DiGraph,
) -> list:
    """
    Compute the minimum background perturbation set B(t) for TF *tf*.

    B(t) is the minimum vertex cut (on the SCC subgraph restricted to
    ``scc_nodes``) that separates every return path from any in-SCC child
    of *tf* back to *tf*, subject to:

    - *tf* itself is NOT in B(t)
    - direct children of *tf* that are in the SCC are NOT in B(t)
      (Interpretation A: we perturb only *intermediate* nodes, not the
      outcome variables themselves)

    The cut is found via ``networkx.minimum_node_cut``, which internally
    uses node-splitting + max-flow.  A virtual super-source is added and
    connected to all in-SCC children so a single call handles all return
    paths simultaneously.

    If *tf* is not in any non-trivial SCC (``len(scc_nodes) <= 1``) or has
    no in-SCC children, returns an empty list (B(t) = empty, no background
    perturbation required).

    Parameters
    ----------
    tf:
        The TF gene name.
    scc_nodes:
        The frozenset (or set) of nodes in the SCC containing *tf*.
    in_scc_children:
        List of *tf*'s direct children that are in the SCC (from
        ``find_in_scc_children``).
    graph:
        The full directed graph.

    Returns
    -------
    list[str]
        Sorted list of node names constituting B(t).

    axiomander:
        ensures:
            tf not in result
            all(c not in result for c in in_scc_children)
            all(n in scc_nodes for n in result)
            implies(len(in_scc_children) == 0, len(result) == 0)
            implies(len(scc_nodes) <= 1, len(result) == 0)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(tf, str), "PRE: tf must be a str"
    assert isinstance(scc_nodes, set | frozenset), "PRE: scc_nodes must be a set or frozenset"
    assert isinstance(in_scc_children, list), "PRE: in_scc_children must be a list"

    if not in_scc_children:
        return []

    if len(scc_nodes) <= 1:
        return []

    # Forbidden nodes: tf itself + its direct in-SCC children
    forbidden = {tf} | set(in_scc_children)

    # Build the SCC subgraph (copy() returns nx.DiGraph when graph is DiGraph)
    scc_sub: nx.DiGraph = graph.subgraph(scc_nodes).copy()  # type: ignore[assignment]

    # Add a virtual super-source outside the SCC that connects to all
    # in-SCC children, so a single min-cut call handles all of them at once.
    aux = scc_sub.copy()
    super_src = "__SUPER_SRC__"
    aux.add_node(super_src)
    for c in in_scc_children:
        aux.add_edge(super_src, c)

    try:
        raw_cut: set = set(nx.minimum_node_cut(aux, s=super_src, t=tf))
        # Exclude the virtual super-source if it sneaked in
        cut_nodes: set = raw_cut - {super_src}
        # If any forbidden node ended up in the cut (shouldn't happen for
        # correct graphs, but guard anyway), fall back to safe heuristic.
        if cut_nodes & forbidden:
            # Heuristic fallback: remove all in-neighbours of tf inside SCC
            # except forbidden nodes -- guaranteed to cut all return paths.
            cut_nodes = {n for n in scc_sub.predecessors(tf) if n not in forbidden}
    except (nx.NetworkXError, nx.NetworkXNoPath):
        # networkx raises when no path exists (already disconnected) or on
        # degenerate graphs; in those cases B is empty.
        cut_nodes = set()

    result = sorted(cut_nodes)  # deterministic order

    # --- POST ---
    assert isinstance(result, list), "POST: result must be a list"
    assert tf not in result, "POST: tf must not be in B(t)"
    assert all(c not in result for c in in_scc_children), (
        "POST: direct in-SCC children must not be in B(t)"
    )
    assert all(n in scc_nodes for n in result), "POST: all cut nodes must be in the SCC"
    return result


# ---------------------------------------------------------------------------
# residual_scc_analysis
# ---------------------------------------------------------------------------


def residual_scc_analysis(  # noqa: C901
    tf: str,
    children: list,
    min_cut: list,
    graph: nx.DiGraph,
) -> dict:
    """
    Analyse residual cyclic structure among *tf*'s children after do(B(t)).

    After applying ``do(B(t))`` (removing in-edges to ``min_cut`` nodes),
    the return paths ``child -> tf`` are severed.  However, the children
    of *tf* may still participate in cycles *among themselves* or with
    other surviving network nodes.  This function identifies which children
    sit in non-trivial SCCs of the post-intervention graph -- the "residually
    cyclic" children whose mutual entanglement explains why the joint
    ``cyclic_id`` query fails even when B(t) is correct.

    Parameters
    ----------
    tf:
        The TF gene name.
    children:
        List of *tf*'s direct children in the post-``do(B(t))`` graph
        (from ``get_direct_children``).
    min_cut:
        B(t) -- the list of nodes to intervene on (passed to
        ``build_intervened_graph``).
    graph:
        The full directed graph (pre-intervention).

    Returns
    -------
    dict with keys:
        ``g_do``              -- the post-intervention nx.DiGraph
        ``child_set``         -- frozenset of child names
        ``node_to_scc_id``    -- dict mapping every node in g_do to its SCC index
        ``scc_sizes``         -- dict mapping SCC index to size
        ``children_cyclic``   -- list of children in non-trivial SCCs (size >= 2)
        ``children_acyclic``  -- list of children in trivial SCCs (singleton)
        ``residual_clusters`` -- list[frozenset] of non-trivial SCCs that contain
                                 >= 2 children of *tf*
        ``tf_still_cyclic``   -- bool: is *tf* itself in a non-trivial SCC of g_do?
        ``cut_verified``      -- bool: True iff no child can reach *tf* in g_do

    axiomander:
        ensures:
            len(result["children_cyclic"]) + len(result["children_acyclic"]) == len(children)
            isinstance(result["tf_still_cyclic"], bool)
            isinstance(result["cut_verified"], bool)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(tf, str), "PRE: tf must be a str"
    assert isinstance(children, list), "PRE: children must be a list"
    assert isinstance(min_cut, list), "PRE: min_cut must be a list"

    g_do = build_intervened_graph(graph, min_cut)
    child_set = frozenset(children)

    # Compute SCCs of the full post-intervention graph
    sccs = list(nx.strongly_connected_components(g_do))
    node_to_scc_id: dict = {}
    scc_sizes: dict = {}
    for idx, scc in enumerate(sccs):
        scc_sizes[idx] = len(scc)
        for node in scc:
            node_to_scc_id[node] = idx

    # Tag each child
    children_cyclic = []
    children_acyclic = []
    for c in children:
        if c in node_to_scc_id:
            scc_id = node_to_scc_id[c]
            if scc_sizes[scc_id] >= 2:
                children_cyclic.append(c)
            else:
                children_acyclic.append(c)
        else:
            children_acyclic.append(c)  # node absent from graph -> acyclic

    # Find residual clusters: non-trivial SCCs containing >=2 children of tf
    residual_clusters = []
    for idx, scc in enumerate(sccs):
        if scc_sizes[idx] >= 2:
            scc_children = child_set & scc
            if len(scc_children) >= 2:
                residual_clusters.append(frozenset(scc_children))

    # Is tf itself still in a non-trivial SCC?
    tf_cyclic = False
    if tf in node_to_scc_id:
        tf_cyclic = scc_sizes[node_to_scc_id[tf]] >= 2

    # Verify that no child can still reach tf
    cut_verified = True
    for c in children:
        if c in g_do and tf in g_do:
            try:
                if nx.has_path(g_do, c, tf):
                    cut_verified = False
                    break
            except nx.NetworkXError:
                pass

    result = {
        "g_do": g_do,
        "child_set": child_set,
        "node_to_scc_id": node_to_scc_id,
        "scc_sizes": scc_sizes,
        "children_cyclic": sorted(children_cyclic),
        "children_acyclic": sorted(children_acyclic),
        "residual_clusters": residual_clusters,
        "tf_still_cyclic": tf_cyclic,
        "cut_verified": cut_verified,
    }

    # --- POST ---
    assert len(result["children_cyclic"]) + len(result["children_acyclic"]) == len(children), (
        "POST: all children must be classified"
    )
    assert isinstance(result["tf_still_cyclic"], bool), "POST: tf_still_cyclic must be bool"
    assert isinstance(result["cut_verified"], bool), "POST: cut_verified must be bool"
    return result


# ---------------------------------------------------------------------------
# residual_cluster_size_distribution
# ---------------------------------------------------------------------------


def verify_cut_complete(
    tf: str,
    in_scc_children: list,
    min_cut: list,
    graph: nx.DiGraph,
) -> dict:
    """
    Verify that ``do(B(t))`` severs **every** return path from any in-SCC child
    of *tf* back to *tf*.

    Under Interpretation A, ``compute_min_cut_b`` excludes *tf* and its direct
    in-SCC children from B(t).  For short cycles (e.g. a 2-cycle ``t ↔ c``),
    no valid intermediate node exists on the return path ``c → t``, so the
    computed B(t) may be incomplete.  This function detects such failures.

    Parameters
    ----------
    tf:
        The TF gene name.
    in_scc_children:
        List of *tf*'s direct children that are in the same SCC (from
        ``find_in_scc_children``).  These are the nodes whose return paths
        must be severed.
    min_cut:
        B(t) — the list of nodes to intervene on.
    graph:
        The full directed graph (pre-intervention).

    Returns
    -------
    dict with keys:
        ``complete``             – bool: True iff NO in-SCC child can reach
                                   *tf* in the post-intervention graph.
        ``surviving_children``   – list of in-SCC children that can still
                                   reach *tf* after ``do(B(t))``; empty
                                   when ``complete=True``.
        ``tf_still_cyclic``      – bool: True iff *tf* sits in a non-trivial
                                   SCC of the post-intervention graph (i.e.
                                   the cut failed to break *tf*'s own loop).

    axiomander:
        ensures:
            result["complete"] == (len(result["surviving_children"]) == 0)
            isinstance(result["tf_still_cyclic"], bool)
            isinstance(result["complete"], bool)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(tf, str), "PRE: tf must be a str"
    assert isinstance(in_scc_children, list), "PRE: in_scc_children must be a list"
    assert isinstance(min_cut, list), "PRE: min_cut must be a list"

    g_do = build_intervened_graph(graph, min_cut)

    # Check which in-SCC children can still reach tf
    surviving: list = []
    for c in in_scc_children:
        if c == tf:
            continue
        if c in g_do and tf in g_do:
            try:
                if nx.has_path(g_do, c, tf):
                    surviving.append(c)
            except nx.NetworkXError:
                pass

    complete = len(surviving) == 0

    # Check whether tf itself is in a non-trivial SCC of g_do
    tf_cyclic = False
    if tf in g_do:
        for scc in nx.strongly_connected_components(g_do):
            if tf in scc and len(scc) >= 2:
                tf_cyclic = True
                break

    result = {
        "complete": complete,
        "surviving_children": sorted(surviving),
        "tf_still_cyclic": tf_cyclic,
    }

    # --- POST ---
    assert result["complete"] == (len(result["surviving_children"]) == 0), (
        "POST: complete must equal (no surviving children)"
    )
    assert isinstance(result["tf_still_cyclic"], bool), (
        "POST: tf_still_cyclic must be bool"
    )
    assert isinstance(result["complete"], bool), (
        "POST: complete must be bool"
    )
    return result


# ---------------------------------------------------------------------------
# min_scc_break_set
# ---------------------------------------------------------------------------


def min_scc_break_set(
    cause: str,
    effect: str,
    graph: nx.DiGraph,
) -> dict:
    """Compute the minimum vertex-intervention set B that makes cause->effect
    single-door identifiable by breaking the residual SCC.

    The single-door criterion applied to edge cause->effect:

    1. Build G' = G - {cause->effect}  (the direct edge is removed).
    2. If cause and effect fall into **different** SCCs of G':
       the O-adjustment already identifies the edge; no intervention needed.
       Returns ``needs_intervention=False``, ``break_set=[]``.
    3. If they are still in the **same** SCC of G':
       find the minimum vertex cut B that, when ``do(B)`` is applied (in-edges
       to B removed) on G', severs every **return path effect ⇝ cause** within
       the shared SCC.  After ``do(B)``, cause and effect will be in different
       SCCs, enabling identification.
       B excludes ``{cause, effect}`` themselves.
       Cut is found via a super-source over effect's in-SCC successors → sink
       cause, node-split max-flow (``networkx.minimum_node_cut``).

    Parameters
    ----------
    cause, effect:
        Source and target of the directed edge being studied.  Both must be
        nodes in *graph* and the edge ``cause->effect`` must exist.
    graph:
        The full directed graph (not mutated).

    Returns
    -------
    dict with keys:
        ``same_scc_after_removal``  – bool: cause & effect same SCC in G'
        ``needs_intervention``      – bool: True iff same SCC (B needed)
        ``break_set``               – sorted list of nodes in B (empty if not needed)
        ``break_size``              – int: len(break_set)
        ``cut_verified``            – bool: after do(B) on G', cause & effect
                                      are in different SCCs

    axiomander:
        requires:
            graph.has_edge(cause, effect)
            isinstance(cause, str)
            isinstance(effect, str)
        ensures:
            cause not in result["break_set"]
            effect not in result["break_set"]
            result["break_size"] == len(result["break_set"])
            result["needs_intervention"] == result["same_scc_after_removal"]
            implies(not result["needs_intervention"], result["break_size"] == 0)
            implies(result["cut_verified"], not result["needs_intervention"] or result["break_size"] >= 0)
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(cause, str) and isinstance(effect, str), (
        "PRE: cause and effect must be strings"
    )
    assert graph.has_edge(cause, effect), (
        f"PRE: edge {cause!r}->{effect!r} must exist in graph"
    )

    # Step 1: G' = G - {cause->effect}
    g_prime = graph.copy()
    g_prime.remove_edge(cause, effect)

    # Step 2: check same-SCC in G'
    sccs = list(nx.strongly_connected_components(g_prime))
    scc_map: dict[str, int] = {}
    for cid, comp in enumerate(sccs):
        for n in comp:
            scc_map[n] = cid

    cause_scc = scc_map.get(cause, -1)
    effect_scc = scc_map.get(effect, -2)
    same_scc = (cause_scc == effect_scc) and (cause_scc != -1)

    if not same_scc:
        result = {
            "same_scc_after_removal": False,
            "needs_intervention": False,
            "break_set": [],
            "break_size": 0,
            "cut_verified": True,  # trivially: already separated
        }
        # --- POST ---
        assert cause not in result["break_set"], "POST: cause not in break_set"
        assert effect not in result["break_set"], "POST: effect not in break_set"
        assert result["break_size"] == 0, "POST: break_size must be 0 when not needed"
        return result

    # Step 3: same SCC — find min vertex cut to sever effect ⇝ cause
    # Restrict to the SCC subgraph in G'
    shared_scc_nodes: set[str] = set(sccs[cause_scc])
    forbidden: set[str] = {cause, effect}

    # In-SCC successors of effect (start of all return paths)
    effect_in_scc_succs = [
        n for n in g_prime.successors(effect)
        if n in shared_scc_nodes and n != effect
    ]

    break_set_nodes: set[str] = set()

    if effect_in_scc_succs:
        # Build SCC subgraph of G' and add super-source
        scc_sub: nx.DiGraph = g_prime.subgraph(shared_scc_nodes).copy()  # type: ignore[assignment]
        aux = scc_sub.copy()
        super_src = "__BREAK_SUPER_SRC__"
        aux.add_node(super_src)
        for succ in effect_in_scc_succs:
            aux.add_edge(super_src, succ)

        try:
            raw_cut: set = set(nx.minimum_node_cut(aux, s=super_src, t=cause))
            cut_nodes: set = raw_cut - {super_src}
            # Guard: forbidden nodes must not appear in the cut
            if cut_nodes & forbidden:
                # Fallback: all in-SCC predecessors of cause except forbidden
                cut_nodes = {
                    n for n in scc_sub.predecessors(cause)
                    if n not in forbidden
                }
        except (nx.NetworkXError, nx.NetworkXNoPath):
            cut_nodes = set()

        break_set_nodes = cut_nodes
    # else: effect has no in-SCC successors — no return path exists, cut is empty

    break_set = sorted(break_set_nodes)

    # Verify: after do(break_set) on G', cause and effect are in different SCCs
    g_do_prime = build_intervened_graph(g_prime, break_set)
    sccs_after = list(nx.strongly_connected_components(g_do_prime))
    scc_map_after: dict[str, int] = {}
    for cid, comp in enumerate(sccs_after):
        for n in comp:
            scc_map_after[n] = cid

    cause_scc_after = scc_map_after.get(cause, -1)
    effect_scc_after = scc_map_after.get(effect, -2)
    cut_verified = (cause_scc_after != effect_scc_after) or (cause_scc_after == -1)

    result = {
        "same_scc_after_removal": True,
        "needs_intervention": True,
        "break_set": break_set,
        "break_size": len(break_set),
        "cut_verified": cut_verified,
    }

    # --- POST ---
    assert cause not in result["break_set"], "POST: cause must not be in break_set"
    assert effect not in result["break_set"], "POST: effect must not be in break_set"
    assert result["break_size"] == len(result["break_set"]), (
        "POST: break_size must equal len(break_set)"
    )
    return result


def residual_cluster_size_distribution(analysis_result: dict) -> dict:
    """
    Summarise the size distribution of residual clusters from ``residual_scc_analysis``.

    A *residual cluster* is a non-trivial SCC in the post-``do(B(t))`` graph
    that contains **>= 2 direct children** of the TF.  This function extracts
    the per-cluster child-counts (not total SCC sizes) -- i.e. how many of the
    TF's own children are entangled inside each cluster -- and derives summary
    statistics over those counts.

    Parameters
    ----------
    analysis_result:
        The dict returned by ``residual_scc_analysis``.

    Returns
    -------
    dict with keys:
        ``n_clusters``                 -- int: number of residual clusters
        ``sizes``                      -- list[int]: sorted child-counts per cluster
        ``max_size``                   -- int: largest cluster child-count (0 if none)
        ``total_children_in_clusters`` -- int: total children across all clusters
        ``has_residual_cluster``       -- bool: True iff n_clusters >= 1

    axiomander:
        ensures:
            result["n_clusters"] == len(result["sizes"])
            isinstance(result["has_residual_cluster"], bool)
            result["has_residual_cluster"] == (result["n_clusters"] >= 1)
            implies(result["n_clusters"] == 0, result["max_size"] == 0)
            implies(result["n_clusters"] >= 1, result["max_size"] >= 2)
            result["total_children_in_clusters"] == sum(result["sizes"])
        modifies:
            none
    """
    # --- PRE ---
    assert isinstance(analysis_result, dict), "PRE: analysis_result must be a dict"
    assert "residual_clusters" in analysis_result, (
        "PRE: analysis_result must contain 'residual_clusters' key"
    )

    clusters: list = analysis_result["residual_clusters"]

    # Each element of residual_clusters is a frozenset of child names
    # (children of tf that share this SCC).  The size is the child-count.
    sizes = sorted(len(c) for c in clusters)
    n_clusters = len(sizes)
    max_size = max(sizes) if sizes else 0
    total_in_clusters = sum(sizes)
    has_cluster = n_clusters >= 1

    result = {
        "n_clusters": n_clusters,
        "sizes": sizes,
        "max_size": max_size,
        "total_children_in_clusters": total_in_clusters,
        "has_residual_cluster": has_cluster,
    }

    # --- POST ---
    assert n_clusters == len(sizes), "POST: n_clusters must equal len(sizes)"
    assert isinstance(has_cluster, bool), "POST: has_residual_cluster must be bool"
    assert has_cluster == (n_clusters >= 1), "POST: has_residual_cluster must match n_clusters >= 1"
    assert total_in_clusters == sum(sizes), "POST: total_children_in_clusters must equal sum(sizes)"
    return result
