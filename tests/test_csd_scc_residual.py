"""tests/test_csd_scc_residual.py — Unit tests for residual_scc_info.

Tests four canonical edge cases on tiny hand-built graphs:
1. Bridge edge     — removal breaks mutual reachability → same_scc_after=False
2. Redundant cycle — parallel path survives → same_scc_after=True
3. Self-loop       — flagged is_self_loop=True, same_scc_after=False
4. Cross-SCC edge  — same_scc_before=False → same_scc_after=False, trivial
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

# Make sure the scripts/ directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from csd_scc_residual_analysis import residual_scc_info


def _build_scc_structures(g: nx.DiGraph):
    """Helper: run Tarjan and return (scc_map, scc_sets, scc_sizes)."""
    scc_map: dict[str, int] = {}
    scc_sets: dict[int, set[str]] = {}
    scc_sizes: dict[int, int] = {}
    for cid, comp in enumerate(nx.strongly_connected_components(g)):
        for n in comp:
            scc_map[n] = cid
        scc_sets[cid] = comp
        scc_sizes[cid] = len(comp)
    return scc_map, scc_sets, scc_sizes


# ---------------------------------------------------------------------------
# 1. Bridge edge: A <-> B (only path A->B->A; removing A->B breaks the cycle)
# ---------------------------------------------------------------------------

def test_bridge_edge_breaks_scc():
    """Removing the only forward edge in a 2-cycle → nodes no longer co-SCC."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "A")])
    scc_map, scc_sets, scc_sizes = _build_scc_structures(g)

    # Before: A and B are in the same SCC of size 2
    assert scc_map["A"] == scc_map["B"]
    assert scc_sizes[scc_map["A"]] == 2

    info = residual_scc_info(g, "A", "B", scc_map, scc_sets, scc_sizes)

    assert info["is_self_loop"] is False
    assert info["same_scc_before"] is True
    assert info["same_scc_after"] is False        # bridge — broken
    assert info["scc_size_u_after"] == 1           # A is now a singleton
    assert info["scc_size_v_after"] == 1           # B is now a singleton


# ---------------------------------------------------------------------------
# 2. Redundant edge: triangle A->B->C->A plus short-cut A->C
#    Removing A->C leaves the cycle A->B->C->A intact.
# ---------------------------------------------------------------------------

def test_redundant_edge_preserves_scc():
    """Removing a non-bridge edge in a cycle leaves nodes still co-SCC."""
    g = nx.DiGraph()
    # Triangle: A->B->C->A
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    # Short-cut: A->C (redundant — the path A->B->C->A already links them)
    g.add_edge("A", "C")

    scc_map, scc_sets, scc_sizes = _build_scc_structures(g)

    # All three in one SCC of size 3
    assert scc_map["A"] == scc_map["B"] == scc_map["C"]
    assert scc_sizes[scc_map["A"]] == 3

    info = residual_scc_info(g, "A", "C", scc_map, scc_sets, scc_sizes)

    assert info["is_self_loop"] is False
    assert info["same_scc_before"] is True
    assert info["same_scc_after"] is True         # redundant — still connected
    # All three nodes still in one component after removing A->C
    assert info["scc_size_u_after"] == 3
    assert info["scc_size_v_after"] == 3


# ---------------------------------------------------------------------------
# 3. Self-loop: A->A
# ---------------------------------------------------------------------------

def test_self_loop_flagged():
    """Self-loop is flagged as degenerate; same_scc_after=False by convention."""
    g = nx.DiGraph()
    g.add_node("A")
    g.add_edge("A", "A")      # self-loop
    g.add_node("B")           # isolated, separate SCC

    scc_map, scc_sets, scc_sizes = _build_scc_structures(g)

    info = residual_scc_info(g, "A", "A", scc_map, scc_sets, scc_sizes)

    assert info["is_self_loop"] is True
    assert info["same_scc_after"] is False        # convention
    # After removing the self-loop, A is a singleton (no other edges)
    assert info["scc_size_u_after"] == 1
    assert info["scc_size_v_after"] == 1


def test_self_loop_in_larger_scc():
    """Self-loop on a node that is also part of a real cycle."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "A"), ("A", "A")])  # 2-cycle + self-loop

    scc_map, scc_sets, scc_sizes = _build_scc_structures(g)

    info = residual_scc_info(g, "A", "A", scc_map, scc_sets, scc_sizes)

    assert info["is_self_loop"] is True
    assert info["same_scc_after"] is False
    # After removing A->A, the A<->B 2-cycle remains: both still in SCC of size 2
    assert info["scc_size_u_after"] == 2


# ---------------------------------------------------------------------------
# 4. Cross-SCC edge: X->Y where X and Y are in different SCCs
# ---------------------------------------------------------------------------

def test_cross_scc_edge():
    """Removing a cross-SCC edge never changes SCC membership."""
    g = nx.DiGraph()
    # SCC1: A<->B
    g.add_edges_from([("A", "B"), ("B", "A")])
    # SCC2: C<->D
    g.add_edges_from([("C", "D"), ("D", "C")])
    # Cross edge: B -> C
    g.add_edge("B", "C")

    scc_map, scc_sets, scc_sizes = _build_scc_structures(g)

    # B and C must be in different SCCs
    assert scc_map["B"] != scc_map["C"]

    info = residual_scc_info(g, "B", "C", scc_map, scc_sets, scc_sizes)

    assert info["is_self_loop"] is False
    assert info["same_scc_before"] is False
    assert info["same_scc_after"] is False
    assert info["scc_size_u_after"] == 2          # SCC1 unchanged
    assert info["scc_size_v_after"] == 2          # SCC2 unchanged
