"""
tests/test_build_coverage_matrix_hypothesis.py
================================================
Property-based tests for scripts/build_coverage_matrix.py using Hypothesis.

Properties verified:
  load_valid_genes
    - Result is always a subset of network_nodes
    - Result never contains names with 'off' (case-insensitive)
    - Result never contains names with 'AAV'
    - Larger network_nodes set can only grow (never shrink) the result

  build_baseline_queries
    - No self-loops in output
    - Every pair (src, tgt) has src in valid_genes and tgt in valid_genes
    - Every pair (src, tgt) corresponds to a real edge in the graph
    - Output length <= number of edges in the graph
    - Output is a list of 2-tuples
    - Idempotent: calling twice with same inputs gives same result
    - Subsetting valid_genes can only reduce (never increase) the output
    - No duplicate pairs
    - Empty valid_genes always gives empty output

Note: load_valid_genes tests use tempfile.TemporaryDirectory (not pytest's
tmp_path fixture) because @given + class methods cannot share pytest fixtures.
"""

import csv
import os
import sys
import tempfile

import networkx as nx
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from build_coverage_matrix import build_baseline_queries, load_valid_genes

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Gene names: short uppercase ASCII identifiers (no 'off' or 'AAV' by default)
gene_name = st.from_regex(r"[A-Z][A-Z0-9]{0,5}", fullmatch=True)

# A frozenset of clean gene names (1–12 genes)
gene_set = st.frozensets(gene_name, min_size=1, max_size=12)

# Gene names that may include "off"/"AAV" variants
dirty_gene_name = st.one_of(
    gene_name,
    gene_name.map(lambda g: g + "_off"),
    gene_name.map(lambda g: g + "_OFF"),
    gene_name.map(lambda g: "AAV_" + g),
    gene_name.map(lambda g: g + "_AAV"),
)
dirty_gene_set = st.frozensets(dirty_gene_name, min_size=0, max_size=15)


def _write_supptable_to(path, genes):
    """Write a minimal supptable CSV to the given path."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Perturbation Name"])
        writer.writeheader()
        for g in genes:
            writer.writerow({"Perturbation Name": g})


@st.composite
def digraph_with_valid_genes(draw):
    """
    Generate a random DiGraph whose nodes are drawn from a gene pool,
    plus a valid_genes set that is a subset of those nodes.
    """
    nodes = draw(st.frozensets(gene_name, min_size=2, max_size=10))
    node_list = sorted(nodes)
    G = nx.DiGraph()
    G.add_nodes_from(node_list)
    possible_edges = [(u, v) for u in node_list for v in node_list]
    edges = draw(
        st.lists(
            st.sampled_from(possible_edges),
            min_size=0,
            max_size=min(20, len(possible_edges)),
            unique=True,
        )
    )
    G.add_edges_from(edges)
    valid = draw(
        st.frozensets(st.sampled_from(node_list), min_size=0, max_size=len(node_list))
    )
    return G, set(valid)


@st.composite
def digraph_with_valid_genes_and_subset(draw):
    """
    Generate a DiGraph + valid_genes + a subset of valid_genes.
    Used to test monotonicity of build_baseline_queries.
    """
    G, valid = draw(digraph_with_valid_genes())
    if not valid:
        subset = set()
    else:
        valid_list = sorted(valid)
        subset = draw(
            st.frozensets(
                st.sampled_from(valid_list),
                min_size=0,
                max_size=len(valid_list),
            )
        )
    return G, valid, set(subset)


# ---------------------------------------------------------------------------
# load_valid_genes properties  (module-level functions — avoids fixture clash)
# ---------------------------------------------------------------------------


@given(dirty_gene_set, gene_set)
@settings(max_examples=200)
def test_load_valid_genes_result_subset_of_network(csv_genes, network_nodes):
    """Result is always a subset of network_nodes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "supptable.csv")
        _write_supptable_to(csv_path, csv_genes)
        result = load_valid_genes(csv_path, set(network_nodes))
    assert result <= set(network_nodes), "Result must be a subset of network_nodes"


@given(dirty_gene_set, gene_set)
@settings(max_examples=200)
def test_load_valid_genes_never_contains_off(csv_genes, network_nodes):
    """Result never contains any name with 'off' (case-insensitive)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "supptable.csv")
        _write_supptable_to(csv_path, csv_genes)
        result = load_valid_genes(csv_path, set(network_nodes))
    for name in result:
        assert "off" not in name.lower(), f"'off' gene leaked into result: {name}"


@given(dirty_gene_set, gene_set)
@settings(max_examples=200)
def test_load_valid_genes_never_contains_aav(csv_genes, network_nodes):
    """Result never contains any name with 'AAV'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "supptable.csv")
        _write_supptable_to(csv_path, csv_genes)
        result = load_valid_genes(csv_path, set(network_nodes))
    for name in result:
        assert "AAV" not in name, f"'AAV' gene leaked into result: {name}"


@given(dirty_gene_set, gene_set)
@settings(max_examples=200)
def test_load_valid_genes_returns_set(csv_genes, network_nodes):
    """Return type is always a set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "supptable.csv")
        _write_supptable_to(csv_path, csv_genes)
        result = load_valid_genes(csv_path, set(network_nodes))
    assert isinstance(result, set)


@given(dirty_gene_set, gene_set)
@settings(max_examples=200)
def test_load_valid_genes_larger_network_never_shrinks_result(csv_genes, network_nodes):
    """Adding more nodes to network_nodes can only grow (never shrink) the result."""
    clean_csv = frozenset(
        g for g in csv_genes if "off" not in g.lower() and "AAV" not in g
    )
    small_network = set(network_nodes)
    extra = clean_csv - small_network
    large_network = small_network | extra

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "supptable.csv")
        _write_supptable_to(csv_path, csv_genes)
        result_small = load_valid_genes(csv_path, small_network)
        result_large = load_valid_genes(csv_path, large_network)

    assert result_small <= result_large, (
        "Larger network should produce a superset of results"
    )


# ---------------------------------------------------------------------------
# build_baseline_queries properties  (class methods — no fixtures needed)
# ---------------------------------------------------------------------------


class TestBuildBaselineQueriesProperties:
    """Property-based tests for build_baseline_queries."""

    @given(digraph_with_valid_genes())
    @settings(max_examples=300)
    def test_no_self_loops(self, graph_and_genes):
        """Output never contains self-loops."""
        G, valid = graph_and_genes
        result = build_baseline_queries(G, valid)
        for src, tgt in result:
            assert src != tgt, f"Self-loop found: ({src}, {tgt})"

    @given(digraph_with_valid_genes())
    @settings(max_examples=300)
    def test_both_endpoints_in_valid_genes(self, graph_and_genes):
        """Both src and tgt in every pair must be in valid_genes."""
        G, valid = graph_and_genes
        result = build_baseline_queries(G, valid)
        for src, tgt in result:
            assert src in valid, f"src {src} not in valid_genes"
            assert tgt in valid, f"tgt {tgt} not in valid_genes"

    @given(digraph_with_valid_genes())
    @settings(max_examples=300)
    def test_edge_exists_in_graph(self, graph_and_genes):
        """Every pair (src, tgt) must correspond to a real edge in the graph."""
        G, valid = graph_and_genes
        result = build_baseline_queries(G, valid)
        for src, tgt in result:
            assert G.has_edge(src, tgt), f"Edge ({src}, {tgt}) not in graph"

    @given(digraph_with_valid_genes())
    @settings(max_examples=300)
    def test_output_is_list_of_2tuples(self, graph_and_genes):
        """Output is a list of 2-tuples."""
        G, valid = graph_and_genes
        result = build_baseline_queries(G, valid)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple) and len(item) == 2

    @given(digraph_with_valid_genes())
    @settings(max_examples=300)
    def test_output_length_bounded_by_edges(self, graph_and_genes):
        """Output length cannot exceed the number of edges in the graph."""
        G, valid = graph_and_genes
        result = build_baseline_queries(G, valid)
        assert len(result) <= G.number_of_edges()

    @given(digraph_with_valid_genes())
    @settings(max_examples=300)
    def test_idempotent(self, graph_and_genes):
        """Calling twice with the same inputs produces the same result."""
        G, valid = graph_and_genes
        result1 = build_baseline_queries(G, valid)
        result2 = build_baseline_queries(G, valid)
        assert sorted(result1) == sorted(result2)

    @given(digraph_with_valid_genes_and_subset())
    @settings(max_examples=300)
    def test_subset_valid_genes_reduces_output(self, args):
        """Restricting valid_genes to a subset can only reduce (never increase) output."""
        G, valid, subset = args
        assert subset <= valid
        result_full = build_baseline_queries(G, valid)
        result_sub = build_baseline_queries(G, subset)
        full_set = set(result_full)
        for pair in result_sub:
            assert pair in full_set, (
                f"Pair {pair} in subset result but not in full result"
            )

    @given(digraph_with_valid_genes())
    @settings(max_examples=300)
    def test_no_duplicate_pairs(self, graph_and_genes):
        """No pair appears more than once (DiGraph has no parallel edges)."""
        G, valid = graph_and_genes
        result = build_baseline_queries(G, valid)
        assert len(result) == len(set(result)), "Duplicate pairs found"

    @given(digraph_with_valid_genes())
    @settings(max_examples=300)
    def test_empty_valid_genes_gives_empty(self, graph_and_genes):
        """Passing empty valid_genes always returns []."""
        G, _ = graph_and_genes
        result = build_baseline_queries(G, set())
        assert result == []
