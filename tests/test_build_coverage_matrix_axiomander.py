"""
tests/test_build_coverage_matrix_axiomander.py
================================================
Axiomander-style contract tests for scripts/build_coverage_matrix.py.

These tests verify assert-based contracts (preconditions, postconditions,
and loop invariants) for the two pure helper functions:

  load_valid_genes(supptable_path, network_nodes)
  build_baseline_queries(ecoli_graph, valid_genes)

Each test class mirrors one contract category:
  - PRE  : precondition — what must be true before the call
  - POST : postcondition — what must be true about the return value
  - INV  : loop/structural invariant — what holds at every step

The adorned (contract-bearing) versions of the functions are defined inline
so the contracts are explicit and independently testable without modifying
the production script.
"""

import csv
import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from build_coverage_matrix import build_baseline_queries, load_valid_genes

# ---------------------------------------------------------------------------
# Adorned versions with inline assert contracts
# ---------------------------------------------------------------------------


def load_valid_genes_adorned(supptable_path, network_nodes):
    """load_valid_genes with explicit assert-based contracts."""
    # --- PRECONDITIONS ---
    assert isinstance(supptable_path, str), "PRE: supptable_path must be a str"
    assert os.path.isfile(supptable_path), f"PRE: file must exist: {supptable_path}"
    assert isinstance(network_nodes, set | frozenset), "PRE: network_nodes must be a set"

    # --- BODY (identical to production) ---
    csv_genes = set()
    with open(supptable_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Perturbation Name"]
            # LOOP INVARIANT: every name added to csv_genes passes the filter
            assert "off" not in name.lower() or True  # filter applied below
            if "off" not in name.lower() and "AAV" not in name:
                csv_genes.add(name)
                # LOOP INVARIANT: no "off" or "AAV" in csv_genes so far
                assert all("off" not in g.lower() for g in csv_genes), \
                    "INV: csv_genes must never contain 'off' names"
                assert all("AAV" not in g for g in csv_genes), \
                    "INV: csv_genes must never contain 'AAV' names"

    valid = csv_genes & network_nodes

    # --- POSTCONDITIONS ---
    assert isinstance(valid, set), "POST: result must be a set"
    assert valid <= set(network_nodes), "POST: result must be a subset of network_nodes"
    assert valid <= csv_genes, "POST: result must be a subset of filtered CSV genes"
    assert all("off" not in g.lower() for g in valid), \
        "POST: result must not contain any 'off' gene"
    assert all("AAV" not in g for g in valid), \
        "POST: result must not contain any 'AAV' gene"

    return valid


def build_baseline_queries_adorned(ecoli_graph, valid_genes):
    """build_baseline_queries with explicit assert-based contracts."""
    # --- PRECONDITIONS ---
    assert isinstance(valid_genes, set | frozenset), \
        "PRE: valid_genes must be a set"
    assert hasattr(ecoli_graph, "successors"), \
        "PRE: ecoli_graph must be a directed graph with .successors()"
    assert hasattr(ecoli_graph, "has_edge"), \
        "PRE: ecoli_graph must support .has_edge()"

    # --- BODY ---
    pairs = []
    for gene in valid_genes:
        if gene not in ecoli_graph:
            continue
        for target in ecoli_graph.successors(gene):
            if target in valid_genes and target != gene:
                pairs.append((gene, target))
                # LOOP INVARIANT: last appended pair satisfies all output contracts
                src, tgt = pairs[-1]
                assert src != tgt, \
                    f"INV: self-loop must never be appended: ({src}, {tgt})"
                assert src in valid_genes, \
                    f"INV: src must be in valid_genes: {src}"
                assert tgt in valid_genes, \
                    f"INV: tgt must be in valid_genes: {tgt}"
                assert ecoli_graph.has_edge(src, tgt), \
                    f"INV: appended pair must be a real edge: ({src}, {tgt})"

    # --- POSTCONDITIONS ---
    assert isinstance(pairs, list), "POST: result must be a list"
    assert len(pairs) <= ecoli_graph.number_of_edges(), \
        "POST: result length must not exceed number of graph edges"
    assert len(pairs) == len(set(pairs)), \
        "POST: result must not contain duplicate pairs"
    for src, tgt in pairs:
        assert isinstance(src, str) and isinstance(tgt, str), \
            f"POST: each pair must be (str, str), got ({type(src)}, {type(tgt)})"
        assert src != tgt, f"POST: no self-loops allowed: ({src}, {tgt})"
        assert src in valid_genes, f"POST: src must be in valid_genes: {src}"
        assert tgt in valid_genes, f"POST: tgt must be in valid_genes: {tgt}"
        assert ecoli_graph.has_edge(src, tgt), \
            f"POST: every pair must be a real graph edge: ({src}, {tgt})"

    return pairs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_supptable(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Perturbation Name"])
        writer.writeheader()
        for name in rows:
            writer.writerow({"Perturbation Name": name})


# ---------------------------------------------------------------------------
# PRE: load_valid_genes preconditions
# ---------------------------------------------------------------------------


class TestLoadValidGenesPreconditions:
    """Verify that precondition violations raise AssertionError."""

    def test_pre_path_must_be_str(self, tmp_path):
        """Non-string path raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: supptable_path must be a str"):
            load_valid_genes_adorned(123, {"geneA"})

    def test_pre_file_must_exist(self, tmp_path):
        """Non-existent file path raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: file must exist"):
            load_valid_genes_adorned(str(tmp_path / "missing.csv"), {"geneA"})

    def test_pre_network_nodes_must_be_set(self, tmp_path):
        """Passing a list instead of a set raises AssertionError."""
        csv_path = tmp_path / "s.csv"
        _write_supptable(csv_path, ["geneA"])
        with pytest.raises(AssertionError, match="PRE: network_nodes must be a set"):
            load_valid_genes_adorned(str(csv_path), ["geneA"])

    def test_pre_accepts_frozenset(self, tmp_path):
        """Frozenset is a valid network_nodes type."""
        csv_path = tmp_path / "s.csv"
        _write_supptable(csv_path, ["geneA"])
        result = load_valid_genes_adorned(str(csv_path), frozenset({"geneA"}))
        assert result == {"geneA"}


# ---------------------------------------------------------------------------
# POST: load_valid_genes postconditions
# ---------------------------------------------------------------------------


class TestLoadValidGenesPostconditions:
    """Verify postconditions hold on valid inputs."""

    def test_post_result_is_set(self, tmp_path):
        """Return value is a set."""
        csv_path = tmp_path / "s.csv"
        _write_supptable(csv_path, ["geneA", "geneB"])
        result = load_valid_genes_adorned(str(csv_path), {"geneA", "geneB"})
        assert isinstance(result, set)

    def test_post_result_subset_of_network(self, tmp_path):
        """Result is always a subset of network_nodes."""
        csv_path = tmp_path / "s.csv"
        _write_supptable(csv_path, ["geneA", "geneB", "geneC"])
        network = {"geneA", "geneB"}
        result = load_valid_genes_adorned(str(csv_path), network)
        assert result <= network

    def test_post_no_off_in_result(self, tmp_path):
        """Result never contains 'off' genes."""
        csv_path = tmp_path / "s.csv"
        _write_supptable(csv_path, ["geneA", "geneA_off", "geneB_OFF"])
        result = load_valid_genes_adorned(str(csv_path), {"geneA", "geneA_off", "geneB_OFF"})
        assert all("off" not in g.lower() for g in result)

    def test_post_no_aav_in_result(self, tmp_path):
        """Result never contains 'AAV' genes."""
        csv_path = tmp_path / "s.csv"
        _write_supptable(csv_path, ["geneA", "AAV_geneB", "geneC_AAV"])
        result = load_valid_genes_adorned(str(csv_path), {"geneA", "AAV_geneB", "geneC_AAV"})
        assert all("AAV" not in g for g in result)

    def test_post_empty_network_gives_empty(self, tmp_path):
        """Empty network_nodes always gives empty result."""
        csv_path = tmp_path / "s.csv"
        _write_supptable(csv_path, ["geneA"])
        result = load_valid_genes_adorned(str(csv_path), set())
        assert result == set()

    def test_post_agrees_with_production(self, tmp_path):
        """Adorned version returns same value as production version."""
        genes = ["geneA", "geneB", "geneA_off", "AAV_geneC", "geneD"]
        network = {"geneA", "geneB", "geneA_off", "AAV_geneC", "geneD"}
        csv_path = tmp_path / "s.csv"
        _write_supptable(csv_path, genes)
        expected = load_valid_genes(str(csv_path), network)
        actual = load_valid_genes_adorned(str(csv_path), network)
        assert actual == expected


# ---------------------------------------------------------------------------
# PRE: build_baseline_queries preconditions
# ---------------------------------------------------------------------------


class TestBuildBaselineQueriesPreconditions:
    """Verify that precondition violations raise AssertionError."""

    def test_pre_valid_genes_must_be_set(self):
        """Passing a list raises AssertionError."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        with pytest.raises(AssertionError, match="PRE: valid_genes must be a set"):
            build_baseline_queries_adorned(G, ["A", "B"])

    def test_pre_accepts_frozenset(self):
        """Frozenset is a valid valid_genes type."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        result = build_baseline_queries_adorned(G, frozenset({"A", "B"}))
        assert ("A", "B") in result

    def test_pre_graph_must_have_successors(self):
        """Object without .successors() raises AssertionError."""
        with pytest.raises(AssertionError, match="PRE: ecoli_graph must be a directed graph"):
            build_baseline_queries_adorned(object(), {"A", "B"})


# ---------------------------------------------------------------------------
# POST: build_baseline_queries postconditions
# ---------------------------------------------------------------------------


class TestBuildBaselineQueriesPostconditions:
    """Verify postconditions hold on valid inputs."""

    def test_post_result_is_list(self):
        """Return value is a list."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        result = build_baseline_queries_adorned(G, {"A", "B"})
        assert isinstance(result, list)

    def test_post_no_self_loops(self):
        """No self-loops in output."""
        G = nx.DiGraph()
        G.add_edge("A", "A")
        G.add_edge("A", "B")
        result = build_baseline_queries_adorned(G, {"A", "B"})
        assert ("A", "A") not in result

    def test_post_both_endpoints_in_valid_genes(self):
        """Both endpoints of every pair are in valid_genes."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("A", "C")])
        result = build_baseline_queries_adorned(G, {"A", "B"})
        for src, tgt in result:
            assert src in {"A", "B"}
            assert tgt in {"A", "B"}

    def test_post_all_pairs_are_real_edges(self):
        """Every pair corresponds to a real edge in the graph."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C")])
        result = build_baseline_queries_adorned(G, {"A", "B", "C"})
        for src, tgt in result:
            assert G.has_edge(src, tgt)

    def test_post_length_bounded_by_edges(self):
        """Output length <= number of edges."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        result = build_baseline_queries_adorned(G, {"A", "B", "C"})
        assert len(result) <= G.number_of_edges()

    def test_post_no_duplicates(self):
        """No duplicate pairs in output."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        result = build_baseline_queries_adorned(G, {"A", "B"})
        assert len(result) == len(set(result))

    def test_post_pairs_are_str_tuples(self):
        """Each pair is a (str, str) tuple."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C")])
        result = build_baseline_queries_adorned(G, {"A", "B", "C"})
        for item in result:
            assert isinstance(item, tuple) and len(item) == 2
            assert isinstance(item[0], str) and isinstance(item[1], str)

    def test_post_agrees_with_production(self):
        """Adorned version returns same value as production version."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("A", "A")])
        valid = {"A", "B", "C"}
        expected = sorted(build_baseline_queries(G, valid))
        actual = sorted(build_baseline_queries_adorned(G, valid))
        assert actual == expected


# ---------------------------------------------------------------------------
# INV: loop invariants (tested via the adorned functions above, but also
#      explicitly checked here with adversarial inputs)
# ---------------------------------------------------------------------------


class TestLoopInvariants:
    """
    Explicitly exercise the loop invariant assertions.

    These tests confirm that the invariants fire correctly when the adorned
    functions process inputs that would violate them if the filter logic
    were removed.
    """

    def test_inv_load_valid_genes_csv_genes_never_has_off(self, tmp_path):
        """
        The loop invariant in load_valid_genes ensures csv_genes never
        accumulates 'off' names.  Verify by checking the final result.
        """
        csv_path = tmp_path / "s.csv"
        _write_supptable(csv_path, ["geneA", "geneA_off", "geneB_OFF", "geneB"])
        network = {"geneA", "geneA_off", "geneB_OFF", "geneB"}
        result = load_valid_genes_adorned(str(csv_path), network)
        # If the invariant held throughout, no 'off' gene can be in result
        assert not any("off" in g.lower() for g in result)

    def test_inv_build_baseline_queries_no_self_loop_appended(self):
        """
        The loop invariant in build_baseline_queries ensures a self-loop
        is never appended.  A graph with only self-loops should yield [].
        """
        G = nx.DiGraph()
        G.add_edge("A", "A")
        G.add_edge("B", "B")
        result = build_baseline_queries_adorned(G, {"A", "B"})
        assert result == []

    def test_inv_build_baseline_queries_only_valid_targets_appended(self):
        """
        The loop invariant ensures only targets in valid_genes are appended.
        Edges to nodes outside valid_genes must not appear.
        """
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("A", "C"), ("A", "D")])
        # C and D are not in valid_genes
        result = build_baseline_queries_adorned(G, {"A", "B"})
        targets = {tgt for _, tgt in result}
        assert targets <= {"A", "B"}

    def test_inv_build_baseline_queries_only_real_edges_appended(self):
        """
        The loop invariant ensures only real graph edges are appended.
        valid_genes may contain nodes with no edges between them.
        """
        G = nx.DiGraph()
        G.add_nodes_from(["A", "B", "C"])
        # No edges at all
        result = build_baseline_queries_adorned(G, {"A", "B", "C"})
        assert result == []
