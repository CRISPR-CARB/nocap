"""
tests/test_build_coverage_matrix.py
=====================================
Unit tests for scripts/build_coverage_matrix.py

Tests cover:
  - Module import smoke test (confirms y0 branch is installed correctly)
  - load_valid_genes: CSV parsing, "off"/"AAV" exclusion, network intersection
  - build_baseline_queries: directed pair construction, self-loop exclusion,
    valid-gene membership filtering, empty-input edge cases
"""

import csv
import os
import sys

import networkx as nx

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from build_coverage_matrix import build_baseline_queries, load_valid_genes

# ---------------------------------------------------------------------------
# Smoke test: module-level imports (cyclic_id, NxMixedGraph, etc.)
# ---------------------------------------------------------------------------


class TestModuleImport:
    """Verify that the module and its y0 dependencies import without error."""

    def test_import_build_coverage_matrix(self):
        """Importing build_coverage_matrix should not raise."""
        import build_coverage_matrix  # noqa: F401

    def test_cyclic_id_available(self):
        """cyclic_id must be importable from the installed y0 branch."""
        from y0.algorithm.identify.cyclic_id import cyclic_id  # noqa: F401

    def test_unidentifiable_available(self):
        """Unidentifiable exception must be importable."""
        from y0.algorithm.identify.utils import Unidentifiable  # noqa: F401

    def test_get_apt_order_available(self):
        """get_apt_order must be importable."""
        from y0.algorithm.ioscm.utils import get_apt_order  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_supptable(path, rows):
    """Write a minimal supptable CSV with a 'Perturbation Name' column."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Perturbation Name", "Other"])
        writer.writeheader()
        for name in rows:
            writer.writerow({"Perturbation Name": name, "Other": "x"})


# ---------------------------------------------------------------------------
# load_valid_genes
# ---------------------------------------------------------------------------


class TestLoadValidGenes:
    """Tests for load_valid_genes(supptable_path, network_nodes)."""

    def test_basic_intersection(self, tmp_path):
        """Genes in both CSV and network are returned."""
        csv_path = tmp_path / "supptable.csv"
        _write_supptable(csv_path, ["geneA", "geneB", "geneC"])
        network_nodes = {"geneA", "geneB", "geneD"}
        result = load_valid_genes(str(csv_path), network_nodes)
        assert result == {"geneA", "geneB"}

    def test_excludes_off_lowercase(self, tmp_path):
        """Rows with 'off' (lowercase) in the name are excluded."""
        csv_path = tmp_path / "supptable.csv"
        _write_supptable(csv_path, ["geneA", "geneA_off", "geneB"])
        network_nodes = {"geneA", "geneA_off", "geneB"}
        result = load_valid_genes(str(csv_path), network_nodes)
        assert "geneA_off" not in result
        assert result == {"geneA", "geneB"}

    def test_excludes_off_uppercase(self, tmp_path):
        """Rows with 'OFF' (uppercase) in the name are excluded (case-insensitive)."""
        csv_path = tmp_path / "supptable.csv"
        _write_supptable(csv_path, ["geneA", "geneA_OFF", "geneB"])
        network_nodes = {"geneA", "geneA_OFF", "geneB"}
        result = load_valid_genes(str(csv_path), network_nodes)
        assert "geneA_OFF" not in result

    def test_excludes_off_mixed_case(self, tmp_path):
        """Rows with 'Off' (mixed case) in the name are excluded."""
        csv_path = tmp_path / "supptable.csv"
        _write_supptable(csv_path, ["geneA_Off", "geneB"])
        network_nodes = {"geneA_Off", "geneB"}
        result = load_valid_genes(str(csv_path), network_nodes)
        assert "geneA_Off" not in result

    def test_excludes_aav(self, tmp_path):
        """Rows containing 'AAV' are excluded."""
        csv_path = tmp_path / "supptable.csv"
        _write_supptable(csv_path, ["geneA", "AAV_geneB", "geneC_AAV", "geneD"])
        network_nodes = {"geneA", "AAV_geneB", "geneC_AAV", "geneD"}
        result = load_valid_genes(str(csv_path), network_nodes)
        assert "AAV_geneB" not in result
        assert "geneC_AAV" not in result
        assert result == {"geneA", "geneD"}

    def test_excludes_both_off_and_aav(self, tmp_path):
        """Rows with both 'off' and 'AAV' are excluded."""
        csv_path = tmp_path / "supptable.csv"
        _write_supptable(csv_path, ["AAV_off_gene", "geneA"])
        network_nodes = {"AAV_off_gene", "geneA"}
        result = load_valid_genes(str(csv_path), network_nodes)
        assert "AAV_off_gene" not in result
        assert result == {"geneA"}

    def test_no_overlap_returns_empty(self, tmp_path):
        """When no CSV gene is in the network, return empty set."""
        csv_path = tmp_path / "supptable.csv"
        _write_supptable(csv_path, ["geneA", "geneB"])
        network_nodes = {"geneX", "geneY"}
        result = load_valid_genes(str(csv_path), network_nodes)
        assert result == set()

    def test_empty_csv_returns_empty(self, tmp_path):
        """CSV with header only returns empty set."""
        csv_path = tmp_path / "supptable.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Perturbation Name"])
            writer.writeheader()
        result = load_valid_genes(str(csv_path), {"geneA"})
        assert result == set()

    def test_empty_network_returns_empty(self, tmp_path):
        """Empty network_nodes set returns empty result."""
        csv_path = tmp_path / "supptable.csv"
        _write_supptable(csv_path, ["geneA", "geneB"])
        result = load_valid_genes(str(csv_path), set())
        assert result == set()

    def test_returns_set(self, tmp_path):
        """Return type must be a set."""
        csv_path = tmp_path / "supptable.csv"
        _write_supptable(csv_path, ["geneA"])
        result = load_valid_genes(str(csv_path), {"geneA"})
        assert isinstance(result, set)

    def test_duplicate_csv_rows_deduplicated(self, tmp_path):
        """Duplicate gene names in CSV should appear only once in result."""
        csv_path = tmp_path / "supptable.csv"
        _write_supptable(csv_path, ["geneA", "geneA", "geneB"])
        network_nodes = {"geneA", "geneB"}
        result = load_valid_genes(str(csv_path), network_nodes)
        assert result == {"geneA", "geneB"}


# ---------------------------------------------------------------------------
# build_baseline_queries
# ---------------------------------------------------------------------------


class TestBuildBaselineQueries:
    """Tests for build_baseline_queries(ecoli_graph, valid_genes)."""

    def test_basic_pair_construction(self):
        """Directed edge between two valid genes produces one pair."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        result = build_baseline_queries(G, {"A", "B"})
        assert ("A", "B") in result
        assert len(result) == 1

    def test_self_loops_excluded(self):
        """Self-loops (gene -> gene) must not appear in the output."""
        G = nx.DiGraph()
        G.add_edge("A", "A")
        G.add_edge("A", "B")
        result = build_baseline_queries(G, {"A", "B"})
        assert ("A", "A") not in result
        assert ("A", "B") in result

    def test_target_not_in_valid_genes_excluded(self):
        """Edge to a gene not in valid_genes must be excluded."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        G.add_edge("A", "C")  # C not in valid_genes
        result = build_baseline_queries(G, {"A", "B"})
        assert ("A", "C") not in result
        assert ("A", "B") in result

    def test_source_not_in_valid_genes_excluded(self):
        """Edges from a gene not in valid_genes must be excluded."""
        G = nx.DiGraph()
        G.add_edge("X", "B")  # X not in valid_genes
        G.add_edge("A", "B")
        result = build_baseline_queries(G, {"A", "B"})
        assert ("X", "B") not in result
        assert ("A", "B") in result

    def test_pairs_are_directed(self):
        """Pairs are ordered (intervention, outcome); reverse is not added."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        result = build_baseline_queries(G, {"A", "B"})
        assert ("A", "B") in result
        assert ("B", "A") not in result

    def test_multiple_successors(self):
        """A gene with multiple valid successors produces multiple pairs."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("A", "C"), ("A", "D")])
        result = build_baseline_queries(G, {"A", "B", "C", "D"})
        assert ("A", "B") in result
        assert ("A", "C") in result
        assert ("A", "D") in result
        assert len(result) == 3

    def test_empty_graph_returns_empty(self):
        """Empty graph (no nodes, no edges) with any valid_genes returns empty list.

        Genes in valid_genes that are not nodes in the graph are silently skipped.
        """
        G = nx.DiGraph()
        result = build_baseline_queries(G, {"A", "B"})
        assert result == []

    def test_empty_valid_genes_returns_empty(self):
        """Empty valid_genes set returns empty list."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        result = build_baseline_queries(G, set())
        assert result == []

    def test_no_edges_in_valid_genes_returns_empty(self):
        """Graph has edges but none connect valid genes to each other."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        # valid_genes has neither A nor B
        result = build_baseline_queries(G, {"C", "D"})
        assert result == []

    def test_returns_list(self):
        """Return type must be a list."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        result = build_baseline_queries(G, {"A", "B"})
        assert isinstance(result, list)

    def test_pairs_are_tuples(self):
        """Each element in the result must be a 2-tuple of strings."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C")])
        result = build_baseline_queries(G, {"A", "B", "C"})
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], str)

    def test_chain_graph(self):
        """A -> B -> C: both (A,B) and (B,C) should appear."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C")])
        result = build_baseline_queries(G, {"A", "B", "C"})
        assert ("A", "B") in result
        assert ("B", "C") in result
        assert len(result) == 2

    def test_cycle_graph_no_self_loops(self):
        """A -> B -> C -> A: three pairs, no self-loops."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        result = build_baseline_queries(G, {"A", "B", "C"})
        assert ("A", "B") in result
        assert ("B", "C") in result
        assert ("C", "A") in result
        assert len(result) == 3
        for src, tgt in result:
            assert src != tgt

    def test_no_duplicate_pairs(self):
        """Even if the graph has parallel edges, pairs should not be duplicated."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        # DiGraph deduplicates parallel edges automatically, but let's confirm
        result = build_baseline_queries(G, {"A", "B"})
        assert result.count(("A", "B")) == 1
