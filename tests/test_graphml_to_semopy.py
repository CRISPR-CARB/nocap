"""Tests for graphml_to_semopy conversion utility.

Tests are written first (TDD) and cover:
- Basic edge → regression line conversion
- Self-loop removal with warning
- Cycle preservation with warning
- Polarity → BOUND constraints (pos / neg / reg)
- Unknown / missing polarity → 'reg' prefix, no bound
- Node name sanitization
- use_polarity=False disables bounds
- Accepts an existing nx.DiGraph (not just a path)
- Raises TypeError for undirected graph
- CLI smoke test (output to file)
- --break-cycles: minimal feedback arc removal to produce a DAG
"""

from __future__ import annotations

import re
import warnings

import networkx as nx
import pytest

from nocap.graphml_to_semopy import _break_cycles, _polarity_prefix, _sanitize, graphml_to_semopy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(edges: list[tuple], polarity_attr: str = "polarity") -> nx.DiGraph:
    """Build a DiGraph from a list of (src, tgt, polarity_or_None) tuples."""
    G = nx.DiGraph()
    for item in edges:
        src, tgt = item[0], item[1]
        pol = item[2] if len(item) > 2 else None
        data = {polarity_attr: pol} if pol is not None else {}
        G.add_edge(src, tgt, **data)
    return G


def _parse_regression_lines(desc: str) -> dict[str, list[str]]:
    """Return {target: [term, ...]} from the structural part of a description."""
    result: dict[str, list[str]] = {}
    for line in desc.splitlines():
        line = line.strip()
        if "~" in line and not line.startswith("#"):
            tgt, rhs = line.split("~", 1)
            result[tgt.strip()] = [t.strip() for t in rhs.strip().split("+")]
    return result


def _parse_bounds(desc: str) -> list[tuple[float, float, list[str]]]:
    """Return [(lo, hi, [param, ...]), ...] from BOUND lines."""
    bounds = []
    for line in desc.splitlines():
        m = re.match(r"BOUND\(([^,]+),\s*([^)]+)\)\s+(.*)", line.strip())
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            params = m.group(3).split()
            bounds.append((lo, hi, params))
    return bounds


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


class TestSanitize:
    def test_clean_name_unchanged(self):
        assert _sanitize("aaeR") == "aaeR"

    def test_hyphen_replaced(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert _sanitize("gene-A") == "gene_A"

    def test_space_replaced(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert _sanitize("gene A") == "gene_A"

    def test_dot_replaced(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert _sanitize("gene.1") == "gene_1"


class TestPolarityPrefix:
    _pos = frozenset({"+", "activation", "positive", "1"})
    _neg = frozenset({"-", "repression", "negative", "-1"})

    def test_plus_is_pos(self):
        assert _polarity_prefix("+", self._pos, self._neg) == "pos"

    def test_activation_is_pos(self):
        assert _polarity_prefix("activation", self._pos, self._neg) == "pos"

    def test_minus_is_neg(self):
        assert _polarity_prefix("-", self._pos, self._neg) == "neg"

    def test_repression_is_neg(self):
        assert _polarity_prefix("repression", self._pos, self._neg) == "neg"

    def test_none_is_reg(self):
        assert _polarity_prefix(None, self._pos, self._neg) == "reg"

    def test_ambiguous_is_reg(self):
        assert _polarity_prefix("+/-", self._pos, self._neg) == "reg"

    def test_unknown_string_is_reg(self):
        assert _polarity_prefix("unknown", self._pos, self._neg) == "reg"


# ---------------------------------------------------------------------------
# Integration tests: graphml_to_semopy
# ---------------------------------------------------------------------------


class TestBasicConversion:
    def test_single_edge_produces_regression_line(self):
        G = _make_graph([("A", "B", "+")])
        desc = graphml_to_semopy(G)
        regs = _parse_regression_lines(desc)
        assert "B" in regs
        assert len(regs["B"]) == 1
        assert regs["B"][0] == "pos_A_B*A"

    def test_multiple_parents_collapsed_on_one_line(self):
        G = _make_graph([("A", "C", "+"), ("B", "C", "-")])
        desc = graphml_to_semopy(G)
        regs = _parse_regression_lines(desc)
        assert "C" in regs
        assert len(regs["C"]) == 2

    def test_multiple_targets_produce_separate_lines(self):
        G = _make_graph([("A", "B", "+"), ("A", "C", "-")])
        desc = graphml_to_semopy(G)
        regs = _parse_regression_lines(desc)
        assert "B" in regs
        assert "C" in regs

    def test_param_name_format(self):
        G = _make_graph([("src", "tgt", "+")])
        desc = graphml_to_semopy(G)
        assert "pos_src_tgt*src" in desc

    def test_neg_param_name_format(self):
        G = _make_graph([("src", "tgt", "-")])
        desc = graphml_to_semopy(G)
        assert "neg_src_tgt*src" in desc

    def test_reg_param_name_format_for_unknown_polarity(self):
        G = _make_graph([("src", "tgt", "+/-")])
        desc = graphml_to_semopy(G)
        assert "reg_src_tgt*src" in desc

    def test_reg_param_name_format_for_missing_polarity(self):
        G = _make_graph([("src", "tgt")])  # no polarity attr
        desc = graphml_to_semopy(G)
        assert "reg_src_tgt*src" in desc


class TestSelfLoops:
    def test_self_loop_dropped(self):
        G = _make_graph([("A", "A", "+"), ("A", "B", "+")])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            desc = graphml_to_semopy(G)
        # A→A should not appear in output
        assert "A*A" not in desc
        # Warning should be emitted
        assert any("self-loop" in str(warning.message).lower() for warning in w)

    def test_self_loop_warning_count(self):
        G = _make_graph([("A", "A", "+"), ("B", "B", "-"), ("A", "B", "+")])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            graphml_to_semopy(G)
        msgs = [
            str(warning.message) for warning in w if "self-loop" in str(warning.message).lower()
        ]
        assert len(msgs) == 1
        assert "2" in msgs[0]  # "Dropping 2 self-loop(s)"

    def test_only_self_loops_produces_empty_structural_part(self):
        G = _make_graph([("A", "A", "+")])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            desc = graphml_to_semopy(G)
        regs = _parse_regression_lines(desc)
        assert len(regs) == 0


class TestCycles:
    def test_cycle_preserved_in_output(self):
        # A → B → A
        G = _make_graph([("A", "B", "+"), ("B", "A", "-")])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            desc = graphml_to_semopy(G)
        regs = _parse_regression_lines(desc)
        assert "A" in regs  # B → A regression present
        assert "B" in regs  # A → B regression present

    def test_cycle_warning_emitted(self):
        G = _make_graph([("A", "B", "+"), ("B", "A", "-")])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            graphml_to_semopy(G)
        assert any("cycle" in str(warning.message).lower() for warning in w)


class TestPolarityConstraints:
    def test_pos_edges_get_lower_bound_zero(self):
        G = _make_graph([("A", "B", "+")])
        desc = graphml_to_semopy(G)
        bounds = _parse_bounds(desc)
        pos_bounds = [(lo, hi, ps) for lo, hi, ps in bounds if lo == 0.0]
        assert len(pos_bounds) == 1
        assert "pos_A_B" in pos_bounds[0][2]

    def test_neg_edges_get_upper_bound_zero(self):
        G = _make_graph([("A", "B", "-")])
        desc = graphml_to_semopy(G)
        bounds = _parse_bounds(desc)
        neg_bounds = [(lo, hi, ps) for lo, hi, ps in bounds if hi == 0.0]
        assert len(neg_bounds) == 1
        assert "neg_A_B" in neg_bounds[0][2]

    def test_reg_edges_produce_no_bound(self):
        G = _make_graph([("A", "B")])  # no polarity
        desc = graphml_to_semopy(G)
        bounds = _parse_bounds(desc)
        assert len(bounds) == 0

    def test_mixed_polarities(self):
        G = _make_graph([("A", "C", "+"), ("B", "C", "-"), ("D", "C")])
        desc = graphml_to_semopy(G)
        bounds = _parse_bounds(desc)
        lo_vals = {lo for lo, hi, _ in bounds}
        hi_vals = {hi for lo, hi, _ in bounds}
        assert 0.0 in lo_vals  # pos bound
        assert 0.0 in hi_vals  # neg bound
        # reg_D_C should not appear in any bound
        all_bound_params = [p for _, _, ps in bounds for p in ps]
        assert "reg_D_C" not in all_bound_params

    def test_custom_inf_value(self):
        G = _make_graph([("A", "B", "+")])
        desc = graphml_to_semopy(G, inf=999.0)
        assert "BOUND(0, 999)" in desc

    def test_no_polarity_flag_disables_bounds(self):
        G = _make_graph([("A", "B", "+"), ("C", "B", "-")])
        desc = graphml_to_semopy(G, use_polarity=False)
        bounds = _parse_bounds(desc)
        assert len(bounds) == 0

    def test_no_polarity_flag_uses_reg_prefix(self):
        G = _make_graph([("A", "B", "+")])
        desc = graphml_to_semopy(G, use_polarity=False)
        assert "reg_A_B*A" in desc


class TestNameSanitization:
    def test_hyphen_in_node_name_sanitized(self):
        G = nx.DiGraph()
        G.add_edge("gene-A", "gene-B", polarity="+")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            desc = graphml_to_semopy(G)
        assert "gene_A" in desc
        assert "gene_B" in desc
        assert "gene-A" not in desc

    def test_sanitization_warning_emitted(self):
        G = nx.DiGraph()
        G.add_edge("gene-A", "B", polarity="+")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            graphml_to_semopy(G)
        assert any("non-identifier" in str(warning.message) for warning in w)


class TestInputTypes:
    def test_accepts_digraph(self):
        G = _make_graph([("A", "B", "+")])
        desc = graphml_to_semopy(G)
        assert "B ~ " in desc

    def test_raises_for_undirected_graph(self):
        G = nx.Graph()
        G.add_edge("A", "B")
        with pytest.raises(TypeError, match="directed"):
            graphml_to_semopy(G)

    def test_accepts_path_string(self, tmp_path):
        G = _make_graph([("X", "Y", "+")])
        path = tmp_path / "test.graphml"
        nx.write_graphml(G, str(path))
        desc = graphml_to_semopy(str(path))
        assert "Y ~ " in desc

    def test_accepts_path_object(self, tmp_path):
        G = _make_graph([("X", "Y", "-")])
        path = tmp_path / "test.graphml"
        nx.write_graphml(G, str(path))
        desc = graphml_to_semopy(path)
        assert "neg_X_Y*X" in desc


class TestOutputStructure:
    def test_starts_with_structural_part_comment(self):
        G = _make_graph([("A", "B", "+")])
        desc = graphml_to_semopy(G)
        assert desc.startswith("# Structural part")

    def test_polarity_constraints_section_present_when_needed(self):
        G = _make_graph([("A", "B", "+")])
        desc = graphml_to_semopy(G)
        assert "# Polarity constraints" in desc

    def test_polarity_constraints_section_absent_when_all_reg(self):
        G = _make_graph([("A", "B")])
        desc = graphml_to_semopy(G)
        assert "# Polarity constraints" not in desc

    def test_targets_sorted_alphabetically(self):
        G = _make_graph([("X", "C", "+"), ("X", "A", "+"), ("X", "B", "+")])
        desc = graphml_to_semopy(G)
        regs = _parse_regression_lines(desc)
        targets = list(regs.keys())
        assert targets == sorted(targets)


class TestCLI:
    def test_cli_writes_output_file(self, tmp_path):
        from nocap.graphml_to_semopy import main

        G = _make_graph([("A", "B", "+"), ("B", "C", "-")])
        graphml_path = tmp_path / "net.graphml"
        nx.write_graphml(G, str(graphml_path))
        out_path = tmp_path / "model.txt"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            main([str(graphml_path), "-o", str(out_path)])

        assert out_path.exists()
        content = out_path.read_text()
        assert "B ~ " in content
        assert "C ~ " in content

    def test_cli_no_polarity_flag(self, tmp_path):
        from nocap.graphml_to_semopy import main

        G = _make_graph([("A", "B", "+")])
        graphml_path = tmp_path / "net.graphml"
        nx.write_graphml(G, str(graphml_path))
        out_path = tmp_path / "model.txt"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            main([str(graphml_path), "-o", str(out_path), "--no-polarity"])

        content = out_path.read_text()
        assert "BOUND" not in content
        assert "reg_A_B*A" in content


# ---------------------------------------------------------------------------
# Unit tests: _break_cycles helper
# ---------------------------------------------------------------------------


class TestBreakCyclesHelper:
    def test_acyclic_graph_unchanged(self):
        G = _make_graph([("A", "B", "+"), ("B", "C", "-")])
        dag, removed = _break_cycles(G)
        assert removed == []
        assert nx.is_directed_acyclic_graph(dag)

    def test_simple_2_cycle_removes_one_edge(self):
        # A → B → A  (2-cycle)
        G = _make_graph([("A", "B", "+"), ("B", "A", "-")])
        dag, removed = _break_cycles(G)
        assert len(removed) == 1
        assert nx.is_directed_acyclic_graph(dag)

    def test_simple_3_cycle_removes_one_edge(self):
        # A → B → C → A
        G = _make_graph([("A", "B", "+"), ("B", "C", "+"), ("C", "A", "-")])
        dag, removed = _break_cycles(G)
        assert len(removed) == 1
        assert nx.is_directed_acyclic_graph(dag)

    def test_two_independent_cycles_removes_two_edges(self):
        # Cycle 1: A → B → A
        # Cycle 2: C → D → C
        G = _make_graph([("A", "B", "+"), ("B", "A", "-"), ("C", "D", "+"), ("D", "C", "-")])
        dag, removed = _break_cycles(G)
        assert len(removed) == 2
        assert nx.is_directed_acyclic_graph(dag)

    def test_removed_edges_are_from_original_graph(self):
        G = _make_graph([("A", "B", "+"), ("B", "A", "-"), ("A", "C", "+")])
        dag, removed = _break_cycles(G)
        for edge in removed:
            assert G.has_edge(*edge[:2])

    def test_dag_has_correct_edge_count(self):
        G = _make_graph([("A", "B", "+"), ("B", "A", "-"), ("A", "C", "+")])
        dag, removed = _break_cycles(G)
        assert dag.number_of_edges() == G.number_of_edges() - len(removed)

    def test_empty_graph_unchanged(self):
        G = nx.DiGraph()
        dag, removed = _break_cycles(G)
        assert removed == []
        assert nx.is_directed_acyclic_graph(dag)

    def test_single_node_no_edges(self):
        G = nx.DiGraph()
        G.add_node("A")
        dag, removed = _break_cycles(G)
        assert removed == []

    def test_edge_data_preserved_in_dag(self):
        G = _make_graph([("A", "B", "+"), ("B", "C", "-")])
        dag, _ = _break_cycles(G)
        assert dag["A"]["B"].get("polarity") == "+"
        assert dag["B"]["C"].get("polarity") == "-"


# ---------------------------------------------------------------------------
# Integration tests: break_cycles=True in graphml_to_semopy
# ---------------------------------------------------------------------------


class TestBreakCyclesIntegration:
    def test_break_cycles_produces_dag_output(self):
        # A → B → A: with break_cycles, only one direction survives
        G = _make_graph([("A", "B", "+"), ("B", "A", "-")])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            desc = graphml_to_semopy(G, break_cycles=True)
        regs = _parse_regression_lines(desc)
        # Exactly one of the two edges should remain
        has_a_reg_b = "A" in regs and any("B" in t for t in regs["A"])
        has_b_reg_a = "B" in regs and any("A" in t for t in regs["B"])
        assert has_a_reg_b ^ has_b_reg_a  # exactly one survives

    def test_break_cycles_no_cycle_warning(self):
        G = _make_graph([("A", "B", "+"), ("B", "A", "-")])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            graphml_to_semopy(G, break_cycles=True)
        # Should NOT emit the "cycle" warning (cycles were removed)
        cycle_warnings = [
            x
            for x in w
            if "cycle" in str(x.message).lower() and "removed" not in str(x.message).lower()
        ]
        assert len(cycle_warnings) == 0

    def test_break_cycles_emits_removal_warning(self):
        G = _make_graph([("A", "B", "+"), ("B", "A", "-")])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            graphml_to_semopy(G, break_cycles=True)
        removal_warnings = [
            x
            for x in w
            if "removed" in str(x.message).lower()
            or "feedback" in str(x.message).lower()
            or "broke" in str(x.message).lower()
            or "break" in str(x.message).lower()
        ]
        assert len(removal_warnings) >= 1

    def test_break_cycles_acyclic_graph_unchanged(self):
        G = _make_graph([("A", "B", "+"), ("B", "C", "-")])
        desc_default = graphml_to_semopy(G)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            desc_break = graphml_to_semopy(G, break_cycles=True)
        assert desc_default == desc_break

    def test_break_cycles_polarity_bounds_only_for_kept_edges(self):
        # A→B (+), B→A (-): one edge removed; only the kept edge's bound should appear
        G = _make_graph([("A", "B", "+"), ("B", "A", "-")])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            desc = graphml_to_semopy(G, break_cycles=True)
        bounds = _parse_bounds(desc)
        # At most 1 bound (for the 1 remaining constrained edge)
        assert len(bounds) <= 1

    def test_break_cycles_default_false_preserves_cycles(self):
        G = _make_graph([("A", "B", "+"), ("B", "A", "-")])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            desc = graphml_to_semopy(G)  # default break_cycles=False
        regs = _parse_regression_lines(desc)
        assert "A" in regs
        assert "B" in regs

    def test_break_cycles_three_cycle(self):
        G = _make_graph([("A", "B", "+"), ("B", "C", "+"), ("C", "A", "-")])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            desc = graphml_to_semopy(G, break_cycles=True)
        # Rebuild a graph from the description to verify it's a DAG
        regs = _parse_regression_lines(desc)
        H = nx.DiGraph()
        for tgt, terms in regs.items():
            for term in terms:
                # term looks like "pos_A_B*A" — extract source after '*'
                src = term.split("*")[-1]
                H.add_edge(src, tgt)
        assert nx.is_directed_acyclic_graph(H)


# ---------------------------------------------------------------------------
# CLI tests: --break-cycles flag
# ---------------------------------------------------------------------------


class TestCLIBreakCycles:
    def test_cli_break_cycles_flag_produces_dag(self, tmp_path):
        from nocap.graphml_to_semopy import main

        G = _make_graph([("A", "B", "+"), ("B", "A", "-")])
        graphml_path = tmp_path / "cyclic.graphml"
        nx.write_graphml(G, str(graphml_path))
        out_path = tmp_path / "model.txt"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            main([str(graphml_path), "-o", str(out_path), "--break-cycles"])

        content = out_path.read_text()
        regs = _parse_regression_lines(content)
        H = nx.DiGraph()
        for tgt, terms in regs.items():
            for term in terms:
                src = term.split("*")[-1]
                H.add_edge(src, tgt)
        assert nx.is_directed_acyclic_graph(H)

    def test_cli_without_break_cycles_preserves_cycles(self, tmp_path):
        from nocap.graphml_to_semopy import main

        G = _make_graph([("A", "B", "+"), ("B", "A", "-")])
        graphml_path = tmp_path / "cyclic.graphml"
        nx.write_graphml(G, str(graphml_path))
        out_path = tmp_path / "model.txt"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            main([str(graphml_path), "-o", str(out_path)])

        content = out_path.read_text()
        regs = _parse_regression_lines(content)
        # Both directions should be present
        assert len(regs) == 2
