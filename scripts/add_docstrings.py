"""Script to add missing docstrings to test files for docstr-coverage compliance."""

import re
import sys

# Map of (class_name, method_name_or_None) -> docstring to insert
# None method_name means the class-level docstring
DOCSTRINGS_GRAPHML = {
    # Classes
    ("TestSanitize", None): "Tests for the _sanitize() node-name helper.",
    ("TestPolarityPrefix", None): "Tests for the _polarity_prefix() helper.",
    (
        "TestBasicConversion",
        None,
    ): "Integration tests for basic edge-to-regression-line conversion.",
    ("TestSelfLoops", None): "Tests that self-loops are dropped with a warning.",
    ("TestCycles", None): "Tests that cycles are preserved (and warned about) by default.",
    ("TestPolarityConstraints", None): "Tests for BOUND constraint generation from edge polarity.",
    ("TestNameSanitization", None): "Tests that non-identifier node names are sanitized.",
    ("TestInputTypes", None): "Tests that graphml_to_semopy accepts various input types.",
    (
        "TestOutputStructure",
        None,
    ): "Tests for the overall structure of the semopy description string.",
    ("TestCLI", None): "Smoke tests for the graphml_to_semopy CLI entry point.",
    ("TestBreakCyclesHelper", None): "Unit tests for the _break_cycles() feedback-arc helper.",
    (
        "TestBreakCyclesIntegration",
        None,
    ): "Integration tests for the break_cycles=True flag in graphml_to_semopy.",
    ("TestCLIBreakCycles", None): "CLI tests for the --break-cycles flag.",
    # Methods
    (
        "TestSanitize",
        "test_clean_name_unchanged",
    ): "Clean alphanumeric names pass through unchanged.",
    (
        "TestSanitize",
        "test_hyphen_replaced",
    ): "Hyphens in node names are replaced with underscores.",
    ("TestSanitize", "test_space_replaced"): "Spaces in node names are replaced with underscores.",
    ("TestSanitize", "test_dot_replaced"): "Dots in node names are replaced with underscores.",
    ("TestPolarityPrefix", "test_plus_is_pos"): "'+' maps to 'pos' prefix.",
    ("TestPolarityPrefix", "test_activation_is_pos"): "'activation' maps to 'pos' prefix.",
    ("TestPolarityPrefix", "test_minus_is_neg"): "'-' maps to 'neg' prefix.",
    ("TestPolarityPrefix", "test_repression_is_neg"): "'repression' maps to 'neg' prefix.",
    ("TestPolarityPrefix", "test_none_is_reg"): "None polarity maps to 'reg' prefix.",
    (
        "TestPolarityPrefix",
        "test_ambiguous_is_reg",
    ): "Ambiguous '+/-' polarity maps to 'reg' prefix.",
    (
        "TestPolarityPrefix",
        "test_unknown_string_is_reg",
    ): "Unknown polarity string maps to 'reg' prefix.",
    (
        "TestBasicConversion",
        "test_single_edge_produces_regression_line",
    ): "A single directed edge produces one regression line.",
    (
        "TestBasicConversion",
        "test_multiple_parents_collapsed_on_one_line",
    ): "Multiple parents of the same target collapse onto one regression line.",
    (
        "TestBasicConversion",
        "test_multiple_targets_produce_separate_lines",
    ): "Multiple targets each get their own regression line.",
    (
        "TestBasicConversion",
        "test_param_name_format",
    ): "Positive-polarity parameter names use 'pos_src_tgt' format.",
    (
        "TestBasicConversion",
        "test_neg_param_name_format",
    ): "Negative-polarity parameter names use 'neg_src_tgt' format.",
    (
        "TestBasicConversion",
        "test_reg_param_name_format_for_unknown_polarity",
    ): "Unknown polarity uses 'reg_src_tgt' format.",
    (
        "TestBasicConversion",
        "test_reg_param_name_format_for_missing_polarity",
    ): "Missing polarity attribute uses 'reg_src_tgt' format.",
    ("TestSelfLoops", "test_self_loop_dropped"): "Self-loop edges are removed from the output.",
    (
        "TestSelfLoops",
        "test_self_loop_warning_count",
    ): "A single warning is emitted listing the count of dropped self-loops.",
    (
        "TestSelfLoops",
        "test_only_self_loops_produces_empty_structural_part",
    ): "A graph with only self-loops produces an empty structural section.",
    (
        "TestCycles",
        "test_cycle_preserved_in_output",
    ): "Cyclic edges are preserved in the output by default.",
    ("TestCycles", "test_cycle_warning_emitted"): "A warning is emitted when cycles are detected.",
    (
        "TestPolarityConstraints",
        "test_pos_edges_get_lower_bound_zero",
    ): "Positive edges get BOUND(0, inf) constraints.",
    (
        "TestPolarityConstraints",
        "test_neg_edges_get_upper_bound_zero",
    ): "Negative edges get BOUND(-inf, 0) constraints.",
    (
        "TestPolarityConstraints",
        "test_reg_edges_produce_no_bound",
    ): "Edges with no polarity produce no BOUND constraints.",
    (
        "TestPolarityConstraints",
        "test_mixed_polarities",
    ): "Mixed polarities produce the correct mix of BOUND constraints.",
    (
        "TestPolarityConstraints",
        "test_custom_inf_value",
    ): "The inf= parameter controls the bound magnitude.",
    (
        "TestPolarityConstraints",
        "test_no_polarity_flag_disables_bounds",
    ): "use_polarity=False suppresses all BOUND constraints.",
    (
        "TestPolarityConstraints",
        "test_no_polarity_flag_uses_reg_prefix",
    ): "use_polarity=False forces 'reg' prefix regardless of edge polarity.",
    (
        "TestNameSanitization",
        "test_hyphen_in_node_name_sanitized",
    ): "Hyphens in node names are replaced in the output description.",
    (
        "TestNameSanitization",
        "test_sanitization_warning_emitted",
    ): "A warning is emitted when a node name is sanitized.",
    ("TestInputTypes", "test_accepts_digraph"): "graphml_to_semopy accepts an nx.DiGraph directly.",
    (
        "TestInputTypes",
        "test_raises_for_undirected_graph",
    ): "graphml_to_semopy raises TypeError for undirected graphs.",
    (
        "TestInputTypes",
        "test_accepts_path_string",
    ): "graphml_to_semopy accepts a path string to a .graphml file.",
    (
        "TestInputTypes",
        "test_accepts_path_object",
    ): "graphml_to_semopy accepts a pathlib.Path to a .graphml file.",
    (
        "TestOutputStructure",
        "test_starts_with_structural_part_comment",
    ): "Output starts with the '# Structural part' comment header.",
    (
        "TestOutputStructure",
        "test_polarity_constraints_section_present_when_needed",
    ): "Polarity constraints section is present when constrained edges exist.",
    (
        "TestOutputStructure",
        "test_polarity_constraints_section_absent_when_all_reg",
    ): "Polarity constraints section is absent when all edges are unconstrained.",
    (
        "TestOutputStructure",
        "test_targets_sorted_alphabetically",
    ): "Regression lines are emitted in alphabetical target order.",
    (
        "TestCLI",
        "test_cli_writes_output_file",
    ): "CLI writes a valid semopy description to the output file.",
    ("TestCLI", "test_cli_no_polarity_flag"): "CLI --no-polarity flag disables BOUND constraints.",
    (
        "TestBreakCyclesHelper",
        "test_acyclic_graph_unchanged",
    ): "An acyclic graph is returned unchanged by _break_cycles.",
    (
        "TestBreakCyclesHelper",
        "test_simple_2_cycle_removes_one_edge",
    ): "A 2-cycle has exactly one edge removed.",
    (
        "TestBreakCyclesHelper",
        "test_simple_3_cycle_removes_one_edge",
    ): "A 3-cycle has exactly one edge removed.",
    (
        "TestBreakCyclesHelper",
        "test_two_independent_cycles_removes_two_edges",
    ): "Two independent cycles each have one edge removed.",
    (
        "TestBreakCyclesHelper",
        "test_removed_edges_are_from_original_graph",
    ): "All removed edges exist in the original graph.",
    (
        "TestBreakCyclesHelper",
        "test_dag_has_correct_edge_count",
    ): "The resulting DAG has the expected edge count.",
    (
        "TestBreakCyclesHelper",
        "test_empty_graph_unchanged",
    ): "An empty graph is returned unchanged.",
    (
        "TestBreakCyclesHelper",
        "test_single_node_no_edges",
    ): "A single-node graph with no edges is returned unchanged.",
    (
        "TestBreakCyclesHelper",
        "test_edge_data_preserved_in_dag",
    ): "Edge attributes are preserved on the edges that remain.",
    (
        "TestBreakCyclesIntegration",
        "test_break_cycles_produces_dag_output",
    ): "break_cycles=True produces a DAG-compatible semopy description.",
    (
        "TestBreakCyclesIntegration",
        "test_break_cycles_no_cycle_warning",
    ): "break_cycles=True suppresses the cycle-detected warning.",
    (
        "TestBreakCyclesIntegration",
        "test_break_cycles_emits_removal_warning",
    ): "break_cycles=True emits a warning about removed edges.",
    (
        "TestBreakCyclesIntegration",
        "test_break_cycles_acyclic_graph_unchanged",
    ): "break_cycles=True on an acyclic graph produces identical output.",
    (
        "TestBreakCyclesIntegration",
        "test_break_cycles_polarity_bounds_only_for_kept_edges",
    ): "BOUND constraints are only emitted for edges that survive cycle-breaking.",
    (
        "TestBreakCyclesIntegration",
        "test_break_cycles_default_false_preserves_cycles",
    ): "Default break_cycles=False preserves all cycle edges.",
    (
        "TestBreakCyclesIntegration",
        "test_break_cycles_three_cycle",
    ): "A 3-cycle is fully broken to a DAG.",
    (
        "TestCLIBreakCycles",
        "test_cli_break_cycles_flag_produces_dag",
    ): "CLI --break-cycles flag produces a DAG-compatible output file.",
    (
        "TestCLIBreakCycles",
        "test_cli_without_break_cycles_preserves_cycles",
    ): "CLI without --break-cycles preserves both directions of a 2-cycle.",
}

DOCSTRINGS_OPTIMIZER = {
    (
        "TestGreedyMaxCoverage",
        None,
    ): "Tests for the greedy_max_coverage budget-constrained optimizer.",
    ("TestGreedyMinSetCover", None): "Tests for the greedy_min_set_cover minimum-cover optimizer.",
    ("TestBuildMarginalGainCurve", None): "Tests for the build_marginal_gain_curve utility.",
    ("TestLoadCoverageMatrix", None): "Tests for the load_coverage_matrix CSV round-trip.",
    (
        "TestCycleBreakingScore",
        "cycle_graph",
    ): "Fixture: directed graph with two overlapping cycles and one acyclic branch.",
}


def add_docstrings_to_file(filepath, docstring_map):
    """Insert missing docstrings into a test file based on the provided map."""
    with open(filepath) as f:
        lines = f.readlines()

    current_class = None
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Track current class
        class_match = re.match(r"^class (\w+)", line)
        if class_match:
            current_class = class_match.group(1)
            result.append(line)
            i += 1
            # Check if next non-empty line is already a docstring
            j = i
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and lines[j].strip().startswith('"""'):
                # Already has docstring, skip
                pass
            else:
                # Insert class docstring if we have one
                key = (current_class, None)
                if key in docstring_map:
                    indent = "    "
                    result.append(f'{indent}"""{docstring_map[key]}"""\n')
            continue

        # Track method definitions
        method_match = re.match(r"^    def (\w+)\(", line)
        if method_match:
            method_name = method_match.group(1)
            result.append(line)
            i += 1
            # Check if next non-empty line is already a docstring
            j = i
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and lines[j].strip().startswith('"""'):
                # Already has docstring, skip
                pass
            else:
                # Insert method docstring if we have one
                key = (current_class, method_name)
                if key in docstring_map:
                    indent = "        "
                    result.append(f'{indent}"""{docstring_map[key]}"""\n')
            continue

        result.append(line)
        i += 1

    with open(filepath, "w") as f:
        f.writelines(result)

    print(f"Updated: {filepath}")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    add_docstrings_to_file(
        f"{base}/tests/test_graphml_to_semopy.py",
        DOCSTRINGS_GRAPHML,
    )
    add_docstrings_to_file(
        f"{base}/tests/test_perturbation_optimizer.py",
        DOCSTRINGS_OPTIMIZER,
    )
    print("Done.")
