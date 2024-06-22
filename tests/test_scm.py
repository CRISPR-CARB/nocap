"""Test functions for the scm module."""

import networkx as nx

# import numpy as np
import sympy as sy
from pgmpy.models import BayesianNetwork

# import y0
from y0.graph import NxMixedGraph

from nocap import (
    convert_to_eqn_array_latex,
    convert_to_latex,
    dagitty_to_digraph,
    dagitty_to_dot,
    dagitty_to_mixed_graph,
    evaluate_lscm,
    generate_lscm_from_dag,
    generate_lscm_from_mixed_graph,
    get_symbols_from_bi_edges,
    get_symbols_from_di_edges,
    get_symbols_from_nodes,
    mixed_graph_to_pgmpy,
    read_dag_file,
)

# TODO: use fixtures!


def test_basic_conversion():
    """Test basic conversion of daggity to dot format."""
    # Test basic conversion
    daggity_string = "dag { A -> B; B -> C; C -> A; }"
    expected_dot = "digraph { A -> B; B -> C; C -> A; }"
    assert dagitty_to_dot(daggity_string) == expected_dot  # noqa: S101


def test_with_latent_variables():
    """Test conversion of daggity to dot format with latent variables."""
    # Test conversion with the 'latent' keyword
    daggity_string = "dag { latent, A -> B; B -> C; C -> A; }"
    expected_dot = 'digraph { observed="no", A -> B; B -> C; C -> A; }'
    assert dagitty_to_dot(daggity_string) == expected_dot  # noqa: S101


def test_with_outcome_variables():
    """Test conversion of daggity to dot format with outcome variables."""
    # Test conversion with the 'outcome' keyword
    daggity_string = "dag { outcome, A -> B; B -> C; C -> A; }"
    expected_dot = "digraph { A -> B; B -> C; C -> A; }"
    assert dagitty_to_dot(daggity_string) == expected_dot  # noqa: S101


def test_with_adjusted_variables():
    """Test conversion of daggity to dot format with adjusted variables."""
    # Test conversion with the 'adjusted' keyword
    daggity_string = "dag { adjusted, A -> B; B -> C; C -> A; }"
    expected_dot = "digraph { A -> B; B -> C; C -> A; }"
    assert dagitty_to_dot(daggity_string) == expected_dot  # noqa: S101


def test_with_exposure_keyword():
    """Test conversion of daggity to dot format with exposure variables."""
    # Test conversion with the 'exposure' keyword
    daggity_string = "dag { exposure, A -> B; B -> C; C -> A; }"
    expected_dot = "digraph { A -> B; B -> C; C -> A; }"
    assert dagitty_to_dot(daggity_string) == expected_dot  # noqa: S101


def test_with_combined_keywords():
    """Test conversion of daggity to dot format with combined keywords."""
    # Test conversion with the 'latent', 'adjusted', 'outcome', and 'exposure' keywords
    daggity_string = "dag { latent, adjusted, outcome, exposure, A -> B; B -> C; C -> A; }"
    expected_dot = 'digraph { observed="no", A -> B; B -> C; C -> A; }'
    assert dagitty_to_dot(daggity_string) == expected_dot  # noqa: S101


def test_with_empty_input():
    """Test conversion of daggity to dot format with an empty string."""
    daggity_string = ""
    expected_dot = ""
    assert dagitty_to_dot(daggity_string) == expected_dot  # noqa: S101


def test_with_complex_graph_structure():
    """Test conversion of daggity to dot format with a complex graph structure."""
    daggity_string = (
        "dag { latent, adjusted, outcome, exposure, A -> B; B -> C; C -> D; D -> A; E -> F; G; }"
    )
    expected_dot = 'digraph { observed="no", A -> B; B -> C; C -> D; D -> A; E -> F; G; }'
    assert dagitty_to_dot(daggity_string) == expected_dot  # noqa: S101


def test_read_dag_file_success():
    """Test reading in a .dag file with a valid file path."""
    fpath = "./tests/test.dag"
    expected = 'dag {\n"Z" -> "Y"\n}'
    actual = read_dag_file(fpath)
    assert expected == actual  # noqa: S101


def test_read_dag_file_io_error():
    """Test reading in a .dag file with an invalid file path."""
    fpath = "./tests/non_existant_graph.dag"
    expected = None  # should return None w/ file IO error
    actual = read_dag_file(fpath)
    assert expected == actual  # noqa: S101


# TODO: update test to use fixture
def test_dagitty_to_mixed_graph():
    """Test if daggity is converted into an NxMixedGraph correctly."""

    def mixed_graphs_equal(graph1: NxMixedGraph, graph2: NxMixedGraph) -> bool:
        """Test if two mixed graphs are equal."""
        if nx.utils.graphs_equal(graph1.undirected, graph2.undirected) and nx.utils.graphs_equal(
            graph1.directed, graph2.directed
        ):
            return True
        else:
            return False

    graph_str = """dag {
                        bb="0,0,1,1"
                        "Z" [pos="0.5,0.9"]
                        "Y" [pos="0.1,0.5"]
                        "X" [pos="0.9,0.5"]
                        "M" [pos="0.5,0.1"]
                        "Z" -> "Y"
                        "Z" -> "X"
                        "M" -> "Y"
                        "X" -> "M"
                        }"""
    graph_fname = "./tests/frontdoor_backdoor.dag"
    mixed_graph1 = dagitty_to_mixed_graph(graph_str)
    mixed_graph2 = dagitty_to_mixed_graph(graph_fname)
    assert mixed_graphs_equal(mixed_graph1, mixed_graph2) is True  # noqa: S101


def test_dagitty_to_digraph():
    """Test if daggity is converted into a directed graph correctly."""
    graph_str = """dag {
                        bb="0,0,1,1"
                        "Z" [pos="0.5,0.9"]
                        "Y" [pos="0.1,0.5"]
                        "X" [pos="0.9,0.5"]
                        "M" [pos="0.5,0.1"]
                        "Z" -> "Y"
                        "Z" -> "X"
                        "M" -> "Y"
                        "X" -> "M"
                        }"""
    graph_fname = "./tests/frontdoor_backdoor.dag"
    mixed_graph1 = dagitty_to_digraph(graph_str)
    mixed_graph2 = dagitty_to_digraph(graph_fname)
    assert nx.utils.graphs_equal(mixed_graph1, mixed_graph2) is True  # noqa: S101


def test_generate_lscm_from_dag():
    """Test the linear structural causal model generated from a directed acyclic graph."""
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C")])
    expected_equations = {
        sy.Symbol("A"): sy.Symbol("epsilon_A"),
        sy.Symbol("B"): sy.Symbol("beta_A_->B") * sy.Symbol("A") + sy.Symbol("epsilon_B"),
        sy.Symbol("C"): sy.Symbol("beta_B_->C") * sy.Symbol("B") + sy.Symbol("epsilon_C"),
    }
    actual_equations = generate_lscm_from_dag(graph)
    for node in expected_equations:  # symbolic equality
        assert sy.simplify(actual_equations[node] - expected_equations[node]) == 0  # noqa: S101


def test_generate_lscm_from_mixed_graph():
    """Test the linear structural causal model generated from an NxMixedGraph."""
    edges = [("A", "B"), ("B", "C")]
    bi_edges = [("A", "B")]
    graph = NxMixedGraph.from_str_edges(directed=edges, undirected=bi_edges)
    expected_equations = {
        sy.Symbol("A"): sy.Symbol("epsilon_A") + sy.Symbol("gamma_A_<->B"),
        sy.Symbol("B"): sy.Symbol("beta_A_->B") * sy.Symbol("A")
        + sy.Symbol("epsilon_B")
        + sy.Symbol("gamma_A_<->B"),
        sy.Symbol("C"): sy.Symbol("beta_B_->C") * sy.Symbol("B") + sy.Symbol("epsilon_C"),
    }
    actual_equations = generate_lscm_from_mixed_graph(graph)
    for node in expected_equations:  # symbolic equality
        assert sy.simplify(actual_equations[node] - expected_equations[node]) == 0  # noqa: S101


def test_get_symbols_from_bi_edges():
    """Test if symbols are gotten from bidirectional edges correctly."""
    edges = [("A", "B"), ("B", "C")]
    bi_edges = [("A", "B")]
    graph = NxMixedGraph.from_str_edges(directed=edges, undirected=bi_edges)
    expected_symbols = {
        ("A", "B"): sy.Symbol("gamma_A_<->B"),
    }
    actual_symbols = get_symbols_from_bi_edges(graph)
    # print(type(list(actual_symbols.keys())[0][0]))
    assert actual_symbols == expected_symbols  # noqa: S101


def test_get_symbols_from_di_edges():
    """Tests if symbols are gotten from directional edges correctly."""
    edges = [("A", "B"), ("B", "C")]
    bi_edges = [("A", "B")]
    graph = NxMixedGraph.from_str_edges(directed=edges, undirected=bi_edges)
    expected_symbols = {
        ("A", "B"): sy.Symbol("beta_A_->B"),
        ("B", "C"): sy.Symbol("beta_B_->C"),
    }
    actual_symbols = get_symbols_from_di_edges(graph)
    # print(type(list(actual_symbols.keys())[0][0]))
    assert actual_symbols == expected_symbols  # noqa: S101


def test_get_symbols_from_nodes():
    """Tests if symbols are gotten from symbols correctly."""
    edges = [("A", "B"), ("B", "C")]
    bi_edges = [("A", "B")]
    graph = NxMixedGraph.from_str_edges(directed=edges, undirected=bi_edges)
    expected_symbols = {
        "A": sy.Symbol("epsilon_A"),
        "B": sy.Symbol("epsilon_B"),
        "C": sy.Symbol("epsilon_C"),
    }
    actual_symbols = get_symbols_from_nodes(graph)
    assert actual_symbols == expected_symbols  # noqa: S101


def test_evaluate_lscm():
    """Tests lscm evaluation given a set of parameters."""
    # use fixture
    edges = [("A", "B"), ("B", "C")]
    bi_edges = [("A", "B")]
    graph = NxMixedGraph.from_str_edges(directed=edges, undirected=bi_edges)
    lscm_dict = generate_lscm_from_mixed_graph(graph)
    epsilon_symbols = get_symbols_from_nodes(graph)
    beta_symbols = get_symbols_from_di_edges(graph)
    gamma_symbols = get_symbols_from_bi_edges(graph)

    epsilon_values = {epsilon: 1.0 for epsilon in epsilon_symbols.values()}
    beta_values = {beta: 1.0 for beta in beta_symbols.values()}
    gamma_values = {gamma: 1.0 for gamma in gamma_symbols.values()}

    param_dict = {**epsilon_values, **beta_values, **gamma_values}

    expected_symbols = {
        sy.Symbol("A"): sy.core.numbers.Float(2.0),
        sy.Symbol("B"): sy.core.numbers.Float(4.0),
        sy.Symbol("C"): sy.core.numbers.Float(5.0),
    }
    actual_symbols = evaluate_lscm(lscm_dict, param_dict)
    # Use a numerical tolerance for comparison
    tolerance = 1e-9
    for key in expected_symbols.keys():
        assert (  # noqa: S101
            abs(float(actual_symbols[key]) - float(expected_symbols[key])) < tolerance
        ), f"Values for {key} are not equal within tolerance."  # noqa: S101


def test_convert_to_latex():
    """Test that lscm is correctly converted into latex."""
    # use fixture
    edges = [("A", "B")]
    graph = NxMixedGraph.from_str_edges(directed=edges)
    lscm_dict = generate_lscm_from_mixed_graph(graph)
    expected = r"$$A = \epsilon_{A}$$" + "\n " + r"$$B = A \beta_{A ->B} + \epsilon_{B}$$"
    actual = convert_to_latex(lscm_dict)
    assert actual == expected  # noqa: S101


def test_convert_to_eqn_array_latex():
    """Test that lscm is correctly converted into latex equation array."""
    # use fixture
    edges = [("A", "B")]
    graph = NxMixedGraph.from_str_edges(directed=edges)
    lscm_dict = generate_lscm_from_mixed_graph(graph)
    expected = r"$$ \begin{array}{rcl}A &=& \epsilon_{A}\\ B &=& A \beta_{A ->B} + \epsilon_{B}\end{array}$$"
    actual = convert_to_eqn_array_latex(lscm_dict)
    assert actual == expected  # noqa: S101


def test_mixed_graph_to_pgmpy():
    """Convert a mixed graph to an equivalent :class:`pgmpy.BayesianNetwork`."""
    # Create a complex mixed graph with mixed edges, disconnected nodes, and a fully disconnected node
    graph = NxMixedGraph()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")
    graph.add_directed_edge("A", "B")
    graph.add_directed_edge("B", "C")
    graph.add_undirected_edge("A", "C")
    graph.add_undirected_edge("D", "E")
    graph.add_node("F")  # Add a fully disconnected node

    # Convert to BayesianNetwork
    bn = mixed_graph_to_pgmpy(graph)

    # Check if the BayesianNetwork has the correct edges and nodes
    assert isinstance(bn, BayesianNetwork)  # noqa: S101
    assert set(bn.edges()) == {  # noqa: S101
        ("A", "B"),
        ("B", "C"),
        ("U_A_C", "A"),
        ("U_A_C", "C"),
        ("U_D_E", "D"),
        ("U_D_E", "E"),
    }
    assert set(bn.nodes()) == {"A", "B", "C", "U_A_C", "D", "E", "U_D_E", "F"}  # noqa: S101


# def test_generate_synthetic_data_from_lscm():
#     """"""
#     ### Use y0 function
#     # input lscm and parameter values
#     # evaluate lscm
#     # compare expected to actual
#     raise NotImplementedError


# def test_regress_lscm():
#     """"""
#     ### Use y0 function
#     # input lscm and synthetic data
#     # regress using single door criterion (from y0)
#     # compare parameter estimates to ground truth
#     raise NotImplementedError
