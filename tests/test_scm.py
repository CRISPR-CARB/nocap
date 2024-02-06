"""Test functions for the scm module."""

from nocap import (
    dagitty_to_dot, 
    read_dag_file, 
    dagitty_to_mixed_graph, 
    dagitty_to_digraph, 
    generate_LSCM_from_DAG, 
    generate_LSCM_from_mixed_graph,
    get_symbols_from_bi_edges,
    get_symbols_from_di_edges,
    get_symbols_from_nodes, 
    evaluate_LSCM, 
    convert_to_eqn_array_latex,
    convert_to_latex
)
import networkx as nx
from y0.graph import NxMixedGraph
import sympy as sy
import numpy as np

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
    assert expected == actual


def test_read_dag_file_IOerror():
    """Test reading in a .dag file with an invalid file path."""
    fpath = "./tests/non_existant_graph.dag"
    expected = None
    actual = read_dag_file(fpath)
    assert expected == actual

# TODO: update test to use fixture
def test_dagitty_to_mixed_graph():
    """Tests if daggity is converted into an NxMixedGraph correctly."""
    def mixed_graphs_equal(G1: NxMixedGraph, G2: NxMixedGraph) -> bool:
        """Tests if two mixed graphs are equal"""
        if nx.utils.graphs_equal(G1.undirected, G2.undirected) and nx.utils.graphs_equal(
            G1.directed, G2.directed
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
    assert mixed_graphs_equal(mixed_graph1, mixed_graph2) == True



def test_dagitty_to_digraph():
    """Tests if daggity is converted into a directed graph correctly."""
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
    assert nx.utils.graphs_equal(mixed_graph1, mixed_graph2) == True
  


def test_generate_LSCM_from_DAG():
    """Tests the linear structural causal model generated from a directed acyclic graph."""
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('B', 'C')]) 
    expected_equations = {
        sy.Symbol('A'): sy.Symbol('epsilon_A'),
        sy.Symbol('B'): sy.Symbol('beta_A_->B') * sy.Symbol('A') + sy.Symbol('epsilon_B'),
        sy.Symbol('C'): sy.Symbol('beta_B_->C') * sy.Symbol('B') + sy.Symbol('epsilon_C'),
    }
    actual_equations = generate_LSCM_from_DAG(G)
    for node in expected_equations:  # symbolic equality
        assert sy.simplify(actual_equations[node] - expected_equations[node]) == 0


def test_generate_LSCM_from_mixed_graph():
    """Tests the linear structural causal model generated from an NxMixedGraph."""

    edges = [("A", "B"), ("B", "C")]
    bi_edges = [("A", "B")]
    G = NxMixedGraph.from_str_edges(directed=edges, undirected=bi_edges)
    expected_equations = {
        sy.Symbol('A'): sy.Symbol('epsilon_A') + sy.Symbol('gamma_A_<->B'),
        sy.Symbol('B'): sy.Symbol('beta_A_->B') * sy.Symbol('A') + sy.Symbol('epsilon_B') + sy.Symbol('gamma_A_<->B'),
        sy.Symbol('C'): sy.Symbol('beta_B_->C') * sy.Symbol('B') + sy.Symbol('epsilon_C'),
    }
    actual_equations = generate_LSCM_from_mixed_graph(G)
    for node in expected_equations:  # symbolic equality
        assert sy.simplify(actual_equations[node] - expected_equations[node]) == 0
  


def test_get_symbols_from_bi_edges():
    """Tests if symbols are gotten from bidirectional edges correctly."""

    edges = [("A", "B"), ("B", "C")]
    bi_edges = [("A", "B")]
    G = NxMixedGraph.from_str_edges(directed=edges, undirected=bi_edges)
    expected_symbols = {
        ('A', 'B'): sy.Symbol('gamma_A_<->B'),
    }
    actual_symbols = get_symbols_from_bi_edges(G)
    assert actual_symbols == expected_symbols
    


def test_get_symbols_from_di_edges():
    """Tests if symbols are gotten from directional edges correctly."""

    edges = [("A", "B"), ("B", "C")]
    bi_edges = [("A", "B")]
    G = NxMixedGraph.from_str_edges(directed=edges, undirected=bi_edges)
    expected_symbols = {
        ("A", "B"): sy.Symbol("beta_A_->B"),
        ("B", "C"): sy.Symbol("beta_B_->C"),
    }
    actual_symbols = get_symbols_from_di_edges(G)
    assert actual_symbols == expected_symbols


def test_get_symbols_from_nodes():
    """Tests if symbols are gotten from symbols correctly."""

    edges = [("A", "B"), ("B", "C")]
    bi_edges = [("A", "B")]
    G = NxMixedGraph.from_str_edges(directed=edges, undirected=bi_edges)
    expected_symbols = {
        "A": sy.Symbol("epsilon_A"),
        "B": sy.Symbol("epsilon_B"),
        "C": sy.Symbol("epsilon_C"),
    }
    actual_symbols = get_symbols_from_nodes(G)
    assert actual_symbols == expected_symbols


def test_evaluate_LSCM():
    """Tests LSCM evaluation given a set of parameters."""

    # use fixture
    edges = [("A", "B"), ("B", "C")]
    bi_edges = [("A", "B")]
    G = NxMixedGraph.from_str_edges(directed=edges, undirected=bi_edges)
    LSCM_dict = generate_LSCM_from_mixed_graph(G)
    epsilon_symbols = get_symbols_from_nodes(G)
    beta_symbols = get_symbols_from_di_edges(G)
    gamma_symbols = get_symbols_from_bi_edges(G)

    epsilon_values = {epsilon: 1.0 for epsilon in epsilon_symbols.values()}
    beta_values = {beta: 1.0 for beta in beta_symbols.values()}
    gamma_values = {gamma: 1.0 for gamma in gamma_symbols.values()}

    param_dict = {**epsilon_values, **beta_values, **gamma_values}

    expected_symbols = {
        'A': sy.core.numbers.Rational(2.0),
        'B': sy.core.numbers.Rational(4.0),
        'C': sy.core.numbers.Rational(5.0),
    }
    actual_symbols = evaluate_LSCM(LSCM_dict, param_dict)
    print(actual_symbols)
    assert actual_symbols == expected_symbols


def test_convert_to_latex():
    """Test that LSCM is correctly converted into latex."""

    # use fixture
    edges = [("A", "B")]
    G = NxMixedGraph.from_str_edges(directed=edges)
    LSCM_dict = generate_LSCM_from_mixed_graph(G)
    expected = r"$$A = \epsilon_{A}$$" + "\n " + r"$$B = A \beta_{A ->B} + \epsilon_{B}$$"
    actual = convert_to_latex(LSCM_dict)
    assert actual == expected


def test_convert_to_eqn_array_latex():
    """Test that LSCM is correctly converted into latex equation array."""
    # use fixture
    edges = [("A", "B")]
    G = NxMixedGraph.from_str_edges(directed=edges)
    LSCM_dict = generate_LSCM_from_mixed_graph(G)
    expected = r"$$ \begin{array}{rcl}A &=& \epsilon_{A}\\ B &=& A \beta_{A ->B} + \epsilon_{B}\end{array}$$"
    actual = convert_to_eqn_array_latex(LSCM_dict)
    assert actual == expected


def test_generate_synthetic_data_from_LSCM():
    """"""
    # given a set of parameters, evaluate LSCM (with noise)
    raise NotImplementedError


def test_regress_LSCM():
    """"""
    # single door criterion
    raise NotImplementedError
