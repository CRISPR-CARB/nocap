"""Test functions for the scm module."""

from functools import partial

import networkx as nx
import numpy as np
import pandas as pd

# import numpy as np
import sympy as sy
from numpy.random import uniform
from pgmpy.models import BayesianNetwork
from y0.dsl import Variable

# import y0
from y0.graph import NxMixedGraph

from nocap import (
    calibrate_lscm,
    compute_average_treatment_effect,
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
    intervene_on_lscm,
    mixed_graph_to_pgmpy,
    read_dag_file,
    simulate_lscm,
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


def test_simulate_lscm_abc():
    """Tests that simulation for LSCM works as expected for a simple ABC model."""
    # Define the simple ABC graph: A -> B, B -> C
    directed_edges = [("A", "B"), ("B", "C")]
    graph = NxMixedGraph.from_str_edges(directed=directed_edges)

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Setup node generators and edge weights with fixed ranges
    node_generators = {
        Variable(node.name): partial(uniform, low=2.0, high=4.0) for node in graph.directed.nodes()
    }
    edge_weights = {edge: uniform(low=1.0, high=2.0) for edge in graph.directed.edges()}

    # Simulate data
    n_samples = 10000
    df = simulate_lscm(
        graph=graph, node_generators=node_generators, edge_weights=edge_weights, n_samples=n_samples
    )

    # Compute expected summary statistics for noise
    def _compute_noise_stats(generator, n_samples):
        """Compute the mean and stdev."""
        samples = [generator() for _ in range(n_samples)]
        return np.mean(samples), np.std(samples)

    mean_a, std_a = _compute_noise_stats(node_generators[Variable("A")], n_samples)
    mean_b, std_b = _compute_noise_stats(node_generators[Variable("B")], n_samples)
    mean_c, std_c = _compute_noise_stats(node_generators[Variable("C")], n_samples)

    # Direct computation of expected summary statistics
    expected_mean_a, expected_std_a = mean_a, std_a
    expected_mean_b = mean_b + (edge_weights[(Variable("A"), Variable("B"))] * mean_a)
    expected_std_b = np.sqrt(std_b**2 + (edge_weights[(Variable("A"), Variable("B"))] * std_a) ** 2)
    expected_mean_c = mean_c + (edge_weights[(Variable("B"), Variable("C"))] * expected_mean_b)
    expected_std_c = np.sqrt(
        std_c**2 + (edge_weights[(Variable("B"), Variable("C"))] * expected_std_b) ** 2
    )

    # Actual summary statistics from simulated data
    actual_mean_a, actual_std_a = df["A"].mean(), df["A"].std()
    actual_mean_b, actual_std_b = df["B"].mean(), df["B"].std()
    actual_mean_c, actual_std_c = df["C"].mean(), df["C"].std()

    # Check the shape of the dataframe
    assert df.shape == (  # noqa: S101
        n_samples,
        len(graph.directed.nodes()),
    ), "DataFrame shape is incorrect"

    # Check the column names
    expected_columns = sorted([node.name for node in graph.directed.nodes()])
    assert sorted(df.columns) == expected_columns, "DataFrame columns are incorrect"  # noqa: S101

    # Verify summary statistics match
    np.testing.assert_allclose(
        actual_mean_a, expected_mean_a, rtol=0.1, err_msg="Mean mismatch for column A"
    )
    np.testing.assert_allclose(
        actual_std_a, expected_std_a, rtol=0.1, err_msg="Std dev mismatch for column A"
    )

    np.testing.assert_allclose(
        actual_mean_b, expected_mean_b, rtol=0.1, err_msg="Mean mismatch for column B"
    )
    np.testing.assert_allclose(
        actual_std_b, expected_std_b, rtol=0.1, err_msg="Std dev mismatch for column B"
    )

    np.testing.assert_allclose(
        actual_mean_c, expected_mean_c, rtol=0.1, err_msg="Mean mismatch for column C"
    )
    np.testing.assert_allclose(
        actual_std_c, expected_std_c, rtol=0.1, err_msg="Std dev mismatch for column C"
    )


def test_calibrate_lscm_abc():
    """Test that calibration of LSCM works as expected for a simple ABC model."""
    # Define the simple ABC graph: A -> B, B -> C
    directed_edges = [("A", "B"), ("B", "C")]
    graph = NxMixedGraph.from_str_edges(directed=directed_edges)

    # Known edge weights
    manual_edge_weights = {(Variable("A"), Variable("B")): 1.5, (Variable("B"), Variable("C")): 1.2}

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Setup node generators and known edge weights
    node_generators = {
        Variable(node.name): partial(uniform, low=2.0, high=4.0) for node in graph.directed.nodes()
    }
    edge_weights = manual_edge_weights

    # Simulate data
    n_samples = 10000
    df = simulate_lscm(
        graph=graph, node_generators=node_generators, edge_weights=edge_weights, n_samples=n_samples
    )

    # Calibrate the model using the simulated data
    calibrated_weights = calibrate_lscm(graph, df)

    # Check that the calibrated weights are within a tolerance level of the actual weights
    for edge in edge_weights:
        actual_weight = edge_weights[edge]
        calibrated_weight = calibrated_weights[edge]
        np.testing.assert_allclose(
            calibrated_weight, actual_weight, rtol=0.1, err_msg=f"Weight mismatch for edge {edge}"
        )


def test_intervene_on_lscm():
    """Test the intervene_on_lscm function using a specific graph and intervention."""
    # Define the graph
    directed_edges = [
        ("V1", "V2"),
        ("V1", "V4"),
        ("V2", "V5"),
        ("V4", "V5"),
        ("V4", "V6"),
        ("V5", "V6"),
        ("V3", "V5"),
    ]
    graph = NxMixedGraph.from_str_edges(directed=directed_edges)

    # Define node generators and edge weights
    node_generators = {
        Variable(node.name): partial(uniform, low=2.0, high=4.0) for node in graph.directed.nodes()
    }
    edge_weights = {edge: uniform(low=1.0, high=2.0) for edge in graph.directed.edges()}

    # Perform the intervention on node V4 by setting it to 10
    intervention_node = (Variable("V4"), 10.0)
    intervened_lscm = intervene_on_lscm(graph, intervention_node, node_generators, edge_weights)

    intervened_graph = intervened_lscm.graph.directed
    intervened_generators = intervened_lscm.generators
    intervened_weights = intervened_lscm.weights

    # Check the edges of the intervened graph
    expected_edges = [
        (Variable("V1"), Variable("V2")),
        (Variable("V2"), Variable("V5")),
        (Variable("V4"), Variable("V5")),
        (Variable("V4"), Variable("V6")),
        (Variable("V5"), Variable("V6")),
        (Variable("V3"), Variable("V5")),
    ]
    actual_edges = list(intervened_graph.edges())

    assert sorted(actual_edges) == sorted(  # noqa: S101
        expected_edges
    ), f"Edges mismatch: expected {expected_edges}, got {actual_edges}"

    # Check that the node generator for V4 returns 10
    assert (  # noqa: S101
        intervened_generators[Variable("V4")]() == 10.0
    ), "Generator for V4 should return the intervention value 10"

    # Check that the edge weights do not include any incoming edge weight for V4
    removed_edge = ("V1", "V4")
    assert (  # noqa: S101
        removed_edge not in intervened_weights
    ), f"Edge {removed_edge} should have been removed from edge weights"


def test_compute_average_treatment_effect():
    """Tests the compute_average_treatment_effect function using known values."""
    # Define variables
    outcome_variables = [Variable("Y1"), Variable("Y2")]

    # Create example DataFrames for treatments
    data_untreated = {"Y1": [1.0, 1.2, 1.1, 1.3, 1.0], "Y2": [2.0, 2.1, 2.0, 2.1, 2.0]}
    data_treated = {"Y1": [1.5, 1.6, 1.7, 1.8, 1.5], "Y2": [2.5, 2.4, 2.5, 2.4, 2.5]}

    df_untreated = pd.DataFrame(data_untreated)
    df_treated = pd.DataFrame(data_treated)

    # Expected average treatment effects
    expected_ate = {Variable("Y1"): 0.5, Variable("Y2"): 0.42}

    # Compute the average treatment effects
    computed_ate = compute_average_treatment_effect(df_untreated, df_treated, outcome_variables)

    # Validate the computed ATE against expected values
    for variable in outcome_variables:
        expected_value = expected_ate[variable]
        computed_value = computed_ate[variable]
        np.testing.assert_almost_equal(
            computed_value,
            expected_value,
            decimal=2,
            err_msg=f"ATE mismatch for variable {variable.name}: expected {expected_value}, got {computed_value}",
        )


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
