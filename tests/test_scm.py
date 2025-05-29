"""Test functions for the scm module."""

import networkx as nx
import pandas as pd

# import numpy as np
import sympy as sy
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork

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
    read_dag_file,
)
from nocap.scm import (
    compile_lgbn_from_lscm,
    create_dag_from_lscm,
    create_lgbn_from_dag,
    estimate_ate,
    fit_model,
    simulate_data_with_outliers,
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


def test_create_lgbn_from_dag_simple():
    """Test create_lgbn_from_dag with a simple DAG."""

    # Simple DAG: A -> B
    dag = nx.DiGraph()
    dag.add_edge("A", "B")
    model = create_lgbn_from_dag(dag)

    assert isinstance(model, LinearGaussianBayesianNetwork)
    assert set(model.nodes()) == {"A", "B"}
    assert set(model.edges()) == {("A", "B")}
    cpds = {cpd.variable: cpd for cpd in model.get_cpds()}
    assert "A" in cpds and "B" in cpds
    assert isinstance(cpds["A"], LinearGaussianCPD)
    assert isinstance(cpds["B"], LinearGaussianCPD)
    # A has no parents, so evidence should be None or empty
    assert cpds["A"].evidence is None or cpds["A"].evidence == []
    # B has A as parent
    assert cpds["B"].evidence == ["A"]
    # Check model validity
    assert model.check_model() is True


def test_create_lgbn_from_dag_disconnected():
    """Test create_lgbn_from_dag with a disconnected DAG."""

    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C"])
    model = create_lgbn_from_dag(dag)
    assert set(list(model.nodes())) == {"A", "B", "C"}
    assert list(model.edges()) == []
    assert model.check_model() is True


def test_create_lgbn_from_dag_cycle_raises():
    """Test create_lgbn_from_dag raises error on cyclic graph."""
    dag = nx.DiGraph()
    dag.add_edges_from([("A", "B"), ("B", "A")])
    try:
        create_lgbn_from_dag(dag)
        assert False, "Should raise an exception for cyclic graph"
    except Exception:
        pass  # Expected


def test_simulate_data_with_outliers_basic():
    """Test simulate_data_with_outliers returns correct shape and non-negative values."""

    # Simple DAG: A -> B
    dag = nx.DiGraph()
    dag.add_edge("A", "B")
    num_samples = 200
    outlier_fraction = 0.1
    outlier_magnitude = 50

    data = simulate_data_with_outliers(
        create_lgbn_from_dag(dag),
        num_samples=num_samples,
        outlier_fraction=outlier_fraction,
        outlier_magnitude=outlier_magnitude,
    )

    # Check shape
    assert data.shape[0] == num_samples
    assert set(data.columns) == {"A", "B"}

    # Check all values are non-negative
    assert (data.values >= 0).all()

    # Check outlier rows are present (at least one value much larger than median)
    num_outliers = int(outlier_fraction * num_samples)
    # Outliers should be present in the data
    assert (data > (data.median() * outlier_magnitude * 0.5)).any(axis=None)

    # Check that the number of outlier rows is as expected (allowing for possible overlap)
    outlier_rows = (data > (data.median() * outlier_magnitude * 0.5)).any(axis=1)
    assert outlier_rows.sum() >= num_outliers // 2  # allow for overlap


def test_simulate_data_with_outliers_invalid_backend():
    """Test simulate_data_with_outliers raises ValueError for unsupported backend."""

    dag = nx.DiGraph()
    dag.add_edge("A", "B")
    try:
        simulate_data_with_outliers(dag, backend="invalid")
        assert False, "Should raise ValueError for unsupported backend"
    except ValueError:
        pass


def test_simulate_data_with_outliers_type_check():
    """Test simulate_data_with_outliers raises AssertionError for wrong model type."""

    # Not a DiGraph
    not_a_dag = {"A": ["B"]}
    try:
        simulate_data_with_outliers(not_a_dag)
        assert False, "Should raise AssertionError for wrong model type"
    except AssertionError:
        pass


def test_fit_model_pgmpy_basic():
    """Test fit_model with pgmpy backend on a simple DAG and data."""
    dag = nx.DiGraph()
    dag.add_edge("A", "B")
    # Simulate some data
    data = pd.DataFrame(
        {
            "A": [0.1, 0.2, 0.3, 0.4],
            "B": [0.5, 0.6, 0.7, 0.8],
        }
    )

    model = fit_model(LinearGaussianBayesianNetwork(dag), data, backend="pgmpy", method="mle")
    # Model should be a LinearGaussianBayesianNetwork
    assert isinstance(model, LinearGaussianBayesianNetwork)
    # Model should have the correct nodes and edges
    assert set(model.nodes()) == {"A", "B"}
    assert set(model.edges()) == {("A", "B")}


def test_fit_model_pgmpy_invalid_type():
    """Test fit_model raises AssertionError if model is not a DiGraph."""
    not_a_dag = {"A": ["B"]}
    data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    try:
        fit_model(not_a_dag, data, backend="pgmpy")
        assert False, "Should raise AssertionError for wrong model type"
    except AssertionError:
        pass


def test_fit_model_invalid_backend():
    """Test fit_model raises ValueError for unsupported backend."""
    dag = nx.DiGraph()
    dag.add_edge("A", "B")
    data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    try:
        fit_model(dag, data, backend="invalid")
        assert False, "Should raise ValueError for unsupported backend"
    except ValueError:
        pass


def test_estimate_ate_pgmpy_type_check():
    """Test estimate_ate raises AssertionError if model is not a DiGraph."""
    not_a_dag = {"A": ["B"]}
    data = pd.DataFrame({"A": [0, 1], "B": [1, 2]})
    try:
        estimate_ate(not_a_dag, data, "A", "B", backend="pgmpy")
        assert False, "Should raise AssertionError for wrong model type"
    except AssertionError:
        pass


def test_estimate_ate_invalid_backend():
    """Test estimate_ate raises ValueError for unsupported backend."""
    dag = nx.DiGraph()
    dag.add_edge("A", "B")
    data = pd.DataFrame({"A": [0, 1], "B": [1, 2]})
    try:
        estimate_ate(dag, data, "A", "B", backend="invalid")
        assert False, "Should raise ValueError for unsupported backend"
    except ValueError:
        pass


def test_compile_lgbn_from_lscm_simple():
    """Test compile_lgbn_from_lscm with a simple LSCM."""

    # LSCM for A -> B
    lscm = {
        sy.Symbol("A"): sy.Symbol("epsilon_A"),
        sy.Symbol("B"): sy.Symbol("beta_A_->B") * sy.Symbol("A") + sy.Symbol("epsilon_B"),
    }
    model = compile_lgbn_from_lscm(lscm)
    assert isinstance(model, LinearGaussianBayesianNetwork)
    assert set(model.nodes()) == {"A", "B"}
    assert set(model.edges()) == {("A", "B")}
    assert model.check_model() is True


def test_compile_lgbn_from_lscm_disconnected():
    """Test compile_lgbn_from_lscm with a disconnected LSCM."""

    lscm = {
        sy.Symbol("A"): sy.Symbol("epsilon_A"),
        sy.Symbol("B"): sy.Symbol("epsilon_B"),
        sy.Symbol("C"): sy.Symbol("epsilon_C"),
    }
    model = compile_lgbn_from_lscm(lscm)
    assert isinstance(model, LinearGaussianBayesianNetwork)
    assert set(model.nodes()) == {"A", "B", "C"}
    assert set(model.edges()) == set()
    assert model.check_model() is True


def test_compile_lgbn_from_lscm_cycle_raises():
    """Test compile_lgbn_from_lscm raises error on cyclic LSCM."""

    # Cyclic: A -> B, B -> A
    lscm = {
        sy.Symbol("A"): sy.Symbol("beta_B_->A") * sy.Symbol("B") + sy.Symbol("epsilon_A"),
        sy.Symbol("B"): sy.Symbol("beta_A_->B") * sy.Symbol("A") + sy.Symbol("epsilon_B"),
    }
    try:
        compile_lgbn_from_lscm(lscm)
        assert False, "Should raise an exception for cyclic LSCM"
    except Exception:
        pass  # Expected


def test_create_dag_from_lscm_simple():
    """Test create_dag_from_lscm with a simple LSCM (A -> B -> C)."""
    lscm = {
        sy.Symbol("A"): sy.Symbol("epsilon_A"),
        sy.Symbol("B"): sy.Symbol("beta_A_->B") * sy.Symbol("A") + sy.Symbol("epsilon_B"),
        sy.Symbol("C"): sy.Symbol("beta_B_->C") * sy.Symbol("B") + sy.Symbol("epsilon_C"),
    }
    dag = create_dag_from_lscm(lscm)
    assert isinstance(dag, nx.DiGraph)
    assert set(dag.nodes()) == {"A", "B", "C"}
    assert set(dag.edges()) == {("A", "B"), ("B", "C")}
    assert nx.is_directed_acyclic_graph(dag)


def test_create_dag_from_lscm_disconnected():
    """Test create_dag_from_lscm with disconnected nodes (no edges)."""
    lscm = {
        sy.Symbol("A"): sy.Symbol("epsilon_A"),
        sy.Symbol("B"): sy.Symbol("epsilon_B"),
        sy.Symbol("C"): sy.Symbol("epsilon_C"),
    }
    dag = create_dag_from_lscm(lscm)
    assert set(dag.nodes()) == {"A", "B", "C"}
    assert set(dag.edges()) == set()
    assert nx.is_directed_acyclic_graph(dag)


def test_create_dag_from_lscm_cycle_raises():
    """Test create_dag_from_lscm raises error on cyclic LSCM."""
    lscm = {
        sy.Symbol("A"): sy.Symbol("beta_B_->A") * sy.Symbol("B") + sy.Symbol("epsilon_A"),
        sy.Symbol("B"): sy.Symbol("beta_A_->B") * sy.Symbol("A") + sy.Symbol("epsilon_B"),
    }
    try:
        create_dag_from_lscm(lscm)
        assert False, "Should raise an exception for cyclic LSCM"
    except AssertionError:
        pass


def test_create_dag_from_lscm_ignores_non_parent_terms():
    """Test create_dag_from_lscm ignores terms that are not parent multiplications."""
    lscm = {
        sy.Symbol("A"): sy.Symbol("epsilon_A") + 2,
        sy.Symbol("B"): sy.Symbol("beta_A_->B") * sy.Symbol("A") + sy.Symbol("epsilon_B") + 3,
    }
    dag = create_dag_from_lscm(lscm)
    assert set(dag.nodes()) == {"A", "B"}
    assert set(dag.edges()) == {("A", "B")}
    assert nx.is_directed_acyclic_graph(dag)


def test_create_dag_from_lscm_multiple_parents():
    """Test create_dag_from_lscm with a node having multiple parents."""
    lscm = {
        sy.Symbol("A"): sy.Symbol("epsilon_A"),
        sy.Symbol("B"): sy.Symbol("epsilon_B"),
        sy.Symbol("C"): (
            sy.Symbol("beta_A_->C") * sy.Symbol("A")
            + sy.Symbol("beta_B_->C") * sy.Symbol("B")
            + sy.Symbol("epsilon_C")
        ),
    }
    dag = create_dag_from_lscm(lscm)
    assert set(dag.nodes()) == {"A", "B", "C"}
    assert set(dag.edges()) == {("A", "C"), ("B", "C")}
    assert nx.is_directed_acyclic_graph(dag)
