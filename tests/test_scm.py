"""Test functions for the scm module."""

from nocap import dagitty_to_dot, read_dag_file


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


def test_dagitty_to_mixed_graph():
    # def mixed_graphs_equal(G1: NxMixedGraph, G2: NxMixedGraph) -> bool:
    # """Tests if two mixed graphs are equal"""
    # if nx.utils.graphs_equal(G1.undirected, G2.undirected) and nx.utils.graphs_equal(
    #     G1.directed, G2.directed
    # ):
    #     return True
    # else:
    #     return False
    raise NotImplementedError


def test_dagitty_to_digraph():
    raise NotImplementedError


def test_generate_LSCM_from_DAG():
    raise NotImplementedError


def test_generate_LSCM_from_mixed_graph():
    raise NotImplementedError


def test_get_symbols_from_bi_edges():
    raise NotImplementedError


def test_get_symbols_from_di_edges():
    raise NotImplementedError


def test_get_symbols_from_nodes():
    raise NotImplementedError


def test_evaluate_LSCM():
    raise NotImplementedError


def test_convert_to_latex():
    raise NotImplementedError


def test_convert_to_eqn_array_latex():
    raise NotImplementedError


def test_generate_synthetic_data_from_LSCM():
    raise NotImplementedError


def test_regress_LSCM():
    raise NotImplementedError
