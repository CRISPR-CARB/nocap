"""Test functions for the scm module."""

from nocap import dagitty_to_dot


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
