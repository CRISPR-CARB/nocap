import pytest

from nocap import daggity_to_dot  # Replace 'your_module' with the actual module name


def test_basic_conversion():
    """Test basic conversion of daggity to dot format."""
    # Test basic conversion
    daggity_string = "dag { A -> B; B -> C; C -> A; }"
    expected_dot = "digraph { A -> B; B -> C; C -> A; }"
    assert daggity_to_dot(daggity_string) == expected_dot


def test_with_latent_variables():
    """Test conversion of daggity to dot format with latent variables."""
    # Test conversion with latent variables
    daggity_string = "dag { latent,* A -> B; B -> C; C -> A; }"
    expected_dot = 'digraph { observed="no", A -> B; B -> C; C -> A; }'
    assert daggity_to_dot(daggity_string) == expected_dot


# Additional test functions here for other scenarios, e.g., with 'outcome', 'adjusted', 'exposure', etc.
