"""Tests for estimate_path_coefficient_for_edge in src/nocap/cyclic_single_door.py."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from nocap.cyclic_single_door import estimate_path_coefficient_for_edge


def test_estimate_path_coefficient_identifiable_no_adj_set():
    """Test estimation on an identifiable edge without providing an adjustment set."""
    # Z -> X -> Y
    # True model:
    # Z = N_Z
    # X = 2 * Z + N_X
    # Y = 3 * X + N_Y
    # The path coefficient for X -> Y is 3.0.
    # No confounders, so adjustment set is empty.
    g = nx.DiGraph()
    g.add_edges_from([("Z", "X"), ("X", "Y")])

    # Generate some synthetic data
    import numpy as np

    np.random.seed(42)
    n_samples = 200
    z = np.random.normal(0, 1, n_samples)
    x = 2.0 * z + np.random.normal(0, 0.5, n_samples)
    y = 3.0 * x + np.random.normal(0, 0.5, n_samples)

    data = pd.DataFrame({"Z": z, "X": x, "Y": y})

    result = estimate_path_coefficient_for_edge(g, "X", "Y", data)
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 4

    path_coef, _, residual_var, t_val = result
    # Path coefficient should be close to 3.0
    assert pytest.approx(path_coef, abs=0.1) == 3.0
    assert residual_var > 0
    assert abs(t_val) > 10  # Highly significant


def test_estimate_path_coefficient_identifiable_with_confounder():
    """Test estimation on an edge with a confounder (Z -> X -> Y, Z -> Y)."""
    # True model:
    # Z = N_Z
    # X = 1.5 * Z + N_X
    # Y = 2.5 * X + 4.0 * Z + N_Y
    # The path coefficient for X -> Y is 2.5.
    # Adjustment set must contain Z.
    g = nx.DiGraph()
    g.add_edges_from([("Z", "X"), ("X", "Y"), ("Z", "Y")])

    import numpy as np

    np.random.seed(42)
    n_samples = 500
    z = np.random.normal(0, 1, n_samples)
    x = 1.5 * z + np.random.normal(0, 0.5, n_samples)
    y = 2.5 * x + 4.0 * z + np.random.normal(0, 0.5, n_samples)

    data = pd.DataFrame({"Z": z, "X": x, "Y": y})

    # 1. Without pre-computed adjustment set (should auto-compute {Z})
    result = estimate_path_coefficient_for_edge(g, "X", "Y", data)
    assert result is not None
    path_coef, residual_var, t_val = result
    assert pytest.approx(path_coef, abs=0.1) == 2.5

    # 2. With pre-computed adjustment set
    result_with_adj = estimate_path_coefficient_for_edge(
        g, "X", "Y", data, adj_set=frozenset(["Z"])
    )
    assert result_with_adj is not None
    assert pytest.approx(result_with_adj[0], abs=0.1) == 2.5


def test_estimate_path_coefficient_unidentifiable():
    """Test that None is returned for an unidentifiable edge (e.g., in a cycle)."""
    # X -> Y -> X (same SCC, unidentifiable)
    g = nx.DiGraph()
    g.add_edges_from([("X", "Y"), ("Y", "X")])

    import numpy as np

    data = pd.DataFrame({"X": np.random.normal(0, 1, 100), "Y": np.random.normal(0, 1, 100)})

    result = estimate_path_coefficient_for_edge(g, "X", "Y", data)
    assert result is None
