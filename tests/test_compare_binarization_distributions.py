"""Tests for per-gene binary recovery diagnostics and output writing."""

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


def _load_script() -> ModuleType:
    """Load the comparison CLI as a module for direct helper testing."""
    path = Path(__file__).parents[1] / "scripts" / "compare_binarization_distributions.py"
    spec = importlib.util.spec_from_file_location("compare_binarization_distributions", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load comparison CLI module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_per_gene_metrics_exclude_unknown_states():
    """Exclude unknown binary entries from per-gene classification metrics."""
    module = _load_script()
    truth = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=bool)
    binary_expression = np.array([[0, 1], [1, np.nan], [0, 0], [1, 0]], dtype=float)

    metrics = module._per_gene_metrics(truth, binary_expression, ["A", "B"]).set_index("gene")

    assert metrics.loc["A", "known_entry_accuracy"] == 0.5
    assert metrics.loc["A", "unknown_fraction"] == 0.0
    assert metrics.loc["B", "known_entry_accuracy"] == 1.0
    assert metrics.loc["B", "unknown_fraction"] == 0.25
    assert metrics.loc["B", "true_on_fraction"] == 0.5
    assert metrics.loc["B", "recovered_on_fraction"] == 1 / 3


def test_output_writer_creates_csv_and_figure(tmp_path):
    """Write the per-gene CSV and ON-fraction comparison figure."""
    module = _load_script()
    metrics = module._per_gene_metrics(
        np.array([[False, True], [True, False]]),
        np.array([[0.0, 1.0], [1.0, np.nan]]),
        ["A", "B"],
    )

    csv_path, figure_path = module._write_outputs(metrics, tmp_path / "result")

    assert csv_path.exists()
    assert figure_path.exists()
    assert {"gene", "unknown_fraction", "known_entry_accuracy"}.issubset(
        set(pd.read_csv(csv_path).columns)
    )


def test_zero_count_state_is_exposed_as_cli_option():
    """Offer a comparison-specific choice for scBoolSeq zero handling."""
    module = _load_script()
    source = inspect.getsource(module.main)

    assert '"--zero-count-state"' in source
    assert '"off": 0.0' in source
    assert '"unknown": np.nan' in source
