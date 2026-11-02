"""Tests for scBoolSeq-based expression binarization."""

import numpy as np
import pandas as pd
import pytest

from nocap import binarization
from nocap.binarization import binarize_anndata, binarize_expression, binarize_h5ad


class _FakeBinarizer:
    """Minimal scBoolSeq substitute for testing binarization behavior."""

    def __init__(self, **kwargs):
        """Store constructor options used by the test."""
        self.kwargs = kwargs

    def fit(self, frame, *, simulation):
        """Capture labels while matching the scBoolSeq fit interface."""
        assert simulation is False
        self.columns = frame.columns
        self.index = frame.index
        return self

    def binarize(self, frame):
        """Return a fixed matrix containing a representative unknown state."""
        return pd.DataFrame([[0.0, np.nan], [1.0, 0.0]], index=self.index, columns=self.columns)


def _install_fake_scboolseq(monkeypatch):
    monkeypatch.setattr(binarization, "scBoolSeq", _FakeBinarizer)


def test_binarize_expression_preserves_labels_and_unknowns(monkeypatch):
    """Preserve DataFrame labels and unknown values without mutating input."""
    _install_fake_scboolseq(monkeypatch)
    expression = pd.DataFrame(
        [[1, 2], [3, 4]], index=["cell-a", "cell-b"], columns=["gene-x", "gene-y"]
    )

    result = binarize_expression(expression)

    assert result.index.equals(expression.index)
    assert result.columns.equals(expression.columns)
    assert np.isnan(result.loc["cell-a", "gene-y"])
    pd.testing.assert_frame_equal(
        expression,
        pd.DataFrame([[1, 2], [3, 4]], index=["cell-a", "cell-b"], columns=["gene-x", "gene-y"]),
    )


def test_binarize_expression_can_return_result_and_fitted_binarizer(monkeypatch):
    """Return the NumPy result alongside its fitted binarizer when requested."""
    _install_fake_scboolseq(monkeypatch)

    result, binarizer = binarize_expression(
        np.ones((2, 2)), binarizer_kwargs={"confidence": 0.9}, return_binarizer=True
    )

    assert isinstance(result, np.ndarray)
    assert isinstance(binarizer, _FakeBinarizer)
    assert binarizer.kwargs == {"confidence": 0.9}
    assert binarizer.columns.equals(pd.Index([0, 1]))


@pytest.mark.parametrize(("unknown", "expected"), [("off", 0.0), ("on", 1.0)])
def test_binarize_expression_maps_unknowns(monkeypatch, unknown, expected):
    """Map unknown values according to the selected ON/OFF policy."""
    _install_fake_scboolseq(monkeypatch)
    result = binarize_expression(np.ones((2, 2)), unknown=unknown)

    assert result[0, 1] == expected
    assert isinstance(result, np.ndarray)


def test_binarize_expression_can_reject_unknowns(monkeypatch):
    """Raise when the selected policy rejects unknown output states."""
    _install_fake_scboolseq(monkeypatch)

    with pytest.raises(ValueError, match="unknown states"):
        binarize_expression(np.ones((2, 2)), unknown="error")


@pytest.mark.parametrize("expression", [np.ones(3), np.array([[1.0, np.inf]])])
def test_binarize_expression_rejects_invalid_input(expression):
    """Reject inputs with invalid shape or non-finite values."""
    with pytest.raises(ValueError):
        binarize_expression(expression)


def test_binarize_expression_allows_duplicate_cell_names(monkeypatch):
    """Allow duplicate cell labels while preserving their row order."""
    _install_fake_scboolseq(monkeypatch)
    expression = pd.DataFrame(
        [[1, 2], [3, 4]], index=["same-cell", "same-cell"], columns=["gene-x", "gene-y"]
    )

    result = binarize_expression(expression)

    assert result.index.equals(expression.index)


def test_binarize_anndata_uses_obs_and_var_names(monkeypatch):
    """Use AnnData labels and write output to a copied object's layer."""
    _install_fake_scboolseq(monkeypatch)

    class FakeAnnData:
        """Small AnnData substitute exposing the attributes under test."""

        def __init__(self):
            """Initialize expression values and their observation labels."""
            self.X = np.ones((2, 2))
            self.obs_names = pd.Index(["cell-a", "cell-b"])
            self.var_names = pd.Index(["gene-x", "gene-y"])
            self.layers = {}

        def copy(self):
            """Return a separate object with copied expression and layers."""
            duplicate = FakeAnnData()
            duplicate.X = self.X.copy()
            duplicate.layers = self.layers.copy()
            return duplicate

    monkeypatch.setattr(binarization, "AnnData", FakeAnnData)
    source = FakeAnnData()

    default_result = binarize_anndata(source)
    result, binarizer = binarize_anndata(source, return_binarizer=True)

    assert "binarized" not in source.layers
    assert default_result.layers["binarized"].shape == (2, 2)
    assert result.layers["binarized"].shape == (2, 2)
    assert np.isnan(result.layers["binarized"][0, 1])
    assert isinstance(binarizer, _FakeBinarizer)


def test_binarize_h5ad_returns_binarizer_and_writes_result(monkeypatch, tmp_path):
    """Return the fitted binarizer and write the resulting H5AD file."""

    class FakeAnnData:
        """Track H5AD writes without requiring a real AnnData object."""

        def __init__(self):
            """Initialize the recorded output path."""
            self.written_path = None

        def write_h5ad(self, path):
            """Record the path requested by the H5AD writer."""
            self.written_path = path

    monkeypatch.setattr(binarization, "AnnData", FakeAnnData)
    source = FakeAnnData()
    monkeypatch.setattr(binarization, "read_h5ad", lambda path: source)

    binarizer = object()
    monkeypatch.setattr(
        binarization, "binarize_anndata", lambda adata, **kwargs: (adata, binarizer)
    )

    output_path = tmp_path / "binary.h5ad"
    result, returned_binarizer = binarize_h5ad("input.h5ad", output_path, return_binarizer=True)

    assert result is source
    assert returned_binarizer is binarizer
    assert source.written_path == output_path
