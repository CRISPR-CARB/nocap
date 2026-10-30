"""Tests for scBoolSeq-based expression binarization."""

import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from nocap.binarization import binarize_anndata, binarize_expression


class _FakeBinarizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, frame, *, simulation):
        assert simulation is False
        self.columns = frame.columns
        self.index = frame.index
        return self

    def binarize(self, frame):
        return pd.DataFrame(
            [[0.0, np.nan], [1.0, 0.0]], index=self.index, columns=self.columns
        )


def _install_fake_scboolseq(monkeypatch):
    module = ModuleType("scboolseq")
    module.scBoolSeq = _FakeBinarizer
    monkeypatch.setitem(sys.modules, "scboolseq", module)


def test_binarize_expression_preserves_labels_and_unknowns(monkeypatch):
    _install_fake_scboolseq(monkeypatch)
    expression = pd.DataFrame(
        [[1, 2], [3, 4]], index=["cell-a", "cell-b"], columns=["gene-x", "gene-y"]
    )

    result = binarize_expression(expression)

    assert result.index.equals(expression.index)
    assert result.columns.equals(expression.columns)
    assert np.isnan(result.loc["cell-a", "gene-y"])
    pd.testing.assert_frame_equal(expression, pd.DataFrame(
        [[1, 2], [3, 4]], index=["cell-a", "cell-b"], columns=["gene-x", "gene-y"]
    ))


@pytest.mark.parametrize(
    ("unknown", "expected"), [("off", 0.0), ("on", 1.0)]
)
def test_binarize_expression_maps_unknowns(monkeypatch, unknown, expected):
    _install_fake_scboolseq(monkeypatch)
    result = binarize_expression(np.ones((2, 2)), unknown=unknown)

    assert result[0, 1] == expected
    assert isinstance(result, np.ndarray)


def test_binarize_expression_can_reject_unknowns(monkeypatch):
    _install_fake_scboolseq(monkeypatch)

    with pytest.raises(ValueError, match="unknown states"):
        binarize_expression(np.ones((2, 2)), unknown="error")


@pytest.mark.parametrize(
    "expression", [np.ones(3), np.array([[1.0, np.inf]])]
)
def test_binarize_expression_rejects_invalid_input(expression):
    with pytest.raises(ValueError):
        binarize_expression(expression)


def test_binarize_expression_allows_duplicate_cell_names(monkeypatch):
    _install_fake_scboolseq(monkeypatch)
    expression = pd.DataFrame(
        [[1, 2], [3, 4]], index=["same-cell", "same-cell"], columns=["gene-x", "gene-y"]
    )

    result = binarize_expression(expression)

    assert result.index.equals(expression.index)


def test_binarize_expression_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "scboolseq", raising=False)
    monkeypatch.setitem(sys.modules, "scboolseq", None)

    with pytest.raises(ImportError, match=r"nocap\[boolean\]"):
        binarize_expression(np.ones((2, 2)))


def test_binarize_anndata_uses_obs_and_var_names(monkeypatch):
    _install_fake_scboolseq(monkeypatch)

    class FakeAnnData:
        def __init__(self):
            self.X = np.ones((2, 2))
            self.obs_names = pd.Index(["cell-a", "cell-b"])
            self.var_names = pd.Index(["gene-x", "gene-y"])
            self.layers = {}

        def copy(self):
            duplicate = FakeAnnData()
            duplicate.X = self.X.copy()
            duplicate.layers = self.layers.copy()
            return duplicate

    module = ModuleType("anndata")
    module.AnnData = FakeAnnData
    monkeypatch.setitem(sys.modules, "anndata", module)
    source = FakeAnnData()

    result = binarize_anndata(source)

    assert "binarized" not in source.layers
    assert result.layers["binarized"].shape == (2, 2)
    assert np.isnan(result.layers["binarized"][0, 1])
