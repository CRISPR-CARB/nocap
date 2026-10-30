"""Generate binary ON/OFF/unknown expression data from gene expression matrices."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

UnknownState = Literal["keep", "off", "on", "error"]
ExpressionMatrix = pd.DataFrame | ArrayLike
BinaryMatrix = pd.DataFrame | NDArray[np.float64]


def _expression_frame(expression: ExpressionMatrix) -> tuple[pd.DataFrame, bool]:
    """Validate input shape and values, returning a numeric DataFrame."""
    is_frame = isinstance(expression, pd.DataFrame)
    if is_frame:
        if not expression.columns.is_unique:
            raise ValueError("expression DataFrame gene labels must be unique")
        frame = expression.copy()
    else:
        values = np.asarray(expression)
        if values.ndim != 2:
            raise ValueError("expression must have shape (n_cells, n_genes)")
        frame = pd.DataFrame(values)

    if frame.empty:
        raise ValueError("expression must contain at least one cell and one gene")
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("expression values must be numeric") from error
    if not np.isfinite(values).all():
        raise ValueError("expression values must be finite")
    return pd.DataFrame(values, index=frame.index, columns=frame.columns), is_frame


def _handle_unknown_states(result: pd.DataFrame, unknown: UnknownState) -> pd.DataFrame:
    """Apply the requested policy to scBoolSeq's NaN/unknown states."""
    if unknown not in {"keep", "off", "on", "error"}:
        raise ValueError("unknown must be one of {'keep', 'off', 'on', 'error'}")
    if unknown == "error" and result.isna().to_numpy().any():
        raise ValueError("scBoolSeq produced unknown states (NaN)")
    if unknown == "off":
        return result.fillna(0.0)
    if unknown == "on":
        return result.fillna(1.0)
    return result


def _validate_unknown_policy(unknown: UnknownState) -> None:
    """Validate the policy for states scBoolSeq leaves undetermined."""
    if unknown not in {"keep", "off", "on", "error"}:
        raise ValueError("unknown must be one of {'keep', 'off', 'on', 'error'}")


def binarize_expression(
    expression: ExpressionMatrix,
    *,
    unknown: UnknownState = "keep",
    binarizer_kwargs: dict[str, Any] | None = None,
) -> BinaryMatrix:
    """Binarize cell-by-gene expression data with scBoolSeq.

    DataFrame row and column labels are preserved. Array-like input returns a
    floating-point NumPy array. scBoolSeq's undetermined values are retained
    as NaN by default; ``unknown`` can instead map them to OFF/ON or raise.

    ``binarizer_kwargs`` is passed directly to ``scBoolSeq``. Supported
    constructor options (with scBoolSeq defaults) are:

    - ``confidence=0.95``: posterior probability required to call a bimodal
      value OFF or ON.
    - ``covariance_type="full"``: Gaussian-mixture covariance model;
      choose ``"full"``, ``"tied"``, ``"diag"``, or ``"spherical"``.
    - ``margin_quantile=0.05`` and ``alpha=0.0``: quantile margin and IQR
      multiplier for unimodal genes.
    - ``zeroinf_binarizer="zero_or_not"``: zero-inflated gene method;
      choose ``"zero_or_not"`` or ``"quantile"``.
    - ``zeroes_are=np.nan``: state assigned to zero counts by the zero-inflated
      binarizer.
    - ``dor_threshold=0.99`` and ``amplitude_threshold=10.0``: criteria for
      discarding genes with high dropout or insufficient amplitude.
    - ``half_life_parameter="MeanNZ"``: expression criterion used for
      simulation metadata. Fitting here uses ``simulation=False``.
    - ``warm_start=True`` and ``require_df=True``: reuse fitted criteria when
      possible and require DataFrame input, respectively.

    For example, pass ``binarizer_kwargs={"confidence": 0.9,
    "covariance_type": "diag"}``. These options control scBoolSeq's
    classification; ``unknown`` separately controls the returned NaN states.
    """
    _validate_unknown_policy(unknown)
    frame, is_frame = _expression_frame(expression)

    try:
        from scboolseq import scBoolSeq
    except ImportError as error:
        raise ImportError(
            "Binarization requires scBoolSeq; install it with `pip install 'nocap[boolean]'`."
        ) from error

    binarizer = scBoolSeq(**(binarizer_kwargs or {}))
    binarizer.fit(frame, simulation=False)
    result = binarizer.binarize(frame)
    if not isinstance(result, pd.DataFrame):
        result = pd.DataFrame(result, index=frame.index, columns=frame.columns)
    else:
        if result.shape != frame.shape:
            raise ValueError("scBoolSeq returned a matrix with an unexpected shape")
        result = pd.DataFrame(result.to_numpy(), index=frame.index, columns=result.columns)
        result = result.reindex(columns=frame.columns)
    result = result.astype(float)

    result = _handle_unknown_states(result, unknown)

    if is_frame:
        return result
    return result.to_numpy()


def binarize_anndata(
    adata,
    *,
    layer: str | None = None,
    output_layer: str = "binarized",
    unknown: UnknownState = "keep",
    binarizer_kwargs: dict[str, object] | None = None,
    copy: bool = True,
):
    """Add scBoolSeq binarized expression to an AnnData layer.

    By default, binarization uses ``adata.X``, stores the result in the
    ``binarized`` layer of a copy, and leaves the source object unchanged.
    Set ``layer`` to binarize an existing layer or ``copy=False`` to modify
    the input object. ``binarizer_kwargs`` is forwarded to the scBoolSeq
    constructor; see :func:`binarize_expression` for supported options.
    """
    try:
        from anndata import AnnData
    except ImportError as error:
        raise ImportError(
            "AnnData support requires anndata; install it with `pip install 'nocap[boolean]'`."
        ) from error
    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an anndata.AnnData instance")
    if not output_layer:
        raise ValueError("output_layer must be a non-empty layer name")
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"AnnData has no layer named {layer!r}")

    matrix = adata.X if layer is None else adata.layers[layer]
    if isinstance(matrix, csr_matrix):  # if sparse matrix
        matrix = matrix.toarray()
    expression = pd.DataFrame(
        np.asarray(matrix), index=adata.obs_names, columns=adata.var_names
    )
    binary = binarize_expression(
        expression, unknown=unknown, binarizer_kwargs=binarizer_kwargs
    )
    result = adata.copy() if copy else adata
    if isinstance(binary, pd.DataFrame):
        result.layers[output_layer] = binary.to_numpy(dtype=float)
    elif isinstance(binary, np.ndarray):
        result.layers[output_layer] = binary
    else:
        raise TypeError(f"Binary data is in unsupported format: {type(binary)}")
    return result


def binarize_h5ad(
    input_path: str | PathLike[str],
    output_path: str | PathLike[str] | None = None,
    *,
    layer: str | None = None,
    output_layer: str = "binarized",
    unknown: UnknownState = "keep",
    binarizer_kwargs: dict[str, object] | None = None,
):
    """Binarize an H5AD file and optionally write the updated AnnData object.

    ``binarizer_kwargs`` is forwarded to the scBoolSeq constructor; see
    :func:`binarize_expression` for supported options.
    """
    try:
        import anndata
    except ImportError as error:
        raise ImportError(
            "H5AD support requires anndata; install it with `pip install 'nocap[boolean]'`."
        ) from error

    result = binarize_anndata(
        anndata.read_h5ad(Path(input_path)),
        layer=layer,
        output_layer=output_layer,
        unknown=unknown,
        binarizer_kwargs=binarizer_kwargs,
        copy=False,
    )
    if output_path is not None:
        result.write_h5ad(output_path)
    return result
