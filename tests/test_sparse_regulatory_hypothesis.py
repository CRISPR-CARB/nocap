"""Property tests for sparse regulatory equilibrium solvers."""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from nocap.sparse_regulatory import SparseRegulatoryLayer

FINITE = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
CONTRACTIVE = st.floats(
    min_value=-0.5,
    max_value=0.5,
    allow_nan=False,
    allow_infinity=False,
)


def _cycle_layer(weights: list[float], *, normalized: bool = False) -> SparseRegulatoryLayer:
    """Build a three-gene directed cycle with generated edge weights."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    return SparseRegulatoryLayer(
        3,
        edge_index,
        torch.tensor(weights, dtype=torch.double),
        stability_bound=0.6,
        weight_mode="normalized" if normalized else "raw",
    )


@settings(max_examples=30, deadline=None)
@given(
    weights=st.lists(FINITE, min_size=3, max_size=3),
    left=st.lists(FINITE, min_size=3, max_size=3),
    right=st.lists(FINITE, min_size=3, max_size=3),
    scale=FINITE,
)
def test_sparse_matvec_matches_dense_linearity(weights, left, right, scale):
    """Match dense multiplication and preserve linearity on generated states."""
    layer = _cycle_layer(weights)
    left_state = torch.tensor(left, dtype=torch.double)
    right_state = torch.tensor(right, dtype=torch.double)

    combined = layer.sparse_matvec(left_state + scale * right_state)
    separate = layer.sparse_matvec(left_state) + scale * layer.sparse_matvec(right_state)
    dense = (left_state + scale * right_state) @ layer.dense_adjacency().T

    torch.testing.assert_close(combined, separate, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(combined, dense, atol=1e-12, rtol=1e-12)


@settings(max_examples=30, deadline=None)
@given(
    weights=st.lists(FINITE, min_size=3, max_size=3),
    drive_values=st.lists(FINITE, min_size=3, max_size=3),
    permutation=st.permutations([0, 1, 2]),
)
def test_message_solver_is_gene_permutation_equivariant(weights, drive_values, permutation):
    """Commute fixed message solving with a relabeling of gene indices."""
    layer = _cycle_layer(weights)
    drive = torch.tensor(drive_values, dtype=torch.double)
    permutation = torch.tensor(permutation, dtype=torch.long)

    permuted_edges = permutation[layer.edge_index]
    permuted_layer = SparseRegulatoryLayer(
        3,
        permuted_edges,
        torch.tensor(weights, dtype=torch.double),
        weight_mode="raw",
    )
    permuted_drive = torch.empty_like(drive)
    permuted_drive[permutation] = drive

    original = layer(drive, method="message", num_steps=4).state
    permuted = permuted_layer(permuted_drive, method="message", num_steps=4).state

    torch.testing.assert_close(permuted[permutation], original, atol=1e-12, rtol=1e-12)


@settings(max_examples=30, deadline=None)
@given(weights=st.lists(FINITE, min_size=3, max_size=3))
def test_normalization_preserves_support_and_sign_and_bounds_rows(weights):
    """Preserve sparse support and coefficient signs under row normalization."""
    layer = _cycle_layer(weights, normalized=True)

    effective = layer.effective_weights()

    assert effective.shape == layer.raw_weights.shape
    assert torch.equal(torch.sign(effective), torch.sign(layer.raw_weights))
    assert layer.incoming_row_norms(effective).max().item() <= 0.6 + 1e-12


@settings(max_examples=30, deadline=None)
@given(
    weights=st.lists(CONTRACTIVE, min_size=3, max_size=3),
    drive_values=st.lists(FINITE, min_size=3, max_size=3),
    steps=st.integers(min_value=0, max_value=6),
)
def test_fixed_message_error_respects_contraction_bound(weights, drive_values, steps):
    """Bound fixed-step error by the induced infinity-norm geometric tail."""
    layer = _cycle_layer(weights, normalized=True)
    drive = torch.tensor(drive_values, dtype=torch.double)
    message = layer(drive, method="message", num_steps=steps).state
    equilibrium = layer(drive, method="linalg").state
    alpha = layer.incoming_row_norms().max().item()

    error = (message - equilibrium).abs().max().item()
    bound = alpha ** (steps + 1) * drive.abs().max().item() / (1.0 - alpha)

    assert error <= bound + 1e-12


@settings(max_examples=20, deadline=None)
@given(
    weights=st.lists(FINITE, min_size=3, max_size=3),
    drive_values=st.lists(FINITE, min_size=3, max_size=3),
)
def test_solve_does_not_mutate_inputs_or_registered_topology(weights, drive_values):
    """Leave the drive and fixed topology unchanged after either backend runs."""
    layer = _cycle_layer(weights, normalized=True)
    drive = torch.tensor(drive_values, dtype=torch.double)
    drive_before = drive.clone()
    edges_before = layer.edge_index.clone()

    layer(drive, method="message", num_steps=3)
    layer(drive, method="linalg")

    torch.testing.assert_close(drive, drive_before, rtol=0, atol=0)
    torch.testing.assert_close(layer.edge_index, edges_before, rtol=0, atol=0)
