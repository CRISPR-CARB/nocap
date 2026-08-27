"""Tests for sparse regulatory equilibrium solvers."""

from __future__ import annotations

import pytest
import torch

from nocap.sparse_regulatory import SparseRegulatoryLayer


def _two_gene_layer(
    *, weight_mode: str = "raw", weights: tuple[float, float] = (0.2, -0.1)
) -> SparseRegulatoryLayer:
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    init_weights = torch.tensor(weights, dtype=torch.double)
    return SparseRegulatoryLayer(
        2,
        edge_index,
        init_weights,
        weight_mode=weight_mode,
    )


def test_sparse_matvec_uses_regulator_to_target_orientation():
    layer = _two_gene_layer()
    state = torch.tensor([[3.0, 5.0]], dtype=torch.double)

    actual = layer.sparse_matvec(state)

    expected = torch.tensor([[-0.5, 0.6]], dtype=torch.double)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("num_steps", [0, 1, 2, 5])
def test_fixed_message_matches_dense_recurrence(num_steps):
    layer = _two_gene_layer()
    drive = torch.tensor([[1.0, 2.0], [-2.0, 0.5]], dtype=torch.double)
    adjacency = layer.dense_adjacency()
    expected = drive
    for _ in range(num_steps):
        expected = drive + expected @ adjacency.T

    result = layer(drive, method="message", num_steps=num_steps)

    torch.testing.assert_close(result.state, expected)
    assert result.steps == num_steps
    assert result.method == "message"


def test_message_converges_to_linalg_on_stable_cycle():
    layer = _two_gene_layer()
    drive = torch.tensor([[1.0, 2.0]], dtype=torch.double)

    message = layer(drive, method="message", num_steps=20)
    linalg = layer(drive, method="linalg")

    torch.testing.assert_close(message.state, linalg.state, atol=1e-12, rtol=1e-12)
    assert message.residual.item() < 1e-12
    assert linalg.residual.item() < 1e-12


def test_normalization_bounds_each_target_row_and_preserves_signs():
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 1]], dtype=torch.long)
    raw_weights = torch.tensor([0.8, -0.7, -0.2], dtype=torch.double)
    layer = SparseRegulatoryLayer(
        3,
        edge_index,
        raw_weights,
        stability_bound=0.6,
        weight_mode="normalized",
    )

    effective = layer.effective_weights()
    row_norms = layer.incoming_row_norms(effective)

    assert row_norms.max().item() <= 0.6 + 1e-12
    assert torch.equal(torch.sign(effective), torch.sign(raw_weights))
    torch.testing.assert_close(effective[2], raw_weights[2])


def test_graph_polarity_hard_constrains_effective_weight_signs():
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 1]], dtype=torch.long)
    edge_signs = torch.tensor([1, -1, 0], dtype=torch.int8)
    layer = SparseRegulatoryLayer(
        3,
        edge_index,
        torch.tensor([-10.0, -10.0, -0.4]),
        edge_signs=edge_signs,
        weight_mode="raw",
    )

    effective = layer.effective_weights()

    assert effective[0] > 0
    assert effective[1] < 0
    assert effective[2] == layer.raw_weights[2]


def test_tolerance_mode_reports_convergence():
    layer = _two_gene_layer()
    drive = torch.tensor([1.0, 2.0], dtype=torch.double)

    result = layer(
        drive,
        method="message",
        stopping="tolerance",
        max_steps=50,
        atol=1e-12,
        rtol=0.0,
    )

    assert result.converged
    assert 0 < result.steps < 50
    assert result.residual.item() <= 1e-12


def test_tolerance_mode_rejects_unstable_raw_weights():
    layer = _two_gene_layer(weights=(2.0, 1.0))
    drive = torch.ones(2, dtype=torch.double)

    with pytest.raises(ValueError, match="row norm"):
        layer(drive, method="message", stopping="tolerance")


def test_linalg_can_solve_nonsingular_unstable_system_without_calling_it_stable():
    layer = _two_gene_layer(weights=(2.0, 1.0))
    drive = torch.ones(2, dtype=torch.double)

    result = layer(drive, method="linalg")

    torch.testing.assert_close(result.residual, torch.tensor(0.0, dtype=torch.double))
    assert result.converged
    assert not result.stable


def test_linalg_dense_limit_is_checked_before_solve():
    edge_index = torch.empty((2, 0), dtype=torch.long)
    layer = SparseRegulatoryLayer(513, edge_index, max_dense_genes=512)

    with pytest.raises(ValueError, match="limited to 512"):
        layer(torch.zeros(513), method="linalg")


def test_empty_graph_is_identity_for_arbitrary_leading_dimensions():
    edge_index = torch.empty((2, 0), dtype=torch.long)
    layer = SparseRegulatoryLayer(4, edge_index)
    drive = torch.randn(2, 3, 4)

    result = layer(drive, method="message", num_steps=5)

    torch.testing.assert_close(result.state, drive, rtol=0, atol=0)
    torch.testing.assert_close(result.residual, torch.zeros(2, 3), rtol=0, atol=0)


@pytest.mark.parametrize(
    "edge_index,match",
    [
        (torch.tensor([[0], [0]], dtype=torch.long), "self-loop"),
        (torch.tensor([[0, 0], [1, 1]], dtype=torch.long), "duplicate"),
        (torch.tensor([[0], [2]], dtype=torch.long), "valid gene indices"),
    ],
)
def test_invalid_topology_is_rejected(edge_index, match):
    with pytest.raises(ValueError, match=match):
        SparseRegulatoryLayer(2, edge_index)


def test_message_and_linalg_gradients_agree():
    drive = torch.tensor([[1.0, 2.0]], dtype=torch.double, requires_grad=True)
    message_layer = _two_gene_layer()
    linalg_layer = _two_gene_layer()

    message_loss = message_layer(drive, method="message", num_steps=20).state.square().sum()
    message_drive_grad, message_weight_grad = torch.autograd.grad(
        message_loss, (drive, message_layer.raw_weights)
    )

    linalg_loss = linalg_layer(drive, method="linalg").state.square().sum()
    linalg_drive_grad, linalg_weight_grad = torch.autograd.grad(
        linalg_loss, (drive, linalg_layer.raw_weights)
    )

    torch.testing.assert_close(message_drive_grad, linalg_drive_grad, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(
        message_weight_grad, linalg_weight_grad, atol=1e-10, rtol=1e-10
    )
