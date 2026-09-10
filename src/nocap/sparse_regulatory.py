"""Differentiable equilibrium solvers for sparse regulatory networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

SolverMethod = Literal["message", "linalg"]
StoppingMode = Literal["fixed", "tolerance"]
WeightMode = Literal["normalized", "raw"]


@dataclass(frozen=True)
class EquilibriumResult:
    """Result and diagnostics from a regulatory equilibrium solve."""

    state: Tensor
    residual: Tensor
    method: SolverMethod
    converged: bool
    steps: int
    max_row_norm: Tensor
    stable: bool


class SparseRegulatoryLayer(nn.Module):
    """Solve ``g = drive + A g`` over a fixed regulator-to-target graph.

    ``edge_index[0]`` contains regulator indices and ``edge_index[1]``
    contains target indices, so ``A[target, regulator]`` is the edge weight.
    """

    def __init__(
        self,
        num_genes: int,
        edge_index: Tensor,
        init_weights: Tensor | None = None,
        *,
        edge_signs: Tensor | None = None,
        stability_bound: float = 0.95,
        weight_mode: WeightMode = "normalized",
        max_dense_genes: int = 512,
    ) -> None:
        """Initialize a layer over a validated, fixed edge topology."""
        super().__init__()
        self._validate_topology(num_genes, edge_index)
        self._validate_initial_weights(edge_index, init_weights)
        self._validate_edge_signs(edge_index, edge_signs)
        self._validate_settings(stability_bound, weight_mode, max_dense_genes)

        num_edges = edge_index.shape[1]
        if init_weights is None:
            init_weights = torch.zeros(num_edges, dtype=torch.get_default_dtype())

        self.num_genes = num_genes
        self.stability_bound = stability_bound
        self.weight_mode = weight_mode
        self.max_dense_genes = max_dense_genes
        self.register_buffer("edge_index", edge_index.detach().clone())
        if edge_signs is None:
            edge_signs = torch.zeros(num_edges, dtype=torch.int8, device=edge_index.device)
        self.register_buffer("edge_signs", edge_signs.detach().clone())
        self.raw_weights = nn.Parameter(init_weights.detach().clone())

    @staticmethod
    def _validate_topology(num_genes: int, edge_index: Tensor) -> None:
        if not isinstance(num_genes, int) or isinstance(num_genes, bool) or num_genes <= 0:
            raise ValueError("num_genes must be a positive integer")
        if not isinstance(edge_index, Tensor):
            raise TypeError("edge_index must be a torch.Tensor")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype torch.long")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, num_edges)")
        if edge_index.numel() and (torch.any(edge_index < 0) or torch.any(edge_index >= num_genes)):
            raise ValueError("edge_index endpoints must be valid gene indices")

        regulators, targets = edge_index
        if torch.any(regulators == targets):
            raise ValueError("self-loop edges are not supported")
        if edge_index.shape[1]:
            edge_keys = regulators * num_genes + targets
            if torch.unique(edge_keys).numel() != edge_keys.numel():
                raise ValueError("duplicate directed edges are not supported")

    @staticmethod
    def _validate_initial_weights(edge_index: Tensor, init_weights: Tensor | None) -> None:
        if init_weights is not None:
            if not isinstance(init_weights, Tensor):
                raise TypeError("init_weights must be a torch.Tensor or None")
            if not init_weights.is_floating_point():
                raise TypeError("init_weights must have a floating-point dtype")
            if init_weights.shape != (edge_index.shape[1],):
                raise ValueError("init_weights must have shape (num_edges,)")
            if not torch.isfinite(init_weights).all():
                raise ValueError("init_weights must contain only finite values")
            if init_weights.device != edge_index.device:
                raise ValueError("edge_index and init_weights must be on the same device")

    @staticmethod
    def _validate_edge_signs(edge_index: Tensor, edge_signs: Tensor | None) -> None:
        if edge_signs is None:
            return
        if not isinstance(edge_signs, Tensor):
            raise TypeError("edge_signs must be a torch.Tensor or None")
        if edge_signs.dtype != torch.int8:
            raise TypeError("edge_signs must have dtype torch.int8")
        if edge_signs.shape != (edge_index.shape[1],):
            raise ValueError("edge_signs must have shape (num_edges,)")
        if not torch.all((edge_signs >= -1) & (edge_signs <= 1)):
            raise ValueError("edge_signs entries must be -1, 0, or 1")
        if edge_signs.device != edge_index.device:
            raise ValueError("edge_index and edge_signs must be on the same device")

    @staticmethod
    def _validate_settings(stability_bound: float, weight_mode: str, max_dense_genes: int) -> None:
        if not isinstance(stability_bound, int | float) or isinstance(stability_bound, bool):
            raise TypeError("stability_bound must be a real number")
        if not 0 <= stability_bound < 1:
            raise ValueError("stability_bound must satisfy 0 <= stability_bound < 1")
        if weight_mode not in ("normalized", "raw"):
            raise ValueError("weight_mode must be 'normalized' or 'raw'")
        if (
            not isinstance(max_dense_genes, int)
            or isinstance(max_dense_genes, bool)
            or max_dense_genes <= 0
        ):
            raise ValueError("max_dense_genes must be a positive integer")

    def incoming_row_norms(self, weights: Tensor | None = None) -> Tensor:
        """Return the absolute incoming-weight sum for each target gene."""
        if weights is None:
            weights = self.effective_weights()
        row_norms = weights.new_zeros(self.num_genes)
        return row_norms.index_add(0, self.edge_index[1], weights.abs())

    def effective_weights(self) -> Tensor:
        """Return raw weights or their stability-normalized counterparts."""
        signs = self.edge_signs.to(dtype=self.raw_weights.dtype)
        signed_weights = torch.where(
            signs > 0,
            torch.nn.functional.softplus(self.raw_weights),
            torch.where(
                signs < 0, -torch.nn.functional.softplus(self.raw_weights), self.raw_weights
            ),
        )
        if self.weight_mode == "raw" or self.raw_weights.numel() == 0:
            return signed_weights

        row_norms = self.incoming_row_norms(signed_weights)
        target_norms = row_norms[self.edge_index[1]]
        scales = torch.where(
            target_norms > self.stability_bound,
            self.stability_bound / target_norms.clamp_min(torch.finfo(target_norms.dtype).tiny),
            torch.ones_like(target_norms),
        )
        return signed_weights * scales

    def sparse_matvec(self, state: Tensor, weights: Tensor | None = None) -> Tensor:
        """Apply the regulatory adjacency without materializing a dense matrix."""
        self._validate_drive(state)
        if weights is None:
            weights = self.effective_weights()
        flat_state = state.reshape(-1, self.num_genes)
        messages = flat_state[:, self.edge_index[0]] * weights
        propagated = flat_state.new_zeros(flat_state.shape)
        propagated = propagated.index_add(1, self.edge_index[1], messages)
        return propagated.reshape(state.shape)

    def dense_adjacency(self) -> Tensor:
        """Materialize ``A`` for guarded direct solves and small references."""
        if self.num_genes > self.max_dense_genes:
            raise ValueError(
                f"linalg is limited to {self.max_dense_genes} genes; got {self.num_genes}"
            )
        adjacency = self.raw_weights.new_zeros((self.num_genes, self.num_genes))
        return adjacency.index_put(
            (self.edge_index[1], self.edge_index[0]), self.effective_weights()
        )

    def forward(
        self,
        drive: Tensor,
        *,
        method: SolverMethod = "message",
        stopping: StoppingMode = "fixed",
        num_steps: int = 10,
        max_steps: int = 100,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> EquilibriumResult:
        """Solve for regulatory equilibrium and return common diagnostics."""
        self._validate_drive(drive)
        self._validate_solve_options(method, stopping, num_steps, max_steps, atol, rtol)

        weights = self.effective_weights()
        max_row_norm = self.incoming_row_norms(weights).max()
        stable = bool(max_row_norm.detach() < 1)

        if method == "linalg":
            state = self._solve_linalg(drive)
            residual = self._residual(drive, state, weights)
            return EquilibriumResult(state, residual, method, True, 0, max_row_norm, stable)

        return self._solve_message(
            drive,
            weights,
            stopping=stopping,
            num_steps=num_steps,
            max_steps=max_steps,
            atol=atol,
            rtol=rtol,
            max_row_norm=max_row_norm,
            stable=stable,
        )

    @staticmethod
    def _validate_solve_options(
        method: str,
        stopping: str,
        num_steps: int,
        max_steps: int,
        atol: float,
        rtol: float,
    ) -> None:
        if method not in ("message", "linalg"):
            raise ValueError("method must be 'message' or 'linalg'")
        if stopping not in ("fixed", "tolerance"):
            raise ValueError("stopping must be 'fixed' or 'tolerance'")
        if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 0:
            raise ValueError("num_steps must be a nonnegative integer")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if atol < 0 or rtol < 0:
            raise ValueError("atol and rtol must be nonnegative")
        if method == "linalg":
            if stopping != "fixed":
                raise ValueError("stopping is only configurable for method='message'")

    def _solve_message(
        self,
        drive: Tensor,
        weights: Tensor,
        *,
        stopping: StoppingMode,
        num_steps: int,
        max_steps: int,
        atol: float,
        rtol: float,
        max_row_norm: Tensor,
        stable: bool,
    ) -> EquilibriumResult:
        if stopping == "tolerance" and not stable:
            raise ValueError("tolerance message solving requires max incoming row norm < 1")

        state = drive
        executed_steps = num_steps if stopping == "fixed" else max_steps
        converged = False
        for step in range(executed_steps):
            state = drive + self.sparse_matvec(state, weights)
            if stopping == "tolerance":
                residual = self._residual(drive, state, weights)
                threshold = atol + rtol * state.abs().amax(dim=-1)
                if bool(torch.all(residual.detach() <= threshold.detach())):
                    executed_steps = step + 1
                    converged = True
                    break

        residual = self._residual(drive, state, weights)
        if stopping == "fixed":
            converged = bool(
                torch.all(residual.detach() <= atol + rtol * state.abs().amax(dim=-1).detach())
            )
        return EquilibriumResult(
            state, residual, "message", converged, executed_steps, max_row_norm, stable
        )

    def _solve_linalg(self, drive: Tensor) -> Tensor:
        adjacency = self.dense_adjacency()
        system = (
            torch.eye(self.num_genes, dtype=adjacency.dtype, device=adjacency.device) - adjacency
        )
        right_hand_side = drive.reshape(-1, self.num_genes).transpose(0, 1)
        try:
            solution = torch.linalg.solve(system, right_hand_side)
        except torch.linalg.LinAlgError as error:
            raise ValueError("I - A is singular; regulatory equilibrium is not unique") from error
        return solution.transpose(0, 1).reshape(drive.shape)

    def _residual(self, drive: Tensor, state: Tensor, weights: Tensor) -> Tensor:
        return (drive + self.sparse_matvec(state, weights) - state).abs().amax(dim=-1)

    def _validate_drive(self, drive: Tensor) -> None:
        if not isinstance(drive, Tensor):
            raise TypeError("drive must be a torch.Tensor")
        if not drive.is_floating_point():
            raise TypeError("drive must have a floating-point dtype")
        if drive.ndim < 1 or drive.shape[-1] != self.num_genes:
            raise ValueError(f"drive must have final dimension {self.num_genes}")
        if not torch.isfinite(drive).all():
            raise ValueError("drive must contain only finite values")
        if drive.dtype != self.raw_weights.dtype or drive.device != self.raw_weights.device:
            raise ValueError("drive must match the layer parameter dtype and device")
