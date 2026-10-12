"""Differentiable SIMP compliance via analytical adjoint sensitivities.

Wraps the cached ``FeaSolver`` with a ``torch.autograd.Function`` so INR
densities can receive ``dc/dρ`` for ``E/E0 = emin + (1-emin) ρ^p``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.autograd import Function

from .fea import FeaSolver, get_default_solver
from .fea.q4 import load_to_forces


class _ComplianceAutograd(Function):
    @staticmethod
    def forward(
        ctx,
        rho: torch.Tensor,
        forces: np.ndarray,
        fixed_dofs: np.ndarray,
        penal: float,
        nelx: int,
        nely: int,
        dx: float,
        dy: float,
        solver: FeaSolver,
    ) -> torch.Tensor:
        if rho.ndim != 2 or rho.shape != (nely, nelx):
            raise ValueError(
                f"rho must have shape {(nely, nelx)}, got {tuple(rho.shape)}"
            )
        rho_np = rho.detach().cpu().numpy().reshape(-1)
        c, dc = solver.compliance(
            rho_np,
            forces,
            fixed_dofs,
            nelx,
            nely,
            dx,
            dy,
            penal=penal,
            sensitivities=True,
        )
        ctx.save_for_backward(
            torch.from_numpy(dc.reshape(nely, nelx)).to(
                device=rho.device, dtype=rho.dtype
            )
        )
        return rho.new_tensor(c)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (dc,) = ctx.saved_tensors
        grad_rho = grad_output * dc
        return grad_rho, None, None, None, None, None, None, None, None


class DifferentiableCompliance:
    """Callable compliance loss for a fixed mesh / BC / load problem."""

    def __init__(
        self,
        sample,
        *,
        penal: float = 3.0,
        solver: Optional[FeaSolver] = None,
    ):
        nelx = int(sample.mesh["nelx"])
        nely = int(sample.mesh["nely"])
        dx = float(sample.mesh["width"]) / nelx
        dy = float(sample.mesh["height"]) / nely
        self.nelx = nelx
        self.nely = nely
        self.dx = dx
        self.dy = dy
        self.penal = float(penal)
        self.solver = solver if solver is not None else get_default_solver()
        self.forces = load_to_forces(sample.load, nelx, nely)
        self.fixed_dofs = np.asarray(
            sample.get_boundary_condition_dofs(), dtype=np.int64
        )
        rho_ref = np.full(nelx * nely, float(sample.volume_fraction), dtype=np.float64)
        c_ref = self.solver.compliance(
            rho_ref, self.forces, self.fixed_dofs, nelx, nely, dx, dy, penal=self.penal
        )
        self.norm_scale = max(float(c_ref), 1e-8)

    def __call__(self, rho: torch.Tensor) -> torch.Tensor:
        """ρ soft density ``(nely, nelx)`` in [0, 1] → scalar compliance."""
        return _ComplianceAutograd.apply(
            rho,
            self.forces,
            self.fixed_dofs,
            self.penal,
            self.nelx,
            self.nely,
            self.dx,
            self.dy,
            self.solver,
        )
