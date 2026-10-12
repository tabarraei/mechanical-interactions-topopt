"""Multi-resolution compliance + volume physics loss for INR training."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from inr_render import remesh_from_coords, render_field
from topopt_utils.diff_compliance import DifferentiableCompliance
from topopt_utils.topology_data import TopOptSample
from topopt_utils.topopt_2d import __create_filter_kernel_nd__

Resolution = Tuple[int, int]


def filter_density(rho: torch.Tensor, rmin: float) -> torch.Tensor:
    """SIMP density filter: ``ndimage.convolve`` with constant-0 pad, element-count ``rmin``.

    ``rmin <= 0`` is a no-op. The kernel is symmetric, so ``conv2d``/``conv3d`` match SciPy.
    """
    if rho.ndim not in (2, 3):
        raise ValueError(
            f"rho must be (nely, nelx) or (nelz, nely, nelx), got {tuple(rho.shape)}"
        )
    rmin = float(rmin)
    if rmin <= 0:
        return rho
    kernel = torch.as_tensor(
        __create_filter_kernel_nd__(*((rmin,) * rho.ndim)),
        device=rho.device,
        dtype=rho.dtype,
    )
    conv = F.conv2d if rho.ndim == 2 else F.conv3d
    return conv(rho[None, None], kernel[None, None], padding=int(kernel.shape[-1]) // 2)[0, 0]


class PhysicsLoss(nn.Module):
    """``L = c / c_ref + λ SmoothL1(mean(ρ), volfrac)`` at one random resolution.

    ``ρ`` is the density-filtered INR field (``rmin`` in element-count units).
    """

    def __init__(
        self,
        resolutions: Sequence[Resolution] = ((48, 48), (64, 64)),
        vol_weight: float = 100.0,
        vol_beta: float = 0.05,
        penal: float = 3.0,
        rmin: float = 2.0,
    ):
        super().__init__()
        self.resolutions = tuple((int(nx), int(ny)) for nx, ny in resolutions)
        if not self.resolutions:
            raise ValueError("resolutions must be non-empty")
        self.vol_weight = float(vol_weight)
        self.vol_beta = float(vol_beta)
        self.penal = float(penal)
        self.rmin = float(rmin)
        self._cache = {}  # (sample_id, nelx, nely) -> (mesh, DifferentiableCompliance)

    def _solver(self, sample: TopOptSample, nelx: int, nely: int):
        key = (sample.sample_id if sample.sample_id is not None else id(sample), nelx, nely)
        if key not in self._cache:
            remeshed = remesh_from_coords(sample, nelx=nelx, nely=nely)
            self._cache[key] = (remeshed.mesh, DifferentiableCompliance(remeshed, penal=self.penal))
        return self._cache[key]

    def forward(
        self,
        fields: Union[nn.Module, Sequence[nn.Module]],
        samples: Union[TopOptSample, Sequence[TopOptSample]],
    ) -> torch.Tensor:
        if not isinstance(fields, (list, tuple)):
            fields, samples = [fields], [samples]
        losses = []
        for field, sample in zip(fields, samples):
            nelx, nely = self.resolutions[int(torch.randint(len(self.resolutions), (1,)))]
            mesh, compliance = self._solver(sample, nelx, nely)
            rho = filter_density(render_field(field, mesh), self.rmin)
            c_norm = compliance(rho) / rho.new_tensor(compliance.norm_scale)
            vol_pen = F.smooth_l1_loss(
                rho.mean(),
                rho.new_tensor(float(sample.volume_fraction)),
                beta=self.vol_beta,
            )
            losses.append(c_norm + self.vol_weight * vol_pen)
        return torch.stack(losses).mean()
