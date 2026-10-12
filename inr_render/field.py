"""Topology field from packed-bias weight matrices."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import FourierFeatures

# Paper default for the *first* sine layer (Sitzmann et al. 2020). Prefer a
# smaller value (e.g. 10) on 64×64 topology grids; ω0=30 is often too
# oscillatory once rmin≈2 smoothing is applied.
SIREN_OMEGA_0 = 30.0
# Hidden sine layers use ω=1 so frequencies do not compound. Official SIREN
# multiplies hidden layers by the same large ω0 *and* divides hidden weights
# by ω0 (a gradient boost). That is unstable for physics-driven TO at ReLU-MLP
# learning rates; keep hidden_omega_0=1 unless you explicitly want paper SIREN.
SIREN_HIDDEN_OMEGA_0 = 1.0


def _linear_packed_bias(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    ones = x.new_ones(x.shape[0], 1)
    x = torch.cat([x, ones], dim=-1)
    return F.linear(x, weight)


def init_siren_packed_weights(
    dims: Sequence[int],
    omega_0: float = SIREN_OMEGA_0,
    hidden_omega_0: float = SIREN_HIDDEN_OMEGA_0,
) -> list[nn.Parameter]:
    """Packed-bias ``(out, in+1)`` SIREN init (Sitzmann et al. 2020).

    First-layer weights ``U(-1/in, 1/in)`` (ω0 is applied in the forward, not
    here). Hidden and last-layer weights
    ``U(-sqrt(6/in)/ω_hidden, sqrt(6/in)/ω_hidden)``. Packed bias uses the
    matching ``nn.Linear`` default ``U(-1/sqrt(in), 1/sqrt(in))``.
    """
    if len(dims) < 2:
        raise ValueError(f"dims must have at least input and output, got {list(dims)}")
    omega_0 = float(omega_0)
    hidden_omega_0 = float(hidden_omega_0)
    if omega_0 <= 0:
        raise ValueError(f"omega_0 must be > 0, got {omega_0}")
    if hidden_omega_0 <= 0:
        raise ValueError(f"hidden_omega_0 must be > 0, got {hidden_omega_0}")
    weights = []
    for i in range(len(dims) - 1):
        in_f, out_f = int(dims[i]), int(dims[i + 1])
        if in_f < 1 or out_f < 1:
            raise ValueError(f"layer dims must be >= 1, got {in_f} -> {out_f}")
        packed = torch.empty(out_f, in_f + 1)
        if i == 0:
            w_bound = 1.0 / in_f
        else:
            w_bound = math.sqrt(6.0 / in_f) / hidden_omega_0
        nn.init.uniform_(packed[:, :-1], -w_bound, w_bound)
        b_bound = 1.0 / math.sqrt(in_f)
        nn.init.uniform_(packed[:, -1], -b_bound, b_bound)
        weights.append(nn.Parameter(packed))
    return weights


def topology_mlp_from_weights(
    weights: Sequence[torch.Tensor],
    encoding: Optional[FourierFeatures] = None,
) -> TopologyField:
    """Build a topology field from weight matrices ``(out, in+1)`` (bias in last col)."""
    return TopologyField(weights, encoding=encoding)

def topology_mlp_from_columns(
    columns: Sequence[torch.Tensor],
    encoding: Optional[FourierFeatures] = None,
) -> TopologyField:
    """Build a topology field from per-layer column groups.

    Each group is ``(n_cols, out)``: one packed-bias column per row (transformer
    token layout). Stacked into weight matrices ``(out, n_cols)``.
    """
    weights = [group.transpose(-2, -1) for group in columns]
    return topology_mlp_from_weights(weights, encoding=encoding)

class TopologyField(nn.Module):
    """Topology MLP parameterized by external weight matrices with packed bias."""

    def __init__(
        self,
        weights: Sequence[torch.Tensor],
        encoding: Optional[FourierFeatures] = None,
    ):
        super().__init__()
        self.weights = list(weights)
        self.encoding = encoding
        self.out_channels = int(weights[-1].shape[0])
        self.in_dim = (
            encoding.in_dim if encoding is not None else int(weights[0].shape[1] - 1)
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = self.encoding(coords) if self.encoding is not None else coords
        for i, weight in enumerate(self.weights):
            x = _linear_packed_bias(x, weight)
            if i < len(self.weights) - 1:
                x = F.relu(x)
        if self.out_channels == 1:
            return torch.sigmoid(x)
        return torch.softmax(x, dim=-1)


def topology_siren_from_weights(
    weights: Sequence[torch.Tensor],
    omega_0: float = SIREN_OMEGA_0,
    hidden_omega_0: float = SIREN_HIDDEN_OMEGA_0,
    map_unit_interval: bool = True,
) -> SirenTopologyField:
    """Build a SIREN topology field from packed-bias matrices ``(out, in+1)``."""
    return SirenTopologyField(
        weights,
        omega_0=omega_0,
        hidden_omega_0=hidden_omega_0,
        map_unit_interval=map_unit_interval,
    )


class SirenTopologyField(nn.Module):
    """SIREN density field (Sitzmann et al. 2020) with packed-bias weights.

    First hidden layer: ``sin(ω0 (Wx + b))`` with large ω0. Subsequent hidden
    layers: ``sin(ω_hidden (Wx + b))`` with ``ω_hidden=1`` by default so
    frequencies do not compound. Last layer is linear, then sigmoid so density
    is in ``(0, 1)``. No Fourier features.

    ``render_field`` evaluates on normalized element centers in ``(0, 1)``.
    With ``map_unit_interval=True`` (default) those coords are mapped to
    ``[-1, 1]`` via ``2x - 1`` before the first layer, which is the usual
    SIREN input domain.
    """

    def __init__(
        self,
        weights: Sequence[torch.Tensor],
        omega_0: float = SIREN_OMEGA_0,
        hidden_omega_0: float = SIREN_HIDDEN_OMEGA_0,
        map_unit_interval: bool = True,
    ):
        super().__init__()
        if len(weights) < 1:
            raise ValueError("SIREN needs at least one weight matrix")
        omega_0 = float(omega_0)
        hidden_omega_0 = float(hidden_omega_0)
        if omega_0 <= 0:
            raise ValueError(f"omega_0 must be > 0, got {omega_0}")
        if hidden_omega_0 <= 0:
            raise ValueError(f"hidden_omega_0 must be > 0, got {hidden_omega_0}")
        self.weights = list(weights)
        self.omega_0 = omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.map_unit_interval = bool(map_unit_interval)
        self.out_channels = int(weights[-1].shape[0])
        self.in_dim = int(weights[0].shape[1] - 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected coords (N, {self.in_dim}), got {tuple(coords.shape)}"
            )
        # Normalized element centers are in (0, 1); SIREN expects ~[-1, 1].
        x = coords * 2.0 - 1.0 if self.map_unit_interval else coords
        last = len(self.weights) - 1
        for i, weight in enumerate(self.weights):
            x = _linear_packed_bias(x, weight)
            if i < last:
                omega = self.omega_0 if i == 0 else self.hidden_omega_0
                x = torch.sin(omega * x)
        if self.out_channels == 1:
            return torch.sigmoid(x)
        return torch.softmax(x, dim=-1)
