"""Fourier positional encoding"""

from __future__ import annotations

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """NeRF-style Fourier features: ``[x, sin(2^k π x), cos(2^k π x), ...]``.

    For each input dimension and frequency ``k = 0 .. num_freqs-1``, appends
    ``sin(2^k π x)`` and ``cos(2^k π x)``. Optionally concatenates the raw
    coordinates.
    """

    def __init__(
        self,
        in_dim: int = 2,
        num_freqs: int = 6,
        include_input: bool = True,
    ):
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}")
        if num_freqs < 1:
            raise ValueError(f"num_freqs must be >= 1, got {num_freqs}")
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.include_input = include_input

        freqs = (2.0 ** torch.arange(num_freqs, dtype=torch.float32)) * torch.pi
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        encoded = self.in_dim * 2 * self.num_freqs
        return encoded + (self.in_dim if self.include_input else 0)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coords ``(N, in_dim)`` → ``(N, out_dim)``."""
        if coords.ndim != 2 or coords.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected coords (N, {self.in_dim}), got {tuple(coords.shape)}"
            )
        # (N, in_dim, L)
        scaled = coords.unsqueeze(-1) * self.freqs.to(
            device=coords.device, dtype=coords.dtype
        )
        sin_feat = torch.sin(scaled)
        cos_feat = torch.cos(scaled)
        # (N, in_dim * 2 * L)
        encoded = torch.cat([sin_feat, cos_feat], dim=-1).reshape(coords.shape[0], -1)
        if self.include_input:
            return torch.cat([coords, encoded], dim=-1)
        return encoded
