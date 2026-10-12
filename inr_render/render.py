"""Render a topology field onto a dense mesh grid."""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Union
import torch
from .coords import get_element_centers, mesh_spatial_dim


def render_field(
    field: Union[torch.nn.Module, Callable[..., torch.Tensor]],
    mesh: Dict[str, Any],
    thresh: bool = False,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Evaluate ``field`` on element centers → ``(*spatial,)`` if C=1 else ``(C, *spatial)``."""
    coords = get_element_centers(mesh, normalized=True)
    dim = mesh_spatial_dim(mesh)
    if dim == 2:
        spatial_shape = (int(mesh["nely"]), int(mesh["nelx"]))
    else:
        spatial_shape = (int(mesh["nelz"]), int(mesh["nely"]), int(mesh["nelx"]))

    if chunk_size is None or chunk_size <= 0 or chunk_size >= coords.shape[0]:
        probs = field(coords)
    else:
        chunks = []
        for start in range(0, coords.shape[0], chunk_size):
            chunks.append(field(coords[start : start + chunk_size]))
        probs = torch.cat(chunks, dim=0)

    if probs.ndim != 2:
        raise ValueError(
            f"field must return (N, C) probs, got shape {tuple(probs.shape)}"
        )

    c = probs.shape[1]
    grid = probs.transpose(0, 1).reshape(c, *spatial_shape)

    if c == 1:
        grid = grid[0]
        if thresh:
            return (grid >= 0.5).to(dtype=grid.dtype)
        return grid

    if thresh:
        return torch.argmax(grid, dim=0)
    return grid
