from typing import Any, Dict, Optional, Sequence, Tuple, Union, overload

import numpy as np
import torch

from topopt_utils.topology_data import TopOptBatch, TopOptSample

SpatialShape = Tuple[int, ...]

def mesh_spatial_dim(mesh: Dict[str, Any]) -> int:
    """2 if nelx/nely only; 3 if nelz and/or depth present."""
    if "nelz" in mesh or "depth" in mesh:
        return 3
    if "nelx" in mesh and "nely" in mesh:
        return 2
    raise ValueError(
        f"mesh needs nelx/nely (+ nelz/depth for 3D); got {sorted(mesh.keys())}"
    )


def get_element_centers(
    mesh: Dict[str, Any],
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
    normalized: bool = False,
) -> torch.Tensor:
    """Return ``(N, D)`` element centers in C-order over ``(nely, nelx)`` / ``(nelz, nely, nelx)``."""
    dtype = dtype or torch.float32
    device = torch.device(device) if device is not None else None

    dim = mesh_spatial_dim(mesh)
    for key in ("nelx", "nely", *(("nelz",) if dim == 3 else ())):
        if key not in mesh:
            raise ValueError(f"mesh missing required key '{key}'")

    nelx, nely = int(mesh["nelx"]), int(mesh["nely"])
    nels_xyz = [nelx, nely]
    width, height = float(mesh.get("width", 1.0)), float(mesh.get("height", 1.0))
    extents = [width, height]

    if dim == 3:
        nelz = int(mesh["nelz"])
        nels_xyz.append(nelz)
        if "depth" in mesh:
            depth = float(mesh["depth"])
        elif "width" in mesh:
            depth = float(nelz)
        else:
            depth = 1.0
        extents.append(depth)

    if any(e <= 0 for e in extents):
        raise ValueError(f"width/height/depth must be positive, got {extents}")
        
    spatial_shape: SpatialShape = tuple(reversed(nels_xyz))
    grids = torch.meshgrid(
        *[torch.arange(n, device=device, dtype=dtype) for n in spatial_shape],
        indexing="ij",
    )
    scale = [1.0] * dim if normalized else extents
    coords = torch.stack(
        [(g + 0.5) / n * s for g, n, s in zip(reversed(grids), nels_xyz, scale)],
        dim=-1,
    ).reshape(-1, dim)
    return coords


def _physical_to_node(
    px: float, py: float, width: float, height: float, nelx: int, nely: int
) -> Tuple[int, int]:
    return (
        int(np.clip(round(px / width * nelx), 0, nelx)),
        int(np.clip(round(py / height * nely), 0, nely)),
    )


def _physical_load_to_elements(
    load: Optional[Dict[str, Any]], nelx: int, nely: int, width: float, height: float
) -> Optional[Dict[str, Any]]:
    if load is None:
        return None
    xs, ys, fxs, fys = [], [], [], []
    x_coords = np.asarray(load["x"]).flatten()
    y_coords = np.asarray(load["y"]).flatten()
    fx_vals = np.asarray(load["fx"]).flatten()
    fy_vals = np.asarray(load["fy"]).flatten()
    for x, y, fx, fy in zip(x_coords, y_coords, fx_vals, fy_vals):
        nx, ny = _physical_to_node(float(x), float(y), width, height, nelx, nely)
        xs.append(min(nelx - 1, nx))
        ys.append(min(nely - 1, ny))
        fxs.append(float(fx))
        fys.append(float(fy))
    return {"x": xs, "y": ys, "fx": fxs, "fy": fys}


def _rasterize_boundary_conditions(
    segments: Optional[Dict[str, Any]], nelx: int, nely: int, width: float, height: float
) -> np.ndarray:
    bc = np.zeros((nely + 1, nelx + 1, 2), dtype=float)
    if not segments:
        return bc
    axis_to_channel = {"x": 0, "y": 1}
    for axis, channel in axis_to_channel.items():
        for seg in segments.get(axis, []):
            x0, y0 = seg["start"]
            x1, y1 = seg["stop"]
            nx0, ny0 = _physical_to_node(float(x0), float(y0), width, height, nelx, nely)
            nx1, ny1 = _physical_to_node(float(x1), float(y1), width, height, nelx, nely)
            if nx0 == nx1:
                yn0, yn1 = sorted((ny0, ny1))
                bc[yn0 : yn1 + 1, nx0, channel] = 1.0
            elif ny0 == ny1:
                xn0, xn1 = sorted((nx0, nx1))
                bc[ny0, xn0 : xn1 + 1, channel] = 1.0
            else:
                raise ValueError(
                    "Support segment is not axis-aligned after snapping to nodes: "
                    f"start=({nx0}, {ny0}), stop=({nx1}, {ny1})"
                )
    return bc


def _element_counts_from_nels(extents: Tuple[float, ...], nels: int) -> Tuple[int, ...]:
    """``nels`` on the longest side; other axes scale by aspect ratio (at least 1)."""
    nels = int(nels)
    if nels < 1:
        raise ValueError(f"nels must be >= 1, got {nels}")
    longest = max(float(e) for e in extents)
    if longest <= 0:
        raise ValueError(f"extents must be positive, got {list(extents)}")
    return tuple(max(1, int(round(nels * float(e) / longest))) for e in extents)


def _resolve_element_counts(
    width: float,
    height: float,
    nels: Optional[int],
    nelx: Optional[int],
    nely: Optional[int],
) -> Tuple[int, int]:
    if (nelx is None) != (nely is None):
        raise ValueError(
            f"nelx and nely must be provided together; got nelx={nelx}, nely={nely}"
        )
    if nelx is not None and nely is not None:
        return int(nelx), int(nely)
    if nels is None:
        raise ValueError("provide nels or both nelx and nely")
    return _element_counts_from_nels((width, height), nels)


def _remesh_from_coords_sample(
    sample: TopOptSample,
    nels: Optional[int],
    nelx: Optional[int],
    nely: Optional[int],
) -> TopOptSample:
    """Rasterize one coord-space sample. Does not mutate ``sample``."""
    width = float(sample.mesh["width"])
    height = float(sample.mesh["height"])
    nelx, nely = _resolve_element_counts(width, height, nels, nelx, nely)
    mesh = dict(sample.mesh)
    mesh["nelx"] = nelx
    mesh["nely"] = nely
    return TopOptSample(
        mesh=mesh,
        load=_physical_load_to_elements(sample.coord_load, nelx, nely, width, height),
        boundary_conditions=_rasterize_boundary_conditions(
            sample.coord_boundary_conditions, nelx, nely, width, height
        ),
        volume_fraction=sample.volume_fraction,
        sample_id=sample.sample_id,
        load_type=sample.load_type,
    )


@overload
def remesh_from_coords(
    sample: TopOptSample,
    nels: Optional[int] = 64,
    *,
    nelx: Optional[int] = None,
    nely: Optional[int] = None,
) -> TopOptSample: ...


@overload
def remesh_from_coords(
    sample: Union[TopOptBatch, Sequence[TopOptSample]],
    nels: Optional[int] = 64,
    *,
    nelx: Optional[int] = None,
    nely: Optional[int] = None,
) -> TopOptBatch: ...


def remesh_from_coords(
    sample: Union[TopOptSample, TopOptBatch, Sequence[TopOptSample]],
    nels: Optional[int] = 64,
    *,
    nelx: Optional[int] = None,
    nely: Optional[int] = None,
) -> Union[TopOptSample, TopOptBatch]:
    """Rasterize ``coord_load`` and ``coord_boundary_conditions`` onto an FEA grid.

    Accepts a single ``TopOptSample`` or a ``TopOptBatch`` / list / tuple of samples.
    Default: ``nels`` elements on the longest physical side; other axes scale by
    aspect ratio so element size is approximately isotropic (``nels=64``).
    Explicit ``nelx`` and ``nely`` override when both are given. 2D only.
    Batch inputs share the same ``nels``/``nelx``/``nely`` kwargs; each sample
    still resolves its own ``nelx``/``nely`` from its width/height when using ``nels``.

    Does not mutate ``sample``. Does not copy topology / stress / strain fields.
    """
    if isinstance(sample, TopOptSample):
        return _remesh_from_coords_sample(sample, nels, nelx, nely)
    if isinstance(sample, TopOptBatch):
        samples = sample.samples
    elif isinstance(sample, (list, tuple)):
        samples = sample
    else:
        raise TypeError(
            "remesh_from_coords expects TopOptSample, TopOptBatch, or a list/tuple "
            f"of samples; got {type(sample).__name__}"
        )
    return TopOptBatch(
        [_remesh_from_coords_sample(s, nels, nelx, nely) for s in samples]
    )


def coords_from_grid(sample: TopOptSample) -> TopOptSample:
    """Convert gridded ``load`` / ``boundary_conditions`` to domain-unit coord fields.

    Does not mutate ``sample``. Does not copy topology / stress / strain fields.
    """
    nelx = int(sample.mesh["nelx"])
    nely = int(sample.mesh["nely"])
    width = float(sample.mesh["width"])
    height = float(sample.mesh["height"])

    coord_load = None
    if sample.load is not None:
        coord_load = {
            "x": [float(x) / nelx * width for x in np.asarray(sample.load["x"]).flatten()],
            "y": [float(y) / nely * height for y in np.asarray(sample.load["y"]).flatten()],
            "fx": [float(fx) for fx in np.asarray(sample.load["fx"]).flatten()],
            "fy": [float(fy) for fy in np.asarray(sample.load["fy"]).flatten()],
        }

    coord_bc = {"x": [], "y": []}
    if sample.boundary_conditions is not None:
        bc = np.asarray(sample.boundary_conditions)
        for channel, axis in ((0, "x"), (1, "y")):
            rows, cols = np.nonzero(bc[:, :, channel])
            for yn, xn in zip(rows, cols):
                px = float(xn) / nelx * width
                py = float(yn) / nely * height
                coord_bc[axis].append({"start": (px, py), "stop": (px, py)})

    return TopOptSample(
        mesh={"width": width, "height": height},
        coord_load=coord_load,
        coord_boundary_conditions=coord_bc if sample.boundary_conditions is not None else None,
        volume_fraction=sample.volume_fraction,
        sample_id=sample.sample_id,
        load_type=sample.load_type,
    )


def remesh_from_grid(sample: TopOptSample, nelx: int, nely: int) -> TopOptSample:
    """Rebuild a gridded sample on a new ``(nelx, nely)`` mesh via physical coords."""
    return remesh_from_coords(coords_from_grid(sample), nelx=nelx, nely=nely)
