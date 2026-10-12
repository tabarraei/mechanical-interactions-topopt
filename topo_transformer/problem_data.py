"""Synthetic topology-optimization problem generator (coord loads / BCs, no GT)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from topopt_utils.topology_data import TopOptBatch, TopOptSample

MAX_LOADS = 4
MAX_BCS = 4
DEFAULT_WIDTH = 2.0
DEFAULT_HEIGHT = 1.0
DEFAULT_SIZE = 10**9
VOLFRAC_RANGE = (0.2, 0.5)
P_TWO_LOADS = 0.0
P_POINT_BC = 0.25
MIN_FIXED_POINTS = 3
POINT_ATOL = 1e-6
MAX_WELL_POSED_TRIES = 10_000
# Snap BCs to this longest-side count so rank-3 restraints survive default remesh.
WELL_POSED_REMESH_NELS = 64
# Minimum load–BC gap as a fraction of the longest domain side.
LOAD_BC_SEPARATION = 0.2
ASPECT_RATIOS: Tuple[Tuple[int, int], ...] = (
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
)

ProblemDict = Dict[str, torch.Tensor]


def _as_generator(seed: Optional[int], generator: Optional[torch.Generator], idx: int) -> torch.Generator:
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(int(seed) + int(idx))
        return g
    if generator is not None:
        return generator
    return torch.Generator()


def _rand(generator: torch.Generator) -> float:
    return float(torch.rand(1, generator=generator).item())


def _randint(high: int, generator: torch.Generator, low: int = 0) -> int:
    return int(torch.randint(low, high, (1,), generator=generator).item())


def _sample_load_count(
    generator: torch.Generator,
    max_loads: int = MAX_LOADS,
    p_two_loads: float = P_TWO_LOADS,
) -> int:
    """Sample a load count uniformly over all valid cases.

    The legacy ``p_two_loads`` parameter is retained only for backwards compatibility,
    but it is intentionally ignored so every load count from 1 to ``max_loads`` has the
    same chance of being selected.
    """
    del p_two_loads
    if max_loads < 1:
        raise ValueError(f"max_loads must be >= 1, got {max_loads}")
    if max_loads == 1:
        return 1
    options = list(range(1, max_loads + 1))
    return options[_randint(len(options), generator)]


def _sample_aspect(generator: torch.Generator) -> Tuple[float, float]:
    w, h = ASPECT_RATIOS[_randint(len(ASPECT_RATIOS), generator)]
    return float(w), float(h)


def _resolve_domain(
    generator: torch.Generator,
    width: Optional[float],
    height: Optional[float],
) -> Tuple[float, float]:
    if width is None and height is None:
        return _sample_aspect(generator)
    if width is None or height is None:
        raise ValueError("width and height must both be set or both omitted")
    return float(width), float(height)


def _sample_edge(generator: torch.Generator, width: float, height: float) -> int:
    """Pick a rectangle edge with probability proportional to its length."""
    lengths = (width, height, width, height)  # bottom, right, top, left
    u = _rand(generator) * (2.0 * (width + height))
    acc = 0.0
    for i, length in enumerate(lengths):
        acc += length
        if u <= acc:
            return i
    return 3


def _point_on_edge(edge: int, t: float, width: float, height: float) -> Tuple[float, float]:
    t = min(1.0, max(0.0, t))
    if edge == 0:
        return t * width, 0.0
    if edge == 1:
        return width, t * height
    if edge == 2:
        return t * width, height
    return 0.0, t * height


def _distinct_fixed_points(
    bc_start: torch.Tensor,
    bc_stop: torch.Tensor,
    bc_fixed: torch.Tensor,
    atol: float = POINT_ATOL,
) -> List[List[float]]:
    """Merge segment endpoints; coincident locations share DOFs.

    A point BC (start≈stop) contributes one location. A non-degenerate segment
    contributes its two endpoints. Returns ``[x, y, fix_x, fix_y]`` rows.
    """
    merged: List[List[float]] = []
    for start, stop, fixed in zip(bc_start, bc_stop, bc_fixed):
        fix_x = float(fixed[0]) > 0.5
        fix_y = float(fixed[1]) > 0.5
        if not fix_x and not fix_y:
            continue
        x0, y0 = float(start[0]), float(start[1])
        x1, y1 = float(stop[0]), float(stop[1])
        locs = [(x0, y0)]
        if abs(x1 - x0) > atol or abs(y1 - y0) > atol:
            locs.append((x1, y1))
        for x, y in locs:
            for row in merged:
                if abs(row[0] - x) <= atol and abs(row[1] - y) <= atol:
                    row[2] = float(row[2] or fix_x)
                    row[3] = float(row[3] or fix_y)
                    break
            else:
                merged.append([x, y, float(fix_x), float(fix_y)])
    return merged


def _rigid_body_constraint_rank(points: Sequence[Sequence[float]]) -> int:
    """Rank of 2D rigid-body restraints (tx, ty, rz) from ``[x, y, fix_x, fix_y]`` rows."""
    rows: List[List[float]] = []
    for x, y, fix_x, fix_y in points:
        if float(fix_x) > 0.5:
            rows.append([1.0, 0.0, -float(y)])
        if float(fix_y) > 0.5:
            rows.append([0.0, 1.0, float(x)])
    if len(rows) < 3:
        return 0
    singular = torch.linalg.svdvals(torch.tensor(rows, dtype=torch.float64))
    scale = max(1.0, float(singular[0]))
    return int((singular > 1e-8 * scale).sum().item())


def _snap_fixed_points(
    points: Sequence[Sequence[float]],
    width: float,
    height: float,
    nels: int = WELL_POSED_REMESH_NELS,
) -> List[List[float]]:
    """Merge fixed points that land on the same node after ``nels`` remesh snapping."""
    longest = max(float(width), float(height))
    nelx = max(1, int(round(nels * float(width) / longest)))
    nely = max(1, int(round(nels * float(height) / longest)))
    merged: Dict[Tuple[int, int], List[float]] = {}
    for x, y, fix_x, fix_y in points:
        nx = int(min(nelx, max(0, round(float(x) / width * nelx))))
        ny = int(min(nely, max(0, round(float(y) / height * nely))))
        key = (nx, ny)
        if key in merged:
            merged[key][2] = float(merged[key][2] or fix_x)
            merged[key][3] = float(merged[key][3] or fix_y)
        else:
            merged[key] = [
                nx / nelx * width,
                ny / nely * height,
                float(fix_x),
                float(fix_y),
            ]
    return list(merged.values())


def _bcs_are_well_posed(
    bc_start: torch.Tensor,
    bc_stop: torch.Tensor,
    bc_fixed: torch.Tensor,
    *,
    width: float,
    height: float,
    atol: float = POINT_ATOL,
) -> bool:
    """Require ≥1 x, ≥1 y, ≥3 points, and rank-3 rigid-body restraint after remesh snap."""
    points = _distinct_fixed_points(bc_start, bc_stop, bc_fixed, atol=atol)
    has_x = any(row[2] > 0.5 for row in points)
    has_y = any(row[3] > 0.5 for row in points)
    if not (has_x and has_y and len(points) >= MIN_FIXED_POINTS):
        return False
    if _rigid_body_constraint_rank(points) < 3:
        return False
    snapped = _snap_fixed_points(points, width, height)
    return _rigid_body_constraint_rank(snapped) >= 3


def _point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    atol: float = POINT_ATOL,
) -> float:
    """Euclidean distance from a point to the closest point on segment AB."""
    vx, vy = bx - ax, by - ay
    len2 = vx * vx + vy * vy
    if len2 <= atol * atol:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / len2
    t = min(1.0, max(0.0, t))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _bcs_separated_from_loads(
    load_xy: torch.Tensor,
    bc_start: torch.Tensor,
    bc_stop: torch.Tensor,
    min_dist: float,
) -> bool:
    """True if every BC segment is at least ``min_dist`` from every load."""
    for load in load_xy:
        px, py = float(load[0]), float(load[1])
        for start, stop in zip(bc_start, bc_stop):
            dist = _point_segment_distance(
                px,
                py,
                float(start[0]),
                float(start[1]),
                float(stop[0]),
                float(stop[1]),
            )
            if dist < min_dist:
                return False
    return True


def _draw_boundary_conditions(
    generator: torch.Generator,
    width: float,
    height: float,
    p_point_bc: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_bcs = _randint(MAX_BCS + 1, generator, low=1)
    bc_start = torch.empty(n_bcs, 2, dtype=torch.float32)
    bc_stop = torch.empty(n_bcs, 2, dtype=torch.float32)
    bc_fixed = torch.zeros(n_bcs, 2, dtype=torch.float32)
    for i in range(n_bcs):
        edge = _sample_edge(generator, width, height)
        if _rand(generator) < p_point_bc:
            t0 = t1 = _rand(generator)
        else:
            t0, t1 = _rand(generator), _rand(generator)
            if t1 < t0:
                t0, t1 = t1, t0
        bc_start[i, 0], bc_start[i, 1] = _point_on_edge(edge, t0, width, height)
        bc_stop[i, 0], bc_stop[i, 1] = _point_on_edge(edge, t1, width, height)
        choice = _randint(3, generator)  # 0=x, 1=y, 2=both
        bc_fixed[i, 0] = float(choice != 1)
        bc_fixed[i, 1] = float(choice != 0)
    return bc_start, bc_stop, bc_fixed


def _generate_problem_dict(
    generator: torch.Generator,
    width: Optional[float] = None,
    height: Optional[float] = None,
    volfrac_range: Tuple[float, float] = VOLFRAC_RANGE,
    p_two_loads: float = P_TWO_LOADS,
    p_point_bc: float = P_POINT_BC,
) -> ProblemDict:
    """Draw one problem as tensors: border loads first, then BCs away from them."""
    width, height = _resolve_domain(generator, width, height)
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"width/height must be positive, got {width}, {height}")
    lo, hi = float(volfrac_range[0]), float(volfrac_range[1])
    if not 0.0 < lo <= hi < 1.0:
        raise ValueError(f"volfrac_range must satisfy 0 < lo <= hi < 1, got {volfrac_range}")

    n_loads = _sample_load_count(generator, max_loads=MAX_LOADS, p_two_loads=p_two_loads)
    load_xy = torch.empty(n_loads, 2, dtype=torch.float32)
    load_f = torch.empty(n_loads, 2, dtype=torch.float32)
    for i in range(n_loads):
        edge = _sample_edge(generator, width, height)
        load_xy[i, 0], load_xy[i, 1] = _point_on_edge(edge, _rand(generator), width, height)
        theta = 2.0 * math.pi * _rand(generator)
        load_f[i, 0] = math.cos(theta)
        load_f[i, 1] = math.sin(theta)

    min_sep = LOAD_BC_SEPARATION * max(width, height)
    for _ in range(MAX_WELL_POSED_TRIES):
        bc_start, bc_stop, bc_fixed = _draw_boundary_conditions(
            generator, width, height, p_point_bc
        )
        if _bcs_are_well_posed(
            bc_start, bc_stop, bc_fixed, width=width, height=height
        ) and _bcs_separated_from_loads(load_xy, bc_start, bc_stop, min_sep):
            break
    else:
        raise RuntimeError(
            f"failed to sample well-posed BCs separated from loads "
            f"in {MAX_WELL_POSED_TRIES} tries"
        )

    return {
        "volume_fraction": torch.tensor(lo + (hi - lo) * _rand(generator), dtype=torch.float32),
        "width": torch.tensor(width, dtype=torch.float32),
        "height": torch.tensor(height, dtype=torch.float32),
        "load_xy": load_xy,
        "load_f": load_f,
        "bc_start": bc_start,
        "bc_stop": bc_stop,
        "bc_fixed": bc_fixed,
    }


def _masked_rows(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values
    return values[mask.bool()]


def problem_to_sample(item: ProblemDict, sample_id: Optional[int] = None) -> TopOptSample:
    """Convert a generated (possibly padded) problem dict to a coord-space TopOptSample."""
    load_xy = _masked_rows(item["load_xy"], item.get("load_mask"))
    load_f = _masked_rows(item["load_f"], item.get("load_mask"))
    bc_start = _masked_rows(item["bc_start"], item.get("bc_mask"))
    bc_stop = _masked_rows(item["bc_stop"], item.get("bc_mask"))
    bc_fixed = _masked_rows(item["bc_fixed"], item.get("bc_mask"))

    coord_load = {
        "x": [float(x) for x in load_xy[:, 0].tolist()],
        "y": [float(y) for y in load_xy[:, 1].tolist()],
        "fx": [float(fx) for fx in load_f[:, 0].tolist()],
        "fy": [float(fy) for fy in load_f[:, 1].tolist()],
    }
    coord_bc: Dict[str, list] = {"x": [], "y": []}
    for start, stop, fixed in zip(bc_start, bc_stop, bc_fixed):
        seg = {
            "start": (float(start[0]), float(start[1])),
            "stop": (float(stop[0]), float(stop[1])),
        }
        if float(fixed[0]) > 0.5:
            coord_bc["x"].append(seg)
        if float(fixed[1]) > 0.5:
            coord_bc["y"].append(seg)

    return TopOptSample(
        mesh={
            "width": float(item["width"].reshape(-1)[0].item()),
            "height": float(item["height"].reshape(-1)[0].item()),
        },
        coord_load=coord_load,
        coord_boundary_conditions=coord_bc,
        volume_fraction=float(item["volume_fraction"].reshape(-1)[0].item()),
        sample_id=sample_id,
    )


def generate_problem(
    generator: torch.Generator,
    width: Optional[float] = None,
    height: Optional[float] = None,
    volfrac_range: Tuple[float, float] = VOLFRAC_RANGE,
    p_two_loads: float = P_TWO_LOADS,
    p_point_bc: float = P_POINT_BC,
    sample_id: Optional[int] = None,
) -> TopOptSample:
    """Draw one well-posed problem: border loads, border BC segments, volume fraction."""
    return problem_to_sample(
        _generate_problem_dict(
            generator,
            width=width,
            height=height,
            volfrac_range=volfrac_range,
            p_two_loads=p_two_loads,
            p_point_bc=p_point_bc,
        ),
        sample_id=sample_id,
    )


def collate_topopt_problems(items: Sequence[TopOptSample]) -> TopOptBatch:
    """Wrap samples in TopOptBatch (default collate cannot stack custom objects)."""
    return TopOptBatch(list(items))


class TopOptProblemDataset(Dataset):
    """Infinite/large synthetic dataset of coordinate-space TO problems.

    Defaults sample an integer aspect ratio (domain ``[0, w] x [0, h]``), unit-magnitude
    loads (random direction on the border), volume fraction in ``[0.2, 0.5]``.
    """

    def __init__(
        self,
        width: Optional[float] = None,
        height: Optional[float] = None,
        size: int = DEFAULT_SIZE,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        volfrac_range: Tuple[float, float] = VOLFRAC_RANGE,
        p_two_loads: float = P_TWO_LOADS,
        p_point_bc: float = P_POINT_BC,
    ):
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        if (width is None) != (height is None):
            raise ValueError("width and height must both be set or both omitted")
        self.width = None if width is None else float(width)
        self.height = None if height is None else float(height)
        self.size = int(size)
        self.seed = seed
        self.generator = generator
        self.volfrac_range = (float(volfrac_range[0]), float(volfrac_range[1]))
        self.p_two_loads = float(p_two_loads)
        self.p_point_bc = float(p_point_bc)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> TopOptSample:
        return generate_problem(
            _as_generator(self.seed, self.generator, idx),
            width=self.width,
            height=self.height,
            volfrac_range=self.volfrac_range,
            p_two_loads=self.p_two_loads,
            p_point_bc=self.p_point_bc,
            sample_id=int(idx),
        )


def get_problem_loader(
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    return DataLoader(
        TopOptProblemDataset(**dataset_kwargs),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_topopt_problems,
    )
