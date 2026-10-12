"""Spatial diagnostics for dense mechanical interaction matrices.

These utilities operate on the verified M1 interaction matrices and element-center
coordinates without altering the FEM assembly, SIMP interpolation, density filter,
optimization, or normalization logic.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np


__all__ = [
    "pairwise_upper_triangle",
    "pairwise_distances",
    "compute_distance_bin_stats",
    "interaction_energy_tail",
    "interaction_energy_tail_curve",
    "directional_coupling_stats",
]


def pairwise_upper_triangle(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return indices of the unique unordered pairs i < j."""
    if n < 0:
        raise ValueError(f"n must be nonnegative; got {n}")
    if n < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.triu_indices(n, k=1)


def pairwise_distances(element_x: np.ndarray, element_y: np.ndarray) -> np.ndarray:
    """Physical distances for unique unordered pairs only."""
    x = np.asarray(element_x, dtype=np.float64).reshape(-1)
    y = np.asarray(element_y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"element_x and element_y must have the same shape; got {x.shape} and {y.shape}")
    if x.size < 2:
        return np.empty(0, dtype=np.float64)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    i_idx, j_idx = pairwise_upper_triangle(x.size)
    return np.hypot(dx[i_idx, j_idx], dy[i_idx, j_idx])


def _pair_values_for_mask(
    P: np.ndarray,
    R: np.ndarray,
    active_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    P = np.asarray(P, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be a square matrix; got shape {P.shape}")
    if R.shape != P.shape:
        raise ValueError(f"R must match P; got shapes {R.shape} and {P.shape}")

    n = P.shape[0]
    i_idx, j_idx = pairwise_upper_triangle(n)
    if active_mask is None:
        valid = np.ones(i_idx.size, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if active_mask.shape[0] != n:
            raise ValueError(f"active_mask must have length {n}; got {active_mask.shape}")
        valid = active_mask[i_idx] & active_mask[j_idx]
        i_idx = i_idx[valid]
        j_idx = j_idx[valid]

    return i_idx, j_idx, P[i_idx, j_idx], R[i_idx, j_idx]


def _empty_bin_record(bin_start: float, bin_end: float) -> dict[str, Any]:
    return {
        "bin_start": float(bin_start),
        "bin_end": float(bin_end),
        "pair_count": 0,
        "bin_status": "empty_bin",
        "mean_abs_R": np.nan,
        "median_abs_R": np.nan,
        "p90_abs_R": np.nan,
        "max_abs_R": np.nan,
        "mean_signed_R": np.nan,
        "fraction_positive_R": np.nan,
        "fraction_negative_R": np.nan,
        "mean_abs_P": np.nan,
        "median_abs_P": np.nan,
        "p90_abs_P": np.nan,
        "max_abs_P": np.nan,
        "mean_signed_P": np.nan,
        "fraction_positive_P": np.nan,
        "fraction_negative_P": np.nan,
    }


def _bin_status_for_count(pair_count: int, min_pairs: int) -> str:
    count = int(pair_count)
    if count == 0:
        return "empty_bin"
    if count < int(min_pairs):
        return "below_min_pairs"
    return "valid"


def _summarize_pairs(values: np.ndarray, signed_values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "mean_abs": np.nan,
            "median_abs": np.nan,
            "p90_abs": np.nan,
            "max_abs": np.nan,
            "mean_signed": np.nan,
            "fraction_positive": np.nan,
            "fraction_negative": np.nan,
        }
    abs_values = np.abs(values)
    return {
        "mean_abs": float(np.mean(abs_values)),
        "median_abs": float(np.median(abs_values)),
        "p90_abs": float(np.quantile(abs_values, 0.90)),
        "max_abs": float(np.max(abs_values)),
        "mean_signed": float(np.mean(signed_values)),
        "fraction_positive": float(np.mean(signed_values > 0.0)),
        "fraction_negative": float(np.mean(signed_values < 0.0)),
    }


def compute_distance_bin_stats(
    P: np.ndarray,
    R: np.ndarray,
    active_mask: Optional[np.ndarray],
    element_x: np.ndarray,
    element_y: np.ndarray,
    distance_bin_edges: Iterable[float],
) -> dict[str, Any]:
    """Distance-binned statistics for unique unordered pairs.

    For active pairs, only entries with both endpoints active are included. When
    ``active_mask`` is None, the full upper triangle is used. The returned structure
    includes pair counts and signed/absolute summaries for both ``R`` and ``|P|``.
    """
    P = np.asarray(P, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    element_x = np.asarray(element_x, dtype=np.float64).reshape(-1)
    element_y = np.asarray(element_y, dtype=np.float64).reshape(-1)
    if element_x.shape != element_y.shape:
        raise ValueError(f"element_x and element_y must match; got {element_x.shape} and {element_y.shape}")
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be square; got shape {P.shape}")
    if R.shape != P.shape:
        raise ValueError(f"R must match P; got shapes {R.shape} and {P.shape}")
    if element_x.size != P.shape[0]:
        raise ValueError(f"element coordinates must have length {P.shape[0]}; got {element_x.size}")

    edges = np.asarray(list(distance_bin_edges), dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("distance_bin_edges must be a 1D array with at least two values")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("distance_bin_edges must be strictly increasing")

    i_idx, j_idx, p_vals, r_vals = _pair_values_for_mask(P, R, active_mask)
    if i_idx.size == 0:
        return {
            "distance_bin_edges": edges.copy(),
            "num_active_pairs": 0,
            "pair_statistics": [_empty_bin_record(edges[k], edges[k + 1]) for k in range(edges.size - 1)],
            "distance_range": {"min": np.nan, "max": np.nan},
        }

    distances = np.hypot(element_x[i_idx] - element_x[j_idx], element_y[i_idx] - element_y[j_idx])
    bin_ids = np.digitize(distances, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, edges.size - 2)

    stats = []
    for idx in range(edges.size - 1):
        in_bin = bin_ids == idx
        if not np.any(in_bin):
            record = _empty_bin_record(edges[idx], edges[idx + 1])
            record["bin_status"] = "empty_bin"
            stats.append(record)
            continue

        r_bin = r_vals[in_bin]
        p_bin = p_vals[in_bin]
        r_summary = _summarize_pairs(r_bin, r_bin)
        p_summary = _summarize_pairs(np.abs(p_bin), p_bin)

        stats.append({
            "bin_start": float(edges[idx]),
            "bin_end": float(edges[idx + 1]),
            "pair_count": int(r_bin.size),
            "bin_status": "valid",
            "mean_abs_R": r_summary["mean_abs"],
            "median_abs_R": r_summary["median_abs"],
            "p90_abs_R": r_summary["p90_abs"],
            "max_abs_R": r_summary["max_abs"],
            "mean_signed_R": r_summary["mean_signed"],
            "fraction_positive_R": r_summary["fraction_positive"],
            "fraction_negative_R": r_summary["fraction_negative"],
            "mean_abs_P": p_summary["mean_abs"],
            "median_abs_P": p_summary["median_abs"],
            "p90_abs_P": p_summary["p90_abs"],
            "max_abs_P": p_summary["max_abs"],
            "mean_signed_P": p_summary["mean_signed"],
            "fraction_positive_P": p_summary["fraction_positive"],
            "fraction_negative_P": p_summary["fraction_negative"],
        })

    return {
        "distance_bin_edges": edges.copy(),
        "num_active_pairs": int(i_idx.size),
        "pair_statistics": stats,
        "distance_range": {"min": float(np.min(distances)), "max": float(np.max(distances))},
    }


def interaction_energy_tail(
    P: np.ndarray,
    element_x: np.ndarray,
    element_y: np.ndarray,
    cutoff_distance: float,
    active_mask: Optional[np.ndarray] = None,
) -> float:
    """Compute the off-diagonal interaction-energy tail fraction.

    F_far(r) = sum_{i<j, d_ij > r} P_ij^2 / sum_{i<j} P_ij^2.
    The all-element version uses the full upper triangle and the active-pair
    version restricts the sum to active-active pairs only.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be square; got shape {P.shape}")
    element_x = np.asarray(element_x, dtype=np.float64).reshape(-1)
    element_y = np.asarray(element_y, dtype=np.float64).reshape(-1)
    if element_x.size != P.shape[0]:
        raise ValueError(f"element coordinates must have length {P.shape[0]}; got {element_x.size}")
    if element_x.shape != element_y.shape:
        raise ValueError("element_x and element_y must have the same shape")

    i_idx, j_idx, p_vals, _ = _pair_values_for_mask(P, P, active_mask)
    if i_idx.size == 0:
        return 0.0

    distances = np.hypot(element_x[i_idx] - element_x[j_idx], element_y[i_idx] - element_y[j_idx])
    denom = float(np.sum(p_vals ** 2))
    if denom == 0.0:
        return 0.0
    far = distances > float(cutoff_distance)
    return float(np.sum(p_vals[far] ** 2) / denom)


def interaction_energy_tail_curve(
    P: np.ndarray,
    element_x: np.ndarray,
    element_y: np.ndarray,
    cutoffs: Iterable[float],
    active_mask: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Evaluate interaction-energy tails across a sequence of cutoffs."""
    cutoffs = np.asarray(list(cutoffs), dtype=np.float64)
    values = []
    for cutoff in cutoffs:
        values.append({
            "cutoff_distance": float(cutoff),
            "tail_fraction": float(interaction_energy_tail(P, element_x, element_y, cutoff, active_mask=active_mask)),
        })
    return {"cutoffs": cutoffs.copy(), "tail_values": values}


def directional_coupling_stats(
    R: np.ndarray,
    active_mask: Optional[np.ndarray],
    element_x: np.ndarray,
    element_y: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Directional statistics on active unordered pairs.

    Angles are computed between each pair and reduced modulo 180 degrees so that
    orientation is symmetric under pair reversal. The angular classes are:

      - horizontal: [0, 22.5) U [157.5, 180)
      - diagonal: [22.5, 67.5) U [112.5, 157.5)
      - vertical: [67.5, 112.5)
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be a square matrix; got shape {R.shape}")
    element_x = np.asarray(element_x, dtype=np.float64).reshape(-1)
    element_y = np.asarray(element_y, dtype=np.float64).reshape(-1)
    if element_x.shape != element_y.shape:
        raise ValueError(f"element_x and element_y must match; got {element_x.shape} and {element_y.shape}")
    if element_x.size != R.shape[0]:
        raise ValueError(f"element coordinates must have length {R.shape[0]}; got {element_x.size}")

    i_idx, j_idx, _, r_vals = _pair_values_for_mask(R, R, active_mask)
    if i_idx.size == 0:
        return {
            "horizontal": {"pair_count": 0, "mean_abs_R": np.nan, "median_abs_R": np.nan, "p90_abs_R": np.nan, "max_abs_R": np.nan},
            "diagonal": {"pair_count": 0, "mean_abs_R": np.nan, "median_abs_R": np.nan, "p90_abs_R": np.nan, "max_abs_R": np.nan},
            "vertical": {"pair_count": 0, "mean_abs_R": np.nan, "median_abs_R": np.nan, "p90_abs_R": np.nan, "max_abs_R": np.nan},
        }

    dx = element_x[j_idx] - element_x[i_idx]
    dy = element_y[j_idx] - element_y[i_idx]
    angles = np.mod(np.degrees(np.arctan2(dy, dx)), 180.0)

    classes = {
        "horizontal": (angles < 22.5) | (angles >= 157.5),
        "diagonal": ((angles >= 22.5) & (angles < 67.5)) | ((angles >= 112.5) & (angles < 157.5)),
        "vertical": (angles >= 67.5) & (angles < 112.5),
    }

    out: dict[str, dict[str, float | int]] = {}
    for label, mask in classes.items():
        vals = r_vals[mask]
        abs_vals = np.abs(vals)
        if vals.size == 0:
            out[label] = {"pair_count": 0, "mean_abs_R": np.nan, "median_abs_R": np.nan, "p90_abs_R": np.nan, "max_abs_R": np.nan}
            continue
        out[label] = {
            "pair_count": int(vals.size),
            "mean_abs_R": float(np.mean(abs_vals)),
            "median_abs_R": float(np.median(abs_vals)),
            "p90_abs_R": float(np.quantile(abs_vals, 0.90)),
            "max_abs_R": float(np.max(abs_vals)),
            "mean_signed_R": float(np.mean(vals)),
            "fraction_positive_R": float(np.mean(vals > 0.0)),
            "fraction_negative_R": float(np.mean(vals < 0.0)),
        }
    return out


def filter_support_geometry(rmin: float, dx: float = 1.0, dy: float = 1.0) -> dict[str, float]:
    """Return the support geometry implied by the existing density-filter kernel.

    The filter kernel is constructed in element-count space as
        dist = sqrt(sum((abs(i - center) / rmin) ** 2))
    with weight max(0, 1 - dist). The support is therefore isotropic in index
    space, not a single fixed physical radius unless dx == dy. In the current
    M1 data, dx = dy = 1.0, so the support is effectively circular in physical
    coordinates with radius rmin. For general anisotropy, the axis and diagonal
    support lengths differ.
    """
    rmin = float(rmin)
    if rmin <= 0.0:
        raise ValueError(f"rmin must be positive; got {rmin}")
    dx = float(dx)
    dy = float(dy)
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError(f"dx and dy must both be positive; got {dx} and {dy}")
    return {
        "rmin_element_count": rmin,
        "rmin_physical_x": rmin * dx,
        "rmin_physical_y": rmin * dy,
        "rmin_physical_diagonal": rmin * np.hypot(dx, dy),
    }


def distance_conditioned_directional_coupling(
    R: np.ndarray,
    active_mask: Optional[np.ndarray],
    element_x: np.ndarray,
    element_y: np.ndarray,
    distance_bin_edges: Iterable[float],
    min_pairs: int = 10,
) -> dict[str, Any]:
    """Directional statistics for each distance bin.

    The output flags insufficient support with ``support_ok`` so directional
    comparisons are only made where there are enough active-active pairs in the
    bin.
    """
    R = np.asarray(R, dtype=np.float64)
    element_x = np.asarray(element_x, dtype=np.float64).reshape(-1)
    element_y = np.asarray(element_y, dtype=np.float64).reshape(-1)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square; got shape {R.shape}")
    if element_x.shape != element_y.shape or element_x.size != R.shape[0]:
        raise ValueError("element_x and element_y must match the element count in R")

    i_idx, j_idx, _, r_vals = _pair_values_for_mask(R, R, active_mask)
    distances = np.hypot(element_x[i_idx] - element_x[j_idx], element_y[i_idx] - element_y[j_idx])
    edges = np.asarray(list(distance_bin_edges), dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("distance_bin_edges must be a 1D array with at least two values")

    bin_ids = np.digitize(distances, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, edges.size - 2)
    dx = element_x[j_idx] - element_x[i_idx]
    dy = element_y[j_idx] - element_y[i_idx]
    angles = np.mod(np.degrees(np.arctan2(dy, dx)), 180.0)
    classes = {
        "horizontal": (angles < 22.5) | (angles >= 157.5),
        "diagonal": ((angles >= 22.5) & (angles < 67.5)) | ((angles >= 112.5) & (angles < 157.5)),
        "vertical": (angles >= 67.5) & (angles < 112.5),
    }

    out = {"distance_bin_edges": edges.copy(), "min_pairs": int(min_pairs), "bins": []}
    for k in range(edges.size - 1):
        in_bin = bin_ids == k
        if not np.any(in_bin):
            out["bins"].append({
                "bin_start": float(edges[k]),
                "bin_end": float(edges[k + 1]),
                "pair_count": 0,
                "bin_status": "empty_bin",
                "support_ok": False,
                "horizontal": {"pair_count": 0, "mean_abs_R": np.nan, "median_abs_R": np.nan, "p90_abs_R": np.nan, "max_abs_R": np.nan, "bin_status": "empty_bin", "support_ok": False},
                "diagonal": {"pair_count": 0, "mean_abs_R": np.nan, "median_abs_R": np.nan, "p90_abs_R": np.nan, "max_abs_R": np.nan, "bin_status": "empty_bin", "support_ok": False},
                "vertical": {"pair_count": 0, "mean_abs_R": np.nan, "median_abs_R": np.nan, "p90_abs_R": np.nan, "max_abs_R": np.nan, "bin_status": "empty_bin", "support_ok": False},
            })
            continue

        count = int(np.sum(in_bin))
        bin_status = _bin_status_for_count(count, min_pairs)
        local = {"bin_start": float(edges[k]), "bin_end": float(edges[k + 1]), "pair_count": count, "bin_status": bin_status, "support_ok": count >= int(min_pairs)}
        for label, mask in classes.items():
            vals = r_vals[in_bin & mask]
            abs_vals = np.abs(vals)
            direction_status = _bin_status_for_count(vals.size, min_pairs)
            if vals.size == 0:
                local[label] = {"pair_count": 0, "mean_abs_R": np.nan, "median_abs_R": np.nan, "p90_abs_R": np.nan, "max_abs_R": np.nan, "bin_status": direction_status, "support_ok": False}
            else:
                local[label] = {
                    "pair_count": int(vals.size),
                    "mean_abs_R": float(np.mean(abs_vals)),
                    "median_abs_R": float(np.median(abs_vals)),
                    "p90_abs_R": float(np.quantile(abs_vals, 0.90)),
                    "max_abs_R": float(np.max(abs_vals)),
                    "bin_status": direction_status,
                    "support_ok": vals.size >= int(min_pairs),
                }
        out["bins"].append(local)
    return out
