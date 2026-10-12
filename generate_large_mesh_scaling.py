#!/usr/bin/env python
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from generate_m1_interactions import _make_topopt_sample, _trajectory_states
from topopt_utils.fea import get_default_solver
from topopt_utils.fea.interactions import normalize_mechanical_interactions
from topopt_utils.fea.spatial_interactions import (
    compute_distance_bin_stats,
    directional_coupling_stats,
    distance_conditioned_directional_coupling,
    interaction_energy_tail,
    pairwise_distances,
)

OUTPUT_ROOT = Path("results") / "m1_interactions_large_48x24"
ANALYSIS_ROOT = Path("m2_spatial_large")


def _uniform_rho(nelx: int, nely: int, volfrac: float = 0.5) -> np.ndarray:
    return np.full(nelx * nely, volfrac, dtype=np.float64)


def _element_centers(nelx: int, nely: int, dx: float = 1.0, dy: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    ex = (np.arange(nelx, dtype=np.float64) + 0.5) * dx
    ey = (np.arange(nely, dtype=np.float64) + 0.5) * dy
    element_x = np.repeat(ex[None, :], nely, axis=0).reshape(-1)
    element_y = np.repeat(ey[:, None], nelx, axis=1).reshape(-1)
    return element_x, element_y


def _make_force_vector(nelx: int, nely: int) -> np.ndarray:
    ndof = (nelx + 1) * (nely + 1) * 2
    forces = np.zeros(ndof, dtype=np.float64)
    tip_node = (nelx, nely // 2)
    tip_id = tip_node[1] * (nelx + 1) + tip_node[0]
    forces[2 * tip_id] = 1.0
    return forces


def _make_fixed_dofs(nelx: int, nely: int) -> np.ndarray:
    fixed_dofs: list[int] = []
    for y in range(nely + 1):
        node_id = y * (nelx + 1)
        fixed_dofs.extend([2 * node_id, 2 * node_id + 1])
    return np.asarray(sorted(set(fixed_dofs)), dtype=np.int64)


def _build_state_record(nelx: int, nely: int, rho: np.ndarray, *, state_type: str, oc_iteration: int | None, seed: int) -> dict[str, Any]:
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    dx = 1.0
    dy = 1.0
    forces = _make_force_vector(nelx, nely)
    fixed_dofs = _make_fixed_dofs(nelx, nely)
    solver = get_default_solver()
    P, D, H, info = solver.compliance_interaction_matrices(
        rho,
        forces,
        fixed_dofs,
        nelx,
        nely,
        dx,
        dy,
        penal=3.0,
        return_info=True,
    )
    R, active_mask = normalize_mechanical_interactions(P)
    element_x, element_y = _element_centers(nelx, nely, dx, dy)
    return {
        "mesh": {"nelx": nelx, "nely": nely, "dx": dx, "dy": dy, "width": dx * nelx, "height": dy * nely},
        "state_type": state_type,
        "oc_iteration": oc_iteration,
        "rho": rho,
        "P": np.asarray(P, dtype=np.float64),
        "D": np.asarray(D, dtype=np.float64),
        "H": np.asarray(H, dtype=np.float64),
        "R": np.asarray(R, dtype=np.float64),
        "active_mask": np.asarray(active_mask, dtype=bool),
        "element_x": element_x,
        "element_y": element_y,
        "forces": np.asarray(forces, dtype=np.float64),
        "fixed_dofs": np.asarray(fixed_dofs, dtype=np.int64),
        "filter_radius": 2.0,
        "seed": seed,
        "info": {"factorizations": int(info.get("factorizations", 0))},
    }


def _save_npz(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rho=record["rho"],
        P=record["P"],
        D=record["D"],
        H=record["H"],
        R=record["R"],
        active_mask=record["active_mask"],
        element_x=record["element_x"],
        element_y=record["element_y"],
        forces=record["forces"],
        fixed_dofs=record["fixed_dofs"],
    )
    (path.with_suffix(".json")).write_text(
        json.dumps(
            {
                "mesh": record["mesh"],
                "state_type": record["state_type"],
                "oc_iteration": record["oc_iteration"],
                "filter_radius": record["filter_radius"],
                "notes": "Inactive entries in R are masked placeholders set to zero; they are not measured zero couplings.",
                "seed": record["seed"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _distance_bin_edges(max_distance: float, num_bins: int = 12) -> np.ndarray:
    if not np.isfinite(max_distance) or max_distance <= 0.0:
        return np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
    return np.linspace(0.0, max_distance, num_bins + 1, dtype=np.float64)


def _vector_random(n: int, rng: np.random.Generator, *, normalization: str = "l2") -> np.ndarray:
    v = rng.standard_normal(n)
    return v / np.linalg.norm(v)


def _vector_smooth(n: int, nelx: int, nely: int, rng: np.random.Generator, *, normalization: str = "l2") -> np.ndarray:
    x = np.arange(nelx, dtype=np.float64) + 0.5
    y = np.arange(nely, dtype=np.float64) + 0.5
    xx, yy = np.meshgrid(x / max(1.0, nelx), y / max(1.0, nely), indexing="xy")
    v = np.sin(np.pi * xx) * np.cos(2.0 * np.pi * yy) + 0.5 * np.sin(2.0 * np.pi * xx)
    v = v.reshape(-1)
    v = v + 1e-6 * rng.standard_normal(v.size)
    return v / np.linalg.norm(v)


def _vector_localized(n: int, nelx: int, nely: int, rng: np.random.Generator, *, normalization: str = "l2") -> np.ndarray:
    x = np.arange(nelx, dtype=np.float64) + 0.5
    y = np.arange(nely, dtype=np.float64) + 0.5
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xc = (nelx - 1) * 0.33 + 0.5 * rng.standard_normal()
    yc = (nely - 1) * 0.67 + 0.5 * rng.standard_normal()
    sigma = max(1.0, min(nelx, nely) / 6.0)
    v = np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / (2.0 * sigma ** 2))
    v = v.reshape(-1)
    return v / np.linalg.norm(v)


def _operator_action_error(P: np.ndarray, distances: np.ndarray, cutoff: float, vectors: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be square; got {P.shape}")
    trunc = P.copy()
    trunc[distances > cutoff] = 0.0
    out: dict[str, dict[str, float]] = {}
    for name, v in vectors.items():
        ref = P @ v
        approx = trunc @ v
        ref_norm = float(np.linalg.norm(ref))
        err_norm = float(np.linalg.norm(ref - approx))
        if ref_norm < 1e-12:
            err = err_norm
            rel = float("nan")
        else:
            err = err_norm
            rel = err_norm / ref_norm
        out[name] = {"reference_norm": ref_norm, "absolute_error": err, "relative_error": rel, "cutoff": float(cutoff)}
    return out


def _vector_batch_random(n: int, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [_vector_random(n, rng) for _ in range(count)]


def _vector_batch_smooth(nelx: int, nely: int, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [_vector_smooth(nelx * nely, nelx, nely, rng) for _ in range(count)]


def _vector_batch_localized(nelx: int, nely: int, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [_vector_localized(nelx * nely, nelx, nely, rng) for _ in range(count)]


def _summarize_family_errors(P: np.ndarray, distances: np.ndarray, cutoff: float, vectors: list[np.ndarray], *, label: str) -> dict[str, Any]:
    trunc = P.copy()
    trunc[distances > cutoff] = 0.0
    rel_errors: list[float] = []
    abs_errors: list[float] = []
    near_zero = 0
    for v in vectors:
        ref = P @ v
        approx = trunc @ v
        ref_norm = float(np.linalg.norm(ref))
        err_norm = float(np.linalg.norm(ref - approx))
        abs_errors.append(err_norm)
        if ref_norm < 1e-12:
            near_zero += 1
            rel_errors.append(np.nan)
        else:
            rel_errors.append(err_norm / ref_norm)
    finite_rel = np.asarray([x for x in rel_errors if np.isfinite(x)], dtype=np.float64)
    summary = {
        "class": label,
        "num_vectors": int(len(vectors)),
        "median_relative_error": float(np.nanmedian(finite_rel)) if finite_rel.size else np.nan,
        "p90_relative_error": float(np.nanquantile(finite_rel, 0.90)) if finite_rel.size else np.nan,
        "max_relative_error": float(np.nanmax(finite_rel)) if finite_rel.size else np.nan,
        "median_absolute_error": float(np.median(abs_errors)) if abs_errors else np.nan,
        "near_zero_reference_actions": int(near_zero),
        "finite_relative_error_count": int(finite_rel.size),
    }
    return summary


def _summarize_state(state_file: Path, shared_edges: np.ndarray | None = None) -> dict[str, Any]:
    data = np.load(state_file)
    P = np.asarray(data["P"], dtype=np.float64)
    R = np.asarray(data["R"], dtype=np.float64)
    active_mask = np.asarray(data["active_mask"], dtype=bool)
    element_x = np.asarray(data["element_x"], dtype=np.float64)
    element_y = np.asarray(data["element_y"], dtype=np.float64)

    pair_indices = np.triu_indices(P.shape[0], k=1)
    active_pair_mask = active_mask[pair_indices[0]] & active_mask[pair_indices[1]]
    pair_distances = np.hypot(
        element_x[pair_indices[0]] - element_x[pair_indices[1]],
        element_y[pair_indices[0]] - element_y[pair_indices[1]],
    )
    active_distances = pair_distances[active_pair_mask]
    max_distance = float(np.max(active_distances)) if active_distances.size else 1.0
    bin_edges = shared_edges if shared_edges is not None else _distance_bin_edges(max_distance, num_bins=12)

    stats = compute_distance_bin_stats(P, R, active_mask, element_x, element_y, bin_edges)
    directionals = directional_coupling_stats(R, active_mask, element_x, element_y)
    directional_by_bin = distance_conditioned_directional_coupling(R, active_mask, element_x, element_y, bin_edges, min_pairs=10)
    info = {
        "state_name": state_file.stem,
        "mesh": {
            "nelx": int(P.shape[0] if P.shape[0] == 1 else np.sqrt(P.shape[0]) if np.isclose(np.sqrt(P.shape[0]) % 1, 0) else P.shape[0]),
            "nely": int(P.shape[0] if P.shape[0] == 1 else P.shape[0] / (np.sqrt(P.shape[0]) if np.isclose(np.sqrt(P.shape[0]) % 1, 0) else 1.0)),
        },
        "distance_bin_edges": bin_edges.copy().tolist(),
        "distance_bin_summary": stats["pair_statistics"],
        "directional_stats": directionals,
        "directional_by_distance_bin": directional_by_bin,
        "P_symmetry_abs_max": float(np.max(np.abs(P - P.T))),
        "P_psd_min_eig": float(np.min(np.linalg.eigvalsh(P))),
        "R_norm_error": float(np.max(np.abs(R - R.T))),
        "R_active_diag_min": float(np.min(np.diag(R)[active_mask])) if np.any(active_mask) else 0.0,
        "R_active_diag_max": float(np.max(np.diag(R)[active_mask])) if np.any(active_mask) else 0.0,
        "active_count": int(np.count_nonzero(active_mask)),
        "all_element_tails": {str(float(cutoff)): float(interaction_energy_tail(P, element_x, element_y, cutoff, active_mask=None)) for cutoff in (2.0, 4.0, 6.0)},
        "active_pair_tails": {str(float(cutoff)): float(interaction_energy_tail(P, element_x, element_y, cutoff, active_mask=active_mask)) for cutoff in (2.0, 4.0, 6.0)},
    }
    return info


def _generate_48x24_states(output_root: Path) -> list[dict[str, Any]]:
    nelx, nely = 48, 24
    output_root.mkdir(parents=True, exist_ok=True)
    states: list[dict[str, Any]] = []
    seed = 123
    rng = np.random.default_rng(seed)

    uniform = _uniform_rho(nelx, nely, volfrac=0.5)
    uniform_record = _build_state_record(nelx, nely, uniform, state_type="uniform", oc_iteration=None, seed=seed)
    uniform_path = output_root / "uniform_state.npz"
    _save_npz(uniform_path, uniform_record)
    states.append({"state_name": "uniform_state", "record": uniform_record, "file": uniform_path})

    traj = _trajectory_states(nelx, nely, seed=seed)
    for label, density, oc_iter in traj:
        if label not in {"middle", "late"}:
            continue
        rec = _build_state_record(nelx, nely, density, state_type=label, oc_iteration=oc_iter, seed=seed)
        path = output_root / f"{label}_state_{oc_iter}.npz"
        _save_npz(path, rec)
        states.append({"state_name": f"{label}_state_{oc_iter}", "record": rec, "file": path})
    return states


def _summaries_for_states(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    n = len(states)
    pair_count = np.triu_indices(states[0]["record"]["P"].shape[0], k=1)
    state_summaries: list[dict[str, Any]] = []
    for item in states:
        record = item["record"]
        P = record["P"]
        R = record["R"]
        active_mask = record["active_mask"]
        element_x = record["element_x"]
        element_y = record["element_y"]
        pair_indices = np.triu_indices(P.shape[0], k=1)
        active_pair_mask = active_mask[pair_indices[0]] & active_mask[pair_indices[1]]
        pair_distances = np.hypot(
            element_x[pair_indices[0]] - element_x[pair_indices[1]],
            element_y[pair_indices[0]] - element_y[pair_indices[1]],
        )
        active_distances = pair_distances[active_pair_mask]
        max_distance = float(np.max(active_distances)) if active_distances.size else 1.0
        bin_edges = _distance_bin_edges(max_distance, num_bins=12)
        stats = compute_distance_bin_stats(P, R, active_mask, element_x, element_y, bin_edges)
        directional = directional_coupling_stats(R, active_mask, element_x, element_y)
        directional_by_bin = distance_conditioned_directional_coupling(R, active_mask, element_x, element_y, bin_edges, min_pairs=10)
        dist_matrix = np.hypot(element_x[:, None] - element_x[None, :], element_y[:, None] - element_y[None, :])
        vector_sets = {
            "random": _vector_batch_random(P.shape[0], 20, seed=7 + n),
            "smooth": _vector_batch_smooth(48, 24, 20, seed=11 + n),
            "localized": _vector_batch_localized(48, 24, 20, seed=13 + n),
        }
        action_errors = {}
        for cutoff in (2.0, 4.0, 6.0):
            action_errors[str(cutoff)] = _operator_action_error(P, dist_matrix, cutoff, {k: v[0] for k, v in vector_sets.items()})
        operator_action_summary = {}
        for cutoff in (2.0, 4.0, 6.0):
            operator_action_summary[str(cutoff)] = {
                "num_vectors_per_class": {k: len(v) for k, v in vector_sets.items()},
                "classes": {
                    k: _summarize_family_errors(P, dist_matrix, cutoff, v, label=k)
                    for k, v in vector_sets.items()
                },
            }
        state_summaries.append(
            {
                "state_name": item["state_name"],
                "state_type": item["record"]["state_type"],
                "oc_iteration": item["record"]["oc_iteration"],
                "mesh": item["record"]["mesh"],
                "active_count": int(np.count_nonzero(active_mask)),
                "active_pair_count": int(np.count_nonzero(active_pair_mask)),
                "max_distance": max_distance,
                "distance_bin_edges": bin_edges.tolist(),
                "distance_bin_summary": stats["pair_statistics"],
                "directional_stats": directional,
                "directional_by_distance_bin": directional_by_bin,
                "all_element_energy_tails": {str(float(cutoff)): float(interaction_energy_tail(P, element_x, element_y, cutoff, active_mask=None)) for cutoff in (2.0, 4.0, 6.0)},
                "active_pair_energy_tails": {str(float(cutoff)): float(interaction_energy_tail(P, element_x, element_y, cutoff, active_mask=active_mask)) for cutoff in (2.0, 4.0, 6.0)},
                "P_symmetry_abs_max": float(np.max(np.abs(P - P.T))),
                "P_psd_min_eig": float(np.min(np.linalg.eigvalsh(P))),
                "R_symmetry_abs_max": float(np.max(np.abs(R - R.T))),
                "R_active_mean_diag": float(np.mean(np.diag(R)[active_mask])) if np.any(active_mask) else 0.0,
                "R_normalization_max_error": float(np.max(np.abs(np.diag(R)[active_mask] - 1.0))) if np.any(active_mask) else 0.0,
                "operator_action_errors": action_errors,
                "operator_action_summary": operator_action_summary,
            }
        )
    return state_summaries


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _write_summary_csv(path: Path, entries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "state_name",
        "state_type",
        "oc_iteration",
        "active_count",
        "max_distance",
        "all_tail_2",
        "all_tail_4",
        "all_tail_6",
        "active_tail_2",
        "active_tail_4",
        "active_tail_6",
        "p_psd_min_eig",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        import csv
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "state_name": entry["state_name"],
                    "state_type": entry["state_type"],
                    "oc_iteration": entry["oc_iteration"],
                    "active_count": entry["active_count"],
                    "max_distance": entry["max_distance"],
                    "all_tail_2": entry["all_element_energy_tails"]["2.0"],
                    "all_tail_4": entry["all_element_energy_tails"]["4.0"],
                    "all_tail_6": entry["all_element_energy_tails"]["6.0"],
                    "active_tail_2": entry["active_pair_energy_tails"]["2.0"],
                    "active_tail_4": entry["active_pair_energy_tails"]["4.0"],
                    "active_tail_6": entry["active_pair_energy_tails"]["6.0"],
                    "p_psd_min_eig": entry["P_psd_min_eig"],
                }
            )


def main() -> None:
    start = time.perf_counter()
    output_root = OUTPUT_ROOT
    analysis_root = ANALYSIS_ROOT
    analysis_root.mkdir(parents=True, exist_ok=True)

    states = _generate_48x24_states(output_root)
    summaries = _summaries_for_states(states)
    manifest = {
        "mesh": [48, 24],
        "dx": 1.0,
        "dy": 1.0,
        "rmin": 2.0,
        "states": [
            {
                "state_name": s["state_name"],
                "state_type": s["state_type"],
                "oc_iteration": s["oc_iteration"],
                "mesh": s["mesh"],
                "active_count": s["active_count"],
                "max_distance": s["max_distance"],
            }
            for s in summaries
        ],
        "vector_normalizations": {
            "random": "unit L2 norm, i.i.d. Gaussian",
            "smooth": "unit L2 norm, low-frequency sine/cosine field on the 48x24 grid",
            "localized": "unit L2 norm, Gaussian bump centered near one-third x and two-third y",
        },
        "cutoffs": [2.0, 4.0, 6.0],
        "notes": "Dense P is stored in full form for the 48x24 exact M1 interaction study; matrix truncation retains entries satisfying d <= cutoff only.",
        "runtime_seconds": time.perf_counter() - start,
    }
    serial_manifest = _to_serializable(manifest)
    serial_summaries = _to_serializable({"states": summaries})
    (analysis_root / "manifest.json").write_text(json.dumps(serial_manifest, indent=2), encoding="utf-8")
    (analysis_root / "summary.json").write_text(json.dumps(serial_summaries, indent=2), encoding="utf-8")
    _write_summary_csv(analysis_root / "summary.csv", summaries)

    # Estimate memory and print a compact report.
    n = 48 * 24
    dense_P_bytes = n * n * 8
    dense_state_bytes = 4 * dense_P_bytes
    print(f"48x24 states generated: {len(states)}")
    print(f"dense P bytes: {dense_P_bytes / (1024**2):.3f} MiB")
    print(f"dense state working set estimate: {dense_state_bytes / (1024**2):.3f} MiB")
    for entry in summaries:
        print(
            f"{entry['state_name']}: active={entry['active_count']} | "
            f"max_d={entry['max_distance']:.3f} | "
            f"tail_2={entry['all_element_energy_tails']['2.0']:.6f} | "
            f"tail_4={entry['all_element_energy_tails']['4.0']:.6f} | "
            f"tail_6={entry['all_element_energy_tails']['6.0']:.6f} | "
            f"sym={entry['P_symmetry_abs_max']:.3e} | psd_min={entry['P_psd_min_eig']:.3e}"
        )
    print(f"Manifest written to {analysis_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
