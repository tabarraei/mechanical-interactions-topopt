#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from topopt_utils.fea.spatial_interactions import (
    compute_distance_bin_stats,
    directional_coupling_stats,
    distance_conditioned_directional_coupling,
    filter_support_geometry,
    interaction_energy_tail,
    interaction_energy_tail_curve,
)


DEFAULT_M1_ROOT = Path("results") / "m1_interactions"
DEFAULT_OUT_ROOT = Path("m2_spatial")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _state_paths(m1_root: Path) -> list[Path]:
    paths: list[Path] = []
    for mesh_dir in sorted(m1_root.glob("*")):
        if mesh_dir.is_dir():
            paths.extend(sorted(mesh_dir.glob("*.npz")))
    return paths


def _select_reference_index(active_mask: np.ndarray, P: np.ndarray) -> int:
    if not np.any(active_mask):
        return 0
    active_idx = np.flatnonzero(active_mask)
    col_strength = np.abs(P[np.ix_(active_mask, active_mask)]).sum(axis=0)
    if col_strength.size == 0:
        return int(active_idx[0])
    return int(active_idx[int(np.argmax(col_strength))])


def _mesh_shape_from_state(state_data: dict[str, Any]) -> tuple[int, int]:
    mesh = state_data.get("mesh", {})
    return int(mesh.get("nelx", 1)), int(mesh.get("nely", 1))


def _distance_bin_edges(max_distance: float, num_bins: int = 12) -> np.ndarray:
    if not np.isfinite(max_distance) or max_distance <= 0.0:
        return np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
    return np.linspace(0.0, max_distance, num_bins + 1, dtype=np.float64)


def _summarize_state(npz_path: Path, shared_bin_edges: np.ndarray | None = None) -> dict[str, Any]:
    data = np.load(npz_path)
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
    bin_edges = np.asarray(shared_bin_edges, dtype=np.float64) if shared_bin_edges is not None else _distance_bin_edges(max_distance, num_bins=12)

    stats = compute_distance_bin_stats(P, R, active_mask, element_x, element_y, bin_edges)
    tail_curve = interaction_energy_tail_curve(P, element_x, element_y, np.linspace(0.0, max_distance, 20), active_mask=active_mask)
    directionals = directional_coupling_stats(R, active_mask, element_x, element_y)
    directional_by_bin = distance_conditioned_directional_coupling(R, active_mask, element_x, element_y, bin_edges, min_pairs=10)
    ref_idx = _select_reference_index(active_mask, P)
    ref_map = np.zeros((P.shape[0],), dtype=np.float64)
    ref_map[active_mask] = R[active_mask, ref_idx] if ref_idx < P.shape[0] else 0.0

    mesh_dir = npz_path.parent
    state_name = npz_path.stem
    json_path = mesh_dir / f"{state_name}.json"
    metadata = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    nelx, nely = _mesh_shape_from_state(metadata)
    if nelx == 1 and nely == 1:
        nelx = int(round(np.sqrt(P.shape[0])))
        nely = int(P.shape[0] / nelx)
    ref_map_2d = ref_map.reshape(nely, nelx)

    filter_geom = filter_support_geometry(2.0, dx=float(metadata.get("mesh", {}).get("dx", 1.0)), dy=float(metadata.get("mesh", {}).get("dy", 1.0)))
    physical_cutoffs = [float(filter_geom["rmin_physical_x"]) * k for k in (1, 2, 3)]
    all_tail = {}
    active_tail = {}
    for cutoff in physical_cutoffs:
        all_tail[float(cutoff)] = interaction_energy_tail(P, element_x, element_y, cutoff, active_mask=None)
        active_tail[float(cutoff)] = interaction_energy_tail(P, element_x, element_y, cutoff, active_mask=active_mask)

    return {
        "state_name": state_name,
        "mesh": metadata.get("mesh", {}),
        "state_type": metadata.get("state_type", "unknown"),
        "oc_iteration": metadata.get("oc_iteration"),
        "num_active_pairs": int(stats["num_active_pairs"]),
        "distance_bin_edges": stats["distance_bin_edges"].tolist(),
        "distance_bin_summary": stats["pair_statistics"],
        "tail_curve": tail_curve,
        "directional_stats": directionals,
        "directional_by_distance_bin": directional_by_bin,
        "filter_support": {
            "rmin_element_count": 2.0,
            "physical_cutoffs": physical_cutoffs,
            "all_element_energy_tails": {str(float(k)): float(v) for k, v in all_tail.items()},
            "active_pair_energy_tails": {str(float(k)): float(v) for k, v in active_tail.items()},
            "support_geometry": filter_geom,
        },
        "reference_index": int(ref_idx),
        "reference_map": ref_map_2d,
        "max_distance": max_distance,
    }


def _write_summary_csv(out_root: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "mesh",
        "state_name",
        "state_type",
        "oc_iteration",
        "active_pairs",
        "max_distance",
        "tail_at_0p25",
        "tail_at_0p50",
        "tail_at_0p75",
        "mean_abs_R_horizontal",
        "mean_abs_R_diagonal",
        "mean_abs_R_vertical",
    ]
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in results:
            tail_values = entry["tail_curve"]["tail_values"]
            r_dir = entry["directional_stats"]
            writer.writerow({
                "mesh": entry["mesh"].get("width", "") if isinstance(entry["mesh"], dict) else entry["mesh"],
                "state_name": entry["state_name"],
                "state_type": entry["state_type"],
                "oc_iteration": entry["oc_iteration"],
                "active_pairs": entry["num_active_pairs"],
                "max_distance": entry["max_distance"],
                "tail_at_0p25": tail_values[0]["tail_fraction"] if tail_values else np.nan,
                "tail_at_0p50": tail_values[len(tail_values) // 2]["tail_fraction"] if len(tail_values) > 1 else np.nan,
                "tail_at_0p75": tail_values[-1]["tail_fraction"] if tail_values else np.nan,
                "mean_abs_R_horizontal": r_dir["horizontal"].get("mean_abs_R", np.nan),
                "mean_abs_R_diagonal": r_dir["diagonal"].get("mean_abs_R", np.nan),
                "mean_abs_R_vertical": r_dir["vertical"].get("mean_abs_R", np.nan),
            })


def _plot_distance_decay(results: list[dict[str, Any]], out_root: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(12, 16), constrained_layout=True)
    axes = axes.ravel()
    for idx, entry in enumerate(results):
        ax = axes[idx]
        stats = entry["distance_bin_summary"]
        bins = [item["bin_start"] for item in stats]
        mean_abs = [item["mean_abs_R"] for item in stats]
        ax.plot(bins, mean_abs, marker="o", linewidth=1.8, label=entry["state_name"])
        ax.set_title(f"{entry['state_name']}")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Mean |R_ij|")
        ax.grid(alpha=0.3)
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.savefig(out_root / "distance_decay_mean_abs_R.png", dpi=200)
    fig.savefig(out_root / "distance_decay_mean_abs_R.pdf")
    plt.close(fig)


def _plot_energy_tail(results: list[dict[str, Any]], out_root: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(12, 16), constrained_layout=True)
    axes = axes.ravel()
    for idx, entry in enumerate(results):
        ax = axes[idx]
        tail_data = entry["tail_curve"]["tail_values"]
        x = [item["cutoff_distance"] for item in tail_data]
        y = [item["tail_fraction"] for item in tail_data]
        ax.plot(x, y, marker="o", linewidth=1.8)
        ax.set_title(f"{entry['state_name']}")
        ax.set_xlabel("Distance cutoff")
        ax.set_ylabel("F_far(r)")
        ax.grid(alpha=0.3)
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.savefig(out_root / "interaction_energy_tail.png", dpi=200)
    fig.savefig(out_root / "interaction_energy_tail.pdf")
    plt.close(fig)


def _plot_directional_summary(results: list[dict[str, Any]], out_root: Path) -> None:
    labels = ["horizontal", "diagonal", "vertical"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    for entry in results[:3]:
        vals = [entry["directional_stats"][k].get("mean_abs_R", np.nan) for k in labels]
        ax.plot(x, vals, marker="o", label=entry["state_name"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean |R_ij|")
    ax.set_title("Directional coupling summary (representative states)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_root / "directional_coupling_summary.png", dpi=200)
    fig.savefig(out_root / "directional_coupling_summary.pdf")
    plt.close(fig)


def _plot_reference_maps(results: list[dict[str, Any]], out_root: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(12, 16), constrained_layout=True)
    axes = axes.ravel()
    for idx, entry in enumerate(results):
        ax = axes[idx]
        ref_map = entry["reference_map"]
        if ref_map.size == 0:
            ax.axis("off")
            continue
        im = ax.imshow(ref_map, cmap="RdBu_r", origin="lower", vmin=-1.0, vmax=1.0)
        ax.set_title(f"{entry['state_name']}: active reference map")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.savefig(out_root / "reference_signed_R_maps.png", dpi=200)
    fig.savefig(out_root / "reference_signed_R_maps.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze M1 spatial nonlocality diagnostics.")
    parser.add_argument("--m1-root", type=Path, default=DEFAULT_M1_ROOT, help="M1 dataset directory containing mesh subfolders.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT, help="Directory for M2 spatial analysis outputs.")
    args = parser.parse_args()

    m1_root = args.m1_root.resolve()
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    state_paths = _state_paths(m1_root)
    if not state_paths:
        raise FileNotFoundError(f"No .npz state files were found under {m1_root}")

    mesh_max_distance: dict[tuple[int, int], float] = {}
    for path in state_paths:
        data = np.load(path)
        active_mask = np.asarray(data["active_mask"], dtype=bool)
        pair_indices = np.triu_indices(data["P"].shape[0], k=1)
        active_pair_mask = active_mask[pair_indices[0]] & active_mask[pair_indices[1]]
        element_x = np.asarray(data["element_x"], dtype=np.float64)
        element_y = np.asarray(data["element_y"], dtype=np.float64)
        pair_distances = np.hypot(
            element_x[pair_indices[0]] - element_x[pair_indices[1]],
            element_y[pair_indices[0]] - element_y[pair_indices[1]],
        )
        active_distances = pair_distances[active_pair_mask]
        max_distance = float(np.max(active_distances)) if active_distances.size else 1.0
        json_path = path.with_suffix(".json")
        metadata = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
        mesh = metadata.get("mesh", {})
        nelx = int(mesh.get("nelx", 1))
        nely = int(mesh.get("nely", 1))
        if nelx == 1 and nely == 1:
            nelx = int(round(np.sqrt(data["P"].shape[0])))
            nely = int(data["P"].shape[0] / nelx)
        mesh_key = (nelx, nely)
        mesh_max_distance[mesh_key] = max(mesh_max_distance.get(mesh_key, 0.0), max_distance)

    results: list[dict[str, Any]] = []
    for path in state_paths:
        json_path = path.with_suffix(".json")
        metadata = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
        mesh = metadata.get("mesh", {})
        nelx = int(mesh.get("nelx", 1))
        nely = int(mesh.get("nely", 1))
        if nelx == 1 and nely == 1:
            data = np.load(path)
            nelx = int(round(np.sqrt(data["P"].shape[0])))
            nely = int(data["P"].shape[0] / nelx)
        shared_edges = _distance_bin_edges(mesh_max_distance.get((nelx, nely), 1.0), num_bins=12)
        summary = _summarize_state(path, shared_bin_edges=shared_edges)
        if not summary["state_name"]:
            continue
        results.append(summary)
        (out_root / summary["state_name"]).mkdir(parents=True, exist_ok=True)
        (out_root / summary["state_name"] / "summary.json").write_text(json.dumps(_to_serializable(summary), indent=2), encoding="utf-8")

    if not results:
        raise RuntimeError(f"No M1 summary states could be generated from {m1_root}")

    _write_summary_csv(out_root, results)
    (out_root / "all_states_summary.json").write_text(json.dumps({"states": _to_serializable(results)}, indent=2), encoding="utf-8")
    _plot_distance_decay(results, out_root)
    _plot_energy_tail(results, out_root)
    _plot_directional_summary(results, out_root)
    _plot_reference_maps(results, out_root)

    print(f"M2 spatial diagnostics completed for {len(results)} states")
    for entry in results:
        print(
            f"{entry['state_name']:>18s} | active_pairs={entry['num_active_pairs']:>4d} | "
            f"max_d={entry['max_distance']:.3f} | F_far(0.5*max_d)={entry['tail_curve']['tail_values'][len(entry['tail_curve']['tail_values']) // 2]['tail_fraction']:.4f}"
        )


if __name__ == "__main__":
    main()
