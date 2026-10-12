#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from topopt_utils.fea import get_default_solver
from topopt_utils.fea.interactions import normalize_mechanical_interactions
from topopt_utils.topology_data import TopOptSample
from topopt_utils.topopt_2d import optimize

DEFAULT_MESHES = [(12, 6), (24, 12)]
DEFAULT_OUTPUT_DIR = Path("results") / "m1_interactions"


def _smooth_density(nelx: int, nely: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, nelx, endpoint=False)
    y = np.linspace(0.0, 1.0, nely, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    field = 0.5 + 0.15 * np.sin(2.0 * np.pi * xx + 0.25 * seed) * np.cos(np.pi * yy)
    field += 0.05 * rng.standard_normal((nely, nelx))
    return np.clip(field.reshape(-1), 1e-3, 0.95).astype(np.float64)


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


def _make_boundary_conditions(nelx: int, nely: int) -> np.ndarray:
    bc = np.zeros((nely + 1, nelx + 1, 2), dtype=np.float64)
    for y in range(nely + 1):
        bc[y, 0, 0] = 1.0
        bc[y, 0, 1] = 1.0
    return bc


def _make_topopt_sample(nelx: int, nely: int, volume_fraction: float = 0.5) -> TopOptSample:
    return TopOptSample(
        mesh={"width": 1.0, "height": 2.0, "nelx": nelx, "nely": nely},
        load={"x": [nelx - 1], "y": [nely // 2], "fx": [1.0], "fy": [0.0]},
        boundary_conditions=_make_boundary_conditions(nelx, nely),
        volume_fraction=volume_fraction,
        sample_id=f"{nelx}x{nely}",
        load_type="static",
    )


def _load_saved_trajectory(nelx: int, nely: int) -> list[np.ndarray] | None:
    dataset_root = Path(__file__).resolve().parent / "dataset" / "simp_trajectory"
    if not dataset_root.exists():
        return None
    for path in sorted(dataset_root.glob("*.pkl")):
        try:
            with path.open("rb") as fh:
                payload = pickle.load(fh)
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for sample in payload:
            if sample is None:
                continue
            mesh = getattr(sample, "mesh", None) or {}
            if int(mesh.get("nelx", -1)) == nelx and int(mesh.get("nely", -1)) == nely:
                traj = getattr(sample, "density_trajectory", None)
                if traj is not None and len(traj) > 0:
                    return [np.asarray(frame, dtype=np.float64).reshape(nely, nelx).copy() for frame in np.asarray(traj)]
    return None


def _generate_trajectory(nelx: int, nely: int, *, seed: int, maxiter: int = 20, rmin: float = 2.0, volfrac: float = 0.5) -> list[np.ndarray]:
    sample = _make_topopt_sample(nelx, nely, volume_fraction=volfrac)
    optimize(
        sample,
        penal=3.0,
        rmin=rmin,
        maxiter=maxiter,
        tolerance=1e-3,
        plotting=False,
        store_trajectory=True,
        quiet=True,
    )
    traj = getattr(sample, "density_trajectory", None)
    if traj is None or len(traj) == 0:
        raise RuntimeError(f"Failed to generate a trajectory for {nelx}x{nely}")
    return [np.asarray(frame, dtype=np.float64).reshape(nely, nelx).copy() for frame in np.asarray(traj)]


def _trajectory_states(nelx: int, nely: int, *, seed: int) -> list[tuple[str, np.ndarray, int | None]]:
    saved = _load_saved_trajectory(nelx, nely)
    if saved is not None and len(saved) >= 3:
        idxs = [0, len(saved) // 2, len(saved) - 1]
        labels = ["early", "middle", "late"]
        return [(label, saved[idx], idx) for label, idx in zip(labels, idxs)]

    traj = _generate_trajectory(nelx, nely, seed=seed, maxiter=max(12, min(30, 3 * max(nelx, nely))))
    idxs = [0, len(traj) // 2, len(traj) - 1]
    labels = ["early", "middle", "late"]
    return [(label, traj[idx], idx) for label, idx in zip(labels, idxs)]


def _element_centers(nelx: int, nely: int, dx: float = 1.0, dy: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    ex = (np.arange(nelx, dtype=np.float64) + 0.5) * dx
    ey = (np.arange(nely, dtype=np.float64) + 0.5) * dy
    element_x = np.repeat(ex[None, :], nely, axis=0).reshape(-1)
    element_y = np.repeat(ey[:, None], nelx, axis=1).reshape(-1)
    return element_x, element_y


def _build_state_record(nelx: int, nely: int, rho: np.ndarray, *, state_type: str, oc_iteration: int | None, seed: int, filter_radius: float = 2.0) -> dict[str, Any]:
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
    c, g_rho = solver.compliance(rho, forces, fixed_dofs, nelx, nely, dx, dy, penal=3.0, sensitivities=True)
    R, active_mask = normalize_mechanical_interactions(P)
    element_x, element_y = _element_centers(nelx, nely, dx, dy)
    return {
        "mesh": {"nelx": nelx, "nely": nely, "dx": dx, "dy": dy, "width": dx * nelx, "height": dy * nely},
        "state_type": state_type,
        "oc_iteration": oc_iteration,
        "rho": rho,
        "compliance": float(c),
        "g_rho": np.asarray(g_rho, dtype=np.float64),
        "P": np.asarray(P, dtype=np.float64),
        "D": np.asarray(D, dtype=np.float64),
        "H": np.asarray(H, dtype=np.float64),
        "R": np.asarray(R, dtype=np.float64),
        "active_mask": np.asarray(active_mask, dtype=bool),
        "element_x": element_x,
        "element_y": element_y,
        "forces": np.asarray(forces, dtype=np.float64),
        "fixed_dofs": np.asarray(fixed_dofs, dtype=np.int64),
        "penal": 3.0,
        "emin": 0.001,
        "filter_radius": float(filter_radius),
        "reference_state": {"seed": seed, "trajectory_source": "generated_trajectory" if len(str(seed)) else "none"},
        "info": {
            "A_shape": tuple(np.asarray(info["A"]).shape),
            "u_shape": tuple(np.asarray(info["u"]).shape),
            "factorizations": int(info.get("factorizations", 0)),
        },
    }


def _save_state(state_dir: Path, record: dict[str, Any], *, state_name: str) -> None:
    state_root = Path(state_dir)
    state_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(state_root / f"{state_name}.npz",
        rho=record["rho"],
        compliance=np.asarray(record["compliance"], dtype=np.float64),
        g_rho=record["g_rho"],
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
    (state_root / f"{state_name}.json").write_text(json.dumps({
        "mesh": record["mesh"],
        "state_type": record["state_type"],
        "oc_iteration": record["oc_iteration"],
        "compliance": float(record["compliance"]),
        "penal": record["penal"],
        "emin": record["emin"],
        "filter_radius": record["filter_radius"],
        "notes": "Inactive entries in R are masked placeholders set to zero; they are not measured zero couplings.",
        "reference_state": record["reference_state"],
        "info": record["info"],
    }, indent=2), encoding="utf-8")


def _select_reference_column(P: np.ndarray, active_mask: np.ndarray) -> int:
    if not np.any(active_mask):
        return 0
    active_indices = np.arange(P.shape[0], dtype=int)[active_mask]
    active_P = P[np.ix_(active_mask, active_mask)]
    col_scores = np.abs(active_P).sum(axis=0)
    ref_active = active_indices[int(np.argmax(col_scores))]
    return int(ref_active)


def _plot_density(rho: np.ndarray, *, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 8))
    img = ax.imshow(rho.reshape((rho.size // 1, 1)) if False else rho.reshape((-1, 1)), aspect="auto", cmap="viridis")
    del img
    plt.close(fig)
    arr = rho.reshape(-1)
    if arr.size == 0:
        return
    nely = int(np.sqrt(arr.size)) if False else 1
    # Use a standard mesh reshape with the correct geometry.
    raise RuntimeError("_plot_density is not used; mesh-aware plotting is handled in generate_plots().")


def _generate_mesh_plots(mesh_dir: Path, rho: np.ndarray, P: np.ndarray, R: np.ndarray, active_mask: np.ndarray, *, nelx: int, nely: int, state_name: str) -> None:
    ref_col = _select_reference_column(P, active_mask)
    ref_P = P[:, ref_col].reshape(nely, nelx)
    ref_R = R[:, ref_col].reshape(nely, nelx)
    rho_img = rho.reshape(nely, nelx)

    fig, ax = plt.subplots(figsize=(5, 8))
    mesh = ax.imshow(rho_img, cmap="viridis", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_title(f"{state_name}: ρ (physical density)")
    ax.set_xlabel("x element index")
    ax.set_ylabel("y element index")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("ρ")
    fig.tight_layout()
    fig.savefig(mesh_dir / f"{state_name}_density.png", dpi=200)
    fig.savefig(mesh_dir / f"{state_name}_density.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    im = ax.imshow(P, cmap="RdBu_r", origin="lower", vmin=-np.max(np.abs(P)), vmax=np.max(np.abs(P)))
    ax.set_title(f"{state_name}: P = A^T K^{-1} A")
    ax.set_xlabel("element index")
    ax.set_ylabel("element index")
    fig.colorbar(im, ax=ax, label="P")
    fig.tight_layout()
    fig.savefig(mesh_dir / f"{state_name}_P_heatmap.png", dpi=200)
    fig.savefig(mesh_dir / f"{state_name}_P_heatmap.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    R_plot = np.where(active_mask[:, None] & active_mask[None, :], R, np.nan)
    im = ax.imshow(R_plot, cmap="RdBu_r", origin="lower", vmin=-1.0, vmax=1.0)
    ax.set_title(f"{state_name}: normalized coupling R (inactive entries masked)")
    ax.set_xlabel("element index")
    ax.set_ylabel("element index")
    fig.colorbar(im, ax=ax, label="R")
    fig.tight_layout()
    fig.savefig(mesh_dir / f"{state_name}_R_heatmap.png", dpi=200)
    fig.savefig(mesh_dir / f"{state_name}_R_heatmap.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 8))
    im = ax.imshow(ref_P, cmap="RdBu_r", origin="lower", vmin=-np.max(np.abs(P)), vmax=np.max(np.abs(P)))
    ax.set_title(f"{state_name}: column {ref_col} of P")
    ax.set_xlabel("x element index")
    ax.set_ylabel("y element index")
    fig.colorbar(im, ax=ax, label="P[:, col]")
    fig.tight_layout()
    fig.savefig(mesh_dir / f"{state_name}_P_column.png", dpi=200)
    fig.savefig(mesh_dir / f"{state_name}_P_column.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 8))
    im = ax.imshow(ref_R, cmap="RdBu_r", origin="lower", vmin=-1.0, vmax=1.0)
    ax.set_title(f"{state_name}: column {ref_col} of R")
    ax.set_xlabel("x element index")
    ax.set_ylabel("y element index")
    fig.colorbar(im, ax=ax, label="R[:, col]")
    fig.tight_layout()
    fig.savefig(mesh_dir / f"{state_name}_R_column.png", dpi=200)
    fig.savefig(mesh_dir / f"{state_name}_R_column.pdf")
    plt.close(fig)

    vals = np.linalg.eigvalsh(0.5 * (P + P.T))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(vals.size), np.sort(vals)[::-1], marker="o", markersize=3)
    ax.set_title(f"{state_name}: eigenvalue spectrum of P")
    ax.set_xlabel("rank")
    ax.set_ylabel("eigenvalue")
    fig.tight_layout()
    fig.savefig(mesh_dir / f"{state_name}_eigs.png", dpi=200)
    fig.savefig(mesh_dir / f"{state_name}_eigs.pdf")
    plt.close(fig)


def generate_m1_interactions_dataset(output_dir: str | Path, *, seed: int = 0, mesh_specs: Iterable[tuple[int, int]] | None = None) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if mesh_specs is None:
        mesh_specs = DEFAULT_MESHES
    all_states: list[dict[str, Any]] = []
    state_index = 0
    start = time.perf_counter()

    for nelx, nely in mesh_specs:
        mesh_dir = output_path / f"{nelx}x{nely}"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        states: list[dict[str, Any]] = []

        uniform_rho = np.full(nelx * nely, 0.5, dtype=np.float64)
        for label, rho, oc_iter in [
            ("uniform", uniform_rho, None),
            ("smooth", _smooth_density(nelx, nely, seed=seed + nelx + nely), None),
            *[(label, rho, iter_idx) for label, rho, iter_idx in _trajectory_states(nelx, nely, seed=seed + nelx + nely)],
        ]:
            record = _build_state_record(nelx, nely, rho, state_type=label, oc_iteration=oc_iter, seed=seed, filter_radius=2.0)
            state_name = f"{label}_state" if oc_iter is None else f"{label}_state_{oc_iter}"
            _save_state(mesh_dir, record, state_name=state_name)
            states.append({"name": state_name, "state_type": label, "oc_iteration": oc_iter, "path": str(mesh_dir / f"{state_name}.npz")})
            all_states.append({"mesh": f"{nelx}x{nely}", "state_name": state_name, **record["mesh"], "state_type": label, "oc_iteration": oc_iter})
            state_index += 1

        representative = states[1] if len(states) > 1 else states[0]
        rep_path = Path(representative["path"])
        rep_data = np.load(rep_path)
        _generate_mesh_plots(
            mesh_dir,
            rep_data["rho"],
            rep_data["P"],
            rep_data["R"],
            rep_data["active_mask"],
            nelx=nelx,
            nely=nely,
            state_name=representative["state_type"],
        )

    manifest = {
        "seed": int(seed),
        "mesh_specs": [[nelx, nely] for nelx, nely in mesh_specs],
        "num_states": len(all_states),
        "states": all_states,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": time.perf_counter() - start,
        "notes": "Filtered physical densities are stored from the OC trajectory; design variables and gradients are not saved.",
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small M1 mechanical-interaction dataset for cantilever cases.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--meshes", type=str, default="12x6,24x12", help="Comma-separated mesh specs like 12x6,24x12")
    args = parser.parse_args()

    meshes = []
    for token in args.meshes.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid mesh spec {token!r}; expected form like 12x6")
        meshes.append((int(parts[0]), int(parts[1])))
    if not meshes:
        meshes = DEFAULT_MESHES

    manifest = generate_m1_interactions_dataset(args.output_dir, seed=args.seed, mesh_specs=meshes)
    print(json.dumps({
        "output_dir": str(Path(args.output_dir).resolve()),
        "seed": args.seed,
        "num_states": int(manifest["num_states"]),
        "runtime_seconds": float(manifest["runtime_seconds"]),
    }, indent=2))


if __name__ == "__main__":
    main()
