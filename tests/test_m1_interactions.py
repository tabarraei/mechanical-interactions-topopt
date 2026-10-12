import json
import numpy as np

from generate_m1_interactions import generate_m1_interactions_dataset
from topopt_utils.fea.solver import FeaSolver


def _load_all_state_paths(mesh_dir):
    return sorted(mesh_dir.glob("*.npz"))


def _state_summary(npz_path):
    data = np.load(npz_path)
    return {
        "rho": data["rho"],
        "P": data["P"],
        "D": data["D"],
        "H": data["H"],
        "R": data["R"],
        "active_mask": data["active_mask"],
        "forces": data["forces"],
        "fixed_dofs": data["fixed_dofs"],
        "element_x": data["element_x"],
        "element_y": data["element_y"],
        "mesh": json.loads((npz_path.parent / (npz_path.stem + ".json")).read_text(encoding="utf-8")),
    }


def test_generate_m1_interactions_dataset(tmp_path):
    out_dir = tmp_path / "m1_dataset"
    manifest = generate_m1_interactions_dataset(out_dir, seed=11)

    assert (out_dir / "manifest.json").exists()
    assert manifest["num_states"] == 10
    assert len(manifest["states"]) == 10

    for nelx, nely in [(12, 6), (24, 12)]:
        mesh_dir = out_dir / f"{nelx}x{nely}"
        assert mesh_dir.is_dir()
        state_paths = _load_all_state_paths(mesh_dir)
        assert len(state_paths) == 5

        rho_states = []
        for state_path in state_paths:
            state = _state_summary(state_path)
            rho = state["rho"]
            P = state["P"]
            D = state["D"]
            H = state["H"]
            R = state["R"]
            active = state["active_mask"]
            n_e = nelx * nely

            assert rho.shape == (n_e,)
            assert P.shape == (n_e, n_e)
            assert D.shape == (n_e,)
            assert H.shape == (n_e, n_e)
            assert R.shape == (n_e, n_e)
            assert active.shape == (n_e,)
            assert state["forces"].shape == ((nelx + 1) * (nely + 1) * 2,)
            assert state["fixed_dofs"].ndim == 1
            assert state["element_x"].shape == (n_e,)
            assert state["element_y"].shape == (n_e,)
            assert np.all(np.isfinite(P))
            assert np.all(np.isfinite(H))
            assert np.all(np.isfinite(R))

            np.testing.assert_allclose(H, 2.0 * P - np.diag(D), rtol=1e-12, atol=1e-12)
            assert np.linalg.norm(P - P.T) / max(np.linalg.norm(P), 1e-14) < 1e-10
            assert np.linalg.norm(H - H.T) / max(np.linalg.norm(H), 1e-14) < 1e-10

            eigvals = np.linalg.eigvalsh(0.5 * (P + P.T))
            tol = 1e-10 * max(float(np.max(np.abs(eigvals))), 1.0)
            assert eigvals.min() >= -tol, (eigvals.min(), tol)

            if np.any(active):
                active_R = R[np.ix_(active, active)]
                assert np.allclose(np.diag(active_R), 1.0, atol=1e-12, rtol=1e-12)
                assert np.all(np.abs(active_R) <= 1.0 + 1e-10)
                assert np.all(R[~active, :] == 0.0)
                assert np.all(R[:, ~active] == 0.0)
                assert np.all(np.isfinite(active_R))
            rho_states.append(rho)

        distinct = any(not np.allclose(rho_states[0], rho) for rho in rho_states[1:])
        assert distinct

        expected_x = np.tile((np.arange(nelx, dtype=np.float64) + 0.5), nely)
        expected_y = np.repeat((np.arange(nely, dtype=np.float64) + 0.5), nelx)
        for state_path in state_paths:
            state = _state_summary(state_path)
            np.testing.assert_allclose(state["element_x"], expected_x, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(state["element_y"], expected_y, rtol=1e-12, atol=1e-12)

            rho = state["rho"]
            forces = state["forces"]
            fixed_dofs = state["fixed_dofs"]
            solver = FeaSolver()
            v = np.random.default_rng(123).standard_normal(rho.size)
            P = state["P"]
            D = state["D"]
            H = state["H"]
            hv_dense = H @ v
            hv_hvp = solver.hessian_vector_product(
                rho,
                v,
                forces,
                fixed_dofs,
                nelx,
                nely,
                1.0,
                1.0,
                penal=3.0,
            )
            rel_err = np.linalg.norm(hv_dense - hv_hvp) / max(np.linalg.norm(hv_hvp), 1e-14)
            assert rel_err < 1e-8, rel_err

    out_dir_repro = tmp_path / "m1_dataset_repro"
    manifest_repro = generate_m1_interactions_dataset(out_dir_repro, seed=11)
    assert manifest["states"] == manifest_repro["states"]
