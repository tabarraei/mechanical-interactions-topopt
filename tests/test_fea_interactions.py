import numpy as np

from topopt_utils.fea.solver import FeaSolver


def _make_cantilever_problem(nelx: int, nely: int):
    dx = 1.0
    dy = 1.0
    ndof = (nelx + 1) * (nely + 1) * 2
    forces = np.zeros(ndof, dtype=np.float64)
    tip_node = (nelx, nely // 2)
    tip_id = tip_node[1] * (nelx + 1) + tip_node[0]
    forces[2 * tip_id] = 1.0

    fixed_dofs = []
    for y in range(nely + 1):
        node_id = y * (nelx + 1)
        fixed_dofs.extend([2 * node_id, 2 * node_id + 1])
    fixed_dofs = np.asarray(sorted(set(fixed_dofs)), dtype=np.int64)
    return forces, fixed_dofs, dx, dy


def _smooth_density(nelx: int, nely: int):
    x = np.linspace(0.0, 1.0, nelx, endpoint=False)
    y = np.linspace(0.0, 1.0, nely, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")
    rho = 0.45 + 0.15 * np.sin(2.0 * np.pi * X) * np.cos(np.pi * Y)
    return np.clip(rho, 1e-3, 0.9).reshape(-1)


def _check_psd(P: np.ndarray):
    S = 0.5 * (P + P.T)
    eig = np.linalg.eigvalsh(S)
    largest = max(float(np.max(np.abs(eig))), 1.0)
    tol = 1e-10 * largest
    return eig, tol


def test_dense_interaction_matrices_for_small_cantilevers():
    solver = FeaSolver()
    rng = np.random.default_rng(123)

    cases = [
        (12, 6, np.full(12 * 6, 0.5, dtype=np.float64)),
        (12, 6, _smooth_density(12, 6)),
        (24, 12, np.full(24 * 12, 0.5, dtype=np.float64)),
    ]

    for nelx, nely, rho in cases:
        forces, fixed_dofs, dx, dy = _make_cantilever_problem(nelx, nely)
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

        ndof = (nelx + 1) * (nely + 1) * 2
        assert P.shape == (nelx * nely, nelx * nely)
        assert D.shape == (nelx * nely,)
        assert H.shape == (nelx * nely, nelx * nely)
        assert info["A"].shape == (ndof, nelx * nely)
        assert np.all(np.isfinite(P))
        assert np.all(np.isfinite(D))
        assert np.all(np.isfinite(H))
        assert np.all(D >= -1e-12)

        symP = np.linalg.norm(P - P.T) / max(np.linalg.norm(P), 1e-14)
        symH = np.linalg.norm(H - H.T) / max(np.linalg.norm(H), 1e-14)
        assert symP < 1e-10, symP
        assert symH < 1e-10, symH

        eig, tol = _check_psd(P)
        assert eig.min() >= -tol, (eig.min(), tol)

        np.testing.assert_allclose(H, 2.0 * P - np.diag(D), rtol=1e-12, atol=1e-12)

        A_dense = info["A"]
        assert np.allclose(A_dense[fixed_dofs, :], 0.0, atol=1e-12)
        assert info["factorizations"] == 1
        assert info["assembly_time"] >= 0.0
        assert info["factorization_time"] >= 0.0
        assert info["equilibrium_solve_time"] >= 0.0
        assert info["a_assembly_time"] >= 0.0
        assert info["rhs_solve_time"] >= 0.0
        assert info["p_construction_time"] >= 0.0
        assert info["total_time"] >= 0.0

        for _ in range(3):
            v = rng.standard_normal(rho.size)
            hv_mat = H @ v
            hv_hvp = solver.hessian_vector_product(
                rho,
                v,
                forces,
                fixed_dofs,
                nelx,
                nely,
                dx,
                dy,
                penal=3.0,
            )
            rel_err = np.linalg.norm(hv_mat - hv_hvp) / max(np.linalg.norm(hv_hvp), 1e-14)
            assert rel_err < 1e-9, (nelx, nely, rel_err)

        # Linearity check on a random direction pair.
        v1 = rng.standard_normal(rho.size)
        v2 = rng.standard_normal(rho.size)
        hv1 = H @ v1
        hv2 = H @ v2
        hv12 = H @ (v1 + v2)
        lin_err = np.linalg.norm(hv12 - (hv1 + hv2)) / max(np.linalg.norm(hv12), 1e-14)
        assert lin_err < 1e-12, lin_err

        assert np.all(np.isfinite(info["A"]))
        assert np.all(np.isfinite(info["u"]))
        assert np.allclose(info["u"][fixed_dofs], 0.0, atol=1e-12)

    print("Dense interaction-matrix verification passed for all requested cantilever cases.")
