import numpy as np

from topopt_utils.fea.solver import FeaSolver


def _make_cantilever_problem(nelx=12, nely=6):
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


def _smooth_density(nelx, nely):
    x = np.linspace(0.0, 1.0, nelx, endpoint=False)
    y = np.linspace(0.0, 1.0, nely, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")
    rho = 0.45 + 0.15 * np.sin(2.0 * np.pi * X) * np.cos(np.pi * Y)
    rho = np.clip(rho, 1e-3, 0.9)
    return rho.reshape(-1)


def test_hessian_vector_product_matches_finite_differences():
    nelx, nely = 12, 6
    forces, fixed_dofs, dx, dy = _make_cantilever_problem(nelx, nely)
    solver = FeaSolver()
    rho_ref = np.full(nelx * nely, 0.5, dtype=np.float64)
    v_ref = np.random.default_rng(0).normal(size=rho_ref.size)

    c0, g0 = solver.compliance(
        rho_ref,
        forces,
        fixed_dofs,
        nelx,
        nely,
        dx,
        dy,
        penal=3.0,
        sensitivities=True,
    )
    assert np.isfinite(c0)
    assert np.all(np.isfinite(g0))

    solver._hvp_stats = {"assemblies": 0, "factorizations": 0, "solves": 0}
    hv_ref = solver.hessian_vector_product(
        rho_ref,
        v_ref,
        forces,
        fixed_dofs,
        nelx,
        nely,
        dx,
        dy,
        penal=3.0,
    )
    assert hv_ref.shape == rho_ref.shape
    assert np.all(np.isfinite(hv_ref))
    assert np.all(np.isfinite(g0))
    assert solver._hvp_stats["assemblies"] == 1
    assert solver._hvp_stats["factorizations"] == 1
    assert solver._hvp_stats["solves"] == 2

    def grad_fn(rho):
        c, g = solver.compliance(
            rho,
            forces,
            fixed_dofs,
            nelx,
            nely,
            dx,
            dy,
            penal=3.0,
            sensitivities=True,
        )
        return g

    for eps in (1e-2, 1e-3, 1e-4, 1e-5):
        hv_fd = (grad_fn(rho_ref + eps * v_ref) - grad_fn(rho_ref - eps * v_ref)) / (2.0 * eps)
        rel_err = np.linalg.norm(hv_ref - hv_fd) / max(np.linalg.norm(hv_fd), 1e-14)
        assert rel_err < 1e-2, (eps, rel_err)

    rho_smooth = _smooth_density(nelx, nely)
    v1 = np.random.default_rng(1).normal(size=rho_smooth.size)
    v2 = np.random.default_rng(2).normal(size=rho_smooth.size)
    hv1 = solver.hessian_vector_product(
        rho_smooth,
        v1,
        forces,
        fixed_dofs,
        nelx,
        nely,
        dx,
        dy,
        penal=3.0,
    )
    hv2 = solver.hessian_vector_product(
        rho_smooth,
        v2,
        forces,
        fixed_dofs,
        nelx,
        nely,
        dx,
        dy,
        penal=3.0,
    )
    hv_sum = solver.hessian_vector_product(
        rho_smooth,
        v1 + v2,
        forces,
        fixed_dofs,
        nelx,
        nely,
        dx,
        dy,
        penal=3.0,
    )
    lin_err = np.linalg.norm(hv_sum - (hv1 + hv2)) / max(np.linalg.norm(hv_sum), 1e-14)
    assert lin_err < 1e-9, lin_err

    sym_err = abs(np.dot(v1, hv2) - np.dot(v2, hv1)) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-14)
    assert sym_err < 1e-8, sym_err

    U = solver.solve(
        rho_ref,
        forces,
        fixed_dofs,
        nelx,
        nely,
        dx,
        dy,
        penal=3.0,
    )
    assert np.allclose(U[fixed_dofs], 0.0, atol=1e-12)

    c_after, g_after = solver.compliance(
        rho_ref,
        forces,
        fixed_dofs,
        nelx,
        nely,
        dx,
        dy,
        penal=3.0,
        sensitivities=True,
    )
    assert np.allclose(c0, c_after)
    assert np.allclose(g0, g_after)
