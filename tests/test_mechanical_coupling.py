import numpy as np

from topopt_utils.fea.interactions import normalize_mechanical_interactions
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


def _report_case_statistics(name: str, P: np.ndarray):
    R, active_mask = normalize_mechanical_interactions(P)
    active = np.arange(P.shape[0])[active_mask]
    active_R = R[np.ix_(active_mask, active_mask)]
    sym_err = np.linalg.norm(R - R.T) / max(np.linalg.norm(R), 1e-14)
    eigvals = np.linalg.eigvalsh(0.5 * (active_R + active_R.T))
    tol = 1e-10 * max(float(np.max(np.abs(eigvals))), 1.0)
    psd_err = max(0.0, -eigvals.min())
    entries = active_R[~np.eye(active_R.shape[0], dtype=bool)] if active_R.shape[0] > 1 else np.empty((0,), dtype=np.float64)
    rmin = float(np.min(entries)) if entries.size else 1.0
    rmax = float(np.max(entries)) if entries.size else 1.0
    return {
        "name": name,
        "active_count": int(active.size),
        "active_fraction": float(active.size / P.shape[0]),
        "symmetry_error": sym_err,
        "psd_error": psd_err,
        "tol": tol,
        "rmin": rmin,
        "rmax": rmax,
        "active_mask": active_mask,
        "R": R,
    }


def test_normalized_mechanical_coupling():
    solver = FeaSolver()
    rng = np.random.default_rng(234)

    cases = [
        ("12x6 uniform", 12, 6, np.full(12 * 6, 0.5, dtype=np.float64)),
        ("12x6 smooth", 12, 6, _smooth_density(12, 6)),
        ("24x12 uniform", 24, 12, np.full(24 * 12, 0.5, dtype=np.float64)),
    ]

    stats = []
    for name, nelx, nely, rho in cases:
        forces, fixed_dofs, dx, dy = _make_cantilever_problem(nelx, nely)
        P, D, H, _ = solver.compliance_interaction_matrices(
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
        P_copy = P.copy()
        R, active_mask = normalize_mechanical_interactions(P)
        assert np.array_equal(P, P_copy)
        assert R.dtype == np.float64
        assert active_mask.dtype == bool
        assert R.shape == P.shape
        assert np.all(np.isfinite(R))
        assert np.all(np.isfinite(active_mask))

        active_R = R[np.ix_(active_mask, active_mask)]
        if active_R.size:
            diag = np.diag(active_R)
            assert np.allclose(diag, 1.0, atol=1e-12, rtol=1e-12)
            assert np.all(np.abs(active_R) <= 1.0 + 1e-10)
            sym_err = np.linalg.norm(active_R - active_R.T) / max(np.linalg.norm(active_R), 1e-14)
            assert sym_err < 1e-12, sym_err
            eigvals = np.linalg.eigvalsh(0.5 * (active_R + active_R.T))
            tol = 1e-10 * max(float(np.max(np.abs(eigvals))), 1.0)
            assert eigvals.min() >= -tol, (eigvals.min(), tol)

        if np.any(~active_mask):
            assert np.all(R[~active_mask, :] == 0.0)
            assert np.all(R[:, ~active_mask] == 0.0)
            assert np.all(np.isfinite(R[~active_mask, :]))

        case_stats = _report_case_statistics(name, P)
        stats.append(case_stats)

    # Synthetic PSD Gram matrix with both positive and negative off-diagonal entries.
    X = np.array(
        [
            [1.0, 0.0],
            [0.5, -1.0],
            [0.2, 1.5],
            [-0.3, 0.8],
        ],
        dtype=np.float64,
    )
    G = X @ X.T
    G = G.astype(np.float64)
    R_G, active_G = normalize_mechanical_interactions(G)
    assert np.allclose(R_G[active_G, :][:, active_G], R_G[np.ix_(active_G, active_G)])
    assert np.allclose(np.diag(R_G[active_G][:, active_G]), 1.0, atol=1e-12)
    eigvals = np.linalg.eigvalsh(0.5 * (R_G[np.ix_(active_G, active_G)] + R_G[np.ix_(active_G, active_G)].T))
    assert eigvals.min() >= -1e-12 * max(float(np.max(np.abs(eigvals))), 1.0)

    # Inactive zero-diagonal matrix.
    P_zero = np.array(
        [[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    R_zero, active_zero = normalize_mechanical_interactions(P_zero)
    assert active_zero.dtype == bool
    assert np.all(np.isfinite(R_zero))
    assert np.all(R_zero[~active_zero, :] == 0.0)
    assert np.all(R_zero[:, ~active_zero] == 0.0)

    # Positive rescaling invariance.
    scale = 7.5
    R_scale, active_scale = normalize_mechanical_interactions(scale * P)
    assert np.array_equal(active_mask, active_scale)
    assert np.allclose(R, R_scale, atol=1e-12, rtol=1e-12)

    # HVP compatibility check for a representative case.
    rho = np.full(12 * 6, 0.5, dtype=np.float64)
    forces, fixed_dofs, dx, dy = _make_cantilever_problem(12, 6)
    P12, D12, H12, _ = solver.compliance_interaction_matrices(
        rho,
        forces,
        fixed_dofs,
        12,
        6,
        dx,
        dy,
        penal=3.0,
        return_info=True,
    )
    v = rng.standard_normal(rho.size)
    H_dense = (2.0 * P12 - np.diag(D12))
    hv_dense = H_dense @ v
    hv_hvp = solver.hessian_vector_product(
        rho,
        v,
        forces,
        fixed_dofs,
        12,
        6,
        dx,
        dy,
        penal=3.0,
    )
    rel_err = np.linalg.norm(hv_dense - hv_hvp) / max(np.linalg.norm(hv_hvp), 1e-14)
    assert rel_err < 1e-9, rel_err

    # Summaries for the final report.
    for item in stats:
        print(
            f"{item['name']}: active={item['active_count']}/{P.shape[0]} "
            f"({item['active_fraction']:.4f}), symmetry={item['symmetry_error']:.3e}, "
            f"psd_error={item['psd_error']:.3e}, rmin={item['rmin']:.6e}, rmax={item['rmax']:.6e}"
        )

    print("Mechanical coupling normalization checks passed.")
