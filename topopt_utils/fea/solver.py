"""Cached FEA solver with mesh-aware connectivity reuse.

Connectivity (element DOF map + COO index pattern) depends only on mesh.
Value assembly and numeric factorization still run when rho or 
boundary conditions change.

Boundary conditions are imposed by zeroing the fixed rows/columns and placing 1 on
those diagonals rather than deleting the constrained DOFs. That keeps the sparsity
structure a function of the mesh alone, so the assembly pattern is built once per
mesh and reused even when the boundary conditions differ on every sample.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse

DEFAULT_MKL_NUM_THREADS = 16 # Dont change unless you redo timing tests for FEA solver


def _configure_mkl_threads(default: int = DEFAULT_MKL_NUM_THREADS) -> None:
    """Cap MKL/OpenMP threads unless the caller already set them."""
    default_s = str(int(default))
    os.environ.setdefault("MKL_NUM_THREADS", default_s)
    os.environ.setdefault("OMP_NUM_THREADS", default_s)


_configure_mkl_threads()

MeshKey = Tuple[int, int]
KeKey = Tuple[float, float]


@dataclass
class MeshConnectivity:
    """Density-independent sparse assembly pattern for a regular Q4 grid."""

    nelx: int
    nely: int
    ndof: int
    element_dofs: np.ndarray  # (n_e, 8)
    row_indices: np.ndarray  # (64 * n_e,)
    col_indices: np.ndarray  # (64 * n_e,)


@dataclass
class MeshPattern:
    """Upper-triangular CSR sparsity over *all* DOFs (mesh-only).

    Dirichlet conditions are imposed by zeroing the fixed rows and columns and
    placing 1 on those diagonals rather than deleting them, which keeps the
    sparsity structure — and hence the symbolic factorization — independent of the
    boundary conditions. ``diag_slots[i]`` is the ``data`` index of entry ``(i, i)``.

    Layout is the upper triangle in CSR, which is what MKL PARDISO expects for a
    symmetric matrix.
    """

    ndof: int
    nnz: int
    indices: np.ndarray  # (nnz,) CSR column indices
    indptr: np.ndarray  # (ndof + 1,) CSR row pointers
    scatter: np.ndarray  # (64 * n_e,)
    diag_slots: np.ndarray  # (ndof,)


@dataclass
class CacheStats:
    connectivity_hits: int = 0
    connectivity_misses: int = 0
    ke_hits: int = 0
    ke_misses: int = 0
    pattern_hits: int = 0
    pattern_misses: int = 0
    symbolic_hits: int = 0
    symbolic_misses: int = 0
    symbolic_evictions: int = 0

    def reset(self) -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, 0)


def _build_connectivity(nelx: int, nely: int) -> MeshConnectivity:
    num_elems = nelx * nely
    ndof = (nelx + 1) * (nely + 1) * 2

    # DOF order per element must match ``__get_e_dofs__``: nodes run clockwise from
    # the top left (ax, ay, bx, by, dx, dy, cx, cy), as ``__get_KE__`` assumes.
    grid_rows = np.arange(nely, dtype=np.int32)
    grid_cols = np.arange(nelx, dtype=np.int32)
    a = ((nelx + 1) * grid_rows[:, None] + grid_cols[None, :]).ravel()  # top-left node
    c = a + nelx + 1  # bottom-left node
    element_dofs = np.empty((num_elems, 8), dtype=np.int32)
    element_dofs[:, 0] = 2 * a
    element_dofs[:, 1] = 2 * a + 1
    element_dofs[:, 2] = 2 * a + 2
    element_dofs[:, 3] = 2 * a + 3
    element_dofs[:, 4] = 2 * c + 2
    element_dofs[:, 5] = 2 * c + 3
    element_dofs[:, 6] = 2 * c
    element_dofs[:, 7] = 2 * c + 1

    # COO expansion: each element contributes an 8×8 block
    rows = np.repeat(element_dofs, 8, axis=1).ravel()
    cols = np.tile(element_dofs, (1, 8)).ravel()
    return MeshConnectivity(
        nelx=nelx,
        nely=nely,
        ndof=ndof,
        element_dofs=element_dofs,
        row_indices=rows,
        col_indices=cols,
    )


def _build_mesh_pattern(conn: MeshConnectivity) -> MeshPattern:
    """All-DOF upper-triangular CSR structure plus scatter and diagonal slot maps."""
    ndof = conn.ndof
    rows = conn.row_indices.astype(np.int64, copy=False)
    cols = conn.col_indices.astype(np.int64, copy=False)
    keep = np.flatnonzero(rows <= cols)
    kept_rows, kept_cols = rows[keep], cols[keep]

    struct = scipy.sparse.coo_matrix(
        (np.ones(keep.size, dtype=np.int8), (kept_rows, kept_cols)),
        shape=(ndof, ndof),
    ).tocsr()
    nnz = int(struct.nnz)

    major = np.repeat(np.arange(ndof, dtype=np.int64), np.diff(struct.indptr))
    lin = major * ndof + struct.indices
    entry_key = kept_rows * ndof + kept_cols
    scatter = np.full(rows.size, nnz, dtype=np.int32)
    scatter[keep] = np.searchsorted(lin, entry_key).astype(np.int32)

    # Every DOF belongs to at least one element, so each diagonal entry exists
    dofs = np.arange(ndof, dtype=np.int64)
    diag_slots = np.searchsorted(lin, dofs * ndof + dofs).astype(np.int32)

    return MeshPattern(
        ndof=ndof,
        nnz=nnz,
        indices=struct.indices.astype(np.int32, copy=False),
        indptr=struct.indptr.astype(np.int32, copy=False),
        scatter=scatter,
        diag_slots=diag_slots,
    )


_MKL_RT_RESOLVED = False


def _ensure_mkl_rt() -> None:
    """Pin the ``mkl_rt`` path in the environment before constructing a handle.

    When ``find_library`` cannot see MKL, pypardiso falls back to a recursive glob
    of the whole prefix on *every* ``PyPardisoSolver()`` construction, which costs
    ~2.2 s here. Resolving it once and exporting ``PYPARDISO_MKL_RT`` makes each
    subsequent construction free.
    """
    global _MKL_RT_RESOLVED
    if _MKL_RT_RESOLVED or os.environ.get("PYPARDISO_MKL_RT"):
        _MKL_RT_RESOLVED = True
        return

    from ctypes.util import find_library

    if find_library("mkl_rt") is None and find_library("mkl_rt.1") is None:
        import glob
        import site

        matches = glob.glob(
            f"{sys.prefix}/[Ll]ib*/**/*mkl_rt*", recursive=True
        ) or glob.glob(f"{site.USER_BASE}/[Ll]ib*/**/*mkl_rt*", recursive=True)
        if matches:
            os.environ["PYPARDISO_MKL_RT"] = sorted(matches, key=len)[0]
    _MKL_RT_RESOLVED = True


class _PardisoContext:
    """One PARDISO handle holding a cached symbolic analysis for a fixed structure."""

    PHASE_SYMBOLIC = 11
    PHASE_NUMERIC = 22
    PHASE_SOLVE = 33

    def __init__(self, A: scipy.sparse.csr_matrix):
        _ensure_mkl_rt()
        import pypardiso

        self.solver = pypardiso.PyPardisoSolver(mtype=2)
        # Configures iparm for CSR input the way the wrapper expects
        self.solver._check_A(A)
        self._scratch = np.zeros(A.shape[0], dtype=np.float64)
        self.solver.set_phase(self.PHASE_SYMBOLIC)
        self.solver._call_pardiso(A, self._scratch)
        self.has_numeric = False
        self._A: Optional[scipy.sparse.csr_matrix] = None

    def factorize(self, A: scipy.sparse.csr_matrix) -> None:
        self.solver.set_phase(self.PHASE_NUMERIC)
        self.solver._call_pardiso(A, self._scratch)
        self.has_numeric = True
        self._A = A

    def solve(self, b: np.ndarray) -> np.ndarray:
        if not self.has_numeric or self._A is None:
            raise RuntimeError("PARDISO context has no numeric factorization")
        self.solver.set_phase(self.PHASE_SOLVE)
        return self.solver._call_pardiso(self._A, np.ascontiguousarray(b))

    def free(self) -> None:
        try:
            self.solver.free_memory(everything=True)
        except Exception:  # noqa: BLE001
            pass
        self._A = None
        self.has_numeric = False


class FeaSolver:
    """Initializable FEA solver that reuses structure across overlapping setups."""

    def __init__(
        self,
        penal: float = 3.0,
        max_symbolic_cache: int = 8,
    ):
        try:
            import pypardiso  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "FeaSolver requires pypardiso (pip install pypardiso)"
            ) from exc
        _ensure_mkl_rt()
        self.penal = float(penal)
        # PARDISO handles hold native memory, so the symbolic cache is bounded
        self.max_symbolic_cache = int(max_symbolic_cache)
        self.stats = CacheStats()
        self._connectivity_cache: Dict[MeshKey, MeshConnectivity] = {}
        self._ke_cache: Dict[KeKey, np.ndarray] = {}
        self._mesh_pattern_cache: Dict[MeshKey, MeshPattern] = {}
        # Keyed by mesh: Dirichlet assembly makes the structure BC-independent
        self._symbolic_cache: "OrderedDict[MeshKey, _PardisoContext]" = OrderedDict()

    def get_connectivity(self, nelx: int, nely: int) -> MeshConnectivity:
        key: MeshKey = (int(nelx), int(nely))
        conn = self._connectivity_cache.get(key)
        if conn is None:
            self.stats.connectivity_misses += 1
            conn = _build_connectivity(key[0], key[1])
            self._connectivity_cache[key] = conn
        else:
            self.stats.connectivity_hits += 1
        return conn

    def get_KE(self, dx: float, dy: float) -> np.ndarray:
        from .q4 import __get_KE__

        key: KeKey = (float(dx), float(dy))
        KE = self._ke_cache.get(key)
        if KE is None:
            self.stats.ke_misses += 1
            KE = __get_KE__(hx=key[0], hy=key[1])
            self._ke_cache[key] = KE
        else:
            self.stats.ke_hits += 1
        return KE

    def get_mesh_pattern(
        self, nelx: int, nely: int, *, conn: Optional[MeshConnectivity] = None
    ) -> MeshPattern:
        key: MeshKey = (int(nelx), int(nely))
        pattern = self._mesh_pattern_cache.get(key)
        if pattern is None:
            self.stats.pattern_misses += 1
            if conn is None:
                conn = self.get_connectivity(nelx, nely)
            pattern = _build_mesh_pattern(conn)
            self._mesh_pattern_cache[key] = pattern
        else:
            self.stats.pattern_hits += 1
        return pattern

    def assemble_dirichlet(
        self,
        rho: np.ndarray,
        pattern: MeshPattern,
        KE: np.ndarray,
        conn: MeshConnectivity,
        fixed_dofs: np.ndarray,
        penal: Optional[float] = None,
    ) -> scipy.sparse.csr_matrix:
        """All-DOF upper CSR with fixed rows/cols zeroed and a unit diagonal there.

        The free-free block is untouched, so the solution there matches the reduced
        system exactly while the fixed DOFs come back as zero.
        """
        from .q4 import __simp_stiffness__

        if penal is None:
            penal = self.penal
        rho = np.asarray(rho, dtype=np.float64).reshape(-1)
        x_penal = __simp_stiffness__(rho, penal)
        blocks = (x_penal[:, None] * KE.reshape(1, 64)).ravel()

        fixed = np.asarray(fixed_dofs, dtype=np.int64).ravel()
        is_fixed = np.zeros(pattern.ndof, dtype=bool)
        is_fixed[fixed] = True
        # Drop every element contribution touching a constrained DOF
        blocks *= ~(is_fixed[conn.row_indices] | is_fixed[conn.col_indices])

        data = np.bincount(pattern.scatter, weights=blocks, minlength=pattern.nnz + 1)
        data[pattern.diag_slots[fixed]] = 1.0
        A = scipy.sparse.csr_matrix(
            (data[: pattern.nnz], pattern.indices, pattern.indptr),
            shape=(pattern.ndof, pattern.ndof),
            copy=False,
        )
        # Built from a canonical ordering, so PARDISO needs no re-sort
        A.has_sorted_indices = True
        return A

    def _get_symbolic(self, key, A: scipy.sparse.csr_matrix) -> _PardisoContext:
        ctx = self._symbolic_cache.get(key)
        if ctx is None:
            self.stats.symbolic_misses += 1
            while len(self._symbolic_cache) >= max(self.max_symbolic_cache, 1):
                _, evicted = self._symbolic_cache.popitem(last=False)
                evicted.free()
                self.stats.symbolic_evictions += 1
            ctx = _PardisoContext(A)
            self._symbolic_cache[key] = ctx
        else:
            self.stats.symbolic_hits += 1
            self._symbolic_cache.move_to_end(key)
        return ctx

    def solve(
        self,
        rho: np.ndarray,
        forces: np.ndarray,
        fixed_dofs: np.ndarray,
        nelx: int,
        nely: int,
        dx: float,
        dy: float,
        *,
        penal: Optional[float] = None,
        KE: Optional[np.ndarray] = None,
        profile: bool = False,
    ):
        if penal is None:
            penal = self.penal
        times: Dict[str, float] = {}
        import time as _time

        def _tic():
            return _time.perf_counter()

        t0 = _tic()
        conn = self.get_connectivity(nelx, nely)
        times["connectivity"] = _tic() - t0

        if KE is None:
            KE = self.get_KE(dx, dy)

        rho_flat = np.asarray(rho, dtype=np.float64).reshape(-1)
        F = np.asarray(forces, dtype=np.float64).reshape(-1)

        t0 = _tic()
        pattern = self.get_mesh_pattern(nelx, nely, conn=conn)
        times["restrict"] = _tic() - t0

        # Fixed DOFs carry no load; they solve to exactly zero
        F_bc = F.copy()
        F_bc[np.asarray(fixed_dofs, dtype=np.int64).ravel()] = 0.0

        t0 = _tic()
        K = self.assemble_dirichlet(
            rho_flat, pattern, KE, conn, fixed_dofs, penal=penal
        )
        times["assemble"] = _tic() - t0

        t0 = _tic()
        ctx = self._get_symbolic((int(nelx), int(nely)), K)
        # Only nonzero when this mesh needed a fresh symbolic analysis
        times["symbolic"] = _tic() - t0

        t0 = _tic()
        ctx.factorize(K)
        times["factorize"] = _tic() - t0

        t0 = _tic()
        U = ctx.solve(F_bc)
        times["solve"] = _tic() - t0

        if profile:
            times["total"] = sum(
                times[k]
                for k in (
                    "connectivity",
                    "assemble",
                    "restrict",
                    "symbolic",
                    "factorize",
                    "solve",
                )
            )
            return U, times
        return U

    def compliance(
        self,
        rho: np.ndarray,
        forces: np.ndarray,
        fixed_dofs: np.ndarray,
        nelx: int,
        nely: int,
        dx: float,
        dy: float,
        *,
        penal: Optional[float] = None,
        profile: bool = False,
        sensitivities: bool = False,
    ):
        """Discrete compliance ``FᵀU`` for the FEA solution.

        If ``sensitivities`` is True, also return unfiltered SIMP ``dc/dρ``.
        """
        from .q4 import __simp_stiffness_sensitivity__

        if profile:
            U, times = self.solve(
                rho,
                forces,
                fixed_dofs,
                nelx,
                nely,
                dx,
                dy,
                penal=penal,
                profile=True,
            )
        else:
            U = self.solve(
                rho, forces, fixed_dofs, nelx, nely, dx, dy, penal=penal, profile=False
            )
            times = None

        import time as _time

        t0 = _time.perf_counter()
        c = float(np.asarray(forces, dtype=np.float64).ravel() @ U)
        dc = None
        if sensitivities:
            if penal is None:
                penal = self.penal
            KE = self.get_KE(dx, dy)
            element_dofs = self.get_connectivity(nelx, nely).element_dofs
            rho_flat = np.asarray(rho, dtype=np.float64).reshape(-1)
            energies = np.einsum(
                "ij,jk,ik->i", U[element_dofs], KE, U[element_dofs]
            )
            dc = (
                -__simp_stiffness_sensitivity__(rho_flat, penal) * energies
            ).astype(np.float64)
        if profile:
            times["compliance"] = _time.perf_counter() - t0
            times["total"] = times.get("total", 0.0) + times["compliance"]
            if sensitivities:
                return c, dc, times
            return c, times
        if sensitivities:
            return c, dc
        return c

    def hessian_vector_product(
        self,
        rho: np.ndarray,
        v: np.ndarray,
        forces: np.ndarray,
        fixed_dofs: np.ndarray,
        nelx: int,
        nely: int,
        dx: float,
        dy: float,
        *,
        penal: Optional[float] = None,
        KE: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Exact compliance Hessian–vector product wrt physical densities rho.

        Implements H_rho v = 2 A^T K^{-1} A v - D v in the element ordering used
        by the existing SIMP compliance sensitivity. The method does not construct
        a dense Hessian or a dense A matrix.
        """
        from .q4 import __simp_stiffness_sensitivity__, __simp_stiffness_second_derivative__

        if not hasattr(self, "_hvp_stats"):
            self._hvp_stats = {"assemblies": 0, "factorizations": 0, "solves": 0}

        if penal is None:
            penal = self.penal

        rho_flat = np.asarray(rho, dtype=np.float64).reshape(-1)
        v_flat = np.asarray(v, dtype=np.float64).reshape(-1)
        if rho_flat.size != nelx * nely:
            raise ValueError(f"rho must have length {nelx * nely}, got {rho_flat.size}")
        if v_flat.size != rho_flat.size:
            raise ValueError(f"v must have length {rho_flat.size}, got {v_flat.size}")

        conn = self.get_connectivity(nelx, nely)
        if KE is None:
            KE = self.get_KE(dx, dy)
        pattern = self.get_mesh_pattern(nelx, nely, conn=conn)

        F = np.asarray(forces, dtype=np.float64).reshape(-1)
        fixed = np.asarray(fixed_dofs, dtype=np.int64).ravel()
        F_bc = F.copy()
        F_bc[fixed] = 0.0

        self._hvp_stats["assemblies"] += 1
        K = self.assemble_dirichlet(
            rho_flat, pattern, KE, conn, fixed, penal=penal
        )
        ctx = self._get_symbolic((int(nelx), int(nely)), K)
        ctx.factorize(K)
        self._hvp_stats["factorizations"] += 1

        u = ctx.solve(F_bc)
        self._hvp_stats["solves"] += 1

        dE = __simp_stiffness_sensitivity__(rho_flat, penal)
        d2E = __simp_stiffness_second_derivative__(rho_flat, penal)
        element_dofs = conn.element_dofs
        Ue = u[element_dofs]
        KE_u = np.einsum("ij,ej->ei", KE, Ue)
        a_local = dE[:, None] * KE_u

        b = np.zeros(K.shape[0], dtype=np.float64)
        np.add.at(b, element_dofs, v_flat[:, None] * a_local)
        b[fixed] = 0.0

        z = ctx.solve(b)
        self._hvp_stats["solves"] += 1

        aTz = np.einsum("ei,ei->e", a_local, z[element_dofs])
        energies = np.einsum("ei,ij,ej->e", Ue, KE, Ue)
        hv = 2.0 * aTz - d2E * v_flat * energies
        return hv.astype(np.float64, copy=False)

    def compliance_interaction_matrices(
        self,
        rho: np.ndarray,
        forces: np.ndarray,
        fixed_dofs: np.ndarray,
        nelx: int,
        nely: int,
        dx: float,
        dy: float,
        *,
        penal: Optional[float] = None,
        KE: Optional[np.ndarray] = None,
        max_dense_bytes: int = 256 * 1024 * 1024,
        max_dense_elems: Optional[int] = None,
        return_info: bool = False,
    ):
        """Dense interaction matrices for small physical-density compliance problems.

        The dense allocation is governed by the footprint of the interaction tensors:
        ``P`` and ``H`` are ``Ne x Ne``; ``A`` and ``Z`` are ``ndof x Ne``; ``D`` has
        length ``Ne``. The guard therefore uses an estimated peak-memory budget in
        bytes rather than a raw mesh-element count to avoid ambiguous behavior.
        """
        from time import perf_counter
        from .q4 import (
            __simp_stiffness_sensitivity__,
            __simp_stiffness_second_derivative__,
        )

        if not hasattr(self, "_interaction_stats"):
            self._interaction_stats = {"assemblies": 0, "factorizations": 0, "solves": 0}

        if penal is None:
            penal = self.penal

        rho_flat = np.asarray(rho, dtype=np.float64).reshape(-1)
        if rho_flat.size != nelx * nely:
            raise ValueError(f"rho must have length {nelx * nely}, got {rho_flat.size}")

        ndof = (nelx + 1) * (nely + 1) * 2
        num_elems = nelx * nely
        dense_matrix_elems = int(num_elems * num_elems)
        dense_elems = dense_matrix_elems
        bytes_per_float64 = np.dtype(np.float64).itemsize

        estimated_peak_bytes = (
            2 * dense_matrix_elems * bytes_per_float64  # P + H
            + 2 * ndof * num_elems * bytes_per_float64  # A + Z
            + num_elems * bytes_per_float64  # D
            + 8 * 1024 * 1024  # modest overhead for temporaries and bookkeeping
        )

        if max_dense_elems is not None:
            legacy_limit = int(max_dense_elems) * bytes_per_float64
            if max_dense_bytes is None or max_dense_bytes > legacy_limit:
                max_dense_bytes = legacy_limit

        if estimated_peak_bytes > int(max_dense_bytes):
            raise ValueError(
                "Dense interaction matrices are disabled for this mesh size: "
                f"estimated_peak_bytes={estimated_peak_bytes} > max_dense_bytes={max_dense_bytes}, "
                f"with Ne^2={dense_matrix_elems}. "
                "The dense guard is based on the P/H/A/Z/D allocation footprint, not only on the number of mesh elements."
            )

        fixed = np.asarray(fixed_dofs, dtype=np.int64).ravel() if fixed_dofs is not None else np.empty(0, dtype=np.int64)
        conn = self.get_connectivity(nelx, nely)
        if KE is None:
            KE = self.get_KE(dx, dy)
        pattern = self.get_mesh_pattern(nelx, nely, conn=conn)

        F = np.asarray(forces, dtype=np.float64).reshape(-1)
        F_bc = F.copy()
        F_bc[fixed] = 0.0

        t0 = perf_counter()
        K = self.assemble_dirichlet(rho_flat, pattern, KE, conn, fixed, penal=penal)
        assemble_time = perf_counter() - t0
        self._interaction_stats["assemblies"] += 1

        t0 = perf_counter()
        ctx = self._get_symbolic((int(nelx), int(nely)), K)
        ctx.factorize(K)
        factorization_time = perf_counter() - t0
        self._interaction_stats["factorizations"] += 1

        t0 = perf_counter()
        u = ctx.solve(F_bc)
        solve_time = perf_counter() - t0
        self._interaction_stats["solves"] += 1

        dE = __simp_stiffness_sensitivity__(rho_flat, penal)
        d2E = __simp_stiffness_second_derivative__(rho_flat, penal)
        element_dofs = conn.element_dofs
        Ue = u[element_dofs]
        energies = np.einsum("ei,ij,ej->e", Ue, KE, Ue)
        local_a = dE[:, None] * np.einsum("ij,ej->ei", KE, Ue)

        t0 = perf_counter()
        A = np.zeros((ndof, num_elems), dtype=np.float64)
        for e in range(num_elems):
            A[element_dofs[e], e] = local_a[e]
        A[fixed, :] = 0.0
        a_assembly_time = perf_counter() - t0

        t0 = perf_counter()
        rhs_blocks = 32
        Z = np.empty((ndof, num_elems), dtype=np.float64)
        for start in range(0, num_elems, rhs_blocks):
            stop = min(start + rhs_blocks, num_elems)
            block_rhs = A[:, start:stop]
            block_sol = np.empty_like(block_rhs)
            for j in range(block_rhs.shape[1]):
                block_sol[:, j] = ctx.solve(block_rhs[:, j])
            Z[:, start:stop] = block_sol
        rhs_solve_time = perf_counter() - t0
        self._interaction_stats["solves"] += num_elems

        t0 = perf_counter()
        P = A.T @ Z
        p_construction_time = perf_counter() - t0

        D = d2E * energies
        H = 2.0 * P - np.diag(D)

        info = {
            "A": A,
            "Z": Z,
            "u": u,
            "K": K,
            "ndof": ndof,
            "num_elems": num_elems,
            "matrix_shape": (ndof, num_elems),
            "dense_elems": dense_elems,
            "assembly_time": assemble_time,
            "factorization_time": factorization_time,
            "equilibrium_solve_time": solve_time,
            "a_assembly_time": a_assembly_time,
            "rhs_solve_time": rhs_solve_time,
            "p_construction_time": p_construction_time,
            "total_time": assemble_time + factorization_time + solve_time + a_assembly_time + rhs_solve_time + p_construction_time,
            "factorizations": 1,
            "solves": 1 + num_elems,
            "memory_bytes": A.nbytes + P.nbytes + H.nbytes + D.nbytes,
        }
        if return_info:
            return P, D, H, info
        return P, D, H


# Default solver. Built on first use so that importing this module never pays
# for locating MKL.
_DEFAULT_SOLVER: Optional[FeaSolver] = None


def get_default_solver() -> FeaSolver:
    global _DEFAULT_SOLVER
    if _DEFAULT_SOLVER is None:
        _DEFAULT_SOLVER = FeaSolver()
    return _DEFAULT_SOLVER
