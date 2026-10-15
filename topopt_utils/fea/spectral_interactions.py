"""Spectral diagnostics of saved mechanical operators; no FEM or optimization calls.

P = A.T K^{-1} A is PSD, unlike the full SIMP compliance Hessian. R is only
meaningful on active elements. All routines use float64 and preserve input arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import eigh, subspace_angles
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

RANK_THRESHOLDS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
ENERGY_TARGETS = (.9, .95, .99, .999, .9999)
SELECTED_RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@dataclass
class Spectrum:
    eigenvalues: np.ndarray  # descending, only tiny negative roundoff set to zero
    eigenvectors: np.ndarray
    validation: dict
    elapsed_seconds: float


def active_principal_submatrix(matrix, mask):
    """Extract a principal submatrix, retaining original C-order element indices."""
    matrix = np.asarray(matrix, dtype=np.float64)
    mask = np.asarray(mask)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Matrix must be square')
    if mask.dtype != bool or mask.shape != (matrix.shape[0],):
        raise ValueError('Mask must be a matching Boolean vector')
    indices = np.flatnonzero(mask)
    return (matrix if np.all(mask) else matrix[np.ix_(indices, indices)]), indices


def symmetric_psd_eigensystem(matrix, *, symmetry_rtol=1e-10, psd_factor=100.):
    """Validate, symmetrize accepted roundoff, and use symmetric LAPACK eigh.

    Negative eigenvalues are accepted only when >= -100*N*eps*max(abs(lambda)).
    No absolute scale of one is imposed, so the test is invariant under rescaling.
    Small positive eigenvalues are retained. Raw minima/negative counts are recorded.
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)):
        raise ValueError('Matrix must be square and finite')
    if symmetry_rtol < 0 or psd_factor <= 0:
        raise ValueError('Invalid validation tolerance')
    n = a.shape[0]
    norm = float(np.linalg.norm(a, 'fro'))
    asym = float(np.linalg.norm(a-a.T, 'fro'))
    relative_asym = asym/norm if norm else 0.
    if relative_asym > symmetry_rtol:
        raise ValueError(f'Matrix is not symmetric: relative error {relative_asym:g}')
    t0 = perf_counter()
    # One Fortran-order workspace; overwrite_a may reuse it. Original a is untouched.
    symmetric = np.array((a+a.T)*.5, order='F')
    if n:
        values, vectors = eigh(symmetric, overwrite_a=True, check_finite=False, driver='evd')
    else:
        values, vectors = np.empty(0), np.empty((0,0))
    elapsed = perf_counter()-t0
    scale = float(np.max(np.abs(values))) if n else 0.
    tolerance = psd_factor*max(n,1)*np.finfo(np.float64).eps*scale
    smallest = float(values[0]) if n else np.nan
    if n and smallest < -tolerance:
        raise ValueError(f'Substantial negative eigenvalue {smallest:g}; PSD tolerance {tolerance:g}')
    negative = values < 0
    validation = dict(dimension=n,finite=True,square=True,
        symmetry_relative_frobenius_error=relative_asym,
        symmetry_max_abs_error=float(np.max(np.abs(a-a.T))) if n else 0.,
        symmetry_relative_tolerance=symmetry_rtol,
        raw_smallest_eigenvalue=smallest,raw_largest_eigenvalue=float(values[-1]) if n else np.nan,
        psd_tolerance=tolerance,roundoff_negative_count=int(negative.sum()),
        largest_clipped_negative_magnitude=float(np.max(-values[negative])) if negative.any() else 0.,
        psd_valid=True,original_frobenius_norm=norm,
        symmetrization_correction_frobenius=.5*asym)
    values[negative] = 0
    return Spectrum(values[::-1].copy(), vectors[:, ::-1], validation, elapsed)


def _validated_values(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError('Eigenvalues must be a finite nonnegative vector')
    if np.any(np.diff(values) > 0):
        raise ValueError('Eigenvalues must be in descending order')
    return values


def spectral_statistics(values, relative_thresholds=RANK_THRESHOLDS):
    """Stable rank, trace-weighted entropy and participation rank are distinct."""
    values = _validated_values(values)
    n = values.size
    largest = float(values[0]) if n else 0.
    scaled = values/largest if largest else np.zeros(n)
    mass, energy = float(scaled.sum()), float(scaled@scaled)
    probabilities = scaled/mass if mass else scaled
    positive = probabilities > 0
    entropy_rank = float(np.exp(-np.sum(probabilities[positive]*np.log(probabilities[positive])))) if mass else 0.
    out = dict(dimension=n,largest_eigenvalue=largest,trace=float(values.sum()),
        frobenius_norm=float(largest*np.sqrt(energy)),stable_rank=energy,
        entropy_effective_rank=entropy_rank,
        participation_ratio=mass**2/energy if energy else 0.)
    for threshold in relative_thresholds:
        if not 0 <= threshold < 1: raise ValueError('Rank thresholds must be in [0,1)')
        retained = values > threshold*largest
        out[f'numerical_rank_{threshold:g}'] = int(retained.sum())
        out[f'effective_condition_{threshold:g}'] = largest/float(values[retained][-1]) if retained.any() else np.nan
    numerical_zero = 100*max(n,1)*np.finfo(float).eps*largest
    nonsingular = n>0 and values[-1]>numerical_zero
    out['condition_full'] = largest/float(values[-1]) if nonsingular else np.nan
    out['singular_within_psd_tolerance'] = not nonsingular
    return out


def cumulative_squared_energy(values):
    values = _validated_values(values)
    if not values.size or values[0]==0: return np.zeros(values.size)
    squares = (values/values[0])**2
    result = np.cumsum(squares)/squares.sum()
    result[-1] = 1.
    return np.clip(result,0,1)


def rank_for_energy(values, target):
    """Smallest k reaching squared spectral energy target; zero operator needs k=0."""
    if not 0 < target <= 1: raise ValueError('Target must lie in (0,1]')
    energy = cumulative_squared_energy(values)
    if not energy.size or energy[-1]==0: return 0
    return int(np.searchsorted(energy,target,side='left')+1)


def relative_frobenius_error(values, rank):
    """Tail norm divided by full spectral norm; compute tail directly for stability."""
    values = _validated_values(values)
    if not isinstance(rank,(int,np.integer)) or not 0 <= rank <= values.size:
        raise ValueError('Rank must lie between zero and dimension')
    if not values.size or values[0]==0: return 0.
    scaled = values/values[0]
    return float(np.sqrt((scaled[rank:]@scaled[rank:])/(scaled@scaled)))


def lowrank_action(values, vectors, rank, perturbations):
    """Apply Q_k diag(lambda_k) Q_k.T to one vector or a batch of columns."""
    values = _validated_values(values)
    q = np.asarray(vectors,dtype=float)
    v = np.asarray(perturbations,dtype=float)
    if q.shape != (values.size,values.size) or v.ndim not in (1,2) or v.shape[0] != values.size:
        raise ValueError('Eigenbasis/perturbation dimensions do not match')
    if not isinstance(rank,(int,np.integer)) or not 0 <= rank <= values.size:
        raise ValueError('Invalid rank')
    weights = values[:rank] if v.ndim==1 else values[:rank,None]
    return q[:,:rank] @ (weights * (q[:,:rank].T @ v))


def action_error_summary(reference, approximation, reference_atol=1e-12):
    """Absolute/relative column-action errors; never regularize a tiny denominator."""
    ref,approx = np.asarray(reference,dtype=float),np.asarray(approximation,dtype=float)
    if ref.shape != approx.shape or ref.ndim not in (1,2) or reference_atol < 0:
        raise ValueError('Matching vector/batch shapes and nonnegative tolerance required')
    if ref.ndim==1: ref,approx=ref[:,None],approx[:,None]
    reference_norm=np.linalg.norm(ref,axis=0)
    absolute=np.linalg.norm(ref-approx,axis=0)
    valid=reference_norm >= reference_atol
    valid &= reference_norm > 0
    relative=np.full(reference_norm.shape,np.nan)
    relative[valid]=absolute[valid]/reference_norm[valid]
    finite=relative[valid]
    out=dict(num_vectors=ref.shape[1],finite_relative_error_count=int(valid.sum()),
        near_zero_reference_actions=int((~valid).sum()),reference_norm_tolerance=reference_atol,
        min_reference_norm=float(reference_norm.min()) if reference_norm.size else np.nan,
        max_reference_norm=float(reference_norm.max()) if reference_norm.size else np.nan)
    for label,data in [('relative',finite),('absolute',absolute)]:
        for stat,fn in [('mean',np.mean),('median',np.median),('p90',lambda x:np.quantile(x,.9)),('max',np.max)]:
            out[f'{stat}_{label}_error'] = float(fn(data)) if data.size else np.nan
    return out, dict(reference_norms=reference_norm,absolute_errors=absolute,relative_errors=relative)


def spatial_truncation(matrix, element_x, element_y, radius, distances=None):
    """Full CSR, both symmetric entries stored. PSD is not guaranteed."""
    a=np.asarray(matrix,dtype=float)
    coordinates=np.column_stack((element_x,element_y))
    if a.shape != (len(coordinates),len(coordinates)) or radius < 0:
        raise ValueError('Invalid matrix, coordinates, or cutoff')
    if distances is None: distances=cdist(coordinates,coordinates)
    if np.shape(distances)!=a.shape: raise ValueError('Distance matrix must match')
    retained=(distances<=radius)&(a!=0)
    rows,cols=np.nonzero(retained)
    return csr_matrix((a[rows,cols],(rows,cols)),shape=a.shape)


def approximation_costs(dimension, *, rank=None, sparse=None):
    """Actual representation bytes and leading multiply/add estimates per vector.

    Low rank stores N*k float64 entries and k eigenvalues; action 4*N*k+k FLOPs.
    CSR stores both triangles: float64 data plus actual index/indptr dtypes;
    action approximately 2*nnz FLOPs. No symmetry-compressed storage is assumed.
    """
    n=int(dimension)
    if rank is not None:
        k=int(rank)
        if not 0<=k<=n: raise ValueError('Invalid rank')
        return dict(storage_bytes=8*k*(n+1),application_flops=4*n*k+k,
                    stored_nonzeros=None,representation='Q float64 + eigenvalues float64')
    if sparse is not None:
        return dict(storage_bytes=int(sparse.data.nbytes+sparse.indices.nbytes+sparse.indptr.nbytes),
            application_flops=2*int(sparse.nnz),stored_nonzeros=int(sparse.nnz),
            representation='full CSR, both symmetric entries stored',
            index_bytes=sparse.indices.dtype.itemsize,indptr_bytes=sparse.indptr.dtype.itemsize,
            diagonal_nonzeros=int(np.count_nonzero(sparse.diagonal())))
    return dict(storage_bytes=8*n*n,application_flops=2*n*n,
                stored_nonzeros=n*n,representation='dense float64')


def leading_eigenspace_comparison(spectrum_a, spectrum_b, rank, gap_rtol=1e-6):
    """Basis-invariant angles and ||Q_a Q_a.T-Q_b Q_b.T||_F, without dense projectors.

    A small kth/(k+1)th gap flags an ambiguous truncated subspace. Angles remain
    numerical observations, but a boundary crossing a degeneracy is not unique.
    """
    a,b=spectrum_a,spectrum_b
    n=a.eigenvalues.size
    if b.eigenvalues.size!=n or not isinstance(rank,(int,np.integer)) or not 1<=rank<=n:
        raise ValueError('Matching spaces and valid rank required')
    qa,qb=a.eigenvectors[:,:rank],b.eigenvectors[:,:rank]
    angles=subspace_angles(qa,qb)
    overlap=float(np.linalg.norm(qa.T@qb,'fro')**2)
    distance=float(np.sqrt(max(0.,2*rank-2*overlap)))
    def gap(s):
        if rank==n: return np.nan
        return float((s.eigenvalues[rank-1]-s.eigenvalues[rank])/s.eigenvalues[0]) if s.eigenvalues[0] else 0.
    ga,gb=gap(a),gap(b)
    return dict(rank=rank,mean_angle_degrees=float(np.degrees(angles).mean()),
        max_angle_degrees=float(np.degrees(angles).max()),min_angle_degrees=float(np.degrees(angles).min()),
        projector_frobenius_distance=distance,normalized_projector_distance=distance/np.sqrt(2*rank),
        boundary_gap_relative_a=ga,boundary_gap_relative_b=gb,
        near_degenerate_boundary_a=bool(rank<n and ga<=gap_rtol),
        near_degenerate_boundary_b=bool(rank<n and gb<=gap_rtol),gap_relative_tolerance=gap_rtol,
        angles_degrees=np.degrees(angles))
