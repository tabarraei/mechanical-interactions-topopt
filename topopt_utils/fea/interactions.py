"""Normalization utilities for dense mechanical interaction matrices."""

from __future__ import annotations

import numpy as np


def normalize_mechanical_interactions(P, relative_threshold: float = 1e-8):
    """Normalize a dense mechanical interaction matrix to a correlation-like form.

    Given a symmetric positive-semidefinite interaction matrix ``P`` with diagonal
    entries representing element-wise self-coupling, define

        R_ij = P_ij / sqrt(P_ii * P_jj)

    on the active elements. The returned ``active_mask`` marks elements whose
    diagonal is strictly above the relative threshold. Inactive entries are left as
    zeros in ``R`` and must not be interpreted as measured zero mechanical coupling.

    Parameters
    ----------
    P : np.ndarray
        Dense square matrix of shape ``(Ne, Ne)``.
    relative_threshold : float, optional
        Relative cutoff for the active diagonal threshold. Default is ``1e-8``.

    Returns
    -------
    R : np.ndarray
        Floats in ``float64`` with the same shape as ``P``. Inactive entries are set
        to zero to keep the representation explicit.
    active_mask : np.ndarray
        Boolean mask with ``True`` on active elements.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be a square 2D matrix, got shape {P.shape}")

    n = P.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=bool)

    diag = np.diag(P).astype(np.float64, copy=True)
    max_diag = float(np.max(diag)) if diag.size else 0.0
    scale = max(abs(max_diag), 1.0)
    tol = 1e-12 * scale

    if np.any(diag < -tol):
        raise ValueError(
            "Substantially negative diagonal entries found in P; "
            "mechanical interaction normalization is undefined."
        )

    diag = np.clip(diag, 0.0, None)

    if max_diag <= 0.0:
        R = np.zeros_like(P, dtype=np.float64)
        active_mask = np.zeros(n, dtype=bool)
        return R, active_mask

    active_mask = diag > relative_threshold * max_diag
    R = np.zeros_like(P, dtype=np.float64)

    if not np.any(active_mask):
        return R, active_mask

    d = diag[active_mask]
    denom = np.sqrt(np.outer(d, d))
    denom[denom == 0.0] = 1.0
    active_block = P[np.ix_(active_mask, active_mask)] / denom
    R[np.ix_(active_mask, active_mask)] = active_block

    active_idx = np.flatnonzero(active_mask)
    R[active_idx, active_idx] = 1.0
    return R.astype(np.float64, copy=False), active_mask.astype(bool, copy=False)


__all__ = ["normalize_mechanical_interactions"]
