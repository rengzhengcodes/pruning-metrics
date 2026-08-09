"""Shared distance-matrix validation for the cluster_stats package."""

from __future__ import annotations

import numpy as np

# Tolerance for symmetry validation: distance matrices computed from
# floating-point pipelines (e.g. summed per-token divergences from two
# directions) can pick up tiny asymmetries from non-associative float
# addition even when the underlying quantity is mathematically symmetric.
_SYMMETRY_TOL: float = 1e-8


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_square_symmetric(mat: np.ndarray, name: str) -> np.ndarray:
    """Validate that ``mat`` is a square, (near-)symmetric 2-D array.

    Parameters
    ----------
    mat:
        Candidate distance matrix.
    name:
        Human-readable name used in error messages.

    Returns
    -------
    np.ndarray
        A symmetrized float64 copy of ``mat``: ``(mat + mat.T) / 2``.
        Symmetrizing (rather than merely checking) absorbs float noise
        within ``_SYMMETRY_TOL`` while still catching real asymmetry.

    Raises
    ------
    ValueError
        If ``mat`` is not 2-D, not square, or its asymmetry exceeds
        ``_SYMMETRY_TOL`` in any entry.
    """
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix, got shape {arr.shape}")
    asymmetry = np.abs(arr - arr.T)
    max_asymmetry = float(asymmetry.max()) if arr.size else 0.0
    if max_asymmetry > _SYMMETRY_TOL:
        raise ValueError(
            f"{name} is not symmetric within tolerance {_SYMMETRY_TOL} "
            f"(max |a - a.T| = {max_asymmetry})"
        )
    # Design: symmetrize by averaging rather than just returning `arr`
    # so that sub-tolerance float noise doesn't leak into downstream
    # rank/correlation computations (e.g. upper-triangle extraction
    # implicitly assumes mat[i, j] == mat[j, i]).
    return (arr + arr.T) / 2.0
