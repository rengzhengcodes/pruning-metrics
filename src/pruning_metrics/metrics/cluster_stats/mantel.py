"""Mantel and partial-Mantel correlation between distance matrices."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from pruning_metrics.metrics.cluster_stats._validation import (
    _validate_square_symmetric,
)


def _upper_triangle(mat: np.ndarray) -> np.ndarray:
    """Flatten the strict upper triangle (k=1) of a square matrix to 1-D."""
    n = mat.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    return mat[rows, cols]


def _correlate(x: np.ndarray, y: np.ndarray, *, method: str) -> float:
    """Pearson or Spearman correlation coefficient between two 1-D vectors.

    Parameters
    ----------
    x, y:
        Equal-length 1-D arrays.
    method:
        ``"spearman"`` rank-transforms both vectors before a Pearson
        correlation; ``"pearson"`` correlates the raw values.

    Returns
    -------
    float
        Correlation coefficient. ``0.0`` if either vector has zero
        variance (undefined correlation, treated as "no relationship"
        rather than raising, since permutations of a constant vector are
        common in degenerate/synthetic test inputs).

    Raises
    ------
    ValueError
        If ``method`` is not one of ``{"spearman", "pearson"}``.
    """
    if method == "spearman":
        x = rankdata(x)
        y = rankdata(y)
    elif method != "pearson":
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def _ols_residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residuals of ``y`` after regressing on ``x`` with an intercept (simple OLS).

    Parameters
    ----------
    y, x:
        Equal-length 1-D arrays (the response and the single predictor).

    Returns
    -------
    np.ndarray
        ``y - (a + b * x)`` where ``a, b`` are the least-squares fit
        coefficients. If ``x`` has zero variance, the fit degenerates to
        the intercept-only mean model (``b = 0``), so residuals are
        simply ``y - mean(y)``.
    """
    x_centered = x - x.mean()
    var_x = np.sum(x_centered**2)
    if var_x == 0.0:
        # Design: a constant control carries no information to regress
        # out; fall back to demeaning y rather than dividing by zero.
        return y - y.mean()
    b = float(np.sum(x_centered * (y - y.mean())) / var_x)
    a = float(y.mean() - b * x.mean())
    return y - (a + b * x)


def mantel(
    d1: np.ndarray,
    d2: np.ndarray,
    *,
    permutations: int = 9999,
    method: str = "spearman",
    seed: int = 0,
) -> tuple[float, float]:
    """Mantel test: correlation between two distance matrices.

    Computes the (Spearman or Pearson) correlation of the upper-triangle
    entries of ``d1`` and ``d2``, then assesses significance by a
    permutation test: ``d2``'s row/column order is jointly shuffled
    (preserving its internal structure as a distance matrix) and the
    correlation recomputed, ``permutations`` times.

    Parameters
    ----------
    d1, d2:
        Square symmetric (n, n) distance matrices. Must have identical
        shape.
    permutations:
        Number of row/column permutations of ``d2`` used to build the
        null distribution. Default 9999 (standard Mantel convention).
    method:
        ``"spearman"`` (default) or ``"pearson"`` correlation of the
        upper-triangle entries.
    seed:
        Seed for the permutation RNG, for reproducibility.

    Returns
    -------
    r, p:
        Observed correlation coefficient and permutation p-value.
        ``p = (1 + #{|r_perm| >= |r_obs|}) / (1 + permutations)``, the
        standard bias-corrected permutation p-value estimator (never
        exactly 0, avoiding overstated significance from a finite
        permutation sample).

    Raises
    ------
    ValueError
        If ``d1``/``d2`` are not square, not (near-)symmetric, or their
        shapes differ.

    Notes
    -----
    Time complexity: O(permutations * n^2) for the shuffles plus
    O(n^2 log n) per correlation from rank-transformation.
    """
    d1 = _validate_square_symmetric(d1, "d1")
    d2 = _validate_square_symmetric(d2, "d2")
    if d1.shape != d2.shape:
        raise ValueError(
            f"d1 and d2 must have the same shape, got {d1.shape} vs {d2.shape}"
        )

    n = d1.shape[0]
    upper_1 = _upper_triangle(d1)
    r_obs = _correlate(upper_1, _upper_triangle(d2), method=method)

    rng = np.random.default_rng(seed)
    exceed_count = 0
    for _ in range(permutations):
        # Permute rows AND columns jointly so the permuted matrix is
        # still a valid distance matrix over the same n points relabeled
        # — this is what makes the null hypothesis "no association
        # between the two point-labelings" rather than "random matrix".
        perm = rng.permutation(n)
        d2_perm = d2[np.ix_(perm, perm)]
        r_perm = _correlate(upper_1, _upper_triangle(d2_perm), method=method)
        if abs(r_perm) >= abs(r_obs):
            exceed_count += 1

    p_value = (1 + exceed_count) / (1 + permutations)
    return r_obs, p_value


def partial_mantel(
    d1: np.ndarray,
    d2: np.ndarray,
    control: np.ndarray,
    *,
    permutations: int = 9999,
    method: str = "spearman",
    seed: int = 0,
) -> tuple[float, float]:
    """Partial Mantel test: correlation between ``d1`` and ``d2`` controlling for ``control``.

    Rank-transforms (if ``method == "spearman"``) the upper-triangle
    vectors of all three matrices, regresses ``control`` out of both
    ``d1`` and ``d2`` via OLS, and Pearson-correlates the residuals.

    Parameters
    ----------
    d1, d2, control:
        Square symmetric (n, n) distance matrices, all the same shape.
    permutations:
        Number of permutations for the null distribution.
    method:
        ``"spearman"`` (default, rank-transform before residualizing) or
        ``"pearson"`` (residualize raw values directly).
    seed:
        Seed for the permutation RNG.

    Returns
    -------
    r, p:
        Observed partial correlation and permutation p-value, using the
        same ``(1 + count) / (1 + permutations)`` estimator as
        :func:`mantel`.

    Raises
    ------
    ValueError
        If the three matrices are not all square, symmetric, and of
        identical shape.

    Notes
    -----
    Permutation strategy (documented per spec): rather than permuting
    the raw upper-triangle vector (which would break the row/column
    distance-matrix structure that ``control`` depends on), we permute
    ``d2`` at the MATRIX level — shuffling its rows/columns jointly —
    and then *re-residualize* the permuted d2 against the (unpermuted)
    control before correlating with the (fixed) residualized d1. This
    keeps the permutation null consistent with the same rank/regression
    pipeline used to compute the observed statistic, and is the
    standard partial-Mantel permutation scheme (permute one raw matrix,
    not the residuals directly).
    """
    d1 = _validate_square_symmetric(d1, "d1")
    d2 = _validate_square_symmetric(d2, "d2")
    control = _validate_square_symmetric(control, "control")
    if not (d1.shape == d2.shape == control.shape):
        raise ValueError(
            "d1, d2, and control must all have the same shape, got "
            f"{d1.shape}, {d2.shape}, {control.shape}"
        )

    n = d1.shape[0]

    def _prepped_vector(mat: np.ndarray) -> np.ndarray:
        """Upper triangle, rank-transformed if method == 'spearman'."""
        vec = _upper_triangle(mat)
        return rankdata(vec) if method == "spearman" else vec

    vec_1 = _prepped_vector(d1)
    vec_control = _prepped_vector(control)
    resid_1 = _ols_residuals(vec_1, vec_control)

    def _observed_r(d2_mat: np.ndarray) -> float:
        vec_2 = _prepped_vector(d2_mat)
        resid_2 = _ols_residuals(vec_2, vec_control)
        return _correlate(resid_1, resid_2, method="pearson")

    r_obs = _observed_r(d2)

    rng = np.random.default_rng(seed)
    exceed_count = 0
    for _ in range(permutations):
        perm = rng.permutation(n)
        d2_perm = d2[np.ix_(perm, perm)]
        r_perm = _observed_r(d2_perm)
        if abs(r_perm) >= abs(r_obs):
            exceed_count += 1

    p_value = (1 + exceed_count) / (1 + permutations)
    return r_obs, p_value
