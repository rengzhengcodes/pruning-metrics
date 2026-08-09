"""Cluster and correlation statistics over precomputed distance matrices.

This module answers a different question than
:mod:`pruning_metrics.metrics.distributions`: instead of computing a
single scalar distance between two models' teacher-forced outputs, it
takes *already-assembled* (n, n) pairwise distance matrices (behavioral
distance, mask-Jaccard distance, ...) over a population of model
variants and asks whether they are statistically related to each other
(Mantel / partial Mantel) or to known group labels (silhouette / ARI /
permutation tests).

Functions
---------
mantel
    Correlation between two distance matrices (Mantel test), with a
    permutation p-value.
partial_mantel
    Mantel correlation between ``d1`` and ``d2`` after regressing out a
    third "control" distance matrix from both.
silhouette_by_label
    Mean silhouette coefficient of known group labels under a
    precomputed distance metric.
ari_vs_labels
    Adjusted Rand Index between known labels and the labels produced by
    agglomerative clustering on the precomputed distances.
label_permutation_pvalue
    Permutation-test p-value for either of the above two label-based
    statistics.

All functions operate purely on ``numpy``/``scipy``/``sklearn`` and take
no torch dependency, per the v2 spec (package P3).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

__all__ = [
    "mantel",
    "partial_mantel",
    "silhouette_by_label",
    "ari_vs_labels",
    "label_permutation_pvalue",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


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


def silhouette_by_label(dist: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient of ``labels`` under a precomputed distance metric.

    Parameters
    ----------
    dist:
        Square symmetric (n, n) precomputed distance matrix.
    labels:
        Length-n array of group labels (any hashable dtype).

    Returns
    -------
    float
        ``sklearn.metrics.silhouette_score(dist, labels, metric="precomputed")``.
        Ranges in [-1, 1]; higher means better-separated groups.

    Raises
    ------
    ValueError
        If ``dist`` is not square/symmetric, if ``len(labels) != dist.shape[0]``,
        or if there are fewer than 2 distinct labels (silhouette is
        undefined for a single cluster) — this is the underlying
        sklearn error, surfaced unchanged.
    """
    dist = _validate_square_symmetric(dist, "dist")
    labels = np.asarray(labels)
    if labels.shape[0] != dist.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} does not match dist shape {dist.shape}"
        )
    return float(silhouette_score(dist, labels, metric="precomputed"))


def ari_vs_labels(
    dist: np.ndarray,
    labels: np.ndarray,
    *,
    linkage: str = "average",
) -> float:
    """Adjusted Rand Index between known ``labels`` and agglomerative clustering.

    Runs ``sklearn.cluster.AgglomerativeClustering`` with
    ``n_clusters = len(unique(labels))`` and ``metric="precomputed"`` on
    ``dist``, then compares the resulting cluster assignment to the
    ground-truth ``labels`` via the Adjusted Rand Index.

    Parameters
    ----------
    dist:
        Square symmetric (n, n) precomputed distance matrix.
    labels:
        Length-n array of ground-truth group labels.
    linkage:
        Linkage criterion passed to ``AgglomerativeClustering``
        (``"average"``, ``"complete"``, or ``"single"``; ``"ward"`` is
        NOT valid with a precomputed metric and will raise inside
        sklearn if passed).

    Returns
    -------
    float
        Adjusted Rand Index in [-1, 1] (1.0 = perfect agreement up to
        label permutation, ~0 = random agreement).

    Raises
    ------
    ValueError
        If ``dist`` is not square/symmetric or ``len(labels) != dist.shape[0]``.
    """
    dist = _validate_square_symmetric(dist, "dist")
    labels = np.asarray(labels)
    if labels.shape[0] != dist.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} does not match dist shape {dist.shape}"
        )
    n_clusters = len(np.unique(labels))
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage=linkage
    )
    predicted = clusterer.fit_predict(dist)
    return float(adjusted_rand_score(labels, predicted))


def label_permutation_pvalue(
    dist: np.ndarray,
    labels: np.ndarray,
    *,
    stat: str = "silhouette",
    permutations: int = 9999,
    seed: int = 0,
) -> tuple[float, float]:
    """Permutation-test p-value for a label-based clustering statistic.

    Computes the observed statistic (silhouette or ARI) for the true
    label assignment, then shuffles ``labels`` ``permutations`` times to
    build a null distribution and reports the fraction of shuffles at
    least as extreme.

    Parameters
    ----------
    dist:
        Square symmetric (n, n) precomputed distance matrix.
    labels:
        Length-n array of ground-truth group labels.
    stat:
        ``"silhouette"`` (default, via :func:`silhouette_by_label`) or
        ``"ari"`` (via :func:`ari_vs_labels`).
    permutations:
        Number of label shuffles used to build the null distribution.
    seed:
        Seed for the permutation RNG.

    Returns
    -------
    observed, p:
        Observed statistic value and its permutation p-value, using the
        one-sided ``(1 + #{stat_perm >= stat_obs}) / (1 + permutations)``
        estimator — both silhouette and ARI are "higher is more
        clustered", so a one-sided upper-tail test is the natural
        alternative hypothesis (unlike Mantel's r, which can be
        meaningfully negative and so uses a two-sided ``|r|`` test).

    Raises
    ------
    ValueError
        If ``stat`` is not one of ``{"silhouette", "ari"}``, or if the
        underlying statistic function raises (e.g. shape mismatch).
    """
    if stat == "silhouette":
        stat_fn = silhouette_by_label
    elif stat == "ari":
        stat_fn = ari_vs_labels
    else:
        raise ValueError(f"stat must be 'silhouette' or 'ari', got {stat!r}")

    dist = _validate_square_symmetric(dist, "dist")
    labels = np.asarray(labels)
    if labels.shape[0] != dist.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} does not match dist shape {dist.shape}"
        )

    observed = stat_fn(dist, labels)

    rng = np.random.default_rng(seed)
    exceed_count = 0
    for _ in range(permutations):
        shuffled = rng.permutation(labels)
        # A shuffle that collapses to a single distinct label is
        # degenerate for both silhouette (undefined) and ARI (trivially
        # matches); skip and retry so the null distribution stays
        # well-defined without biasing the permutation count.
        if len(np.unique(shuffled)) < 2:
            continue
        stat_perm = stat_fn(dist, shuffled)
        if stat_perm >= observed:
            exceed_count += 1

    p_value = (1 + exceed_count) / (1 + permutations)
    return observed, p_value
