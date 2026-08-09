"""Scalar quality measures for a 2-D embedding of a precomputed distance matrix.

A 2-D picture of 232 model variants is a lossy summary, and the lossiness is
not visible in the picture itself: t-SNE and UMAP will happily draw crisp,
well-separated blobs from data that has no cluster structure at all. These
functions put numbers on what a given embedding did and did not preserve, so a
figure can be read with the right amount of trust.

Two complementary families:

**Neighbourhood preservation** (:func:`trustworthiness`, :func:`continuity`)
    Rank-based and local. Trustworthiness penalises *false* neighbours -- points
    the embedding drew together that were far apart in the real distances.
    Continuity penalises *missing* neighbours -- genuinely close points that the
    embedding tore apart. They are the same statistic with the two spaces
    swapped, and both live in ``[0, 1]`` with 1 best.

**Global geometry** (:func:`kruskal_stress1`, :func:`shepard_rho`)
    Distance-based. Stress-1 is a metric criterion (how well 2-D distances
    reproduce the originals, after optimal uniform rescaling; lower is better);
    Shepard rho is its monotone counterpart (Spearman correlation between the
    two sets of pairwise distances; higher is better).

The combination is what makes an embedding interpretable. High trustworthiness
with low Shepard rho -- the usual t-SNE/UMAP signature -- means local
neighbourhoods are real but distances between well-separated groups are not, so
"these two clusters are far apart" is not a supportable reading of the figure.

Every function takes the *original* distance matrix, never a preprocessed one.
Scoring a PCA or LLE embedding against its own classical-MDS input would grade
it on the very information that step discarded.

Usage
-----
>>> from pruning_metrics.metrics.embedding_quality import embedding_quality
>>> embedding_quality(D, coords, k=12)["trustworthiness"]
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr
except ImportError as _exc:  # pragma: no cover - mirrors distributions.py
    raise ImportError(
        "scipy is required for embedding quality metrics: pip install scipy"
    ) from _exc

__all__ = [
    "baseline_distances",
    "linear_r2",
    "effective_k",
    "trustworthiness",
    "continuity",
    "kruskal_stress1",
    "shepard_rho",
    "embedding_quality",
]

_DEFAULT_K = 12


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _as_distance_matrix(D: np.ndarray, name: str = "D") -> np.ndarray:
    """Validate and coerce a square, symmetric, finite distance matrix."""
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"{name} must be square, got shape {D.shape}")
    if not np.isfinite(D).all():
        raise ValueError(f"{name} contains non-finite entries")
    return D


def _embedding_distances(Y: np.ndarray, n: int) -> np.ndarray:
    """Euclidean distance matrix of an embedding, checked against ``n``."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[0] != n:
        raise ValueError(f"Y must have shape ({n}, d), got {Y.shape}")
    if not np.isfinite(Y).all():
        raise ValueError("Y contains non-finite entries")
    return squareform(pdist(Y))


def effective_k(k: int, n: int) -> int:
    """Largest usable neighbourhood size at or below ``k``.

    The trustworthiness normalisation ``2 / (n k (2n - 3k - 1))`` is only
    positive while ``k < (2n - 1) / 3``, and scikit-learn refuses ``k >= n / 2``
    outright. Rather than raising -- these metrics are computed in bulk across
    benchmarks whose ``n`` varies -- the requested ``k`` is clamped and the
    value actually used is reported back to the caller.

    Parameters
    ----------
    k:
        Requested neighbourhood size.
    n:
        Number of points.

    Returns
    -------
    int
        ``min(k, (n - 1) // 2)``, floored at 1. Returns 0 when ``n < 3``, which
        callers treat as "too small to score".
    """
    if n < 3:
        return 0
    return max(1, min(int(k), (n - 1) // 2))


def _rank_penalty(D_rank: np.ndarray, D_neighbors: np.ndarray, k: int) -> float:
    """Shared kernel of trustworthiness and continuity.

    Takes the ``k`` nearest neighbours of each point in ``D_neighbors``, looks
    up how far away they rank in ``D_rank``, and penalises by how far beyond
    ``k`` those ranks fall::

        1 - 2 / (n k (2n - 3k - 1)) * sum_i sum_{j in kNN(i)} max(rank(i,j) - k, 0)

    Trustworthiness is ``_rank_penalty(D, D_embedding, k)`` and continuity is
    the same call with the arguments swapped. Rank conventions (diagonal
    excluded, ranks starting at 1, arbitrary-but-deterministic tie-breaking via
    ``argsort``) match ``sklearn.manifold.trustworthiness`` exactly.
    """
    n = D_rank.shape[0]

    ranked = D_rank.copy()
    np.fill_diagonal(ranked, np.inf)
    ind_rank = np.argsort(ranked, axis=1)

    neighbors = D_neighbors.copy()
    np.fill_diagonal(neighbors, np.inf)
    ind_neighbors = np.argsort(neighbors, axis=1)[:, :k]

    # inverted[i, j] = 1-based rank of point j from point i in D_rank.
    # ind_rank is a per-row permutation; this scatter builds its per-row
    # inverse: ordered[:-1, np.newaxis] (column [0..n-1]) broadcasts against
    # ind_rank (n x n) to address (row i, column ind_rank[i, m]), and the
    # assigned ordered[1:] ([1..n]) broadcasts along each row, so the point
    # sorted into position m receives rank m + 1.
    inverted = np.zeros((n, n), dtype=int)
    ordered = np.arange(n + 1)
    inverted[ordered[:-1, np.newaxis], ind_rank] = ordered[1:]

    # Gather each point's k nearest D_neighbors neighbours and look up their
    # D_rank ranks; subtracting k leaves the "how far beyond k" excess.
    ranks = inverted[ordered[:-1, np.newaxis], ind_neighbors] - k
    # Only neighbours ranked beyond k contribute: max(rank - k, 0) summed.
    penalty = float(np.sum(ranks[ranks > 0]))
    return 1.0 - penalty * (2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0)))


# ---------------------------------------------------------------------------
# Neighbourhood preservation
# ---------------------------------------------------------------------------


def trustworthiness(D: np.ndarray, Y: np.ndarray, k: int = _DEFAULT_K) -> float:
    """Fraction of the embedding's neighbourhoods that are genuine.

    Penalises points that appear close in ``Y`` but are far apart in ``D`` --
    i.e. structure the embedding invented.

    Parameters
    ----------
    D:
        Symmetric ``(n, n)`` matrix of original distances.
    Y:
        ``(n, d)`` embedding coordinates.
    k:
        Neighbourhood size, clamped by :func:`effective_k`.

    Returns
    -------
    float
        ``1.0`` when every embedded neighbourhood is a true neighbourhood,
        ~0.5 for a random embedding, ``nan`` when ``n`` is too small to score.

    Notes
    -----
    Numerically identical to
    ``sklearn.manifold.trustworthiness(D, Y, n_neighbors=k, metric="precomputed")``
    for any ``k`` that scikit-learn accepts; this implementation exists so that
    :func:`continuity` -- which scikit-learn does not provide, and which cannot
    be obtained by swapping that function's arguments when the original space
    has no coordinates -- can share the same rank conventions.
    """
    D = _as_distance_matrix(D)
    n = D.shape[0]
    k_eff = effective_k(k, n)
    if k_eff < 1:
        return float("nan")
    return _rank_penalty(D, _embedding_distances(Y, n), k_eff)


def continuity(D: np.ndarray, Y: np.ndarray, k: int = _DEFAULT_K) -> float:
    """Fraction of the original neighbourhoods the embedding kept together.

    The mirror image of :func:`trustworthiness`: penalises points that are close
    in ``D`` but were pulled apart in ``Y`` -- i.e. structure the embedding
    destroyed.

    Parameters
    ----------
    D:
        Symmetric ``(n, n)`` matrix of original distances.
    Y:
        ``(n, d)`` embedding coordinates.
    k:
        Neighbourhood size, clamped by :func:`effective_k`.

    Returns
    -------
    float
        ``1.0`` when every true neighbourhood survives, ``nan`` when ``n`` is
        too small to score.
    """
    D = _as_distance_matrix(D)
    n = D.shape[0]
    k_eff = effective_k(k, n)
    if k_eff < 1:
        return float("nan")
    return _rank_penalty(_embedding_distances(Y, n), D, k_eff)


# ---------------------------------------------------------------------------
# Global geometry
# ---------------------------------------------------------------------------


def kruskal_stress1(D: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    """Kruskal stress-1 between original and embedded distances, optimally scaled.

    ``sigma_1 = sqrt( sum (d - a*y)^2 / sum d^2 )`` over all point pairs, where
    ``a = <d, y> / <y, y>`` is the scale factor minimising the numerator.

    The rescaling is not cosmetic. t-SNE's output scale is an artefact of early
    exaggeration and LLE normalises its embedding to unit covariance, so raw
    stress would rank those methods on their choice of units rather than on
    their geometry. With ``a`` fitted, stress-1 measures shape alone.

    Parameters
    ----------
    D:
        Symmetric ``(n, n)`` matrix of original distances.
    Y:
        ``(n, d)`` embedding coordinates.

    Returns
    -------
    stress : float
        ``0.0`` for a perfect (up to scale) reproduction, growing toward 1 as
        the embedding distorts distances. ``nan`` if ``D`` is all zeros.
    alpha : float
        The fitted scale factor, reported so the number is auditable.
    """
    D = _as_distance_matrix(D)
    n = D.shape[0]
    upper = np.triu_indices(n, k=1)
    d = D[upper]
    y = _embedding_distances(Y, n)[upper]

    denom = float(np.dot(d, d))
    y_sq = float(np.dot(y, y))
    if denom == 0.0:
        return float("nan"), float("nan")
    if y_sq == 0.0:
        # A collapsed embedding reproduces nothing; stress is maximal.
        return 1.0, 0.0

    alpha = float(np.dot(d, y) / y_sq)
    residual = float(np.sum((d - alpha * y) ** 2))
    return float(np.sqrt(residual / denom)), alpha


def shepard_rho(D: np.ndarray, Y: np.ndarray) -> float:
    """Spearman rank correlation between original and embedded pairwise distances.

    The Shepard diagram reduced to one number. Being rank-based it is invariant
    to any monotone rescaling of either space, which makes it the fair way to
    ask "does this picture get the *ordering* of distances right?" even for
    methods that deliberately distort magnitudes.

    Parameters
    ----------
    D:
        Symmetric ``(n, n)`` matrix of original distances.
    Y:
        ``(n, d)`` embedding coordinates.

    Returns
    -------
    float
        ``1.0`` for a perfectly rank-preserving embedding, ~0 for an unrelated
        one, ``nan`` when either set of distances is constant (no ranks to
        correlate) or there are fewer than 2 pairs.
    """
    D = _as_distance_matrix(D)
    n = D.shape[0]
    upper = np.triu_indices(n, k=1)
    d = D[upper]
    y = _embedding_distances(Y, n)[upper]

    if d.size < 2 or np.ptp(d) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    rho = spearmanr(d, y).statistic
    return float(rho)


# ---------------------------------------------------------------------------
# Predictive validity
# ---------------------------------------------------------------------------


def baseline_distances(Y: np.ndarray, baseline_idx: int = 0) -> np.ndarray:
    """Euclidean distance from one reference point to every point in an embedding.

    In this experiment the reference is the unpruned baseline model, so the
    result is "how far did the picture move this network from where it
    started" -- the embedding's own claim about how damaged each network is.
    Regressing a real degradation measure on it (see :func:`linear_r2`) asks
    whether that claim holds up.

    Parameters
    ----------
    Y:
        ``(n, d)`` embedding coordinates.
    baseline_idx:
        Row index of the reference point.

    Returns
    -------
    np.ndarray
        Length-``n`` distances; entry ``baseline_idx`` is 0.

    Raises
    ------
    IndexError
        If ``baseline_idx`` is out of range.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D, got shape {Y.shape}")
    if not -Y.shape[0] <= baseline_idx < Y.shape[0]:
        raise IndexError(
            f"baseline_idx {baseline_idx} out of range for {Y.shape[0]} points"
        )
    return np.linalg.norm(Y - Y[baseline_idx], axis=1)


def linear_r2(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Least-squares fit of ``y`` on ``x``, reported as R-squared plus the line.

    R-squared here is the squared Pearson correlation, matching the convention
    already used for the metric-versus-accuracy analysis in
    ``04_metric_spaces.ipynb`` so the numbers are comparable.

    Non-finite pairs are dropped rather than poisoning the fit -- the v2 grid
    has variants with no performance record at all -- and the surviving count
    is returned so a high R-squared on three points cannot be mistaken for a
    result.

    Parameters
    ----------
    x, y:
        Equal-length 1-D arrays.

    Returns
    -------
    dict
        ``n`` (pairs actually used), ``r2``, ``r``, ``slope``, ``intercept``.
        Every statistic is ``nan`` when fewer than 3 pairs survive or either
        variable is constant.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different lengths.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"x and y must be the same length, got {x.size} and {y.size}")

    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    nan = float("nan")
    if x.size < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return {"n": int(x.size), "r2": nan, "r": nan, "slope": nan, "intercept": nan}

    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return {
        "n": int(x.size),
        "r2": float(r**2),
        "r": r,
        "slope": float(slope),
        "intercept": float(intercept),
    }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def embedding_quality(
    D: np.ndarray, Y: np.ndarray, k: int = _DEFAULT_K
) -> dict[str, Any]:
    """All four quality measures in one pass, for one ``(D, Y)`` pair.

    Parameters
    ----------
    D:
        Symmetric ``(n, n)`` matrix of original distances.
    Y:
        ``(n, d)`` embedding coordinates.
    k:
        Requested neighbourhood size for the rank-based measures.

    Returns
    -------
    dict
        ``n``, ``k`` (the *effective* value after clamping),
        ``trustworthiness``, ``continuity``, ``stress1``, ``alpha`` and
        ``shepard_rho``.
    """
    D = _as_distance_matrix(D)
    n = D.shape[0]
    D_embedded = _embedding_distances(Y, n)
    k_eff = effective_k(k, n)

    if k_eff < 1:
        trust = cont = float("nan")
    else:
        trust = _rank_penalty(D, D_embedded, k_eff)
        cont = _rank_penalty(D_embedded, D, k_eff)

    stress, alpha = kruskal_stress1(D, Y)
    return {
        "n": n,
        "k": k_eff,
        "trustworthiness": trust,
        "continuity": cont,
        "stress1": stress,
        "alpha": alpha,
        "shepard_rho": shepard_rho(D, Y),
    }
