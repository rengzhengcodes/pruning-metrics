"""Label-based clustering statistics over precomputed distance matrices."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

from pruning_metrics.metrics.cluster_stats._validation import (
    _validate_square_symmetric,
)


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
