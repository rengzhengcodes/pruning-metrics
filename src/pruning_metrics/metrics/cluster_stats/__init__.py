"""Cluster and correlation statistics over precomputed distance matrices.

This package answers a different question than
:mod:`pruning_metrics.metrics.distributions`: instead of computing a
single scalar distance between two models' teacher-forced outputs, it
takes *already-assembled* (n, n) pairwise distance matrices (behavioral
distance, mask-Jaccard distance, ...) over a population of model
variants and asks whether they are statistically related to each other
(Mantel / partial Mantel) or to known group labels (silhouette / ARI /
permutation tests).

Layout: :mod:`~pruning_metrics.metrics.cluster_stats.mantel` holds the
distance-matrix correlation tests,
:mod:`~pruning_metrics.metrics.cluster_stats.label_stats` the label-based
statistics, and the shared matrix validation lives in ``_validation``.

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

from pruning_metrics.metrics.cluster_stats.label_stats import (
    ari_vs_labels,
    label_permutation_pvalue,
    silhouette_by_label,
)
from pruning_metrics.metrics.cluster_stats.mantel import mantel, partial_mantel

__all__ = [
    "mantel",
    "partial_mantel",
    "silhouette_by_label",
    "ari_vs_labels",
    "label_permutation_pvalue",
]
