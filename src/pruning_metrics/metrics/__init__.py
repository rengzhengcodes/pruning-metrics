"""Aggregator package for distance, mask, and embedding-quality diagnostics.

This package holds no logic of its own. It re-exports four sibling
modules so downstream notebooks and tests have one import surface:

* **Distributional distances** -- all sixteen measures implemented in
  :mod:`pruning_metrics.prob_measures` (``compute_kld``, ``compute_jsd``,
  ``compute_emd``, ``compute_chamfer``, and twelve more), reached here
  through the backwards-compatible
  :mod:`pruning_metrics.metrics.distributions` facade. Use
  :func:`compute_all` rather than looping over the individual functions
  when more than one measure is wanted: it shares the per-position
  alignment work, so all sixteen cost about as much as one.
* **Pruning-mask utilities**, from :mod:`pruning_metrics.metrics.masks`
  -- extracting per-layer pruning masks (:func:`extract_pruning_masks`),
  packed digests for cheap storage and comparison
  (:func:`make_mask_digest`, :class:`PackedDigest`), and pairwise
  Jaccard distance (:func:`jaccard_distance`,
  :func:`jaccard_matrix_packed`).
* **Permutation and cluster statistics**, from
  :mod:`pruning_metrics.metrics.cluster_stats` -- Mantel and
  partial-Mantel correlation tests, label-permutation p-values,
  silhouette scores, and adjusted Rand index (ARI) against known labels.
* **Embedding-quality diagnostics**, from
  :mod:`pruning_metrics.metrics.embedding_quality` -- trustworthiness,
  continuity, Kruskal stress-1, Shepard rho, and the composite
  :func:`embedding_quality` score used to grade a 2-D reduction (see
  :mod:`pruning_metrics.dim_reduction`) against the distance matrix it
  was built from.

Usage
-----
>>> from pruning_metrics.metrics import compute_all, jaccard_distance, mantel, trustworthiness
"""

from __future__ import annotations

from pruning_metrics import prob_measures as _prob_measures
from pruning_metrics.metrics import cluster_stats as _cluster_stats
from pruning_metrics.metrics import embedding_quality as _embedding_quality
from pruning_metrics.metrics import masks as _masks

# Wildcards mirror each source module's export surface, so a name added to
# a source's ``__all__`` is available here with no edit to this package.
from pruning_metrics.metrics.cluster_stats import *
from pruning_metrics.metrics.embedding_quality import *
from pruning_metrics.metrics.masks import *
from pruning_metrics.prob_measures import *

#: Composed from the source modules' ``__all__`` lists; each source owns
#: its own canonical list, so no name is maintained by hand twice.
__all__ = (
    list(_prob_measures.__all__)
    + list(_masks.__all__)
    + list(_cluster_stats.__all__)
    + list(_embedding_quality.__all__)
)
