"""Distributional distance metrics for comparing token-prediction distributions.

This package provides sixteen distance measures that operate on the per-token
teacher-forced output stored in ``per_token.json`` files produced by
:mod:`pruning_metrics.evals.coding.teacher_forcing`.  The four described below
are the originals, used by notebooks 04, 05 and 07; the other twelve
(``rkld``, ``jeffreys``, ``tv``, ``hellinger``, ``bhattacharyya``, ``renyi05``,
``chisq``, ``renyi2``, ``triangular``, ``l2``, ``cosine``, ``wasserstein2``)
are documented in :mod:`pruning_metrics.metrics.distributions` and compared
against each other in notebook 08.

Use :func:`compute_all` rather than a loop over the individual functions when
more than one measure is wanted: it shares the per-position alignment work, so
all sixteen cost about as much as one.  :data:`METRIC_INFO` carries the
family, symmetry, boundedness and formula of each, for tables and axis labels.

Metrics
-------
KLD
    Sum of KL(P_base ‖ P_pruned) over all token positions.  Directional
    (base → pruned); measures how surprised the base-model distribution is
    by the pruned one.
JSD
    Sum of √JSD(P_base, P_pruned) over all token positions.  Symmetric,
    bounded in [0, 1] per position when computed in log₂ units.
EMD
    Sum of Wasserstein-1 distances over all token positions, treating
    each model's logprob values as 1-D atom positions and renormalised
    probabilities as weights.
Chamfer
    Symmetric Chamfer distance between two *sets* of sparse probability
    vectors in R^|vocab|.  Each token position contributes one sparse
    point (the top-5 predictions as non-zero coordinates indexed by
    token-id).  Unlike the position-aligned metrics above, nearest-
    neighbour matching is across all positions, capturing geometric
    similarity of the prediction manifold.

Usage
-----
>>> from pruning_metrics.metrics import compute_kld, compute_jsd, compute_emd, compute_chamfer
>>> kld = compute_kld(tokens_level_0, tokens_level_20)
"""

from __future__ import annotations

from pruning_metrics.metrics.cluster_stats import (
    ari_vs_labels,
    label_permutation_pvalue,
    mantel,
    partial_mantel,
    silhouette_by_label,
)
from pruning_metrics.metrics.embedding_quality import (
    baseline_distances,
    continuity,
    effective_k,
    embedding_quality,
    kruskal_stress1,
    linear_r2,
    shepard_rho,
    trustworthiness,
)
from pruning_metrics.metrics.distributions import (
    METRIC_FUNCS,
    METRIC_INFO,
    METRIC_NAMES,
    MetricInfo,
    TokenStepDict,
    TopAlternativeDict,
    compute_all,
    compute_bhattacharyya,
    compute_chamfer,
    compute_chisq,
    compute_cosine,
    compute_emd,
    compute_hellinger,
    compute_jeffreys,
    compute_jsd,
    compute_kld,
    compute_l2,
    compute_renyi05,
    compute_renyi2,
    compute_rkld,
    compute_triangular,
    compute_tv,
    compute_wasserstein2,
)
from pruning_metrics.metrics.masks import (
    PackedDigest,
    extract_pruning_masks,
    jaccard_distance,
    jaccard_distance_packed,
    jaccard_matrix_packed,
    load_digest,
    load_digest_packed,
    load_packed_masks,
    make_mask_digest,
    save_digest,
    save_packed_masks,
)

__all__ = [
    "compute_kld",
    "compute_rkld",
    "compute_jeffreys",
    "compute_jsd",
    "compute_tv",
    "compute_hellinger",
    "compute_bhattacharyya",
    "compute_renyi05",
    "compute_chisq",
    "compute_renyi2",
    "compute_triangular",
    "compute_l2",
    "compute_cosine",
    "compute_emd",
    "compute_wasserstein2",
    "compute_chamfer",
    "compute_all",
    "METRIC_FUNCS",
    "METRIC_INFO",
    "METRIC_NAMES",
    "MetricInfo",
    "TopAlternativeDict",
    "TokenStepDict",
    "extract_pruning_masks",
    "save_packed_masks",
    "load_packed_masks",
    "make_mask_digest",
    "save_digest",
    "load_digest",
    "PackedDigest",
    "load_digest_packed",
    "jaccard_distance",
    "jaccard_distance_packed",
    "jaccard_matrix_packed",
    "mantel",
    "partial_mantel",
    "silhouette_by_label",
    "ari_vs_labels",
    "label_permutation_pvalue",
    "trustworthiness",
    "continuity",
    "kruskal_stress1",
    "shepard_rho",
    "embedding_quality",
    "effective_k",
    "baseline_distances",
    "linear_r2",
]
