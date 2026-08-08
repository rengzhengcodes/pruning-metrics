"""Distributional distance metrics for comparing token-prediction distributions.

This package provides four symmetric or directional distance measures that
operate on the per-token teacher-forced output stored in ``per_token.json``
files produced by
:mod:`pruning_metrics.evals.coding.teacher_forcing`.

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
    continuity,
    effective_k,
    embedding_quality,
    kruskal_stress1,
    shepard_rho,
    trustworthiness,
)
from pruning_metrics.metrics.distributions import (
    TokenStepDict,
    TopAlternativeDict,
    compute_chamfer,
    compute_emd,
    compute_jsd,
    compute_kld,
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
    "compute_jsd",
    "compute_emd",
    "compute_chamfer",
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
]
