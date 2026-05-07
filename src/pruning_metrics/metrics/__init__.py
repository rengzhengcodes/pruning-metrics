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

from pruning_metrics.metrics.distributions import (
    TokenStepDict,
    TopAlternativeDict,
    compute_chamfer,
    compute_emd,
    compute_jsd,
    compute_kld,
)

__all__ = [
    "compute_kld",
    "compute_jsd",
    "compute_emd",
    "compute_chamfer",
    "TopAlternativeDict",
    "TokenStepDict",
]
