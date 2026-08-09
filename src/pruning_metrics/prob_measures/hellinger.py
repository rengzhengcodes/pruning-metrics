"""Hellinger distance, ||sqrt(p) - sqrt(q)||_2 / sqrt(2)."""

from __future__ import annotations

import math

import numpy as np

from pruning_metrics.prob_measures.base import MetricInfo, TokenStepDict, aligned_sum
from pruning_metrics.prob_measures.l2 import kernel as l2_kernel

NAME = "hellinger"

INFO = MetricInfo("Hellinger", "f-divergence", True, 1.0, "‖√p − √q‖₂ / √2")


def kernel(p: np.ndarray, q: np.ndarray) -> float:
    """Per-position Hellinger distance on two aligned probability vectors.

    Composed from the Euclidean kernel by definition.  Hellinger is the
    Euclidean distance between the square-root embeddings, rescaled to
    [0, 1].  That is the "Euclidean distance in disguise" the compute
    docstring describes, made literal.
    """
    return l2_kernel(np.sqrt(p), np.sqrt(q)) / math.sqrt(2.0)


def compute_hellinger(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sums the Hellinger distance, ||sqrt(p) - sqrt(q)||_2 / sqrt(2), over aligned positions.

    Hellinger is a true metric bounded in [0, 1] per position.  It is the
    one measure here that is a genuine Euclidean distance in disguise
    (between the square-root embeddings of the two distributions on the
    unit sphere).  That matters downstream.  A Hellinger distance matrix is
    much closer to being embeddable without loss than a KL one.  So
    classical MDS discards less of it.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed Hellinger distance (non-negative).
    """
    return aligned_sum(tokens_0, tokens_k, kernel)


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_hellinger
