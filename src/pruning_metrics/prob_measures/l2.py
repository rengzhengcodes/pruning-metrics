"""Euclidean distance between two aligned probability vectors."""

from __future__ import annotations

import math

import numpy as np

from pruning_metrics.prob_measures.base import MetricInfo, TokenStepDict, aligned_sum

NAME = "l2"

INFO = MetricInfo("Euclidean", "geometry", True, math.sqrt(2.0), "‖p − q‖₂")


def kernel(p: np.ndarray, q: np.ndarray) -> float:
    """Per-position Euclidean distance on two aligned probability vectors."""
    return float(np.linalg.norm(p - q))


def compute_l2(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sums Euclidean distance ‖p − q‖₂ over aligned positions.

    It is not an f-divergence, and it is not information-theoretic at all.
    It treats the two probability vectors as plain points in Euclidean
    space. It is included as the null hypothesis of this whole comparison.
    If the information-theoretic measures do not beat straight Euclidean
    distance at predicting degradation, their extra machinery is not
    earning its place.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed Euclidean distance (non-negative).
    """
    return aligned_sum(tokens_0, tokens_k, kernel)


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_l2
