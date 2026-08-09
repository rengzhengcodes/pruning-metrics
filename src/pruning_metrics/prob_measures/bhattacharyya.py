"""Bhattacharyya distance, -ln Sigma sqrt(p*q)."""

from __future__ import annotations

import math

import numpy as np

from pruning_metrics.prob_measures.base import (
    EPS,
    MetricInfo,
    TokenStepDict,
    aligned_sum,
)

NAME = "bhattacharyya"

INFO = MetricInfo("Bhattacharyya", "f-divergence", True, None, "−ln Σ√(p·q)")


def bhattacharyya_coefficient(p: np.ndarray, q: np.ndarray) -> float:
    """Computes the Bhattacharyya coefficient Sigma sqrt(p*q) of two vectors.

    Parameters
    ----------
    p:
        First aligned probability vector.
    q:
        Second aligned probability vector.

    Returns
    -------
    float
        The overlap coefficient, in [0, 1].
    """
    return float(np.sum(np.sqrt(p * q)))


def kernel(p: np.ndarray, q: np.ndarray) -> float:
    """Per-position Bhattacharyya distance on two aligned probability vectors."""
    return -math.log(max(bhattacharyya_coefficient(p, q), EPS))


def compute_bhattacharyya(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sums the Bhattacharyya distance, -ln Sigma sqrt(p*q), over aligned positions.

    The unbounded log-form of the same overlap coefficient that Hellinger
    uses in bounded form.  Per position, the two measures are related by a
    fixed monotone map.  That map is nonlinear.  So once summed over a
    sequence, the two measures order model pairs differently.  Bhattacharyya
    lets a handful of near-disjoint positions dominate an answer's score.
    Hellinger, by contrast, caps each position's contribution at 1.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed Bhattacharyya distance (non-negative).
    """
    return aligned_sum(tokens_0, tokens_k, kernel)


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_bhattacharyya
