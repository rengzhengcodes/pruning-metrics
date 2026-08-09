"""Triangular discrimination, Sigma (p - q)^2 / (p + q)."""

from __future__ import annotations

import numpy as np

from pruning_metrics.prob_measures.base import (
    EPS,
    MetricInfo,
    TokenStepDict,
    aligned_sum,
)

NAME = "triangular"

INFO = MetricInfo("Triangular", "f-divergence", True, 2.0, "Σ (p − q)²/(p + q)")


def kernel(p: np.ndarray, q: np.ndarray) -> float:
    """Per-position triangular discrimination on two aligned probability vectors."""
    return float(np.sum((p - q) ** 2 / (p + q + EPS)))


def compute_triangular(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sums triangular discrimination, Σ (p − q)²/(p + q), over aligned positions.

    This measure is also called Le Cam's divergence. It is χ²'s bounded
    companion, obtained by symmetrising the denominator. It is bounded by
    2 per position. As a result, it cannot be dominated by a single
    disjoint-support token the way χ² is. It also keeps the quadratic
    numerator, which makes both triangular discrimination and χ² more
    sensitive than total variation to a large disagreement on one token.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed triangular discrimination (non-negative).
    """
    return aligned_sum(tokens_0, tokens_k, kernel)


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_triangular
