"""Forward Kullback-Leibler divergence, KL(P_base ‖ P_pruned)."""

from __future__ import annotations

import numpy as np

from pruning_metrics.prob_measures.base import (
    EPS,
    MetricInfo,
    TokenStepDict,
    aligned_sum,
)

NAME = "kld"

INFO = MetricInfo("Forward KL", "f-divergence", False, None, "Σ p·ln(p/q)")


def kernel(p: np.ndarray, q: np.ndarray) -> float:
    """Per-position forward KL on two aligned probability vectors."""
    return float(np.sum(p * np.log((p + EPS) / (q + EPS))))


def compute_kld(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sums KL(P_base ‖ P_pruned) over all aligned token positions.

    Directional: measures how surprised the base-model distribution is by
    the pruned model.  Values are large when the pruned model has moved
    probability mass to tokens that the base model considered very
    unlikely.

    If the two lists have different lengths only the first
    ``min(len(tokens_0), len(tokens_k))`` positions are used.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed KL divergence (non-negative).
    """
    return aligned_sum(tokens_0, tokens_k, kernel)


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_kld
