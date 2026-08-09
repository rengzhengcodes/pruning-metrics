"""Reverse Kullback-Leibler divergence, KL(P_pruned ‖ P_base)."""

from __future__ import annotations

import numpy as np

from pruning_metrics.prob_measures.base import (
    EPS,
    MetricInfo,
    TokenStepDict,
    aligned_sum,
)

NAME = "rkld"

INFO = MetricInfo("Reverse KL", "f-divergence", False, None, "Σ q·ln(q/p)")


def kernel(p: np.ndarray, q: np.ndarray) -> float:
    """Per-position reverse KL on two aligned probability vectors."""
    return float(np.sum(q * np.log((q + EPS) / (p + EPS))))


def compute_rkld(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sums the *reverse* KL, KL(P_pruned ‖ P_base), over aligned positions.

    The mirror image of ``compute_kld``. Forward KL is mass-covering. It
    punishes the pruned model for putting no mass where the base model
    does. Reverse KL is mode-seeking. It punishes the pruned model for
    putting mass where the base model does not. A pruned model that has
    collapsed onto a single confident token scores low forward and high
    reverse. One that has flattened into hedging scores the other way
    round. The gap between the two is therefore a readout of *how* the
    model broke, not just how much.

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
        Summed reverse KL divergence (non-negative).
    """
    return aligned_sum(tokens_0, tokens_k, kernel)


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_rkld
