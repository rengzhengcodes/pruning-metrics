"""Renyi divergence of order 1/2, -2*ln Sigma sqrt(p*q)."""

from __future__ import annotations

import numpy as np

from pruning_metrics.prob_measures.bhattacharyya import kernel as bhattacharyya_kernel
from pruning_metrics.prob_measures.base import (
    MetricInfo,
    TokenStepDict,
    aligned_sum,
)

NAME = "renyi05"

INFO = MetricInfo("Rényi α=½", "f-divergence", True, None, "−2·ln Σ√(p·q)")


def kernel(p: np.ndarray, q: np.ndarray) -> float:
    """Per-position order-1/2 Renyi divergence on two aligned probability vectors."""
    return 2.0 * bhattacharyya_kernel(p, q)


def compute_renyi05(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sums the Renyi divergence of order 1/2 over aligned positions.

    Included for completeness of the Rényi family. Be aware that
    ``D_½(P‖Q) = −2 ln Σ√(p·q) = 2 · D_B(P, Q)`` exactly. This identity is
    *linear*. Unlike the Bhattacharyya/Hellinger pair, it therefore does
    survive summation. This function returns exactly twice
    :func:`compute_bhattacharyya` on every input. It therefore carries no
    information the Bhattacharyya column does not. In a metric-agreement
    analysis it should show a rank correlation of exactly 1 with it. This
    makes it a useful built-in check that such an analysis is wired up
    right.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed order-½ Rényi divergence (non-negative).
    """
    return aligned_sum(tokens_0, tokens_k, kernel)


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_renyi05
