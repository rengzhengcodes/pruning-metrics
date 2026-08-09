"""Jeffreys divergence, the symmetrised KL, KL(P‖Q) + KL(Q‖P)."""

from __future__ import annotations

import numpy as np

from pruning_metrics.prob_measures.base import (
    MetricInfo,
    TokenStepDict,
    aligned_sum,
)
from pruning_metrics.prob_measures.kld import kernel as kld_kernel
from pruning_metrics.prob_measures.rkld import kernel as rkld_kernel

NAME = "jeffreys"

INFO = MetricInfo("Jeffreys", "f-divergence", True, None, "KL(p‖q) + KL(q‖p)")


def kernel(p: np.ndarray, q: np.ndarray) -> float:
    """Per-position Jeffreys divergence on two aligned probability vectors."""
    return kld_kernel(p, q) + rkld_kernel(p, q)


def compute_jeffreys(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sums the Jeffreys divergence, KL(P‖Q) + KL(Q‖P), over aligned positions.

    The symmetrised KL.  Unlike JSD, it is unbounded.  So it stays
    sensitive in the regime where JSD has saturated at its ceiling.  That
    regime is exactly the heavily-pruned regime this study cares about.

    Equal to ``compute_kld(a, b) + compute_rkld(a, b)`` up to the order in
    which the positions are summed.  This function adds the two directions
    together within each position.  So the two routes differ in the last
    few ulps on a long sequence.

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
        Summed Jeffreys divergence (non-negative).
    """
    return aligned_sum(tokens_0, tokens_k, kernel)


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_jeffreys
