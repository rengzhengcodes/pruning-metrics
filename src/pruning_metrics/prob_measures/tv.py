"""Total variation distance, ½‖p − q‖₁."""

from __future__ import annotations

import numpy as np

from pruning_metrics.prob_measures.base import (
    MetricInfo,
    TokenStepDict,
    aligned_sum,
)

NAME = "tv"

INFO = MetricInfo("Total variation", "f-divergence", True, 1.0, "½·Σ|p − q|")


def kernel(p: np.ndarray, q: np.ndarray) -> float:
    """Per-position total variation distance on two aligned probability vectors."""
    return 0.5 * float(np.sum(np.abs(p - q)))


def compute_tv(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sums total variation distance, ½‖p − q‖₁, over aligned positions.

    This is the canonical statistical distance. It measures the largest
    possible disagreement between the two models about the probability of
    any single event. It is also a true metric bounded in [0, 1] per
    position. Pinsker's inequality makes it a lower bound on √(KL/2). So
    where TV saturates and KL does not, the extra KL is coming from tail
    ratios rather than from mass actually moving.

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
        Summed total variation distance (non-negative).
    """
    return aligned_sum(tokens_0, tokens_k, kernel)


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_tv
