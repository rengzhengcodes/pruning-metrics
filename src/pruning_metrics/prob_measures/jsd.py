"""Jensen-Shannon divergence, in bounded square-root form."""

from __future__ import annotations

import math

import numpy as np

from pruning_metrics.prob_measures.base import (
    LN2,
    MetricInfo,
    TokenStepDict,
    aligned_sum,
)
from pruning_metrics.prob_measures.kld import kernel as kld_kernel

NAME = "jsd"

INFO = MetricInfo("Jensen-Shannon", "f-divergence", True, 1.0, "√JSD(p, q), log₂ units")


def kernel(p: np.ndarray, q: np.ndarray) -> float:
    """Per-position √JSD (log₂ units) on two aligned probability vectors.

    Composed from the forward-KL kernel by definition.  JSD is the mean of
    the two KL divergences to the midpoint distribution.
    """
    m = 0.5 * (p + q)
    jsd_nat = 0.5 * kld_kernel(p, m) + 0.5 * kld_kernel(q, m)
    return math.sqrt(max(0.0, jsd_nat / LN2))


def compute_jsd(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sums √JSD(P_base, P_pruned) over all aligned token positions.

    Uses the log₂ convention so that √JSD ∈ [0, 1] per position.
    Symmetric: ``compute_jsd(a, b) == compute_jsd(b, a)``.

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
        Summed square-root JSD (non-negative).
    """
    return aligned_sum(tokens_0, tokens_k, kernel)


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_jsd
