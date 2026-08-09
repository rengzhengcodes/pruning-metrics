"""Renyi divergence of order 2, ln Sigma p^2 / q."""

from __future__ import annotations

import math

import numpy as np

from pruning_metrics.prob_measures.base import (
    EPS,
    MetricInfo,
    TokenStepDict,
    aligned_sum,
)

NAME = "renyi2"

INFO = MetricInfo("Rényi α=2", "f-divergence", False, None, "ln Σ p²/q")


def kernel(p: np.ndarray, q: np.ndarray) -> float:
    """Per-position order-2 Rényi divergence on two aligned probability vectors."""
    return math.log(max(float(np.sum(p**2 / (q + EPS))), EPS))


def compute_renyi2(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sums the Rényi divergence of order 2, ln Σ p²/q, over aligned positions.

    Equal to ``ln(1 + χ²)`` per position. That map is monotone but
    nonlinear. Unlike the order-½ case, it therefore does not survive
    summation. This makes it a genuinely different measure from
    :func:`compute_chisq`, rather than a rescaling of it. Practically, it
    is the usable heavy-tail divergence for this data. It keeps χ²'s
    sensitivity to the pruned model's tails. At the same time, it
    compresses a disjoint-support position from order 1e12 down to order
    28. That compressed value is comparable with the other unbounded
    measures.

    It is also the only measure in this module that behaves differently
    from the rest on real data. Notebook 08 finds all sixteen agree at
    ρ ≥ 0.84 except this one. This one is the least correlated with the
    others. It is also the only one whose predictive R² varies wildly
    across benchmarks.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed order-2 Rényi divergence (non-negative).
    """
    return aligned_sum(tokens_0, tokens_k, kernel)


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_renyi2
