"""Pearson chi-squared divergence, Sigma (p - q)^2 / q."""

from __future__ import annotations

import numpy as np

from pruning_metrics.prob_measures.base import (
    EPS,
    MetricInfo,
    TokenStepDict,
    aligned_sum,
)

NAME = "chisq"

INFO = MetricInfo("Pearson χ²", "f-divergence", False, None, "Σ (p − q)²/q")


def kernel(p: np.ndarray, q: np.ndarray) -> float:
    """Per-position Pearson chi-squared on two aligned probability vectors."""
    return float(np.sum((p - q) ** 2 / (q + EPS)))


def compute_chisq(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sums the Pearson χ² divergence, Σ (p − q)²/q, over aligned positions.

    This is the most tail-sensitive f-divergence here.  For that reason, it
    is the one to treat with suspicion on this data.  Its denominator is
    the *pruned* model's probability.  A token that the pruned model
    dropped out of its top-k has ``q`` set by the −50 logprob fill.  So the
    ``eps = 1e-12`` denominator guard is what sets the scale: a position
    where the two support sets are disjoint contributes on the order of
    1e12, versus order 1 for every other measure in this module.  A χ²
    distance matrix is consequently close to a count of disjoint-support
    positions multiplied by a large constant.

    That count does correlate with damage.  Notebook 08 measures R² ≈ 0.98
    against pass@1 drop, in line with the rest.  So the *regression* is not
    the problem.  What is not meaningful is the magnitude.  A χ² of 1e13
    should be read as "roughly twenty positions disagreed completely",
    never as an amount of divergence.  Differences between two such values
    say more about the epsilon than about the models.  It is also the
    second-worst measure to feed through classical MDS.  Notebook 08
    measures 31% of the eigenvalue mass discarded, against exactly 0% for
    JSD, Hellinger and total variation.  Use :func:`compute_renyi2`, the
    same quantity on a log scale, when a heavy-tailed measure is genuinely
    wanted.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed χ² divergence (non-negative).
    """
    return aligned_sum(tokens_0, tokens_k, kernel)


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_chisq
