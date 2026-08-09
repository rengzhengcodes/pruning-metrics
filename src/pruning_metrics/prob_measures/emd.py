"""Wasserstein-1 (earth mover's) distance over logprob atoms."""

from __future__ import annotations

import numpy as np

try:
    from scipy.stats import wasserstein_distance as _scipy_wasserstein
except ImportError as _exc:
    raise ImportError("scipy is required for EMD: pip install scipy") from _exc

from pruning_metrics.prob_measures.base import (
    MetricInfo,
    TokenStepDict,
    transport_sum,
)

NAME = "emd"

INFO = MetricInfo("Wasserstein-1", "transport", True, None, "W₁ over logprob atoms")


def transport_distance(
    u_values: np.ndarray,
    u_weights: np.ndarray,
    v_values: np.ndarray,
    v_weights: np.ndarray,
) -> float:
    """Computes the Wasserstein-1 distance between two weighted point masses.

    :func:`compute_all` calls this directly (rather than reimplementing the
    scipy call), which is what guarantees the batch and single-metric routes
    agree bit-for-bit.

    Parameters
    ----------
    u_values:
        Atom positions of the first distribution.
    u_weights:
        Non-negative masses at ``u_values``.
    v_values:
        Atom positions of the second distribution.
    v_weights:
        Non-negative masses at ``v_values``.

    Returns
    -------
    float
        The Wasserstein-1 distance (non-negative).
    """
    return float(
        _scipy_wasserstein(u_values, v_values, u_weights=u_weights, v_weights=v_weights)
    )


def compute_emd(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sums Wasserstein-1 distances over all aligned token positions.

    Treats each model's top-k logprob values as atom *positions* on the real
    line.  Treats the renormalised probabilities as *weights*.  The EMD
    between two positions then reflects the "work" required to transport
    the base model's probability mass to the locations (logprob values)
    preferred by the pruned model.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed Wasserstein-1 distance (non-negative).
    """
    return transport_sum(tokens_0, tokens_k, transport_distance)


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_emd
