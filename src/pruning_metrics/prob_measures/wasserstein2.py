"""Wasserstein-2 distance over logprob atoms, the quadratic sibling of EMD."""

from __future__ import annotations

import numpy as np

from pruning_metrics.prob_measures.base import (
    MetricInfo,
    TokenStepDict,
    transport_sum,
)

NAME = "wasserstein2"

INFO = MetricInfo("Wasserstein-2", "transport", True, None, "W₂ over logprob atoms")


def wasserstein_p(
    u_values: np.ndarray,
    u_weights: np.ndarray,
    v_values: np.ndarray,
    v_weights: np.ndarray,
    p: float,
) -> float:
    r"""Order-``p`` Wasserstein distance between two weighted 1-D point masses.

    Uses the quantile form :math:`W_p^p = \int_0^1 |F^{-1}(t) - G^{-1}(t)|^p dt`.
    Note this is *not* the CDF-difference integral :math:`\int |F - G|`,
    which coincides with it only at ``p = 1``.
    ``scipy.stats.wasserstein_distance`` implements the latter, so it
    cannot be reused for ``p = 2``.

    Both quantile functions are step functions that are constant between
    consecutive levels of the merged CDF. The integral is therefore an
    exact finite sum over those intervals.

    Parameters
    ----------
    u_values:
        Atom positions on the real line. Need not be sorted.
    u_weights:
        Non-negative masses at ``u_values``; normalised internally.
    v_values:
        Atom positions on the real line. Need not be sorted.
    v_weights:
        Non-negative masses at ``v_values``; normalised internally.
    p:
        Order of the distance; ``p >= 1``.

    Returns
    -------
    float
        The Wasserstein-``p`` distance (non-negative).
    """
    # Sort each distribution's atoms so the running mass below is a CDF.
    u_order = np.argsort(u_values)
    v_order = np.argsort(v_values)
    u_val, u_w = u_values[u_order], u_weights[u_order]
    v_val, v_w = v_values[v_order], v_weights[v_order]

    # Cumulative mass per atom, normalised so each CDF tops out at exactly 1
    # (the incoming weights are not required to sum to 1).
    u_cdf = np.cumsum(u_w)
    v_cdf = np.cumsum(v_w)
    u_cdf = u_cdf / u_cdf[-1]
    v_cdf = v_cdf / v_cdf[-1]

    # Breakpoints of the merged quantile function. On the interval
    # (levels[i-1], levels[i]] both quantile functions are constant.
    levels = np.unique(np.concatenate([u_cdf, v_cdf]))
    widths = np.diff(np.concatenate([[0.0], levels]))

    # Right-continuous inverse: the first atom whose cumulative mass reaches
    # the level. Clipping absorbs float drift at the top of the CDF.
    u_idx = np.clip(np.searchsorted(u_cdf, levels, side="left"), 0, u_val.size - 1)
    v_idx = np.clip(np.searchsorted(v_cdf, levels, side="left"), 0, v_val.size - 1)

    gaps = np.abs(u_val[u_idx] - v_val[v_idx])
    return float(np.sum(widths * gaps**p) ** (1.0 / p))


def transport_distance(
    u_values: np.ndarray,
    u_weights: np.ndarray,
    v_values: np.ndarray,
    v_weights: np.ndarray,
) -> float:
    """Computes the Wasserstein-2 distance between two weighted point masses.

    :func:`compute_all` calls this directly. This is what guarantees the
    batch and single-metric routes agree bit-for-bit.

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
        The Wasserstein-2 distance (non-negative).
    """
    return wasserstein_p(u_values, u_weights, v_values, v_weights, 2.0)


def compute_wasserstein2(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sums Wasserstein-2 distances over all aligned token positions.

    This is the quadratic sibling of ``compute_emd``, using the same
    representation. Logprob values serve as atom positions, and
    renormalised probabilities serve as weights. The cost of moving mass
    grows quadratically with distance. W2 therefore weights a few
    far-flung disagreements more heavily than many small ones. W1, by
    contrast, charges the same total for either. Comparing the two
    therefore says whether a pruned model's drift is diffuse or
    concentrated.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed Wasserstein-2 distance (non-negative).
    """
    return transport_sum(tokens_0, tokens_k, transport_distance)


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_wasserstein2
