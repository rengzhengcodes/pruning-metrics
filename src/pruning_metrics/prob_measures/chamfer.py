"""Symmetric Chamfer distance between sparse per-position vocabulary points."""

from __future__ import annotations

import numpy as np

from pruning_metrics.prob_measures.base import (
    MetricInfo,
    TokenStepDict,
    TopAlternativeDict,
    atom_weights,
)

NAME = "chamfer"

INFO = MetricInfo("Chamfer", "point-cloud", True, None, "mean-NN both directions")


def sparse_point(alts: list[TopAlternativeDict]) -> dict[int, float]:
    """Converts a top-k alternative list to a sparse probability dict.

    Used only by Chamfer, which is why it lives in this module rather than
    in ``base.py``: the other measures work on dense union-support vectors
    or atom/weight pairs, not on sparse vocabulary-indexed points.

    Returns ``{token_id: renormalised_probability}`` with at most
    ``len(alts)`` non-zero entries.  This represents a point in R^|vocab|
    with the token-id as the coordinate index.

    Parameters
    ----------
    alts:
        A top-k alternative list.

    Returns
    -------
    dict[int, float]
        Sparse mapping from token id to renormalised probability.
    """
    if not alts:
        return {}
    # Same renormalisation as the transport measures; only the weights are
    # kept — the token ids, not the logprob values, index this point.
    _, probs = atom_weights(alts)
    return {a["token_id"]: float(p) for a, p in zip(alts, probs)}


def sparse_l2_sq(a: dict[int, float], b: dict[int, float]) -> float:
    """Squared L2 distance between two sparse probability vectors in R^|vocab|.

    Used only by Chamfer.  Uses the identity
    ``‖a - b‖² = ‖a‖² + ‖b‖² - 2⟨a, b⟩`` to avoid materialising the full
    vocabulary dimension.  The dot product only touches tokens in the
    intersection of the two support sets (≤ top_k entries).

    Parameters
    ----------
    a:
        Sparse probability vector (token id -> probability).
    b:
        Sparse probability vector (token id -> probability).

    Returns
    -------
    float
        Squared Euclidean distance (non-negative).
    """
    norm_a_sq = sum(v * v for v in a.values())
    norm_b_sq = sum(v * v for v in b.values())
    dot = sum(a[t] * b[t] for t in a if t in b)
    return max(0.0, norm_a_sq + norm_b_sq - 2.0 * dot)


def compute_chamfer(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Symmetric Chamfer distance between two sets of sparse vocabulary vectors.

    Each token position in a sequence contributes one sparse point in
    R^|vocab|.  The coordinates are the top-k predictions' renormalised
    probabilities, indexed by token-id.  The full set of N such points (one
    per answer token) forms a "point cloud" representing the model's
    prediction behaviour on that prompt.

    Unlike the position-aligned metrics (KLD/JSD/EMD), Chamfer uses
    nearest-neighbour matching across *all* positions.  This captures the
    geometric similarity of the prediction manifold regardless of which
    specific position produced which distribution.

    Distance between two sparse points ``a`` and ``b`` in R^|vocab| is
    computed as ``‖a - b‖ = √(‖a‖² + ‖b‖² - 2⟨a, b⟩)``.  This exploits the
    fact that the dot product only touches tokens in the intersection of
    the two support sets (≤ top_k entries per pair).

    The final value uses *mean* rather than sum over each direction of the
    Chamfer formula.  This normalises the result for sequence length.  That
    makes it comparable across datasets with different answer lengths.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Symmetric Chamfer distance (non-negative).
    """
    pts_0 = [sparse_point(s.get("top_alternatives", [])) for s in tokens_0]
    pts_k = [sparse_point(s.get("top_alternatives", [])) for s in tokens_k]

    # Drop each cloud's empty points (positions with no top-alternatives)
    pts_0 = [p for p in pts_0 if p]
    pts_k = [p for p in pts_k if p]

    n, m = len(pts_0), len(pts_k)
    if n == 0 or m == 0:
        return 0.0

    # Build N×M matrix of squared L2 distances
    dist_sq = np.zeros((n, m), dtype=np.float64)
    for i, a in enumerate(pts_0):
        for j, b in enumerate(pts_k):
            dist_sq[i, j] = sparse_l2_sq(a, b)

    dist = np.sqrt(dist_sq)
    fwd = float(np.mean(dist.min(axis=1)))  # each point in A to nearest in B
    bwd = float(np.mean(dist.min(axis=0)))  # each point in B to nearest in A
    return fwd + bwd


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_chamfer
