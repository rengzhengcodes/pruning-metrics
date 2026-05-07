"""Distributional distance functions for teacher-forced token predictions.

Each public function accepts two ``list[TokenStepDict]`` arguments — one from
the base (level=0) model and one from a pruned model — and returns a single
non-negative float aggregated over all token positions.

The input dicts match the serialised form of
:class:`pruning_metrics.evals.coding.teacher_forcing.TeacherForcedTokenStep`
so callers can pass ``json.load(per_token_path)["per_token"]`` directly.

Implementation notes
--------------------
- Only the ``top_alternatives`` field is used (not ``target_logprob``).
  The top-5 alternatives represent the model's predicted distribution;
  we renormalise them locally so they sum to 1 over the top-k support.
- Tokens absent from one model's top-5 receive logprob -50 before
  renormalisation (≈ 2e-22 probability), so KLD/JSD divergence is large
  but finite when the two support sets are disjoint.
- ``scipy`` is required for EMD; an ``ImportError`` with an install hint
  is raised at import time if it is missing.
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np

try:
    from scipy.stats import wasserstein_distance as _scipy_wasserstein
except ImportError as _exc:
    raise ImportError(
        "scipy is required for EMD: pip install scipy"
    ) from _exc


# ---------------------------------------------------------------------------
# Public TypedDicts
# ---------------------------------------------------------------------------


class TopAlternativeDict(TypedDict):
    """One entry in a per-position top-k alternative list."""

    token_id: int
    token_text: str
    logprob: float


class TokenStepDict(TypedDict):
    """Serialised form of TeacherForcedTokenStep."""

    position: int
    target_token_id: int
    target_token_text: str
    target_logprob: float
    target_prob: float
    rank: int
    top_alternatives: list[TopAlternativeDict]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_MISSING_LOGPROB: float = -50.0  # stand-in for tokens absent from a top-k list


def _renorm(logprobs: np.ndarray) -> np.ndarray:
    """Local-softmax renormalisation: convert logprobs to a probability vector.

    Subtracts the max for numerical stability, exponentiates, then normalises
    so the output sums to 1 over the provided entries (i.e. renormalises within
    the top-k slice rather than the full vocabulary).
    """
    shifted = logprobs - logprobs.max()
    p = np.exp(shifted)
    return p / p.sum()


def _union_probs(
    alts_0: list[TopAlternativeDict],
    alts_k: list[TopAlternativeDict],
) -> tuple[np.ndarray, np.ndarray]:
    """Build aligned probability vectors over the union of two top-k support sets.

    Tokens appearing in one list but not the other receive ``_MISSING_LOGPROB``
    before renormalisation, giving them near-zero probability mass.

    Parameters
    ----------
    alts_0:
        Top-k alternatives from the base model at one token position.
    alts_k:
        Top-k alternatives from the pruned model at the same position.

    Returns
    -------
    p0, pk:
        Renormalised probability vectors of equal length (union vocabulary size).
        Entries correspond to the same token in both vectors.
    """
    map_0 = {a["token_id"]: a["logprob"] for a in alts_0}
    map_k = {a["token_id"]: a["logprob"] for a in alts_k}
    all_ids = list(set(map_0) | set(map_k))
    lp_0 = np.array([map_0.get(tid, _MISSING_LOGPROB) for tid in all_ids])
    lp_k = np.array([map_k.get(tid, _MISSING_LOGPROB) for tid in all_ids])
    return _renorm(lp_0), _renorm(lp_k)


def _sparse_point(alts: list[TopAlternativeDict]) -> dict[int, float]:
    """Convert a top-k alternative list to a sparse probability dict.

    Returns ``{token_id: renormalised_probability}`` with at most ``len(alts)``
    non-zero entries.  This represents a point in R^|vocab| with the token-id
    as the coordinate index.
    """
    if not alts:
        return {}
    logprobs = np.array([a["logprob"] for a in alts], dtype=np.float64)
    probs = _renorm(logprobs)
    return {a["token_id"]: float(p) for a, p in zip(alts, probs)}


def _sparse_l2_sq(a: dict[int, float], b: dict[int, float]) -> float:
    """Squared L2 distance between two sparse probability vectors in R^|vocab|.

    Uses the identity  ‖a - b‖² = ‖a‖² + ‖b‖² - 2⟨a, b⟩  to avoid
    materialising the full vocabulary dimension.  The dot product only touches
    tokens in the intersection of the two support sets (≤ top_k entries).
    """
    norm_a_sq = sum(v * v for v in a.values())
    norm_b_sq = sum(v * v for v in b.values())
    dot = sum(a[t] * b[t] for t in a if t in b)
    return max(0.0, norm_a_sq + norm_b_sq - 2.0 * dot)


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------


def compute_kld(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sum of KL(P_base ‖ P_pruned) over all aligned token positions.

    Directional: measures how surprised the base-model distribution is by
    the pruned model.  Values are large when the pruned model has moved
    probability mass to tokens that the base model considered very unlikely.

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
        Summed KL divergence (non-negative).
    """
    eps = 1e-12
    total = 0.0
    for step_0, step_k in zip(tokens_0, tokens_k):
        alts_0 = step_0.get("top_alternatives", [])
        alts_k = step_k.get("top_alternatives", [])
        if not alts_0 or not alts_k:
            continue
        p0, pk = _union_probs(alts_0, alts_k)
        total += float(np.sum(p0 * np.log((p0 + eps) / (pk + eps))))
    return total


def compute_jsd(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sum of √JSD(P_base, P_pruned) over all aligned token positions.

    Uses the log₂ convention so that √JSD ∈ [0, 1] per position.
    Symmetric: ``compute_jsd(a, b) == compute_jsd(b, a)``.

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
    eps = 1e-12
    ln2 = math.log(2.0)
    total = 0.0
    for step_0, step_k in zip(tokens_0, tokens_k):
        alts_0 = step_0.get("top_alternatives", [])
        alts_k = step_k.get("top_alternatives", [])
        if not alts_0 or not alts_k:
            continue
        p0, pk = _union_probs(alts_0, alts_k)
        m = 0.5 * (p0 + pk)
        kl_p_m = float(np.sum(p0 * np.log((p0 + eps) / (m + eps))))
        kl_q_m = float(np.sum(pk * np.log((pk + eps) / (m + eps))))
        jsd_nat = 0.5 * kl_p_m + 0.5 * kl_q_m
        jsd_log2 = jsd_nat / ln2
        total += math.sqrt(max(0.0, jsd_log2))
    return total


def compute_emd(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sum of Wasserstein-1 distances over all aligned token positions.

    Treats each model's top-k logprob values as atom *positions* on the real
    line and the renormalised probabilities as *weights*.  The EMD between two
    positions then reflects the "work" required to transport the base model's
    probability mass to the locations (logprob values) preferred by the pruned
    model.

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
    total = 0.0
    for step_0, step_k in zip(tokens_0, tokens_k):
        alts_0 = step_0.get("top_alternatives", [])
        alts_k = step_k.get("top_alternatives", [])
        if not alts_0 or not alts_k:
            continue
        lp_0 = np.array([a["logprob"] for a in alts_0], dtype=np.float64)
        lp_k = np.array([a["logprob"] for a in alts_k], dtype=np.float64)
        w0 = _renorm(lp_0)
        wk = _renorm(lp_k)
        total += float(_scipy_wasserstein(lp_0, lp_k, u_weights=w0, v_weights=wk))
    return total


def compute_chamfer(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Symmetric Chamfer distance between two sets of sparse vocabulary vectors.

    Each token position in a sequence contributes one sparse point in R^|vocab|:
    the top-k predictions with their renormalised probabilities as coordinates
    indexed by token-id.  The full set of N such points (one per answer token)
    forms a "point cloud" representing the model's prediction behaviour on that
    prompt.

    Unlike the position-aligned metrics (KLD/JSD/EMD), Chamfer uses nearest-
    neighbour matching across *all* positions.  This captures the geometric
    similarity of the prediction manifold regardless of which specific position
    produced which distribution.

    Distance between two sparse points ``a`` and ``b`` in R^|vocab| is computed
    as  ``‖a - b‖ = √(‖a‖² + ‖b‖² - 2⟨a, b⟩)``  exploiting the fact that
    the dot product only touches tokens in the intersection of the two support
    sets (≤ top_k entries per pair).

    The final value uses *mean* rather than sum over each direction of the
    Chamfer formula so that the result is normalised for sequence length,
    making it comparable across datasets with different answer lengths.

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
    pts_0 = [_sparse_point(s.get("top_alternatives", [])) for s in tokens_0]
    pts_k = [_sparse_point(s.get("top_alternatives", [])) for s in tokens_k]

    # Drop positions where both models have empty top-alternatives
    pts_0 = [p for p in pts_0 if p]
    pts_k = [p for p in pts_k if p]

    n, m = len(pts_0), len(pts_k)
    if n == 0 or m == 0:
        return 0.0

    # Build N×M matrix of squared L2 distances
    dist_sq = np.zeros((n, m), dtype=np.float64)
    for i, a in enumerate(pts_0):
        for j, b in enumerate(pts_k):
            dist_sq[i, j] = _sparse_l2_sq(a, b)

    dist = np.sqrt(dist_sq)
    fwd = float(np.mean(dist.min(axis=1)))  # each point in A to nearest in B
    bwd = float(np.mean(dist.min(axis=0)))  # each point in B to nearest in A
    return fwd + bwd
