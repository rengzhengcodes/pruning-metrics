"""Distributional distance functions for teacher-forced token predictions.

Each public function accepts two ``list[TokenStepDict]`` arguments — one from
the base (level=0) model and one from a pruned model — and returns a single
non-negative float aggregated over all token positions.

The input dicts match the serialised form of
:class:`pruning_metrics.evals.coding.teacher_forcing.TeacherForcedTokenStep`
so callers can pass ``json.load(per_token_path)["per_token"]`` directly.

Sixteen measures are provided, in four families (see :data:`METRIC_INFO` for
the machine-readable version of this table):

*Aligned f-divergences and vector geometry* — computed on the two probability
vectors at each position after they have been aligned onto the union of the
two top-k support sets: ``kld``, ``rkld``, ``jeffreys``, ``jsd``, ``tv``,
``hellinger``, ``bhattacharyya``, ``renyi05``, ``chisq``, ``renyi2``,
``triangular``, ``l2``, ``cosine``.

*Optimal transport* — computed on each model's own logprob values as atom
positions on the real line: ``emd`` (Wasserstein-1), ``wasserstein2``.

*Point cloud* — ``chamfer``, which matches positions by nearest neighbour
rather than by index.

Implementation notes
--------------------
- Only the ``top_alternatives`` field is used (not ``target_logprob``).
  The top-5 alternatives represent the model's predicted distribution;
  we renormalise them locally so they sum to 1 over the top-k support.
- Tokens absent from one model's top-5 receive logprob -50 before
  renormalisation (≈ 2e-22 probability), so KLD/JSD divergence is large
  but finite when the two support sets are disjoint.
- All ratio-form divergences guard their denominator with ``eps = 1e-12``,
  which is far larger than the 2e-22 fill above.  For the bounded measures
  this is numerically irrelevant; for ``chisq`` it is what keeps a
  disjoint-support position finite (see that function's docstring).
- ``scipy`` is required for EMD; an ``ImportError`` with an install hint
  is raised at import time if it is missing.

Computing several measures separately re-parses the same JSON and rebuilds
the same union-support vectors once per measure.  :func:`compute_all` does
one pass and returns every measure, which is what the notebooks use; the
single-metric functions remain the readable reference implementations and
are what the unit tests check ``compute_all`` against.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, NamedTuple, TypedDict

import numpy as np

try:
    from scipy.stats import wasserstein_distance as _scipy_wasserstein
except ImportError as _exc:
    raise ImportError("scipy is required for EMD: pip install scipy") from _exc


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


def _atom_weights(alts: list[TopAlternativeDict]) -> tuple[np.ndarray, np.ndarray]:
    """Convert a top-k alternative list to (atom positions, weights) on the line.

    The *positions* are the raw logprob values and the *weights* are those same
    logprobs renormalised into a probability vector.  This is the representation
    the optimal-transport metrics work in: unlike the union-support alignment
    used by the f-divergences, each model keeps its own support, and the ground
    distance between two predictions is how far apart their logprobs are.
    """
    logprobs = np.array([a["logprob"] for a in alts], dtype=np.float64)
    return logprobs, _renorm(logprobs)


def _wasserstein_p(
    u_values: np.ndarray,
    u_weights: np.ndarray,
    v_values: np.ndarray,
    v_weights: np.ndarray,
    p: float,
) -> float:
    r"""Order-``p`` Wasserstein distance between two weighted 1-D point masses.

    Uses the quantile form  :math:`W_p^p = \int_0^1 |F^{-1}(t) - G^{-1}(t)|^p dt`.
    Note this is *not* the CDF-difference integral :math:`\int |F - G|`, which
    coincides with it only at ``p = 1``; ``scipy.stats.wasserstein_distance``
    implements the latter and so cannot be reused for ``p = 2``.

    Both quantile functions are step functions that are constant between
    consecutive levels of the merged CDF, so the integral is an exact finite
    sum over those intervals.

    Parameters
    ----------
    u_values, v_values:
        Atom positions on the real line.  Need not be sorted.
    u_weights, v_weights:
        Non-negative masses at those positions; each is normalised internally.
    p:
        Order of the distance; ``p >= 1``.

    Returns
    -------
    float
        The Wasserstein-``p`` distance (non-negative).
    """
    u_order = np.argsort(u_values)
    v_order = np.argsort(v_values)
    u_val, u_w = u_values[u_order], u_weights[u_order]
    v_val, v_w = v_values[v_order], v_weights[v_order]

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


# ---------------------------------------------------------------------------
# Per-position kernels
#
# Each takes two aligned probability vectors over the union support and
# returns that position's contribution.  The public metric functions and
# ``compute_all`` share these, so a value is computed by exactly the same
# float operations in exactly the same order no matter which route is taken.
# ---------------------------------------------------------------------------

_EPS: float = 1e-12
_LN2: float = math.log(2.0)


def _k_kld(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log((p + _EPS) / (q + _EPS))))


def _k_rkld(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(q * np.log((q + _EPS) / (p + _EPS))))


def _k_jeffreys(p: np.ndarray, q: np.ndarray) -> float:
    return _k_kld(p, q) + _k_rkld(p, q)


def _k_jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl_p_m = float(np.sum(p * np.log((p + _EPS) / (m + _EPS))))
    kl_q_m = float(np.sum(q * np.log((q + _EPS) / (m + _EPS))))
    jsd_nat = 0.5 * kl_p_m + 0.5 * kl_q_m
    return math.sqrt(max(0.0, jsd_nat / _LN2))


def _k_tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(p - q)))


def _k_hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(np.sqrt(p) - np.sqrt(q))) / math.sqrt(2.0)


def _bhattacharyya_coefficient(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.sqrt(p * q)))


def _k_bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    return -math.log(max(_bhattacharyya_coefficient(p, q), _EPS))


def _k_renyi05(p: np.ndarray, q: np.ndarray) -> float:
    return 2.0 * _k_bhattacharyya(p, q)


def _k_chisq(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum((p - q) ** 2 / (q + _EPS)))


def _k_renyi2(p: np.ndarray, q: np.ndarray) -> float:
    return math.log(max(float(np.sum(p**2 / (q + _EPS))), _EPS))


def _k_triangular(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum((p - q) ** 2 / (p + q + _EPS)))


def _k_l2(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))


def _k_cosine(p: np.ndarray, q: np.ndarray) -> float:
    denom = float(np.linalg.norm(p)) * float(np.linalg.norm(q))
    if denom <= 0.0:
        return 0.0
    return float(np.clip(1.0 - float(np.dot(p, q)) / denom, 0.0, 2.0))


#: Per-position kernels keyed by metric name, for metrics computed on the
#: union-support probability vectors.  Insertion order is the reporting order.
_UNION_KERNELS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "kld": _k_kld,
    "rkld": _k_rkld,
    "jeffreys": _k_jeffreys,
    "jsd": _k_jsd,
    "tv": _k_tv,
    "hellinger": _k_hellinger,
    "bhattacharyya": _k_bhattacharyya,
    "renyi05": _k_renyi05,
    "chisq": _k_chisq,
    "renyi2": _k_renyi2,
    "triangular": _k_triangular,
    "l2": _k_l2,
    "cosine": _k_cosine,
}


def _aligned_sum(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
    kernel: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Sum a per-position kernel over aligned positions with non-empty top-k.

    If the two lists have different lengths only the first
    ``min(len(tokens_0), len(tokens_k))`` positions are used.
    """
    total = 0.0
    for step_0, step_k in zip(tokens_0, tokens_k):
        alts_0 = step_0.get("top_alternatives", [])
        alts_k = step_k.get("top_alternatives", [])
        if not alts_0 or not alts_k:
            continue
        p0, pk = _union_probs(alts_0, alts_k)
        total += kernel(p0, pk)
    return total


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
    return _aligned_sum(tokens_0, tokens_k, _k_kld)


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
    return _aligned_sum(tokens_0, tokens_k, _k_jsd)


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
        lp_0, w0 = _atom_weights(alts_0)
        lp_k, wk = _atom_weights(alts_k)
        total += float(_scipy_wasserstein(lp_0, lp_k, u_weights=w0, v_weights=wk))
    return total


def compute_wasserstein2(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sum of Wasserstein-2 distances over all aligned token positions.

    The quadratic sibling of :func:`compute_emd`, on the same representation:
    logprob values as atom positions, renormalised probabilities as weights.
    Because the cost of moving mass grows quadratically with distance, W2
    weights a few far-flung disagreements more heavily than many small ones,
    where W1 charges the same total for either.  Comparing the two therefore
    says whether a pruned model's drift is diffuse or concentrated.

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
    total = 0.0
    for step_0, step_k in zip(tokens_0, tokens_k):
        alts_0 = step_0.get("top_alternatives", [])
        alts_k = step_k.get("top_alternatives", [])
        if not alts_0 or not alts_k:
            continue
        lp_0, w0 = _atom_weights(alts_0)
        lp_k, wk = _atom_weights(alts_k)
        total += _wasserstein_p(lp_0, w0, lp_k, wk, 2.0)
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


def compute_rkld(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sum of the *reverse* KL, KL(P_pruned ‖ P_base), over aligned positions.

    The mirror image of :func:`compute_kld`.  Forward KL is mass-covering — it
    punishes the pruned model for putting no mass where the base model does.
    Reverse KL is mode-seeking — it punishes the pruned model for putting mass
    where the base model does not.  A pruned model that has collapsed onto a
    single confident token scores low forward and high reverse; one that has
    flattened into hedging scores the other way round, so the gap between the
    two is a readout of *how* the model broke, not just how much.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed reverse KL divergence (non-negative).
    """
    return _aligned_sum(tokens_0, tokens_k, _k_rkld)


def compute_jeffreys(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sum of the Jeffreys divergence, KL(P‖Q) + KL(Q‖P), over aligned positions.

    The symmetrised KL.  Unlike JSD it is unbounded, so it stays sensitive in
    the regime where JSD has saturated at its ceiling — which is exactly the
    heavily-pruned regime this study cares about.

    Equal to ``compute_kld(a, b) + compute_rkld(a, b)`` up to the order in
    which the positions are summed (this function adds the two directions
    together within each position, so the two routes differ in the last few
    ulps on a long sequence).

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
    return _aligned_sum(tokens_0, tokens_k, _k_jeffreys)


def compute_tv(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sum of total variation distance, ½‖p − q‖₁, over aligned positions.

    The canonical statistical distance: the largest possible disagreement
    between the two models about the probability of any single event, and a
    true metric bounded in [0, 1] per position.  Pinsker's inequality makes it
    a lower bound on √(KL/2), so where TV saturates and KL does not, the extra
    KL is coming from tail ratios rather than from mass actually moving.

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
    return _aligned_sum(tokens_0, tokens_k, _k_tv)


def compute_hellinger(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sum of Hellinger distance, ‖√p − √q‖₂ / √2, over aligned positions.

    A true metric bounded in [0, 1] per position, and the one measure here that
    is a genuine Euclidean distance in disguise (between the square-root
    embeddings of the two distributions on the unit sphere).  That matters
    downstream: a Hellinger distance matrix is much closer to being embeddable
    without loss than a KL one, so classical MDS discards less of it.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed Hellinger distance (non-negative).
    """
    return _aligned_sum(tokens_0, tokens_k, _k_hellinger)


def compute_bhattacharyya(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sum of the Bhattacharyya distance, −ln Σ√(p·q), over aligned positions.

    The unbounded log-form of the same overlap coefficient that Hellinger uses
    in bounded form.  Per position the two are related by a fixed monotone map,
    but that map is nonlinear, so once summed over a sequence the two measures
    order model pairs differently: Bhattacharyya lets a handful of
    near-disjoint positions dominate an answer's score, while Hellinger caps
    each position's contribution at 1.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed Bhattacharyya distance (non-negative).
    """
    return _aligned_sum(tokens_0, tokens_k, _k_bhattacharyya)


def compute_renyi05(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sum of the Rényi divergence of order ½ over aligned positions.

    Included for completeness of the Rényi family.  Be aware that
    ``D_½(P‖Q) = −2 ln Σ√(p·q) = 2 · D_B(P, Q)`` exactly, and that this
    identity is *linear*, so unlike the Bhattacharyya/Hellinger pair it does
    survive summation: this function returns exactly twice
    :func:`compute_bhattacharyya` on every input.  It therefore carries no
    information the Bhattacharyya column does not, and in a metric-agreement
    analysis it should show a rank correlation of exactly 1 with it — which
    makes it a useful built-in check that such an analysis is wired up right.

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
    return _aligned_sum(tokens_0, tokens_k, _k_renyi05)


def compute_chisq(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sum of the Pearson χ² divergence, Σ (p − q)²/q, over aligned positions.

    The most tail-sensitive f-divergence here, and for that reason the one to
    treat with suspicion on this data.  Its denominator is the *pruned* model's
    probability, and a token that the pruned model dropped out of its top-k has
    ``q`` set by the −50 logprob fill, so the ``eps = 1e-12`` denominator guard
    is what sets the scale: a position where the two support sets are disjoint
    contributes on the order of 1e12, versus order 1 for every other measure in
    this module.  A χ² distance matrix is consequently close to a count of
    disjoint-support positions multiplied by a large constant.

    That count does correlate with damage — notebook 08 measures R² ≈ 0.98
    against pass@1 drop, in line with the rest — so the *regression* is not the
    problem.  What is not meaningful is the magnitude: a χ² of 1e13 should be
    read as "roughly twenty positions disagreed completely", never as an amount
    of divergence, and differences between two such values say more about the
    epsilon than about the models.  It is also the second-worst measure to feed
    through classical MDS (notebook 08 measures 31% of the eigenvalue mass
    discarded, against exactly 0% for JSD, Hellinger and total variation).  Use
    :func:`compute_renyi2`, the same quantity on a log scale, when a
    heavy-tailed measure is genuinely wanted.

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
    return _aligned_sum(tokens_0, tokens_k, _k_chisq)


def compute_renyi2(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sum of the Rényi divergence of order 2, ln Σ p²/q, over aligned positions.

    Equal to ``ln(1 + χ²)`` per position.  That map is monotone but nonlinear,
    so — unlike the order-½ case — it does not survive summation, and this is a
    genuinely different measure from :func:`compute_chisq` rather than a
    rescaling of it.  Practically it is the usable heavy-tail divergence for
    this data: it keeps χ²'s sensitivity to the pruned model's tails while
    compressing a disjoint-support position from order 1e12 down to order 28,
    which is comparable with the other unbounded measures.

    It is also the only measure in this module that behaves differently from
    the rest on real data: notebook 08 finds all sixteen agree at ρ ≥ 0.84
    except this one, which is both the least correlated with the others and the
    only one whose predictive R² varies wildly across benchmarks.

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
    return _aligned_sum(tokens_0, tokens_k, _k_renyi2)


def compute_triangular(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sum of triangular discrimination, Σ (p − q)²/(p + q), over aligned positions.

    Also called Le Cam's divergence: χ²'s bounded companion, obtained by
    symmetrising the denominator.  Bounded by 2 per position, so it cannot be
    dominated by a single disjoint-support token the way χ² is, while keeping
    the quadratic numerator that makes both of them more sensitive than total
    variation to a large disagreement on one token.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed triangular discrimination (non-negative).
    """
    return _aligned_sum(tokens_0, tokens_k, _k_triangular)


def compute_l2(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sum of Euclidean distance ‖p − q‖₂ over aligned positions.

    Not an f-divergence and not information-theoretic at all: it treats the two
    probability vectors as plain points in Euclidean space.  It is included as
    the null hypothesis of this whole comparison — if the information-theoretic
    measures do not beat straight Euclidean distance at predicting degradation,
    their extra machinery is not earning its place.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed Euclidean distance (non-negative).
    """
    return _aligned_sum(tokens_0, tokens_k, _k_l2)


def compute_cosine(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sum of cosine distance 1 − ⟨p, q⟩/(‖p‖·‖q‖) over aligned positions.

    Purely angular: it asks whether the two models rank the alternatives in
    the same proportions, and is blind to how peaked either distribution is.
    Where cosine stays small but the divergences grow, the pruned model has
    kept the base model's preference ordering and only changed its confidence.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Returns
    -------
    float
        Summed cosine distance (non-negative).
    """
    return _aligned_sum(tokens_0, tokens_k, _k_cosine)


# ---------------------------------------------------------------------------
# Registry and single-pass batch evaluation
# ---------------------------------------------------------------------------


class MetricInfo(NamedTuple):
    """Descriptive metadata for one distance, for tables and figure labels.

    Attributes
    ----------
    label:
        Human-readable name.
    family:
        One of ``"f-divergence"``, ``"geometry"``, ``"transport"``,
        ``"point-cloud"`` — the representation the measure works in.
    symmetric:
        Whether ``d(a, b) == d(b, a)``.
    bounded:
        Per-position upper bound, or ``None`` if the measure is unbounded.
    formula:
        The per-position expression, before summing over positions.
    """

    label: str
    family: str
    symmetric: bool
    bounded: float | None
    formula: str


#: Every distance in this module, in reporting order.
METRIC_FUNCS: dict[str, Callable[[list, list], float]] = {
    "kld": compute_kld,
    "rkld": compute_rkld,
    "jeffreys": compute_jeffreys,
    "jsd": compute_jsd,
    "tv": compute_tv,
    "hellinger": compute_hellinger,
    "bhattacharyya": compute_bhattacharyya,
    "renyi05": compute_renyi05,
    "chisq": compute_chisq,
    "renyi2": compute_renyi2,
    "triangular": compute_triangular,
    "l2": compute_l2,
    "cosine": compute_cosine,
    "emd": compute_emd,
    "wasserstein2": compute_wasserstein2,
    "chamfer": compute_chamfer,
}

#: Metadata for each entry of :data:`METRIC_FUNCS`, same keys and order.
METRIC_INFO: dict[str, MetricInfo] = {
    "kld": MetricInfo("Forward KL", "f-divergence", False, None, "Σ p·ln(p/q)"),
    "rkld": MetricInfo("Reverse KL", "f-divergence", False, None, "Σ q·ln(q/p)"),
    "jeffreys": MetricInfo("Jeffreys", "f-divergence", True, None, "KL(p‖q) + KL(q‖p)"),
    "jsd": MetricInfo(
        "Jensen-Shannon", "f-divergence", True, 1.0, "√JSD(p, q), log₂ units"
    ),
    "tv": MetricInfo("Total variation", "f-divergence", True, 1.0, "½·Σ|p − q|"),
    "hellinger": MetricInfo("Hellinger", "f-divergence", True, 1.0, "‖√p − √q‖₂ / √2"),
    "bhattacharyya": MetricInfo(
        "Bhattacharyya", "f-divergence", True, None, "−ln Σ√(p·q)"
    ),
    "renyi05": MetricInfo("Rényi α=½", "f-divergence", True, None, "−2·ln Σ√(p·q)"),
    "chisq": MetricInfo("Pearson χ²", "f-divergence", False, None, "Σ (p − q)²/q"),
    "renyi2": MetricInfo("Rényi α=2", "f-divergence", False, None, "ln Σ p²/q"),
    "triangular": MetricInfo(
        "Triangular", "f-divergence", True, 2.0, "Σ (p − q)²/(p + q)"
    ),
    "l2": MetricInfo("Euclidean", "geometry", True, math.sqrt(2.0), "‖p − q‖₂"),
    "cosine": MetricInfo("Cosine", "geometry", True, 1.0, "1 − ⟨p, q⟩/(‖p‖·‖q‖)"),
    "emd": MetricInfo(
        "Wasserstein-1", "transport", True, None, "W₁ over logprob atoms"
    ),
    "wasserstein2": MetricInfo(
        "Wasserstein-2", "transport", True, None, "W₂ over logprob atoms"
    ),
    "chamfer": MetricInfo(
        "Chamfer", "point-cloud", True, None, "mean-NN both directions"
    ),
}

METRIC_NAMES: tuple[str, ...] = tuple(METRIC_FUNCS)


def compute_all(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
    *,
    metrics: Iterable[str] | None = None,
) -> dict[str, float]:
    """Compute every distance in :data:`METRIC_FUNCS` in a single pass.

    The union-support alignment is the dominant per-position cost and is shared
    by thirteen of the sixteen measures, so computing all of them costs barely
    more than computing one — whereas calling the individual functions in a
    loop repeats that work once per measure.

    Results are identical to calling the individual functions: the same kernels
    are applied to the same vectors in the same order, so the floats agree
    bit-for-bit, which is what lets a batch build reuse distance matrices that
    were originally produced one metric at a time.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.
    metrics:
        Restrict the output to these names.  Defaults to all of them.  Worth
        using to drop ``"chamfer"``, which alone costs more than the other
        fifteen combined because it compares every position against every other
        rather than pairing them up by index.

    Returns
    -------
    dict[str, float]
        One entry per requested metric name, in :data:`METRIC_FUNCS` order.

    Raises
    ------
    KeyError
        If ``metrics`` names a measure this module does not define.
    """
    wanted = list(METRIC_FUNCS) if metrics is None else list(metrics)
    unknown = [name for name in wanted if name not in METRIC_FUNCS]
    if unknown:
        raise KeyError(f"unknown metric(s): {sorted(unknown)}")
    wanted_set = set(wanted)

    totals: dict[str, float] = {name: 0.0 for name in wanted}
    union_wanted = [name for name in _UNION_KERNELS if name in wanted_set]
    want_emd = "emd" in wanted_set
    want_w2 = "wasserstein2" in wanted_set

    if union_wanted or want_emd or want_w2:
        for step_0, step_k in zip(tokens_0, tokens_k):
            alts_0 = step_0.get("top_alternatives", [])
            alts_k = step_k.get("top_alternatives", [])
            if not alts_0 or not alts_k:
                continue
            if union_wanted:
                p0, pk = _union_probs(alts_0, alts_k)
                for name in union_wanted:
                    totals[name] += _UNION_KERNELS[name](p0, pk)
            if want_emd or want_w2:
                lp_0, w0 = _atom_weights(alts_0)
                lp_k, wk = _atom_weights(alts_k)
                if want_emd:
                    totals["emd"] += float(
                        _scipy_wasserstein(lp_0, lp_k, u_weights=w0, v_weights=wk)
                    )
                if want_w2:
                    totals["wasserstein2"] += _wasserstein_p(lp_0, w0, lp_k, wk, 2.0)

    if "chamfer" in wanted_set:
        totals["chamfer"] = compute_chamfer(tokens_0, tokens_k)

    return {name: totals[name] for name in wanted}
