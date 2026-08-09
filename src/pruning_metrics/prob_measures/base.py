"""Shared types and per-position machinery for the probability measures.

Design decision — one module per measure:
    This package replaces a single 1000-line module that held all sixteen
    distributional distances.  Each measure now lives in its own module
    (``kld.py``, ``jsd.py``, ...) and the registry
    (:mod:`pruning_metrics.prob_measures.registry`) composes them.  Every
    measure module exposes:

    ``NAME``:
        The registry key (also the CSV column / filename token).
    ``INFO``:
        A :class:`MetricInfo` describing the measure for tables and labels.
    ``compute(tokens_0, tokens_k)``:
        The public metric function (also exported under its descriptive
        name, e.g. ``compute_kld``).
    ``kernel(p, q)`` (union-support measures only):
        The per-position kernel on two aligned probability vectors.
        Sharing the kernel between the single-metric function and the
        batch ``compute_all`` is what guarantees the two routes agree
        bit-for-bit — the same float operations run in the same order no
        matter which route is taken.
    ``transport_distance(u_values, u_weights, v_values, v_weights)``
    (transport measures only):
        The per-position distance on two weighted 1-D atom sets produced
        by :func:`atom_weights`.  Shared with ``compute_all`` for the same
        bit-for-bit reason as ``kernel``.

    A measure that defines neither hook (currently only Chamfer, which
    matches positions by nearest neighbour rather than by index) is
    evaluated by ``compute_all`` through its own ``compute`` instead of
    being folded into the shared per-position pass.

Representation notes (shared by all measures):
    * Only the ``top_alternatives`` field of a token step is used (not
      ``target_logprob``).  The top-5 alternatives represent the model's
      predicted distribution; they are renormalised locally so they sum to
      1 over the top-k support.
    * Tokens absent from one model's top-5 receive logprob
      :data:`MISSING_LOGPROB` (-50, ~2e-22 probability) before
      renormalisation, so divergences are large but finite when the two
      support sets are disjoint.
    * All ratio-form divergences guard their denominator with
      :data:`EPS` (1e-12), which is far larger than the 2e-22 fill above.
      For the bounded measures this is numerically irrelevant; for the
      chi-squared divergence it is what keeps a disjoint-support position
      finite (see ``chisq.py``).
"""

from __future__ import annotations

import math
from typing import Callable, Iterator, NamedTuple, TypedDict

import numpy as np

__all__ = [
    "EPS",
    "LN2",
    "MISSING_LOGPROB",
    "MetricInfo",
    "TokenStepDict",
    "TopAlternativeDict",
    "aligned_alternatives",
    "aligned_sum",
    "atom_weights",
    "renorm",
    "transport_sum",
    "union_probs",
]

#: Denominator guard for ratio-form divergences.
EPS: float = 1e-12

#: ln(2), for converting nats to bits in the Jensen-Shannon divergence.
LN2: float = math.log(2.0)

#: Stand-in logprob for tokens absent from a top-k list (~2e-22 probability).
MISSING_LOGPROB: float = -50.0


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


def renorm(logprobs: np.ndarray) -> np.ndarray:
    """Local-softmax renormalisation: logprobs to a probability vector.

    Subtracts the max for numerical stability, exponentiates, then
    normalises so the output sums to 1 over the provided entries (i.e.
    renormalises within the top-k slice rather than the full vocabulary).

    Parameters
    ----------
    logprobs:
        1-D array of log-probabilities.

    Returns
    -------
    np.ndarray
        Probability vector of the same length, summing to 1.
    """
    shifted = logprobs - logprobs.max()
    p = np.exp(shifted)
    return p / p.sum()


def union_probs(
    alts_0: list[TopAlternativeDict],
    alts_k: list[TopAlternativeDict],
) -> tuple[np.ndarray, np.ndarray]:
    """Builds aligned probability vectors over the union of two supports.

    Tokens appearing in one list but not the other receive
    :data:`MISSING_LOGPROB` before renormalisation, giving them near-zero
    probability mass.

    Parameters
    ----------
    alts_0:
        Top-k alternatives from the base model at one position.
    alts_k:
        Top-k alternatives from the pruned model at the same
        position.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A ``(p0, pk)`` pair of renormalised probability vectors of equal
        length (union vocabulary size).  Entries correspond to the same
        token in both vectors.
    """
    map_0 = {a["token_id"]: a["logprob"] for a in alts_0}
    map_k = {a["token_id"]: a["logprob"] for a in alts_k}
    all_ids = list(set(map_0) | set(map_k))
    lp_0 = np.array([map_0.get(tid, MISSING_LOGPROB) for tid in all_ids])
    lp_k = np.array([map_k.get(tid, MISSING_LOGPROB) for tid in all_ids])
    return renorm(lp_0), renorm(lp_k)


def atom_weights(alts: list[TopAlternativeDict]) -> tuple[np.ndarray, np.ndarray]:
    """Converts a top-k list to (atom positions, weights) on the real line.

    The *positions* are the raw logprob values and the *weights* are those
    same logprobs renormalised into a probability vector.  This is the
    representation the optimal-transport measures work in: unlike the
    union-support alignment used by the f-divergences, each model keeps
    its own support, and the ground distance between two predictions is
    how far apart their logprobs are.

    Parameters
    ----------
    alts:
        A top-k alternative list.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A ``(positions, weights)`` pair of equal-length 1-D arrays.
    """
    logprobs = np.array([a["logprob"] for a in alts], dtype=np.float64)
    return logprobs, renorm(logprobs)


def aligned_alternatives(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> Iterator[tuple[list[TopAlternativeDict], list[TopAlternativeDict]]]:
    """Yields the aligned ``(alts_0, alts_k)`` pairs of two step lists.

    The shared iteration skeleton of every position-aligned measure:
    positions are paired up by index, and a position is skipped when either
    model recorded no top-k alternatives there.  If the two lists have
    different lengths only the first ``min(len(tokens_0), len(tokens_k))``
    positions are used.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.

    Yields
    ------
    tuple[list[TopAlternativeDict], list[TopAlternativeDict]]
        ``(alts_0, alts_k)`` pairs of non-empty top-k alternative lists.
    """
    for step_0, step_k in zip(tokens_0, tokens_k):
        alts_0 = step_0.get("top_alternatives", [])
        alts_k = step_k.get("top_alternatives", [])
        if not alts_0 or not alts_k:
            continue
        yield alts_0, alts_k


def aligned_sum(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
    kernel: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Sums a per-position kernel over aligned positions with non-empty top-k.

    If the two lists have different lengths only the first
    ``min(len(tokens_0), len(tokens_k))`` positions are used.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.
    kernel:
        Per-position function of two aligned probability vectors.

    Returns
    -------
    float
        The kernel summed over all usable positions (non-negative for
        every kernel in this package).
    """
    total = 0.0
    for alts_0, alts_k in aligned_alternatives(tokens_0, tokens_k):
        p0, pk = union_probs(alts_0, alts_k)
        total += kernel(p0, pk)
    return total


def transport_sum(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
    distance: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
) -> float:
    """Sums a per-position transport distance over aligned positions.

    The transport-family analogue of :func:`aligned_sum`: the same
    iteration skeleton, but each model keeps its own support (see
    :func:`atom_weights`) instead of being aligned onto the union.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.
    distance:
        Per-position function of two weighted 1-D atom sets,
        called as ``distance(u_values, u_weights, v_values,
        v_weights)``.

    Returns
    -------
    float
        The distance summed over all usable positions (non-negative).
    """
    total = 0.0
    for alts_0, alts_k in aligned_alternatives(tokens_0, tokens_k):
        lp_0, w0 = atom_weights(alts_0)
        lp_k, wk = atom_weights(alts_k)
        total += distance(lp_0, w0, lp_k, wk)
    return total
