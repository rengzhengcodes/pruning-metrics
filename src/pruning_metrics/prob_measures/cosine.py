"""Cosine distance, 1 - <p, q> / (||p|| * ||q||)."""

from __future__ import annotations

import numpy as np

from pruning_metrics.prob_measures.base import MetricInfo, TokenStepDict, aligned_sum

NAME = "cosine"

INFO = MetricInfo("Cosine", "geometry", True, 1.0, "1 − ⟨p, q⟩/(‖p‖·‖q‖)")


def kernel(p: np.ndarray, q: np.ndarray) -> float:
    """Per-position cosine distance on two aligned probability vectors."""
    denom = float(np.linalg.norm(p)) * float(np.linalg.norm(q))
    if denom <= 0.0:
        return 0.0
    return float(np.clip(1.0 - float(np.dot(p, q)) / denom, 0.0, 2.0))


def compute_cosine(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
) -> float:
    """Sums the cosine distance, 1 - <p, q> / (||p|| * ||q||), over aligned positions.

    Cosine distance is purely angular.  It asks whether the two models rank
    the alternatives in the same proportions.  It is blind to how peaked
    either distribution is.  Where cosine stays small but the divergences
    grow, the pruned model has kept the base model's preference ordering.
    It has only changed its confidence.

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
    return aligned_sum(tokens_0, tokens_k, kernel)


#: Uniform registry hook (see the module contract in ``base.py``).
compute = compute_cosine
