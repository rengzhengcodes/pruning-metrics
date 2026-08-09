"""Shared synthetic token-step builders for distance-measure tests.

Three suites (``tests/metrics/test_distributions*.py`` and
``tests/prob_measures/test_measure_modules.py``) construct the same
minimal ``TokenStepDict`` inputs; this module is their single home.
Importable as ``helpers.token_steps`` because pytest puts ``tests/`` on
``sys.path`` (the first ancestor without an ``__init__.py``).
"""

from __future__ import annotations

import math

from pruning_metrics.prob_measures import TokenStepDict, TopAlternativeDict


def make_alt(token_id: int, logprob: float) -> TopAlternativeDict:
    """Build one top-alternative entry with a synthetic token text."""
    return TopAlternativeDict(
        token_id=token_id, token_text=f"tok{token_id}", logprob=logprob
    )


def make_step(
    logprobs: list[float], token_ids: list[int] | None = None
) -> TokenStepDict:
    """Build a minimal TokenStepDict with the given top-alternatives."""
    if token_ids is None:
        token_ids = list(range(len(logprobs)))
    return TokenStepDict(
        position=0,
        target_token_id=token_ids[0] if token_ids else 0,
        target_token_text="",
        target_logprob=logprobs[0] if logprobs else 0.0,
        target_prob=math.exp(logprobs[0]) if logprobs else 1.0,
        rank=1,
        top_alternatives=[make_alt(tid, lp) for tid, lp in zip(token_ids, logprobs)],
    )
