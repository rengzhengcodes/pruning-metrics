"""Unit tests for pruning_metrics.metrics.distributions."""

from __future__ import annotations

import math

import pytest

from pruning_metrics.metrics.distributions import (
    TokenStepDict,
    TopAlternativeDict,
    compute_chamfer,
    compute_emd,
    compute_jsd,
    compute_kld,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alt(token_id: int, logprob: float) -> TopAlternativeDict:
    return TopAlternativeDict(
        token_id=token_id,
        token_text=f"tok{token_id}",
        logprob=logprob,
    )


def _make_step(
    logprobs: list[float],
    token_ids: list[int] | None = None,
) -> TokenStepDict:
    """Build a minimal TokenStepDict with the given top-alternatives."""
    if token_ids is None:
        token_ids = list(range(len(logprobs)))
    alts = [_make_alt(tid, lp) for tid, lp in zip(token_ids, logprobs)]
    return TokenStepDict(
        position=0,
        target_token_id=token_ids[0] if token_ids else 0,
        target_token_text="",
        target_logprob=logprobs[0] if logprobs else 0.0,
        target_prob=math.exp(logprobs[0]) if logprobs else 1.0,
        rank=1,
        top_alternatives=alts,
    )


def _uniform_steps(
    n_positions: int = 3,
    n_tokens: int = 5,
    logprob: float = -1.0,
) -> list[TokenStepDict]:
    """All positions have the same n_tokens alternatives at equal logprob."""
    lps = [logprob] * n_tokens
    return [_make_step(lps) for _ in range(n_positions)]


# ---------------------------------------------------------------------------
# KLD tests
# ---------------------------------------------------------------------------


def test_kld_identical_distributions_is_zero() -> None:
    steps = _uniform_steps()
    assert compute_kld(steps, steps) == pytest.approx(0.0, abs=1e-9)


def test_kld_is_asymmetric() -> None:
    """KL(P||Q) ≠ KL(Q||P) when distributions have different concentration shapes.

    a is peaked (60% on token 0, 40% on token 1, ~0% on token 2).
    b is spread (37% / 33% / 30% across the same three tokens).
    KL(a||b) ≈ 0.35 but KL(b||a) ≈ 1.06 because b places significant mass
    on token 2, which a considers near-impossible.
    """
    # Peaked distribution
    a = [_make_step([-0.1, -0.5, -5.0], token_ids=[0, 1, 2])]
    # Roughly uniform distribution
    b = [_make_step([-0.5, -0.6, -0.7], token_ids=[0, 1, 2])]
    kl_ab = compute_kld(a, b)
    kl_ba = compute_kld(b, a)
    assert kl_ba > kl_ab * 2  # b→a much larger because b has mass where a has near-zero


def test_kld_non_negative() -> None:
    a = [_make_step([-0.5, -1.0, -2.0, -3.0, -4.0])]
    b = [_make_step([-4.0, -3.0, -0.5, -2.0, -1.0])]
    assert compute_kld(a, b) >= 0.0
    assert compute_kld(b, a) >= 0.0


def test_kld_disjoint_supports_is_large() -> None:
    """When the two top-5 sets are completely disjoint, KLD should be high."""
    a = [_make_step([-1.0] * 5, token_ids=[0, 1, 2, 3, 4])]
    b = [_make_step([-1.0] * 5, token_ids=[5, 6, 7, 8, 9])]
    kld = compute_kld(a, b)
    assert kld > 5.0  # all mass moved to completely different tokens


def test_kld_length_mismatch_uses_shorter() -> None:
    steps_5 = _uniform_steps(n_positions=5)
    steps_3 = _uniform_steps(n_positions=3)
    # Should process 3 positions without error
    result = compute_kld(steps_5, steps_3)
    assert math.isfinite(result)


# ---------------------------------------------------------------------------
# JSD tests
# ---------------------------------------------------------------------------


def test_jsd_identical_distributions_is_zero() -> None:
    steps = _uniform_steps()
    assert compute_jsd(steps, steps) == pytest.approx(0.0, abs=1e-9)


def test_jsd_is_symmetric() -> None:
    a = [_make_step([-1.0, -2.0, -3.0])]
    b = [_make_step([-3.0, -2.0, -1.0])]
    assert compute_jsd(a, b) == pytest.approx(compute_jsd(b, a), rel=1e-9)


def test_jsd_per_position_bounded_by_one() -> None:
    """√JSD ≤ 1 per position in log₂ units; total is bounded by n_positions."""
    n = 4
    a = [_make_step([-0.1, -5.0, -5.0])] * n
    b = [_make_step([-5.0, -5.0, -0.1])] * n
    total_jsd = compute_jsd(a, b)
    assert total_jsd <= n + 1e-9


def test_jsd_non_negative() -> None:
    a = [_make_step([-1.0, -2.0, -3.0, -4.0, -5.0])]
    b = [_make_step([-5.0, -4.0, -3.0, -2.0, -1.0])]
    assert compute_jsd(a, b) >= 0.0


# ---------------------------------------------------------------------------
# EMD tests
# ---------------------------------------------------------------------------


def test_emd_identical_distributions_is_zero() -> None:
    steps = _uniform_steps()
    assert compute_emd(steps, steps) == pytest.approx(0.0, abs=1e-9)


def test_emd_shifts_with_logprob_distance() -> None:
    """A single-atom distribution shifted by Δ in logprob → EMD ≈ Δ."""
    delta = 3.0
    a = [_make_step([-1.0], token_ids=[0])]
    b = [_make_step([-1.0 - delta], token_ids=[0])]
    assert compute_emd(a, b) == pytest.approx(delta, rel=1e-6)


def test_emd_non_negative() -> None:
    a = [_make_step([-0.5, -1.5, -3.0])]
    b = [_make_step([-3.0, -1.5, -0.5])]
    assert compute_emd(a, b) >= 0.0


# ---------------------------------------------------------------------------
# Chamfer tests
# ---------------------------------------------------------------------------


def test_chamfer_identical_predictions_is_zero() -> None:
    """Same top-k token IDs and probabilities → Chamfer = 0."""
    steps = _uniform_steps()
    assert compute_chamfer(steps, steps) == pytest.approx(0.0, abs=1e-9)


def test_chamfer_disjoint_tokens() -> None:
    """Disjoint support sets → all pairwise distances equal; Chamfer > 0.

    Model A: tokens {0,1,2,3,4} each with prob 0.2 (uniform over 5 tokens).
    Model B: tokens {5,6,7,8,9} each with prob 0.2 (uniform over 5 tokens).

    In R^10, point a = [0.2,0.2,0.2,0.2,0.2, 0,0,0,0,0]
                point b = [0,0,0,0,0, 0.2,0.2,0.2,0.2,0.2]

    ‖a - b‖² = 5*(0.2²) + 5*(0.2²) = 0.4  →  ‖a - b‖ = √0.4

    Both nearest-neighbour distances equal √0.4, so Chamfer = 2*√0.4.
    """
    lp = [-1.6094379] * 5  # log(0.2) ≈ -1.609
    a = [_make_step(lp, token_ids=[0, 1, 2, 3, 4])]
    b = [_make_step(lp, token_ids=[5, 6, 7, 8, 9])]
    expected = 2.0 * math.sqrt(0.4)
    assert compute_chamfer(a, b) == pytest.approx(expected, rel=1e-4)


def test_chamfer_non_negative() -> None:
    a = _uniform_steps(n_positions=4)
    b = [_make_step([-1.0, -2.0, -3.0, -4.0, -5.0])] * 4
    assert compute_chamfer(a, b) >= 0.0


def test_chamfer_is_symmetric() -> None:
    a = [_make_step([-1.0, -2.0, -3.0], token_ids=[0, 1, 2])]
    b = [_make_step([-3.0, -1.0, -2.0], token_ids=[3, 0, 1])]
    assert compute_chamfer(a, b) == pytest.approx(compute_chamfer(b, a), rel=1e-9)


# ---------------------------------------------------------------------------
# Edge-case tests shared across all metrics
# ---------------------------------------------------------------------------


def test_empty_top_alternatives_returns_zero() -> None:
    empty_step = _make_step([], token_ids=[])
    empty_step["top_alternatives"] = []
    steps = [empty_step]
    assert compute_kld(steps, steps) == 0.0
    assert compute_jsd(steps, steps) == 0.0
    assert compute_emd(steps, steps) == 0.0
    assert compute_chamfer(steps, steps) == 0.0


def test_empty_sequence_returns_zero() -> None:
    assert compute_kld([], []) == 0.0
    assert compute_jsd([], []) == 0.0
    assert compute_emd([], []) == 0.0
    assert compute_chamfer([], []) == 0.0
