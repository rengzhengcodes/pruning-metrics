"""Unit tests for the twelve distances added alongside KLD/JSD/EMD/Chamfer.

Split from ``test_distributions.py`` so the original four keep their own
regression suite untouched; this file covers the extended registry
(:data:`METRIC_FUNCS`), the shared per-position kernels, the generic
Wasserstein-p implementation, and :func:`compute_all`.

The anchors that matter most are the ones tying new code to something
independent: ``_wasserstein_p`` at ``p=1`` against ``scipy``, and
``compute_all`` against the individual functions bit-for-bit — the latter is
what lets a batch build reuse distance matrices originally produced one metric
at a time.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from pruning_metrics.metrics.distributions import (
    METRIC_FUNCS,
    METRIC_INFO,
    METRIC_NAMES,
    TokenStepDict,
    TopAlternativeDict,
    _wasserstein_p,
    compute_all,
    compute_bhattacharyya,
    compute_chisq,
    compute_cosine,
    compute_emd,
    compute_hellinger,
    compute_jeffreys,
    compute_kld,
    compute_l2,
    compute_renyi05,
    compute_renyi2,
    compute_rkld,
    compute_triangular,
    compute_tv,
    compute_wasserstein2,
)

# ---------------------------------------------------------------------------
# Helpers (mirrors test_distributions.py so the two files read alike)
# ---------------------------------------------------------------------------


def _make_alt(token_id: int, logprob: float) -> TopAlternativeDict:
    return TopAlternativeDict(
        token_id=token_id, token_text=f"tok{token_id}", logprob=logprob
    )


def _make_step(
    logprobs: list[float], token_ids: list[int] | None = None
) -> TokenStepDict:
    if token_ids is None:
        token_ids = list(range(len(logprobs)))
    return TokenStepDict(
        position=0,
        target_token_id=token_ids[0] if token_ids else 0,
        target_token_text="",
        target_logprob=logprobs[0] if logprobs else 0.0,
        target_prob=math.exp(logprobs[0]) if logprobs else 1.0,
        rank=1,
        top_alternatives=[_make_alt(tid, lp) for tid, lp in zip(token_ids, logprobs)],
    )


def _peaked(n_positions: int = 4) -> list[TokenStepDict]:
    """Confident: almost all mass on token 0."""
    return [_make_step([0.0, -6.0, -7.0, -8.0, -9.0]) for _ in range(n_positions)]


def _flat(n_positions: int = 4) -> list[TokenStepDict]:
    """Hedging: uniform over the same five tokens."""
    return [_make_step([-1.0] * 5) for _ in range(n_positions)]


def _shifted(n_positions: int = 4) -> list[TokenStepDict]:
    """Same shape as ``_peaked`` but on a disjoint set of token ids."""
    return [
        _make_step([0.0, -6.0, -7.0, -8.0, -9.0], token_ids=[10, 11, 12, 13, 14])
        for _ in range(n_positions)
    ]


ALL_NEW = [
    compute_rkld,
    compute_jeffreys,
    compute_tv,
    compute_hellinger,
    compute_bhattacharyya,
    compute_renyi05,
    compute_chisq,
    compute_renyi2,
    compute_triangular,
    compute_l2,
    compute_cosine,
    compute_wasserstein2,
]


# ---------------------------------------------------------------------------
# Metric axioms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", ALL_NEW, ids=lambda f: f.__name__)
def test_identical_inputs_give_zero(fn) -> None:
    """d(x, x) == 0 for every new measure, including the directional ones."""
    assert fn(_peaked(), _peaked()) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("fn", ALL_NEW, ids=lambda f: f.__name__)
def test_non_negative_on_differing_inputs(fn) -> None:
    assert fn(_peaked(), _flat()) >= 0.0


@pytest.mark.parametrize("name", [n for n in METRIC_NAMES if METRIC_INFO[n].symmetric])
def test_declared_symmetry_holds(name: str) -> None:
    """Every metric METRIC_INFO calls symmetric actually is."""
    fn = METRIC_FUNCS[name]
    assert fn(_peaked(), _flat()) == pytest.approx(fn(_flat(), _peaked()), rel=1e-12)


def test_declared_asymmetry_holds() -> None:
    """And the ones it calls directional actually differ in the two directions."""
    for name in (n for n in METRIC_NAMES if not METRIC_INFO[n].symmetric):
        fn = METRIC_FUNCS[name]
        forward, reverse = fn(_peaked(), _flat()), fn(_flat(), _peaked())
        assert forward != pytest.approx(reverse, rel=1e-6), name


@pytest.mark.parametrize(
    "name", [n for n in METRIC_NAMES if METRIC_INFO[n].bounded is not None]
)
def test_per_position_bound_is_respected(name: str) -> None:
    """A declared bound must hold per position — checked on disjoint support,
    which is the worst case reachable with this data's -50 logprob fill."""
    n_positions = 4
    total = METRIC_FUNCS[name](_peaked(n_positions), _shifted(n_positions))
    assert total <= METRIC_INFO[name].bounded * n_positions + 1e-9


def test_summation_scales_with_length() -> None:
    """All aligned measures are sums, so doubling the sequence doubles them."""
    for name in METRIC_NAMES:
        if name == "chamfer":  # means over positions, deliberately length-normalised
            continue
        fn = METRIC_FUNCS[name]
        assert fn(_peaked(2), _flat(2)) * 2 == pytest.approx(
            fn(_peaked(4), _flat(4)), rel=1e-9
        ), name


# ---------------------------------------------------------------------------
# Exact identities between measures
# ---------------------------------------------------------------------------


def test_jeffreys_is_the_sum_of_both_kl_directions() -> None:
    a, b = _peaked(), _flat()
    assert compute_jeffreys(a, b) == pytest.approx(
        compute_kld(a, b) + compute_rkld(a, b), rel=1e-12
    )


def test_reverse_kl_is_forward_kl_with_arguments_swapped() -> None:
    a, b = _peaked(), _flat()
    assert compute_rkld(a, b) == pytest.approx(compute_kld(b, a), rel=1e-12)


def test_renyi_half_is_exactly_twice_bhattacharyya() -> None:
    """A linear identity, so unlike the other monotone relations between these
    measures it survives summation exactly. Documented in compute_renyi05."""
    a, b = _peaked(), _flat()
    assert compute_renyi05(a, b) == pytest.approx(
        2.0 * compute_bhattacharyya(a, b), rel=1e-12
    )


def test_renyi2_is_log1p_of_chi_squared_per_position() -> None:
    """Holds per position; the test uses a single position so the summation
    that breaks the identity for longer sequences cannot interfere."""
    a, b = _peaked(1), _flat(1)
    assert compute_renyi2(a, b) == pytest.approx(
        math.log1p(compute_chisq(a, b)), rel=1e-9
    )


def test_renyi2_and_chisq_diverge_once_summed() -> None:
    """The same identity must *not* hold across several positions — that is the
    whole reason both are kept in the registry."""
    a, b = _peaked(4), _flat(4)
    assert compute_renyi2(a, b) != pytest.approx(
        math.log1p(compute_chisq(a, b)), rel=1e-3
    )


def test_pinsker_inequality() -> None:
    """TV <= sqrt(KL/2) per position, a textbook bound the implementations must
    satisfy if the normalisation of either is right."""
    a, b = _peaked(1), _flat(1)
    assert compute_tv(a, b) <= math.sqrt(compute_kld(a, b) / 2.0) + 1e-12


def test_hellinger_bounds_total_variation() -> None:
    """H^2 <= TV <= H*sqrt(2 - H^2), per position."""
    h = compute_hellinger(_peaked(1), _flat(1))
    tv = compute_tv(_peaked(1), _flat(1))
    assert h**2 <= tv + 1e-12
    assert tv <= h * math.sqrt(2.0 - h**2) + 1e-12


def test_hellinger_and_bhattacharyya_share_a_coefficient() -> None:
    """Per position, H^2 = 1 - exp(-D_B)."""
    h = compute_hellinger(_peaked(1), _shifted(1))
    d_b = compute_bhattacharyya(_peaked(1), _shifted(1))
    assert h**2 == pytest.approx(1.0 - math.exp(-d_b), rel=1e-9)


def test_chi_squared_is_scale_degenerate_on_disjoint_support() -> None:
    """The documented pathology: a disjoint-support position costs chi-squared
    ~1e11 while every other measure here stays order 1. Guards the docstring
    warning against someone silently 'fixing' the epsilon."""
    chisq = compute_chisq(_peaked(1), _shifted(1))
    assert chisq > 1e9
    assert compute_renyi2(_peaked(1), _shifted(1)) < 100.0
    assert compute_tv(_peaked(1), _shifted(1)) <= 1.0


# ---------------------------------------------------------------------------
# Wasserstein-p
# ---------------------------------------------------------------------------


def test_wasserstein_p1_matches_scipy() -> None:
    """The independent anchor for the generic optimal-transport code."""
    rng = np.random.default_rng(20260808)
    for _ in range(200):
        n_u, n_v = int(rng.integers(1, 8)), int(rng.integers(1, 8))
        u, v = rng.normal(scale=5.0, size=n_u), rng.normal(scale=5.0, size=n_v)
        w_u, w_v = rng.random(n_u) + 1e-3, rng.random(n_v) + 1e-3
        assert _wasserstein_p(
            u, w_u / w_u.sum(), v, w_v / w_v.sum(), 1.0
        ) == pytest.approx(
            wasserstein_distance(u, v, u_weights=w_u, v_weights=w_v), abs=1e-12
        )


def test_wasserstein_between_point_masses_is_the_gap() -> None:
    """W_p between two Diracs is |a - b| for every p."""
    one = np.array([1.0])
    for p in (1.0, 2.0, 3.0):
        assert _wasserstein_p(
            np.array([0.0]), one, np.array([3.0]), one, p
        ) == pytest.approx(3.0, rel=1e-12)


def test_wasserstein2_dominates_wasserstein1() -> None:
    """Jensen: W1 <= W2 always."""
    rng = np.random.default_rng(7)
    w = np.full(4, 0.25)
    for _ in range(100):
        u, v = rng.normal(size=4), rng.normal(size=4)
        assert (
            _wasserstein_p(u, w, v, w, 2.0) >= _wasserstein_p(u, w, v, w, 1.0) - 1e-12
        )


def test_wasserstein_is_translation_equivariant() -> None:
    """Shifting both supports by the same amount leaves the distance alone;
    shifting one by delta with matched supports gives exactly delta."""
    rng = np.random.default_rng(11)
    u, v = rng.normal(size=5), rng.normal(size=5)
    w = np.full(5, 0.2)
    base = _wasserstein_p(u, w, v, w, 2.0)
    assert _wasserstein_p(u + 4.0, w, v + 4.0, w, 2.0) == pytest.approx(base, rel=1e-12)
    assert _wasserstein_p(u, w, u + 2.5, w, 2.0) == pytest.approx(2.5, rel=1e-9)


def test_wasserstein2_beats_wasserstein1_on_concentrated_disagreement() -> None:
    """W2's reason for existing: with the same total mass moved, it charges
    more for one far move than for several near ones, and W1 charges the same."""
    half = np.array([0.5, 0.5])
    diffuse = _wasserstein_p(
        np.array([0.0, 0.0]), half, np.array([1.0, 1.0]), half, 2.0
    )
    concentrated = _wasserstein_p(
        np.array([0.0, 0.0]), half, np.array([0.0, 2.0]), half, 2.0
    )
    w1_diffuse = _wasserstein_p(
        np.array([0.0, 0.0]), half, np.array([1.0, 1.0]), half, 1.0
    )
    w1_concentrated = _wasserstein_p(
        np.array([0.0, 0.0]), half, np.array([0.0, 2.0]), half, 1.0
    )
    assert w1_diffuse == pytest.approx(w1_concentrated, rel=1e-12)
    assert concentrated > diffuse


def test_wasserstein2_on_token_steps_is_non_negative_and_zero_on_self() -> None:
    assert compute_wasserstein2(_peaked(), _peaked()) == pytest.approx(0.0, abs=1e-12)
    assert compute_wasserstein2(_peaked(), _flat()) > 0.0


def test_transport_metrics_ignore_token_identity() -> None:
    """EMD and W2 work on logprob values, not token ids, so relabelling the
    support cannot change them — whereas the aligned divergences explode."""
    assert compute_emd(_peaked(), _shifted()) == pytest.approx(0.0, abs=1e-12)
    assert compute_wasserstein2(_peaked(), _shifted()) == pytest.approx(0.0, abs=1e-12)
    assert compute_tv(_peaked(), _shifted()) > 0.5


# ---------------------------------------------------------------------------
# Geometry measures
# ---------------------------------------------------------------------------


def test_cosine_is_blind_to_confidence_but_l2_is_not() -> None:
    """Two distributions with the same ordering but different sharpness: cosine
    stays small, Euclidean distance does not."""
    sharp = [_make_step([0.0, -2.0, -4.0, -6.0, -8.0])]
    sharper = [_make_step([0.0, -4.0, -8.0, -12.0, -16.0])]
    assert compute_cosine(sharp, sharper) < compute_l2(sharp, sharper)


def test_l2_equals_numpy_norm_of_the_difference() -> None:
    """Sanity-check the geometry path against a hand computation on uniform
    versus one-hot-ish input over identical support."""
    a = [_make_step([0.0, 0.0])]  # renormalises to p = [0.5, 0.5]
    b = [_make_step([0.0, -50.0])]  # renormalises to q = [1.0, 2e-22]
    expected = float(np.linalg.norm(np.array([0.5, 0.5]) - np.array([1.0, 0.0])))
    assert compute_l2(a, b) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Registry and compute_all
# ---------------------------------------------------------------------------


def test_registry_and_info_agree() -> None:
    assert tuple(METRIC_FUNCS) == METRIC_NAMES
    assert tuple(METRIC_INFO) == METRIC_NAMES
    assert len(METRIC_NAMES) == 16


def test_compute_all_matches_individual_functions_bit_for_bit() -> None:
    """The property the batch notebook build relies on: switching to the
    single-pass API must not perturb a single float, or cached matrices built
    the old way would stop being comparable with new ones."""
    a, b = _peaked(), _flat()
    batch = compute_all(a, b)
    for name, fn in METRIC_FUNCS.items():
        assert batch[name] == fn(a, b), name


def test_compute_all_matches_individually_on_disjoint_support() -> None:
    """Same check in the numerically nastiest regime available."""
    a, b = _peaked(3), _shifted(3)
    batch = compute_all(a, b)
    for name, fn in METRIC_FUNCS.items():
        assert batch[name] == fn(a, b), name


def test_compute_all_subset_is_returned_in_request_order() -> None:
    subset = ["chamfer", "tv", "kld"]
    out = compute_all(_peaked(), _flat(), metrics=subset)
    assert list(out) == subset


def test_compute_all_subset_values_match_the_full_run() -> None:
    """Skipping chamfer must not change anything else — the union-support loop
    and the transport loop have to stay independent of what was requested."""
    a, b = _peaked(), _flat()
    full = compute_all(a, b)
    partial = compute_all(a, b, metrics=[n for n in METRIC_NAMES if n != "chamfer"])
    for name, value in partial.items():
        assert value == full[name], name


def test_compute_all_rejects_unknown_metric() -> None:
    with pytest.raises(KeyError, match="wasserstein3"):
        compute_all(_peaked(), _flat(), metrics=["kld", "wasserstein3"])


def test_compute_all_handles_empty_and_ragged_input() -> None:
    """Positions with no alternatives are skipped, and a length mismatch
    truncates to the shorter sequence rather than raising."""
    assert all(v == 0.0 for v in compute_all([], []).values())

    ragged = compute_all(_peaked(2), _flat(5))
    assert ragged["tv"] == pytest.approx(compute_tv(_peaked(2), _flat(2)), rel=1e-12)

    empty_step = _make_step([])
    with_gap = compute_all([empty_step, *_peaked(2)], [empty_step, *_flat(2)])
    assert with_gap["tv"] == pytest.approx(compute_tv(_peaked(2), _flat(2)), rel=1e-12)
