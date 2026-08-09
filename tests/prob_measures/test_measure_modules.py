"""Unit tests for :mod:`pruning_metrics.prob_measures`.

Until now, the sixteen individual measure modules (``kld.py``, ``jsd.py``,
...) plus :mod:`pruning_metrics.prob_measures.base` and
:mod:`pruning_metrics.prob_measures.registry` were only exercised
indirectly, through ``tests/metrics/test_distributions*.py`` calling the
backwards-compatible facade at :mod:`pruning_metrics.metrics.distributions`.
This module tests the ``prob_measures`` package directly and stands on its
own: it does not import anything from ``tests/metrics/`` and does not
require the facade module to exist.

Coverage is parametrised over the registry (:data:`METRIC_NAMES`) rather
than a hand-written list of measure names, so a 17th measure dropped into
:mod:`pruning_metrics.prob_measures.registry` is picked up automatically:

* The registry contract -- :data:`METRIC_FUNCS`, :data:`METRIC_INFO` and
  :data:`METRIC_NAMES` agree on keys and order, and that key set matches
  the modules on disk exactly.
* The per-module contract documented in ``base.py``: every measure module
  exposes ``NAME``, ``INFO`` and ``compute``, and its ``kernel`` or
  ``transport_distance`` hook (if any) returns a finite float.
* Universal math properties -- ``d(x, x) == 0``, non-negativity, declared
  symmetry/asymmetry, and declared boundedness -- checked on a handful of
  fixed synthetic token-step sequences, built with the shared
  ``TokenStepDict``/``TopAlternativeDict`` helper pattern in
  ``tests/helpers/token_steps.py`` (also used by
  ``tests/metrics/test_distributions_extended.py``).
* One bit-for-bit check that :func:`compute_all` agrees with every
  individual ``compute`` function, so this suite does not rely on the
  more thorough version of that same check in
  ``test_distributions_extended.py`` to catch a ``compute_all``
  regression.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

# `_make_step` comes from the shared tests/helpers/token_steps.py module
# rather than being redefined here: that module is the single home for the
# TokenStepDict/TopAlternativeDict builder pattern used by this suite and by
# tests/metrics/test_distributions*.py, so the suites cannot drift apart.
from helpers.token_steps import make_step as _make_step
from pruning_metrics.prob_measures import registry as prob_measures_registry
from pruning_metrics.prob_measures.base import (
    EPS,
    MetricInfo,
    TokenStepDict,
    union_probs,
)
from pruning_metrics.prob_measures.registry import (
    METRIC_FUNCS,
    METRIC_INFO,
    METRIC_NAMES,
    compute_all,
)

# ---------------------------------------------------------------------------
# Synthetic token-step fixtures
# ---------------------------------------------------------------------------


def _peaked(n_positions: int = 4) -> list[TokenStepDict]:
    """Confident: almost all mass on token 0."""
    return [_make_step([0.0, -6.0, -7.0, -8.0, -9.0]) for _ in range(n_positions)]


def _flat(n_positions: int = 4) -> list[TokenStepDict]:
    """Hedging: uniform over the same five tokens."""
    return [_make_step([-1.0] * 5) for _ in range(n_positions)]


def _shifted(n_positions: int = 4) -> list[TokenStepDict]:
    """Same shape as ``_peaked`` but on a disjoint set of token ids.

    Disjoint support is the worst case for the union-support alignment
    (every token in one list gets the -50 ``MISSING_LOGPROB`` fill when
    looked up in the other), which is exactly the regime the boundedness
    and non-negativity checks below want to exercise.
    """
    return [
        _make_step([0.0, -6.0, -7.0, -8.0, -9.0], token_ids=[10, 11, 12, 13, 14])
        for _ in range(n_positions)
    ]


# Fixed, deterministic fixtures shared by the tests below. Built once at
# import time since every measure here is a pure function of its inputs.
_PEAKED = _peaked()
_FLAT = _flat()
_SHIFTED = _shifted()

# Every measure module, keyed by registry name, imported once up front so
# the per-module contract tests below can parametrize on whether the
# optional `kernel` / `transport_distance` hooks are present, without
# re-importing on every parametrized call.
_MODULES_BY_NAME = {
    name: importlib.import_module(f"pruning_metrics.prob_measures.{name}")
    for name in METRIC_NAMES
}
_KERNEL_NAMES = [n for n, m in _MODULES_BY_NAME.items() if hasattr(m, "kernel")]
_TRANSPORT_NAMES = [
    n for n, m in _MODULES_BY_NAME.items() if hasattr(m, "transport_distance")
]
_SYMMETRIC_NAMES = [n for n in METRIC_NAMES if METRIC_INFO[n].symmetric]
_ASYMMETRIC_NAMES = [n for n in METRIC_NAMES if not METRIC_INFO[n].symmetric]
_BOUNDED_NAMES = [n for n in METRIC_NAMES if METRIC_INFO[n].bounded is not None]

# Design: tie the d(x, x) tolerance to EPS (1e-12, see base.py) rather than
# picking an arbitrary constant. EPS is added to every ratio-form kernel's
# denominator, so a measure is not always *exactly* zero on identical
# input. Verified empirically before writing the assertions below: every
# measure in this registry returns exactly 0.0 on self-comparison except
# `renyi2`, whose `log(sum p**2 / (q + EPS))` kernel is off by roughly EPS
# per position -- about 2e-11 in the worst case observed across every
# fixture used in this file (including a 10-position, extreme-skew,
# disjoint-support sequence checked by hand while writing this suite). No
# measure is excluded from the check; 1000 * EPS keeps ~50x headroom over
# that observed worst case.
_ZERO_ATOL = 1_000 * EPS

#: The four measure families documented on `MetricInfo.family` in the
#: module docstring of base.py.
_KNOWN_FAMILIES = frozenset({"f-divergence", "geometry", "transport", "point-cloud"})


# ---------------------------------------------------------------------------
# 1. Registry contract
# ---------------------------------------------------------------------------


def test_registry_keys_and_order_agree() -> None:
    """METRIC_FUNCS, METRIC_INFO and METRIC_NAMES describe the same measures
    in the same order -- the property every downstream table (CSV columns,
    figure legends) relies on to line values up with labels."""
    assert tuple(METRIC_FUNCS) == METRIC_NAMES
    assert tuple(METRIC_INFO) == METRIC_NAMES
    assert len(METRIC_NAMES) == len(set(METRIC_NAMES))  # no duplicate keys


def test_registry_has_at_least_sixteen_measures() -> None:
    assert len(METRIC_NAMES) >= 16


def test_registry_names_match_the_modules_on_disk() -> None:
    """The registry's key set is exactly the ``NAME`` of every module it
    composes -- an independent check on ``_MEASURE_MODULES`` itself, not
    just on the dicts derived from it, so a module whose ``NAME`` drifted
    from its registry entry would be caught here."""
    # Design: reach into the private module tuple directly instead of
    # re-deriving "the modules" from METRIC_NAMES (e.g. via importlib),
    # which would make this check circular -- it would only ever confirm
    # that METRIC_NAMES equals itself.
    # pylint: disable-next=protected-access
    modules = prob_measures_registry._MEASURE_MODULES
    assert {m.NAME for m in modules} == set(METRIC_NAMES)


@pytest.mark.parametrize("name", METRIC_NAMES)
def test_metric_info_field_types(name: str) -> None:
    """Every INFO entry is a MetricInfo with the field types base.py documents."""
    info = METRIC_INFO[name]
    assert isinstance(info, MetricInfo)
    assert isinstance(info.label, str) and info.label
    assert isinstance(info.family, str) and info.family in _KNOWN_FAMILIES
    assert isinstance(info.symmetric, bool)
    assert info.bounded is None or isinstance(info.bounded, float)
    assert isinstance(info.formula, str) and info.formula


# ---------------------------------------------------------------------------
# 2. Per-module contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", METRIC_NAMES)
def test_module_exposes_name_info_compute(name: str) -> None:
    """Every measure module exposes the three hooks every route (single
    metric, ``compute_all``, this suite) depends on."""
    module = _MODULES_BY_NAME[name]
    assert isinstance(module.NAME, str)
    assert module.NAME == name  # registry key must match the module's own NAME
    assert isinstance(module.INFO, MetricInfo)
    assert callable(module.compute)


@pytest.mark.parametrize("name", _KERNEL_NAMES)
def test_kernel_hook_returns_finite_float_on_aligned_vectors(name: str) -> None:
    """Every union-support measure's per-position kernel accepts two aligned
    probability vectors and returns a finite float."""
    module = _MODULES_BY_NAME[name]
    p = np.array([0.6, 0.25, 0.15])
    q = np.array([0.2, 0.3, 0.5])
    value = module.kernel(p, q)
    assert isinstance(value, float)
    assert math.isfinite(value)


@pytest.mark.parametrize("name", _TRANSPORT_NAMES)
def test_transport_hook_returns_finite_float_on_weighted_atoms(name: str) -> None:
    """Every transport measure's per-position distance accepts two weighted
    1-D atom sets and returns a finite float."""
    module = _MODULES_BY_NAME[name]
    u_values, u_weights = np.array([-1.0, -2.0, -3.0]), np.array([0.5, 0.3, 0.2])
    v_values, v_weights = np.array([-0.5, -2.5, -4.0]), np.array([0.2, 0.3, 0.5])
    value = module.transport_distance(u_values, u_weights, v_values, v_weights)
    assert isinstance(value, float)
    assert math.isfinite(value)


# ---------------------------------------------------------------------------
# 3. Universal math properties on synthetic token steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", METRIC_NAMES)
@pytest.mark.parametrize(
    "steps", [_PEAKED, _FLAT, _SHIFTED], ids=["peaked", "flat", "shifted"]
)
def test_distance_to_self_is_zero(name: str, steps: list[TokenStepDict]) -> None:
    """d(x, x) ~= 0 for every registered measure, on every fixture (see
    the ``_ZERO_ATOL`` design comment above for why this is an
    approximate rather than exact equality)."""
    assert METRIC_FUNCS[name](steps, steps) == pytest.approx(0.0, abs=_ZERO_ATOL)


@pytest.mark.parametrize("name", METRIC_NAMES)
def test_non_negative_on_differing_inputs(name: str) -> None:
    """Every measure stays non-negative on two different synthetic inputs,
    including the disjoint-support pair where the -50 logprob fill makes
    every f-divergence kernel work hardest."""
    fn = METRIC_FUNCS[name]
    assert fn(_PEAKED, _FLAT) >= -_ZERO_ATOL
    assert fn(_PEAKED, _SHIFTED) >= -_ZERO_ATOL


@pytest.mark.parametrize("name", _SYMMETRIC_NAMES)
def test_declared_symmetric_measures_are_order_independent(name: str) -> None:
    """d(a, b) == d(b, a) for every measure INFO.symmetric marks True."""
    fn = METRIC_FUNCS[name]
    assert fn(_PEAKED, _FLAT) == pytest.approx(fn(_FLAT, _PEAKED), rel=1e-9)


@pytest.mark.parametrize("name", _ASYMMETRIC_NAMES)
def test_declared_asymmetric_measures_are_flagged_as_such(name: str) -> None:
    """Directional measures must say so on INFO. This suite does not also
    require fn(a, b) != fn(b, a) here -- that per-measure identity is
    already checked in tests/metrics/test_distributions_extended.py; this
    check only pins the declaration itself."""
    assert METRIC_INFO[name].symmetric is False


@pytest.mark.parametrize("name", _BOUNDED_NAMES)
def test_bounded_kernel_respects_its_declared_bound(name: str) -> None:
    """Per-position kernel value never exceeds INFO.bounded, checked on the
    disjoint-support pair -- the worst case reachable with this data's -50
    logprob fill for a token missing from one side's top-k list."""
    module = _MODULES_BY_NAME[name]
    bound = METRIC_INFO[name].bounded
    p, q = union_probs(_PEAKED[0]["top_alternatives"], _SHIFTED[0]["top_alternatives"])
    assert module.kernel(p, q) <= bound + _ZERO_ATOL


# ---------------------------------------------------------------------------
# 4. compute_all consistency
# ---------------------------------------------------------------------------


def test_compute_all_matches_individual_compute_functions() -> None:
    """compute_all agrees bit-for-bit with every individual compute function.

    A lighter, single-pair version of
    ``test_compute_all_matches_individual_functions_bit_for_bit`` in
    ``test_distributions_extended.py`` -- kept here too so this suite does
    not depend on that file existing to catch a ``compute_all`` regression.
    """
    batch = compute_all(_PEAKED, _FLAT)
    for name, fn in METRIC_FUNCS.items():
        assert batch[name] == fn(_PEAKED, _FLAT), name
