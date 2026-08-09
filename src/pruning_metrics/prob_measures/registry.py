"""Measure registry and the single-pass batch evaluator ``compute_all``.

Design decision — registry over hand-maintained dicts:
    ``_MEASURE_MODULES`` below is the single source of truth for which
    measures exist and for their reporting order (the column order of
    every CSV and figure).  :data:`METRIC_FUNCS`, :data:`METRIC_INFO` and
    the private union-kernel table are all derived from it, so they cannot
    fall out of sync with each other or with the modules on disk.
"""

from __future__ import annotations

from typing import Callable, Iterable

from pruning_metrics.prob_measures import (
    bhattacharyya,
    chamfer,
    chisq,
    cosine,
    emd,
    hellinger,
    jeffreys,
    jsd,
    kld,
    l2,
    renyi05,
    renyi2,
    rkld,
    triangular,
    tv,
    wasserstein2,
)
from pruning_metrics.prob_measures.base import (
    MetricInfo,
    TokenStepDict,
    aligned_alternatives,
    atom_weights,
    union_probs,
)

__all__ = ["METRIC_FUNCS", "METRIC_INFO", "METRIC_NAMES", "compute_all"]

#: The measures, in reporting order: the thirteen union-support measures
#: first (f-divergences and vector geometry), then optimal transport, then
#: the point-cloud measure.  This order is the column order of every
#: downstream table.
_MEASURE_MODULES = (
    kld,
    rkld,
    jeffreys,
    jsd,
    tv,
    hellinger,
    bhattacharyya,
    renyi05,
    chisq,
    renyi2,
    triangular,
    l2,
    cosine,
    emd,
    wasserstein2,
    chamfer,
)

#: Every distance in this package, in reporting order.
METRIC_FUNCS: dict[str, Callable[[list, list], float]] = {
    m.NAME: m.compute for m in _MEASURE_MODULES
}

#: Metadata for each entry of :data:`METRIC_FUNCS`, same keys and order.
METRIC_INFO: dict[str, MetricInfo] = {m.NAME: m.INFO for m in _MEASURE_MODULES}

METRIC_NAMES: tuple[str, ...] = tuple(METRIC_FUNCS)

#: Per-position kernels for the measures computed on the union-support
#: probability vectors (the transport and point-cloud measures work in
#: other representations and define no kernel).
_UNION_KERNELS: dict[str, Callable[..., float]] = {
    m.NAME: m.kernel for m in _MEASURE_MODULES if hasattr(m, "kernel")
}

#: Per-position distances for the transport measures, which work on each
#: model's own logprob atoms rather than the union support.
_TRANSPORT_KERNELS: dict[str, Callable[..., float]] = {
    m.NAME: m.transport_distance
    for m in _MEASURE_MODULES
    if hasattr(m, "transport_distance")
}

#: Measures defining neither per-position hook; ``compute_all`` falls back
#: to their own ``compute``, so a hook-less measure can never be silently
#: returned as 0.0.
_STANDALONE_MODULES = tuple(
    m
    for m in _MEASURE_MODULES
    if not hasattr(m, "kernel") and not hasattr(m, "transport_distance")
)

# A duplicated NAME would silently drop a measure from every table above.
if len(METRIC_FUNCS) != len(_MEASURE_MODULES):
    raise ImportError("duplicate NAME among prob_measures modules")


def compute_all(
    tokens_0: list[TokenStepDict],
    tokens_k: list[TokenStepDict],
    *,
    metrics: Iterable[str] | None = None,
) -> dict[str, float]:
    """Computes every distance in :data:`METRIC_FUNCS` in a single pass.

    The union-support alignment is the dominant per-position cost and is
    shared by thirteen of the sixteen measures, so computing all of them
    costs barely more than computing one — whereas calling the individual
    functions in a loop repeats that work once per measure.

    Results are identical to calling the individual functions: the same
    kernels are applied to the same vectors in the same order, so the
    floats agree bit-for-bit, which is what lets a batch build reuse
    distance matrices that were originally produced one metric at a time.

    Parameters
    ----------
    tokens_0:
        Per-token steps from the base (level=0) model.
    tokens_k:
        Per-token steps from the pruned model.
    metrics:
        Restrict the output to these names.  Defaults to all of
        them.  Worth using to drop ``"chamfer"``, which alone costs
        more than the other fifteen combined because it compares every
        position against every other rather than pairing them up by
        index.

    Returns
    -------
    dict[str, float]
        One entry per requested metric name, in :data:`METRIC_FUNCS`
        order.

    Raises
    ------
    KeyError
        If ``metrics`` names a measure this package does not
        define.
    """
    wanted = list(METRIC_FUNCS) if metrics is None else list(metrics)
    unknown = [name for name in wanted if name not in METRIC_FUNCS]
    if unknown:
        raise KeyError(f"unknown metric(s): {sorted(unknown)}")
    wanted_set = set(wanted)

    totals: dict[str, float] = {name: 0.0 for name in wanted}
    union_wanted = [name for name in _UNION_KERNELS if name in wanted_set]
    transport_wanted = [name for name in _TRANSPORT_KERNELS if name in wanted_set]

    # One pass over the aligned positions serves both the union-support
    # measures (shared probability vectors) and the transport measures
    # (shared atom/weight extraction).
    if union_wanted or transport_wanted:
        for alts_0, alts_k in aligned_alternatives(tokens_0, tokens_k):
            if union_wanted:
                p0, pk = union_probs(alts_0, alts_k)
                for name in union_wanted:
                    totals[name] += _UNION_KERNELS[name](p0, pk)
            if transport_wanted:
                lp_0, w0 = atom_weights(alts_0)
                lp_k, wk = atom_weights(alts_k)
                for name in transport_wanted:
                    totals[name] += _TRANSPORT_KERNELS[name](lp_0, w0, lp_k, wk)

    # Measures with no per-position hook (currently only Chamfer, whose
    # nearest-neighbour matching cannot share the aligned pass) run whole.
    for module in _STANDALONE_MODULES:
        if module.NAME in wanted_set:
            totals[module.NAME] = module.compute(tokens_0, tokens_k)

    # `totals` was seeded in `wanted` order and only updated in place, so it
    # already is the requested mapping in the requested order.
    return totals
