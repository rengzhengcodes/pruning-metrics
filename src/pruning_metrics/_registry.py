"""Shared helper for building name-keyed module registries.

Both :mod:`pruning_metrics.dim_reduction.registry` and
:mod:`pruning_metrics.prob_measures.registry` compose their packages from an
ordered tuple of algorithm modules. This module owns the one shape they
share: build the name -> module map and refuse duplicate keys, which would
otherwise silently drop an algorithm from every derived table.
"""

from __future__ import annotations

from types import ModuleType
from typing import Callable, Sequence


def build_module_registry(
    modules: Sequence[ModuleType],
    key_fn: Callable[[ModuleType], str],
    *,
    what: str,
) -> dict[str, ModuleType]:
    """Build an insertion-ordered ``key -> module`` map with a duplicate guard.

    Parameters
    ----------
    modules:
        Algorithm modules in reporting order; the returned dict preserves
        this order.
    key_fn:
        Extracts each module's registry key (e.g. ``m.SPEC.key`` or
        ``m.NAME``).
    what:
        Human-readable description of the key used in the error message,
        e.g. ``"SPEC.key among dim_reduction modules"``.

    Returns
    -------
    dict[str, ModuleType]
        Maps each module's key to the module, in ``modules`` order.

    Raises
    ------
    ImportError
        If two modules share a key. Raised at import time so a duplicate
        fails the build loudly instead of silently dropping an algorithm.
    """
    registry = {key_fn(m): m for m in modules}
    if len(registry) != len(modules):
        raise ImportError(f"duplicate {what}")
    return registry
