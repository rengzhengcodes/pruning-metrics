"""Backwards-compatible facade for :mod:`pruning_metrics.dim_reduction`.

All fourteen reducers used to live in this one module; each algorithm now
has its own file under ``pruning_metrics/dim_reduction/`` (``tsne.py``,
``lle.py``, ...), composed by a registry — see
:mod:`pruning_metrics.dim_reduction` for the package overview and
:mod:`pruning_metrics.dim_reduction.base` for the per-module contract.

This module remains so that existing imports (tests, notebooks, cached
analysis code) keep working unchanged.  New code should import from
:mod:`pruning_metrics.dim_reduction` directly.
"""

from __future__ import annotations

from pruning_metrics import dim_reduction as _dim_reduction

# The wildcard mirrors the package's export surface so a reducer added there
# is available here with no edit to this facade.
from pruning_metrics.dim_reduction import *  # pylint: disable=wildcard-import,unused-wildcard-import

# Historical private helper, re-exported under its old name for the unit
# tests that import it from this module.  Its canonical public home is
# pruning_metrics.dim_reduction.spectral.
from pruning_metrics.dim_reduction.spectral import (  # pylint: disable=unused-import
    spectral_affinity as _spectral_affinity,
)

#: Mirrors ``dim_reduction.__all__``; the package owns the canonical list.
__all__ = list(_dim_reduction.__all__)
