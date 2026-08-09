"""Backwards-compatible facade for :mod:`pruning_metrics.prob_measures`.

All sixteen distributional distances used to live in this one module; each
measure now has its own file under ``pruning_metrics/prob_measures/``
(``kld.py``, ``jsd.py``, ...), composed by a registry — see
:mod:`pruning_metrics.prob_measures` for the package overview and
:mod:`pruning_metrics.prob_measures.base` for the per-module contract and
the shared representation notes.

This module remains so that existing imports (tests, notebooks, cached
analysis code) keep working unchanged.  New code should import from
:mod:`pruning_metrics.prob_measures` directly.
"""

from __future__ import annotations

from pruning_metrics import prob_measures as _prob_measures

# The wildcard mirrors the package's export surface so a measure added there
# is available here with no edit to this facade.
from pruning_metrics.prob_measures import *  # pylint: disable=wildcard-import,unused-wildcard-import

# Historical private helper, re-exported under its old name for callers
# (the unit tests among them) that imported it from this module.  Its
# canonical public home is pruning_metrics.prob_measures.wasserstein2.
from pruning_metrics.prob_measures.wasserstein2 import (  # pylint: disable=unused-import
    wasserstein_p as _wasserstein_p,
)

#: Mirrors ``prob_measures.__all__``; the package owns the canonical list.
__all__ = list(_prob_measures.__all__)
