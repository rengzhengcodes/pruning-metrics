"""Modified locally linear embedding on classical-MDS coordinates.

Shares its fitter with :mod:`pruning_metrics.dim_reduction.lle`: scikit-learn
implements the whole LLE family as one class parameterised by ``method``, so
this module contributes only the ``SPEC`` for ``method="modified"`` and reuses
the sibling module's :func:`fit` outright rather than duplicating it.
"""

from __future__ import annotations

from pruning_metrics.dim_reduction.base import ReducerSpec

from pruning_metrics.dim_reduction.lle import fit

SPEC = ReducerSpec(
    key="lle_modified",
    title="Modified LLE",
    needs_coords=True,
    defaults={"n_neighbors": 15, "method": "modified", "eigen_solver": "dense"},
)

__all__ = ["SPEC", "fit"]
