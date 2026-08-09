"""Hessian locally linear embedding on classical-MDS coordinates.

Shares its fitter with :mod:`pruning_metrics.dim_reduction.lle`: scikit-learn
implements the whole LLE family as one class parameterised by ``method``, so
this module contributes only the ``SPEC`` for ``method="hessian"`` and reuses
the sibling module's :func:`fit` outright rather than duplicating it.
"""

from __future__ import annotations

from pruning_metrics.dim_reduction.base import ReducerSpec

from pruning_metrics.dim_reduction.lle import fit

SPEC = ReducerSpec(
    key="lle_hessian",
    title="Hessian LLE",
    needs_coords=True,
    # Hessian eigenmapping requires n_neighbors > n_components*(n_components+3)/2,
    # i.e. > 5 at n_components=2; the shared lle.fit raises the value if a
    # caller lowers it below that.
    defaults={"n_neighbors": 15, "method": "hessian", "eigen_solver": "dense"},
)

__all__ = ["SPEC", "fit"]
