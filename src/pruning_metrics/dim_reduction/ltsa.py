"""Local tangent space alignment on classical-MDS coordinates.

Shares its fitter with :mod:`pruning_metrics.dim_reduction.lle`: scikit-learn
implements the whole LLE family as one class parameterised by ``method``, so
this module contributes only the ``SPEC`` for ``method="ltsa"`` and reuses the
sibling module's :func:`fit` outright rather than duplicating it.
"""

from __future__ import annotations

from pruning_metrics.dim_reduction.base import ReducerSpec

from pruning_metrics.dim_reduction.lle import fit

SPEC = ReducerSpec(
    key="ltsa",
    title="Local tangent space alignment",
    needs_coords=True,
    defaults={"n_neighbors": 15, "method": "ltsa", "eigen_solver": "dense"},
)

__all__ = ["SPEC", "fit"]
