"""Non-metric MDS on a precomputed distance matrix.

Fits an arbitrary monotone transform of the distances rather than the
distances themselves. It therefore optimises for exactly what Shepard rho
measures, and is the fair comparison for any claim that a neighbour-graph
method "preserves ordering".
"""

from __future__ import annotations

from pruning_metrics.dim_reduction.base import ReducerSpec

# Reuse the shared SMACOF fitter: sklearn.manifold.MDS itself switches from
# metric to non-metric stress minimisation based on the "metric" keyword in
# SPEC.defaults below, so there is nothing left for a separate fit() to do.
from pruning_metrics.dim_reduction.mds import fit

SPEC = ReducerSpec(
    key="nmds",
    title="Non-metric MDS",
    needs_coords=False,
    defaults={"metric": False, "n_init": 4, "max_iter": 300},
)

__all__ = ["SPEC", "fit"]
