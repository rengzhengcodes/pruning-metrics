"""Locally linear embedding (standard variant) on classical-MDS coordinates.

Design decision — one fitter for the whole LLE family:
    ``lle``, ``lle_modified``, ``lle_hessian``, and ``ltsa`` are, in
    scikit-learn, one class (:class:`~sklearn.manifold.LocallyLinearEmbedding`)
    parameterised by its ``method`` argument, not four independent
    algorithms.  Rather than duplicate the fitting logic four times, this
    module owns the shared :func:`fit` and the sibling modules
    (``lle_modified.py``, ``lle_hessian.py``, ``ltsa.py``) re-export it.  The
    method-specific ``n_neighbors`` floors below are keyed off
    ``params["method"]``, exactly mirroring the branching the monolithic
    ``embed_2d`` used to do inline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pruning_metrics.dim_reduction.base import ReducerSpec

SPEC = ReducerSpec(
    key="lle",
    title="LLE",
    needs_coords=True,
    defaults={"n_neighbors": 15, "method": "standard"},
)


def fit(
    X: np.ndarray,
    *,
    n_components: int,
    random_state: int,
    params: dict[str, Any],
    info: dict[str, Any],
) -> np.ndarray:
    """Fits an LLE-family embedding on classical-MDS coordinates ``X``.

    Clamps ``params["n_neighbors"]`` below ``X.shape[0]`` in place, then
    applies a method-specific floor: Hessian eigenmapping requires
    ``n_neighbors > n_components*(n_components+3)/2`` (sklearn raises
    otherwise), and LTSA requires ``n_neighbors >= n_components + 1``.
    Records the model's reconstruction error in ``info`` — the number to
    compare across LLE-family runs.

    Parameters
    ----------
    X:
        The ``(n, k)`` classical-MDS coordinates (``SPEC.needs_coords``
        is True).
    n_components:
        Output dimensionality.
    random_state:
        Seed for the stochastic solver.
    params:
        Merged hyperparameters; ``n_neighbors`` and ``method`` select
        which LLE variant runs; ``n_neighbors`` may be raised or lowered
        here.
    info:
        Provenance dict; receives ``reconstruction_error`` and the
        scikit-learn version.

    Returns
    -------
    np.ndarray
        ``(n, n_components)`` embedding coordinates.
    """
    # Deferred so the package imports without scikit-learn (see base.py).
    import sklearn  # pylint: disable=import-outside-toplevel
    from sklearn.manifold import (  # pylint: disable=import-outside-toplevel
        LocallyLinearEmbedding,
    )

    params["n_neighbors"] = min(params["n_neighbors"], X.shape[0] - 1)
    if params["method"] == "hessian":
        # sklearn requires n_neighbors > n_components*(n_components+3)/2.
        floor = n_components * (n_components + 3) // 2 + 1
        params["n_neighbors"] = max(params["n_neighbors"], floor)
    elif params["method"] == "ltsa":
        params["n_neighbors"] = max(params["n_neighbors"], n_components + 1)
    model = LocallyLinearEmbedding(
        n_components=n_components, random_state=random_state, **params
    )
    coords = model.fit_transform(X)
    info["reconstruction_error"] = float(model.reconstruction_error_)
    info["versions"]["sklearn"] = sklearn.__version__
    return coords
