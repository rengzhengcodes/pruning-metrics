"""Isomap on a precomputed distance matrix.

Distance-native: scikit-learn's Isomap accepts the distance matrix directly
via ``metric="precomputed"``, so nothing is lost to a coordinate conversion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pruning_metrics.dim_reduction.base import ReducerSpec

SPEC = ReducerSpec(
    key="isomap",
    title="Isomap",
    needs_coords=False,
    # eigen_solver="dense": the "auto" default picks ARPACK at this n and
    # fails with "Starting vector is zero" on these geodesic kernels. A
    # dense eigendecomposition of a few-hundred-square matrix is both exact
    # and instant, so there is nothing to gain from the iterative solver.
    defaults={"n_neighbors": 15, "eigen_solver": "dense"},
)


def connected_n_neighbors(D: np.ndarray, n_neighbors: int) -> int:
    """Finds the smallest ``k >= n_neighbors`` whose kNN graph is connected.

    Isomap replaces distances with shortest paths through the kNN graph. If
    that graph has more than one component, some pairs have no path at all
    and scikit-learn silently substitutes very large geodesics, which warps
    the embedding without any error being raised. Growing the neighbourhood
    until the graph connects is the standard fix; the caller records the
    value that was actually used.

    Parameters
    ----------
    D:
        Symmetric ``(n, n)`` distance matrix.
    n_neighbors:
        Requested neighbourhood size.

    Returns
    -------
    int
        The smallest ``k >= n_neighbors`` (and ``< n``) whose kNN graph on
        ``D`` is connected, or ``n - 1`` (floored at 1) if none connects.
    """
    from scipy.sparse.csgraph import (  # pylint: disable=import-outside-toplevel
        connected_components,
    )
    from sklearn.neighbors import (  # pylint: disable=import-outside-toplevel
        kneighbors_graph,
    )

    n = D.shape[0]
    k = min(n_neighbors, n - 1)
    while k < n - 1:
        graph = kneighbors_graph(
            D, n_neighbors=k, metric="precomputed", mode="connectivity"
        )
        n_components, _ = connected_components(graph, directed=False)
        if n_components == 1:
            return k
        k += 1
    return max(k, 1)


def fit(  # pylint: disable=unused-argument  # random_state: uniform contract
    X: np.ndarray,
    *,
    n_components: int,
    random_state: int,
    params: dict[str, Any],
    info: dict[str, Any],
) -> np.ndarray:
    """Fits Isomap on the distance matrix ``X``.

    Grows ``params["n_neighbors"]`` in place until the kNN graph connects
    (see :func:`connected_n_neighbors`) and records the scikit-learn version
    in ``info``. ``random_state`` is accepted for signature uniformity with
    the other fitters but ignored: Isomap is deterministic given its input.

    Parameters
    ----------
    X:
        The ``(n, n)`` distance matrix (``SPEC.needs_coords`` is False).
    n_components:
        Output dimensionality.
    random_state:
        Unused; Isomap is deterministic.
    params:
        Merged hyperparameters; ``n_neighbors`` may be raised here.
    info:
        Provenance dict; receives the scikit-learn version.

    Returns
    -------
    np.ndarray
        ``(n, n_components)`` embedding coordinates.
    """
    # Deferred so the package imports without scikit-learn (see base.py).
    import sklearn  # pylint: disable=import-outside-toplevel
    from sklearn.manifold import Isomap  # pylint: disable=import-outside-toplevel

    params["n_neighbors"] = connected_n_neighbors(X, params["n_neighbors"])
    model = Isomap(n_components=n_components, metric="precomputed", **params)
    coords = model.fit_transform(X)
    info["versions"]["sklearn"] = sklearn.__version__
    return coords
