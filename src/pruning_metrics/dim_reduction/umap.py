"""UMAP on a precomputed distance matrix.

Distance-native: umap-learn accepts the distance matrix directly via
``metric="precomputed"``, so nothing is lost to a coordinate conversion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pruning_metrics.dim_reduction.base import ReducerSpec

SPEC = ReducerSpec(
    key="umap",
    title="UMAP",
    needs_coords=False,
    defaults={"n_neighbors": 15, "min_dist": 0.1},
)


def fit(
    X: np.ndarray,
    *,
    n_components: int,
    random_state: int,
    params: dict[str, Any],
    info: dict[str, Any],
) -> np.ndarray:
    """Fits UMAP on the distance matrix ``X``.

    Clamps ``params["n_neighbors"]`` below ``n`` in place and records the
    umap-learn version in ``info`` for reproducibility.

    Parameters
    ----------
    X:
        The ``(n, n)`` distance matrix (``SPEC.needs_coords`` is False).
    n_components:
        Output dimensionality.
    random_state:
        Seed for the stochastic optimisation.
    params:
        Merged hyperparameters; ``n_neighbors`` may be lowered here.
    info:
        Provenance dict; receives the umap-learn version.

    Returns
    -------
    np.ndarray
        ``(n, n_components)`` embedding coordinates.
    """
    # Imported lazily: umap pulls in numba, which costs seconds of JIT on
    # first use and would be paid by every forked worker that imports this
    # module for classical_mds_coords alone.
    import umap  # pylint: disable=import-outside-toplevel

    params["n_neighbors"] = min(params["n_neighbors"], X.shape[0] - 1)
    model = umap.UMAP(
        n_components=n_components,
        metric="precomputed",
        random_state=random_state,
        **params,
    )
    coords = model.fit_transform(X)
    info["versions"]["umap"] = umap.__version__
    return coords
