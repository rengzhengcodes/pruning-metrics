"""PCA on classical-MDS coordinates.

Coordinate-requiring: scikit-learn's PCA has no ``metric="precomputed"``
mode, so it is fed the classical-MDS coordinates derived from the distance
matrix rather than the distance matrix itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pruning_metrics.dim_reduction.base import ReducerSpec

SPEC = ReducerSpec(
    key="pca",
    title="PCA (on classical-MDS coords)",
    needs_coords=True,
    defaults={},
)


def fit(  # pylint: disable=unused-argument  # random_state: uniform contract
    X: np.ndarray,
    *,
    n_components: int,
    random_state: int,
    params: dict[str, Any],
    info: dict[str, Any],
) -> np.ndarray:
    """Fits PCA on the classical-MDS coordinates ``X``.

    Parameters
    ----------
    X:
        ``(n, k)`` classical-MDS coordinates (``SPEC.needs_coords`` is
        True).
    n_components:
        Output dimensionality.
    random_state:
        Unused; PCA is deterministic given its input.
    params:
        Merged hyperparameters, passed through to ``PCA`` unchanged.
    info:
        Provenance dict; receives ``explained_variance_ratio`` and the
        scikit-learn version.

    Returns
    -------
    np.ndarray
        ``(n, n_components)`` embedding coordinates.
    """
    # Deferred so the package imports without scikit-learn (see base.py).
    import sklearn  # pylint: disable=import-outside-toplevel
    from sklearn.decomposition import PCA  # pylint: disable=import-outside-toplevel

    model = PCA(n_components=n_components, **params)
    coords = model.fit_transform(X)
    info["explained_variance_ratio"] = model.explained_variance_ratio_.tolist()
    info["versions"]["sklearn"] = sklearn.__version__
    return coords
