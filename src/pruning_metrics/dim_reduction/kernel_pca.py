"""Kernel PCA (RBF) on classical-MDS coordinates.

Coordinate-requiring: scikit-learn's KernelPCA has no ``metric="precomputed"``
mode for arbitrary kernels, so it is fed the classical-MDS coordinates
derived from the distance matrix rather than the distance matrix itself.

Kernel PCA with a linear kernel on double-centered coordinates is classical
MDS exactly; the RBF kernel is what makes this a distinct algorithm rather
than a second name for the pca row.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pruning_metrics.dim_reduction.base import ReducerSpec

SPEC = ReducerSpec(
    key="kpca_rbf",
    title="Kernel PCA (RBF)",
    needs_coords=True,
    defaults={"kernel": "rbf", "gamma": None},
)


def fit(  # pylint: disable=unused-argument  # random_state: uniform contract
    X: np.ndarray,
    *,
    n_components: int,
    random_state: int,
    params: dict[str, Any],
    info: dict[str, Any],
) -> np.ndarray:
    """Fits Kernel PCA (RBF) on the classical-MDS coordinates ``X``.

    When ``params["gamma"]`` is ``None``, derives it from ``X``'s per-axis
    variance in place before fitting.

    Parameters
    ----------
    X:
        ``(n, k)`` classical-MDS coordinates (``SPEC.needs_coords`` is
        True).
    n_components:
        Output dimensionality.
    random_state:
        Unused; Kernel PCA is deterministic given its input.
    params:
        Merged hyperparameters; ``gamma`` may be filled in here.
    info:
        Provenance dict; receives the scikit-learn version.

    Returns
    -------
    np.ndarray
        ``(n, n_components)`` embedding coordinates.
    """
    # Deferred so the package imports without scikit-learn (see base.py).
    import sklearn  # pylint: disable=import-outside-toplevel

    # pylint: disable-next=import-outside-toplevel
    from sklearn.decomposition import KernelPCA

    if params.get("gamma") is None:
        # 1 / (2 * total variance) puts the typical squared distance at
        # order 1 in the exponent, whatever scale the metric works in.
        spread = float(np.var(X, axis=0).sum())
        params["gamma"] = 1.0 / (2.0 * spread) if spread > 0 else 1.0
    model = KernelPCA(n_components=n_components, **params)
    coords = model.fit_transform(X)
    info["versions"]["sklearn"] = sklearn.__version__
    return coords
