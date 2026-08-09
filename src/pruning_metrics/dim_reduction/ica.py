"""FastICA on classical-MDS coordinates.

Coordinate-requiring: scikit-learn's FastICA has no ``metric="precomputed"``
mode, so it is fed the classical-MDS coordinates derived from the distance
matrix rather than the distance matrix itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pruning_metrics.dim_reduction.base import ReducerSpec

SPEC = ReducerSpec(
    key="ica",
    title="FastICA",
    needs_coords=True,
    # Read this row carefully: a rotation cannot change pairwise distances,
    # so ICA's *unmixing* is invisible to every quality score here. What
    # separates this from the pca row is FastICA's whitening, which gives
    # both output axes unit variance. Its scores are "equal-variance PCA"
    # scores, and the gap between the two rows is the cost of throwing away
    # the relative scale of the first two components.
    defaults={"max_iter": 2000, "whiten": "unit-variance"},
)


def fit(
    X: np.ndarray,
    *,
    n_components: int,
    random_state: int,
    params: dict[str, Any],
    info: dict[str, Any],
) -> np.ndarray:
    """Fits FastICA on the classical-MDS coordinates ``X``.

    Parameters
    ----------
    X:
        ``(n, k)`` classical-MDS coordinates (``SPEC.needs_coords`` is
        True).
    n_components:
        Output dimensionality.
    random_state:
        Seed for the stochastic optimisation.
    params:
        Merged hyperparameters, passed through to ``FastICA``
        unchanged.
    info:
        Provenance dict; receives ``n_iter`` and the scikit-learn
        version.

    Returns
    -------
    np.ndarray
        ``(n, n_components)`` embedding coordinates.
    """
    # Deferred so the package imports without scikit-learn (see base.py).
    import sklearn  # pylint: disable=import-outside-toplevel
    from sklearn.decomposition import FastICA  # pylint: disable=import-outside-toplevel

    model = FastICA(n_components=n_components, random_state=random_state, **params)
    coords = model.fit_transform(X)
    info["n_iter"] = int(getattr(model, "n_iter_", 0))
    info["versions"]["sklearn"] = sklearn.__version__
    return coords
