"""Gaussian random projection on classical-MDS coordinates.

Coordinate-requiring: scikit-learn's ``GaussianRandomProjection`` has no
``metric="precomputed"`` mode, so it is fed the classical-MDS coordinates
derived from the distance matrix rather than the distance matrix itself.

The control. Johnson-Lindenstrauss says a random projection preserves
distances in expectation, so this is the empirical noise floor: any reducer
that does not beat this row has bought nothing with its algorithm. A single
draw at the given random_state, not an average.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pruning_metrics.dim_reduction.base import ReducerSpec

SPEC = ReducerSpec(
    key="random",
    title="Gaussian random projection",
    needs_coords=True,
    defaults={},
)


def fit(
    X: np.ndarray,
    *,
    n_components: int,
    random_state: int,
    params: dict[str, Any],
    info: dict[str, Any],
) -> np.ndarray:
    """Fits a Gaussian random projection on the classical-MDS coords ``X``.

    Parameters
    ----------
    X:
        ``(n, k)`` classical-MDS coordinates (``SPEC.needs_coords`` is
        True).
    n_components:
        Output dimensionality.
    random_state:
        Seed for the random projection matrix.
    params:
        Merged hyperparameters, passed through to
        ``GaussianRandomProjection`` unchanged.
    info:
        Provenance dict; receives the scikit-learn version.

    Returns
    -------
    np.ndarray
        ``(n, n_components)`` embedding coordinates.
    """
    # Deferred so the package imports without scikit-learn (see base.py).
    import sklearn  # pylint: disable=import-outside-toplevel
    from sklearn.random_projection import (  # pylint: disable=import-outside-toplevel
        GaussianRandomProjection,
    )

    model = GaussianRandomProjection(
        n_components=n_components, random_state=random_state, **params
    )
    coords = model.fit_transform(X)
    info["versions"]["sklearn"] = sklearn.__version__
    return coords
