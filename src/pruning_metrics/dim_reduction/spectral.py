"""Laplacian eigenmaps (spectral embedding) on a precomputed distance matrix.

Needs an affinity, not a distance. ``exp(-(d/sigma)^2)`` with sigma from the
median heuristic (see :func:`spectral_affinity`) is the standard choice; the
effective sigma is recorded in the returned info.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pruning_metrics.dim_reduction.base import ReducerSpec

SPEC = ReducerSpec(
    key="spectral",
    title="Laplacian eigenmaps",
    needs_coords=False,
    defaults={"bandwidth": None},
)


def spectral_affinity(
    D: np.ndarray, bandwidth: float | None
) -> tuple[np.ndarray, float]:
    """Turns a distance matrix into an RBF affinity for Laplacian eigenmaps.

    ``SpectralEmbedding(affinity="precomputed")`` wants a *similarity*, and
    feeding it a distance matrix silently inverts the geometry -- near
    points get the smallest weights. The conversion is
    ``exp(-(d/sigma)^2)``, and sigma matters: too small and the graph
    disconnects, too large and every point is equally similar.

    Parameters
    ----------
    D:
        Square symmetric distance matrix.
    bandwidth:
        Sigma to use, or ``None`` for the median heuristic -- the
        median of the non-zero off-diagonal distances, which puts the
        typical pair at ``exp(-1)`` and adapts automatically to
        measures whose scales differ by twelve orders of magnitude.

    Returns
    -------
    tuple[np.ndarray, float]
        A ``(affinity, sigma)`` tuple: the affinity matrix and the
        bandwidth actually used.
    """
    off = D[~np.eye(D.shape[0], dtype=bool)]
    off = off[off > 0]
    sigma = (
        float(bandwidth) if bandwidth else float(np.median(off)) if off.size else 1.0
    )
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return np.exp(-((D / sigma) ** 2)), sigma


def fit(
    X: np.ndarray,
    *,
    n_components: int,
    random_state: int,
    params: dict[str, Any],
    info: dict[str, Any],
) -> np.ndarray:
    """Fits Laplacian eigenmaps on the distance matrix ``X``.

    Converts ``X`` to an RBF affinity via :func:`spectral_affinity`, then
    fits ``sklearn.manifold.SpectralEmbedding`` on that affinity matrix.
    Note that other entries of ``params`` are deliberately NOT forwarded to
    ``SpectralEmbedding`` -- it receives none of them in the original code
    either, only ``n_components``, ``affinity``, and ``random_state``.

    Parameters
    ----------
    X:
        The ``(n, n)`` distance matrix (``SPEC.needs_coords`` is False).
    n_components:
        Output dimensionality.
    random_state:
        Seed for the spectral embedding solver.
    params:
        Merged hyperparameters; ``params["bandwidth"]`` is popped
        and replaced in place with the sigma actually used.
    info:
        Provenance dict; receives the scikit-learn version.

    Returns
    -------
    np.ndarray
        ``(n, n_components)`` embedding coordinates.
    """
    # Deferred so the package imports without scikit-learn (see base.py).
    import sklearn  # pylint: disable=import-outside-toplevel
    from sklearn.manifold import (  # pylint: disable=import-outside-toplevel
        SpectralEmbedding,
    )

    sigma_source = params.pop("bandwidth", None)
    affinity, sigma = spectral_affinity(X, sigma_source)
    params["bandwidth"] = sigma
    model = SpectralEmbedding(
        n_components=n_components,
        affinity="precomputed",
        random_state=random_state,
    )
    coords = model.fit_transform(affinity)
    info["versions"]["sklearn"] = sklearn.__version__
    return coords
