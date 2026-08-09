"""Metric MDS (SMACOF) on a precomputed distance matrix.

The iterative stress-minimising cousin of classical MDS: instead of
eigendecomposing the double-centered matrix it directly minimises Kruskal
stress, so it is not forced to discard negative eigenvalues and can fit a
non-Euclidean ``D`` that classical MDS mangles.

Shared fitter: this module's :func:`fit` is reused verbatim by
``nmds.py`` for the non-metric row. The two rows differ only in
``SPEC.defaults`` -- ``nmds`` sets ``"metric": False``, which
``sklearn.manifold.MDS`` reads to switch from metric to non-metric
stress minimisation. There is no separate non-metric fitter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pruning_metrics.dim_reduction.base import ReducerSpec

SPEC = ReducerSpec(
    key="mds",
    title="Metric MDS (SMACOF)",
    needs_coords=False,
    defaults={"n_init": 4, "max_iter": 300},
)


def fit(
    X: np.ndarray,
    *,
    n_components: int,
    random_state: int,
    params: dict[str, Any],
    info: dict[str, Any],
) -> np.ndarray:
    """Fits SMACOF MDS on the distance matrix ``X``.

    Parameters
    ----------
    X:
        The ``(n, n)`` distance matrix (``SPEC.needs_coords`` is False).
    n_components:
        Output dimensionality.
    random_state:
        Seed for the SMACOF initialisation(s).
    params:
        Merged hyperparameters, forwarded to ``sklearn.manifold.MDS``
        unchanged. Shared with ``nmds.py``, whose ``SPEC.defaults`` sets
        ``"metric": False`` here to switch to non-metric stress.
    info:
        Provenance dict; receives ``stress``, ``n_iter``, and the
        scikit-learn version.

    Returns
    -------
    np.ndarray
        ``(n, n_components)`` embedding coordinates.
    """
    # Deferred so the package imports without scikit-learn (see base.py).
    import sklearn  # pylint: disable=import-outside-toplevel
    from sklearn.manifold import MDS  # pylint: disable=import-outside-toplevel

    model = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=random_state,
        normalized_stress=False,
        **params,
    )
    coords = model.fit_transform(X)
    info["stress"] = float(model.stress_)
    info["n_iter"] = int(model.n_iter_)
    info["versions"]["sklearn"] = sklearn.__version__
    return coords
