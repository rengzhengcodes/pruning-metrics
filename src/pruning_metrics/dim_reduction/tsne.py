"""t-SNE on a precomputed distance matrix.

Distance-native: scikit-learn's t-SNE accepts the distance matrix directly
via ``metric="precomputed"``, so nothing is lost to a coordinate
conversion.  The trade-off is documented in the ``SPEC`` defaults below —
``init="pca"`` is unavailable in this mode, leaving the less stable random
initialisation as the only option.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pruning_metrics.dim_reduction.base import ReducerSpec

SPEC = ReducerSpec(
    key="tsne",
    title="t-SNE",
    needs_coords=False,
    # init="pca" is rejected outright with metric="precomputed" (sklearn
    # cannot run PCA on a distance matrix), so "random" is the only option.
    defaults={"perplexity": 30.0, "max_iter": 2000, "init": "random", "n_jobs": 1},
)


def fit(
    X: np.ndarray,
    *,
    n_components: int,
    random_state: int,
    params: dict[str, Any],
    info: dict[str, Any],
) -> np.ndarray:
    """Fits t-SNE on the distance matrix ``X``.

    Clamps ``params["perplexity"]`` below ``n`` in place (scikit-learn
    raises otherwise) and records the final Kullback-Leibler divergence of
    the embedding in ``info`` — the number to compare when judging whether
    two t-SNE runs converged equally well.

    Parameters
    ----------
    X:
        The ``(n, n)`` distance matrix (``SPEC.needs_coords`` is False).
    n_components:
        Output dimensionality.
    random_state:
        Seed for the stochastic optimisation.
    params:
        Merged hyperparameters; ``perplexity`` may be lowered here.
    info:
        Provenance dict; receives ``kl_divergence`` and the
        scikit-learn version.

    Returns
    -------
    np.ndarray
        ``(n, n_components)`` embedding coordinates.
    """
    # Deferred so the package imports without scikit-learn (see base.py).
    import sklearn  # pylint: disable=import-outside-toplevel
    from sklearn.manifold import TSNE  # pylint: disable=import-outside-toplevel

    # Perplexity must stay below n; sklearn raises otherwise.
    params["perplexity"] = min(params["perplexity"], max(2.0, X.shape[0] / 3.0))
    model = TSNE(
        n_components=n_components,
        metric="precomputed",
        random_state=random_state,
        **params,
    )
    coords = model.fit_transform(X)
    info["kl_divergence"] = float(model.kl_divergence_)
    info["versions"]["sklearn"] = sklearn.__version__
    return coords
