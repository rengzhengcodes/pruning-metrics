"""Two-dimensional embeddings of precomputed distance matrices.

The experiment compares model variants only through *pairwise distances* —
there is no native coordinate space, because a "point" is a pruned model
and the distance between two of them is a mean over per-token
distributional divergences (see :mod:`pruning_metrics.prob_measures`).
Every reducer here therefore takes an ``(n, n)`` distance matrix ``D``,
not a feature matrix.

Two families:

* **Distance-native** (``tsne``, ``umap``, ``isomap``, ``mds``, ``nmds``,
  ``spectral``) accept ``D`` directly.
* **Coordinate-requiring** (``pca``, ``lle`` and its variants,
  ``kpca_rbf``, ``ica``, ``random``) have no precomputed mode, so
  :func:`classical_mds_coords` first turns ``D`` into Euclidean
  coordinates by Torgerson double-centering.

That second step is lossy whenever ``D`` is not Euclidean: the negative
eigenvalues of the double-centered Gram matrix are discarded outright.
For the v2 KLD matrices this throws away ~25% of the eigenvalue mass and
over half the dimensions, which is enough to visibly distort the
resulting picture.  Use :func:`mds_spectrum` to quantify it and report
the number alongside any PCA or LLE figure, rather than letting a reader
assume all reducers saw the same object.

Layout:
    One module per algorithm (``tsne.py``, ``lle.py``, ...), each exposing
    a ``SPEC`` and a ``fit`` — see :mod:`pruning_metrics.dim_reduction.base`
    for the contract.  :mod:`pruning_metrics.dim_reduction.registry`
    assembles them into :data:`REDUCERS` and dispatches ``embed_2d``;
    :mod:`pruning_metrics.dim_reduction.classical_mds` holds the shared
    distance-to-coordinates preprocessing.

This package holds no plotting code: notebooks own their own styling.

Usage:
    >>> from pruning_metrics.dim_reduction import embed_2d, REDUCERS
    >>> for name in REDUCERS:
    ...     coords, info = embed_2d(D, name)
"""

from __future__ import annotations

from pruning_metrics.dim_reduction.base import ReducerSpec
from pruning_metrics.dim_reduction.classical_mds import (
    classical_mds_coords,
    complete_submatrix_indices,
    mds_spectrum,
)
from pruning_metrics.dim_reduction.registry import REDUCERS, embed_2d

__all__ = [
    "classical_mds_coords",
    "complete_submatrix_indices",
    "mds_spectrum",
    "ReducerSpec",
    "REDUCERS",
    "embed_2d",
]
