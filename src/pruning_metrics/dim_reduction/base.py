"""Shared contract for the dimensionality-reduction sublibrary.

Design decision — one module per algorithm:
    This package replaces a single 600-line module whose ``embed_2d``
    dispatched all fourteen reducers through one if/elif chain.  Each
    algorithm now lives in its own module (``tsne.py``, ``lle.py``, ...)
    and the registry (:mod:`pruning_metrics.dim_reduction.registry`)
    composes them, so adding a reducer means adding a file, not editing a
    dispatch chain.  Every algorithm module exposes exactly two names:

    ``SPEC``:
        A :class:`ReducerSpec` holding the algorithm's registry key,
        display title, input requirement, and default hyperparameters.

    ``fit(X, *, n_components, random_state, params, info)``:
        Fits the algorithm and returns an ``(n, n_components)`` coordinate
        array.  The registry's ``embed_2d`` is the only intended caller.

Fitter contract:
    * ``X`` is the ``(n, n)`` distance matrix itself when
      ``SPEC.needs_coords`` is ``False``, and classical-MDS coordinates
      derived from it (see :func:`pruning_metrics.dim_reduction.
      classical_mds.classical_mds_coords`) when ``True``.  Input validation
      happens once in the registry; fitters may assume a square, symmetric,
      finite input.
    * ``params`` is the caller's keyword overrides merged over
      ``SPEC.defaults``.  A fitter that must adjust a hyperparameter to
      keep the algorithm well posed (clamping t-SNE's perplexity below
      ``n``, growing Isomap's neighbourhood until the kNN graph connects)
      mutates ``params`` in place, so the registry records the
      hyperparameters that *actually ran* rather than the ones requested.
    * ``info`` is the provenance dict eventually returned to the caller.
      Fitters add algorithm-specific diagnostics (stress, reconstruction
      error, KL divergence, ...) and the version of the library that
      produced the embedding — t-SNE and UMAP output is not portable
      across library versions, so recording versions is what makes a saved
      embedding reproducible.
    * Heavyweight libraries (scikit-learn, umap-learn) are imported inside
      ``fit``, never at module import time, so importing this package for
      ``classical_mds_coords`` alone stays cheap and works without the
      optional dependencies installed.
"""

from __future__ import annotations

from typing import Any, NamedTuple

__all__ = ["ReducerSpec"]


class ReducerSpec(NamedTuple):
    """Static description of one dimensionality-reduction algorithm.

    Attributes
    ----------
    key:
        Short lowercase identifier, used in filenames and CSV rows.
    title:
        Display name for figure titles.
    needs_coords:
        ``True`` if the algorithm has no
        ``metric="precomputed"`` mode and must be fed classical-MDS
        coordinates instead of the distance matrix itself.
    defaults:
        Hyperparameters applied unless the caller overrides them.
        Tuned for the v2 grid (n ~ 232); the v1 notebook's much smaller
        n = 13 forced far lower neighbourhood sizes, which should *not*
        be carried over.
    """

    key: str
    title: str
    needs_coords: bool
    defaults: dict[str, Any]
