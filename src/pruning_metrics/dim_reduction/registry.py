"""Reducer registry and the shared ``embed_2d`` entry point.

Design decision — registry over dispatch chain:
    ``embed_2d`` used to be a 200-line if/elif chain; every algorithm now
    contributes its ``SPEC`` and ``fit`` (see
    :mod:`pruning_metrics.dim_reduction.base` for the contract) and this
    module does the three things they all share: validate the input once,
    derive classical-MDS coordinates for the algorithms that need them,
    and record provenance.  ``_REDUCER_MODULES`` is the single source of
    truth for which algorithms exist *and* for their reporting order —
    notebooks iterate ``REDUCERS`` directly, so the tuple below is the
    order figures appear in.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pruning_metrics.dim_reduction import (
    ica,
    isomap,
    kernel_pca,
    lle,
    lle_hessian,
    lle_modified,
    ltsa,
    mds,
    nmds,
    pca,
    random_projection,
    spectral,
    tsne,
    umap,
)
from pruning_metrics._registry import build_module_registry
from pruning_metrics.dim_reduction.base import ReducerSpec
from pruning_metrics.dim_reduction.classical_mds import classical_mds_coords

__all__ = ["REDUCERS", "embed_2d"]

#: The algorithms, in reporting order (the order the original monolithic
#: registry used, which the sweep notebooks rely on for figure layout).
_REDUCER_MODULES = (
    pca,
    tsne,
    umap,
    isomap,
    lle,
    mds,
    nmds,
    spectral,
    kernel_pca,
    lle_modified,
    lle_hessian,
    ltsa,
    ica,
    random_projection,
)

#: Modules keyed by reducer key — the single lookup table behind both the
#: public :data:`REDUCERS` dict and ``embed_2d``'s dispatch. The shared
#: helper raises ImportError on a duplicated key, which would otherwise
#: silently drop a reducer from the registry.
_MODULES_BY_KEY = build_module_registry(
    _REDUCER_MODULES,
    lambda m: m.SPEC.key,
    what="SPEC.key among dim_reduction modules",
)

#: Public registry: reducer key -> static description.
REDUCERS: dict[str, ReducerSpec] = {key: m.SPEC for key, m in _MODULES_BY_KEY.items()}


def embed_2d(
    D: np.ndarray,
    reducer: str,
    *,
    random_state: int = 42,
    n_components: int = 2,
    **params: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reduces a precomputed distance matrix to ``n_components`` dimensions.

    Parameters
    ----------
    D:
        Symmetric ``(n, n)`` distance matrix.  Must be finite.
    reducer:
        A key of :data:`REDUCERS`.
    random_state:
        Seed for the stochastic reducers (t-SNE, UMAP, LLE).
        PCA and Isomap are deterministic given their input and ignore
        it.
    n_components:
        Output dimensionality; 2 for every figure in this
        project.
    **params:
        Overrides merged over the reducer's
        :attr:`ReducerSpec.defaults`.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        A ``(coords, info)`` tuple.  ``coords`` is the ``(n, n_components)``
        embedding.  ``info`` records what actually ran: ``reducer``,
        ``needs_coords``, the *effective* ``params`` (after any in-fitter
        adjustment such as the Isomap connectivity bump), ``mds_dims`` when
        coordinates were derived, and the library versions that produced
        it.  Recording this matters because t-SNE and UMAP output is not
        portable across library versions.

    Raises
    ------
    KeyError
        If ``reducer`` is not a key of :data:`REDUCERS`.
    ValueError
        If ``D`` is not square, not symmetric, or not finite.
    """
    if reducer not in REDUCERS:
        raise KeyError(
            f"unknown reducer {reducer!r}; expected one of {sorted(REDUCERS)}"
        )
    spec = REDUCERS[reducer]

    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square, got shape {D.shape}")
    if not np.isfinite(D).all():
        raise ValueError("D contains non-finite entries")
    if not np.allclose(D, D.T, atol=1e-8):
        raise ValueError("D must be symmetric")

    effective = {**spec.defaults, **params}
    info: dict[str, Any] = {
        "reducer": reducer,
        "needs_coords": spec.needs_coords,
        "n": int(D.shape[0]),
        "random_state": random_state,
        "versions": {"numpy": np.__version__},
    }

    # Coordinate-requiring algorithms are fed classical-MDS coordinates;
    # mds_dims records how many dimensions survived the (lossy) conversion.
    if spec.needs_coords:
        X = classical_mds_coords(D)
        info["mds_dims"] = int(X.shape[1])
    else:
        X = D

    # The fitter may mutate `effective` (clamps, connectivity bumps), so the
    # dict recorded below is the hyperparameters that actually ran.
    coords = _MODULES_BY_KEY[reducer].fit(
        X,
        n_components=n_components,
        random_state=random_state,
        params=effective,
        info=info,
    )

    info["params"] = effective
    return np.asarray(coords, dtype=np.float64), info
