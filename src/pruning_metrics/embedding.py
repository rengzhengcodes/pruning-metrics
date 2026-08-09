"""Two-dimensional embeddings of precomputed distance matrices.

The experiment compares model variants only through *pairwise distances* --
there is no native coordinate space, because a "point" is a pruned model and
the distance between two of them is a mean over per-token distributional
divergences (see :mod:`pruning_metrics.metrics.distributions`). Every reducer
here therefore takes an ``(n, n)`` distance matrix ``D``, not a feature matrix.

Two families:

- **Distance-native** (``tsne``, ``umap``, ``isomap``) accept ``D`` directly
  via their ``metric="precomputed"`` mode.
- **Coordinate-requiring** (``pca``, ``lle``) have no precomputed mode, so
  :func:`classical_mds_coords` first turns ``D`` into Euclidean coordinates by
  Torgerson double-centering.

That second step is lossy whenever ``D`` is not Euclidean: the negative
eigenvalues of the double-centered Gram matrix are discarded outright. For the
v2 KLD matrices this throws away ~25% of the eigenvalue mass and over half the
dimensions, which is enough to visibly distort the resulting picture. Use
:func:`mds_spectrum` to quantify it and report the number alongside any PCA or
LLE figure, rather than letting a reader assume all five reducers saw the same
object.

This module holds no plotting code: notebooks own their own styling.

Usage
-----
>>> from pruning_metrics.embedding import embed_2d, REDUCERS
>>> for name in REDUCERS:
...     coords, info = embed_2d(D, name)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "classical_mds_coords",
    "complete_submatrix_indices",
    "mds_spectrum",
    "ReducerSpec",
    "REDUCERS",
    "embed_2d",
]


# ---------------------------------------------------------------------------
# Classical MDS (PCoA)
# ---------------------------------------------------------------------------


def classical_mds_coords(D: np.ndarray) -> np.ndarray:
    """Embed a distance matrix into Euclidean coordinates via classical MDS.

    Torgerson double-centering: B = -1/2 * J D^2 J is the Gram matrix of a
    centered configuration whose pairwise Euclidean distances reproduce D
    (exactly when D is Euclidean).  Dimensions with non-positive eigenvalues —
    the non-Euclidean part of D — are dropped.

    Parameters
    ----------
    D : np.ndarray
        Symmetric (n, n) distance matrix.

    Returns
    -------
    np.ndarray
        (n, k) coordinates, k = number of positive eigenvalues (padded to >= 2).
    """
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D**2) @ J
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    keep = evals > max(evals.max(), 0.0) * 1e-9
    X = evecs[:, keep] * np.sqrt(evals[keep])
    if X.shape[1] < 2:  # degenerate D — pad so 2-D reducers still run
        X = np.hstack([X, np.zeros((n, 2 - X.shape[1]))])
    return X


def mds_spectrum(D: np.ndarray) -> dict[str, float]:
    """Measure how far a distance matrix is from being Euclidean.

    Runs the same double-centering as :func:`classical_mds_coords` but reports
    the eigenvalue spectrum instead of the coordinates. ``neg_ratio`` is the
    headline number: the total magnitude of the *discarded* negative
    eigenvalues relative to the retained positive ones. It is 0 for a perfectly
    Euclidean ``D`` and grows as ``D`` violates the triangle inequality.

    Parameters
    ----------
    D : np.ndarray
        Symmetric (n, n) distance matrix.

    Returns
    -------
    dict[str, float]
        ``n_pos``
            Number of positive eigenvalues, i.e. how many dimensions survive
            into :func:`classical_mds_coords`' output (out of ``n``).
        ``pos_mass``, ``neg_mass``
            Summed magnitudes of the positive and negative eigenvalues.
        ``neg_ratio``
            ``neg_mass / pos_mass``; ``0.0`` when ``pos_mass`` is 0.
        ``var_2d``
            Fraction of ``pos_mass`` captured by the top 2 dimensions -- an
            upper bound on how much of the (Euclidean part of the) structure
            any 2-D picture can possibly show.
    """
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    evals = np.linalg.eigvalsh(-0.5 * J @ (D**2) @ J)

    # Same relative cutoff classical_mds_coords uses, so n_pos really is the
    # number of dimensions its output will have. An absolute `> 0` test would
    # instead count every ~1e-14 rounding artefact as a genuine dimension
    # (e.g. 24 "dimensions" for data that is exactly 4-dimensional).
    tol = max(float(evals.max()), 0.0) * 1e-9
    pos = evals[evals > tol]
    pos_mass = float(pos.sum())
    # abs(), not negation: summing an empty selection yields -0.0, which would
    # print as a nonsensical "-0.0000" neg_ratio for a perfectly Euclidean D.
    neg_mass = abs(float(evals[evals < -tol].sum()))
    top2 = float(np.sort(pos)[::-1][:2].sum()) if pos.size else 0.0
    return {
        "n_pos": int(pos.size),
        "pos_mass": pos_mass,
        "neg_mass": neg_mass,
        "neg_ratio": (neg_mass / pos_mass) if pos_mass > 0 else 0.0,
        "var_2d": (top2 / pos_mass) if pos_mass > 0 else 0.0,
    }


def complete_submatrix_indices(
    D: np.ndarray, *, counts: np.ndarray | None = None
) -> np.ndarray:
    """Indices of a large principal submatrix with no unobserved pairs.

    The v2 distance matrices are accumulated over evaluation tasks, and a pair
    of variants that never appeared on a common task is left at ``0.0`` --
    indistinguishable, numerically, from two identical models. That is far from
    harmless: a row of zeros makes one point sit at distance 0 from everything,
    which collapses Isomap's geodesic graph entirely and silently corrupts every
    neighbourhood-based measure.

    This greedily drops the row with the most unobserved partners until none
    remain, which is a good approximation to the maximum complete submatrix
    (exactly solving it is the NP-hard maximum-clique problem) and is exact in
    the common case where a few variants are missing outright.

    Parameters
    ----------
    D:
        Symmetric ``(n, n)`` distance matrix.
    counts:
        Optional ``(n, n)`` matrix of how many tasks contributed to each entry.
        When given, "unobserved" means ``counts == 0``, which is exact. Without
        it, a zero off-diagonal distance is treated as unobserved.

    Returns
    -------
    np.ndarray
        Sorted indices of the retained rows/columns. May be empty.
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    missing = (np.asarray(counts) == 0) if counts is not None else (D == 0.0)
    missing = missing.copy()
    np.fill_diagonal(missing, False)

    keep = np.ones(n, dtype=bool)
    while keep.any():
        sub = missing[np.ix_(keep, keep)]
        per_row = sub.sum(axis=1)
        if per_row.max(initial=0) == 0:
            break
        kept_idx = np.flatnonzero(keep)
        keep[kept_idx[int(np.argmax(per_row))]] = False
    return np.flatnonzero(keep)


# ---------------------------------------------------------------------------
# Reducer registry
# ---------------------------------------------------------------------------


class ReducerSpec(NamedTuple):
    """Static description of one dimensionality-reduction algorithm.

    Attributes
    ----------
    key:
        Short lowercase identifier, used in filenames and CSV rows.
    title:
        Display name for figure titles.
    needs_coords:
        ``True`` if the algorithm has no ``metric="precomputed"`` mode and must
        be fed :func:`classical_mds_coords` output instead of ``D`` itself.
    defaults:
        Hyperparameters applied unless the caller overrides them. Tuned for the
        v2 grid (n ~ 232); the v1 notebook's much smaller n = 13 forced far
        lower neighbourhood sizes, which should *not* be carried over.
    """

    key: str
    title: str
    needs_coords: bool
    defaults: dict[str, Any]


REDUCERS: dict[str, ReducerSpec] = {
    "pca": ReducerSpec(
        key="pca",
        title="PCA (on classical-MDS coords)",
        needs_coords=True,
        defaults={},
    ),
    "tsne": ReducerSpec(
        key="tsne",
        title="t-SNE",
        needs_coords=False,
        # init="pca" is rejected outright with metric="precomputed" (sklearn
        # cannot run PCA on a distance matrix), so "random" is the only option.
        defaults={"perplexity": 30.0, "max_iter": 2000, "init": "random", "n_jobs": 1},
    ),
    "umap": ReducerSpec(
        key="umap",
        title="UMAP",
        needs_coords=False,
        defaults={"n_neighbors": 15, "min_dist": 0.1},
    ),
    "isomap": ReducerSpec(
        key="isomap",
        title="Isomap",
        needs_coords=False,
        # eigen_solver="dense": the "auto" default picks ARPACK at this n and
        # fails with "Starting vector is zero" on these geodesic kernels. A
        # dense eigendecomposition of a few-hundred-square matrix is both exact
        # and instant, so there is nothing to gain from the iterative solver.
        defaults={"n_neighbors": 15, "eigen_solver": "dense"},
    ),
    "lle": ReducerSpec(
        key="lle",
        title="LLE",
        needs_coords=True,
        defaults={"n_neighbors": 15, "method": "standard"},
    ),
    "mds": ReducerSpec(
        key="mds",
        title="Metric MDS (SMACOF)",
        needs_coords=False,
        # The iterative stress-minimising cousin of classical MDS: instead of
        # eigendecomposing the double-centered matrix it directly minimises
        # Kruskal stress, so it is not forced to discard negative eigenvalues
        # and can fit a non-Euclidean D that classical MDS mangles.
        defaults={"n_init": 4, "max_iter": 300},
    ),
    "nmds": ReducerSpec(
        key="nmds",
        title="Non-metric MDS",
        needs_coords=False,
        # Fits an arbitrary monotone transform of the distances rather than the
        # distances themselves. It therefore optimises for exactly what Shepard
        # rho measures, and is the fair comparison for any claim that a
        # neighbour-graph method "preserves ordering".
        defaults={"metric": False, "n_init": 4, "max_iter": 300},
    ),
    "spectral": ReducerSpec(
        key="spectral",
        title="Laplacian eigenmaps",
        needs_coords=False,
        # Needs an affinity, not a distance. exp(-(d/sigma)^2) with sigma from
        # the median heuristic (see _spectral_affinity) is the standard choice;
        # the effective sigma is recorded in the returned info.
        defaults={"bandwidth": None},
    ),
    "kpca_rbf": ReducerSpec(
        key="kpca_rbf",
        title="Kernel PCA (RBF)",
        needs_coords=True,
        # Kernel PCA with a linear kernel on double-centered coordinates is
        # classical MDS exactly; the RBF kernel is what makes this a distinct
        # algorithm rather than a second name for the pca row.
        defaults={"kernel": "rbf", "gamma": None},
    ),
    "lle_modified": ReducerSpec(
        key="lle_modified",
        title="Modified LLE",
        needs_coords=True,
        defaults={"n_neighbors": 15, "method": "modified", "eigen_solver": "dense"},
    ),
    "lle_hessian": ReducerSpec(
        key="lle_hessian",
        title="Hessian LLE",
        needs_coords=True,
        # Hessian eigenmapping requires n_neighbors > n_components*(n_components+3)/2,
        # i.e. > 5 at n_components=2; embed_2d raises the value if a caller
        # lowers it below that.
        defaults={"n_neighbors": 15, "method": "hessian", "eigen_solver": "dense"},
    ),
    "ltsa": ReducerSpec(
        key="ltsa",
        title="Local tangent space alignment",
        needs_coords=True,
        defaults={"n_neighbors": 15, "method": "ltsa", "eigen_solver": "dense"},
    ),
    "ica": ReducerSpec(
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
    ),
    "random": ReducerSpec(
        key="random",
        title="Gaussian random projection",
        needs_coords=True,
        # The control. Johnson-Lindenstrauss says a random projection preserves
        # distances in expectation, so this is the empirical noise floor: any
        # reducer that does not beat this row has bought nothing with its
        # algorithm. A single draw at the given random_state, not an average.
        defaults={},
    ),
}


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _connected_n_neighbors(D: np.ndarray, n_neighbors: int) -> int:
    """Smallest ``k >= n_neighbors`` whose kNN graph on ``D`` is connected.

    Isomap replaces distances with shortest paths through the kNN graph. If
    that graph has more than one component, some pairs have no path at all and
    scikit-learn silently substitutes very large geodesics, which warps the
    embedding without any error being raised. Growing the neighbourhood until
    the graph connects is the standard fix; the caller records the value that
    was actually used.
    """
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import kneighbors_graph

    n = D.shape[0]
    k = min(n_neighbors, n - 1)
    while k < n - 1:
        graph = kneighbors_graph(
            D, n_neighbors=k, metric="precomputed", mode="connectivity"
        )
        n_components, _ = connected_components(graph, directed=False)
        if n_components == 1:
            return k
        k += 1
    return max(k, 1)


def _spectral_affinity(
    D: np.ndarray, bandwidth: float | None
) -> tuple[np.ndarray, float]:
    """Turn a distance matrix into an RBF affinity for Laplacian eigenmaps.

    ``SpectralEmbedding(affinity="precomputed")`` wants a *similarity*, and
    feeding it a distance matrix silently inverts the geometry — near points get
    the smallest weights. The conversion is ``exp(-(d/sigma)^2)``, and sigma
    matters: too small and the graph disconnects, too large and every point is
    equally similar.

    Parameters
    ----------
    D:
        Square symmetric distance matrix.
    bandwidth:
        Sigma to use, or ``None`` for the median heuristic — the median of the
        non-zero off-diagonal distances, which puts the typical pair at
        ``exp(-1)`` and adapts automatically to measures whose scales differ by
        twelve orders of magnitude.

    Returns
    -------
    affinity, sigma:
        The affinity matrix and the bandwidth actually used.
    """
    off = D[~np.eye(D.shape[0], dtype=bool)]
    off = off[off > 0]
    sigma = (
        float(bandwidth) if bandwidth else float(np.median(off)) if off.size else 1.0
    )
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return np.exp(-((D / sigma) ** 2)), sigma


def embed_2d(
    D: np.ndarray,
    reducer: str,
    *,
    random_state: int = 42,
    n_components: int = 2,
    **params: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reduce a precomputed distance matrix to ``n_components`` dimensions.

    Parameters
    ----------
    D:
        Symmetric ``(n, n)`` distance matrix. Must be finite.
    reducer:
        A key of :data:`REDUCERS`.
    random_state:
        Seed for the stochastic reducers (t-SNE, UMAP, LLE). PCA and Isomap are
        deterministic given their input and ignore it.
    n_components:
        Output dimensionality; 2 for every figure in this project.
    **params:
        Overrides merged over the reducer's :attr:`ReducerSpec.defaults`.

    Returns
    -------
    coords : np.ndarray
        ``(n, n_components)`` embedding.
    info : dict
        What actually ran: ``reducer``, ``needs_coords``, the *effective*
        ``params`` (after any Isomap connectivity bump), ``mds_dims`` when
        coordinates were derived, and the library versions that produced it.
        Recording this matters because t-SNE and UMAP output is not portable
        across library versions.

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

    if spec.needs_coords:
        X = classical_mds_coords(D)
        info["mds_dims"] = int(X.shape[1])
    else:
        X = D

    if reducer == "pca":
        import sklearn
        from sklearn.decomposition import PCA

        model = PCA(n_components=n_components, **effective)
        coords = model.fit_transform(X)
        info["explained_variance_ratio"] = model.explained_variance_ratio_.tolist()
        info["versions"]["sklearn"] = sklearn.__version__

    elif reducer == "tsne":
        import sklearn
        from sklearn.manifold import TSNE

        # Perplexity must stay below n; sklearn raises otherwise.
        effective["perplexity"] = min(
            effective["perplexity"], max(2.0, D.shape[0] / 3.0)
        )
        model = TSNE(
            n_components=n_components,
            metric="precomputed",
            random_state=random_state,
            **effective,
        )
        coords = model.fit_transform(X)
        info["kl_divergence"] = float(model.kl_divergence_)
        info["versions"]["sklearn"] = sklearn.__version__

    elif reducer == "umap":
        # Imported lazily: umap pulls in numba, which costs seconds of JIT on
        # first use and would be paid by every forked worker that imports this
        # module for classical_mds_coords alone.
        import umap

        effective["n_neighbors"] = min(effective["n_neighbors"], D.shape[0] - 1)
        model = umap.UMAP(
            n_components=n_components,
            metric="precomputed",
            random_state=random_state,
            **effective,
        )
        coords = model.fit_transform(X)
        info["versions"]["umap"] = umap.__version__

    elif reducer == "isomap":
        import sklearn
        from sklearn.manifold import Isomap

        effective["n_neighbors"] = _connected_n_neighbors(D, effective["n_neighbors"])
        model = Isomap(n_components=n_components, metric="precomputed", **effective)
        coords = model.fit_transform(X)
        info["versions"]["sklearn"] = sklearn.__version__

    elif reducer in ("lle", "lle_modified", "lle_hessian", "ltsa"):
        import sklearn
        from sklearn.manifold import LocallyLinearEmbedding

        effective["n_neighbors"] = min(effective["n_neighbors"], X.shape[0] - 1)
        if effective["method"] == "hessian":
            # sklearn requires n_neighbors > n_components*(n_components+3)/2.
            floor = n_components * (n_components + 3) // 2 + 1
            effective["n_neighbors"] = max(effective["n_neighbors"], floor)
        elif effective["method"] == "ltsa":
            effective["n_neighbors"] = max(effective["n_neighbors"], n_components + 1)
        model = LocallyLinearEmbedding(
            n_components=n_components, random_state=random_state, **effective
        )
        coords = model.fit_transform(X)
        info["reconstruction_error"] = float(model.reconstruction_error_)
        info["versions"]["sklearn"] = sklearn.__version__

    elif reducer in ("mds", "nmds"):
        import sklearn
        from sklearn.manifold import MDS

        model = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
            normalized_stress=False,
            **effective,
        )
        coords = model.fit_transform(X)
        info["stress"] = float(model.stress_)
        info["n_iter"] = int(model.n_iter_)
        info["versions"]["sklearn"] = sklearn.__version__

    elif reducer == "spectral":
        import sklearn
        from sklearn.manifold import SpectralEmbedding

        affinity, sigma = _spectral_affinity(D, effective.pop("bandwidth", None))
        effective["bandwidth"] = sigma
        model = SpectralEmbedding(
            n_components=n_components,
            affinity="precomputed",
            random_state=random_state,
        )
        coords = model.fit_transform(affinity)
        info["versions"]["sklearn"] = sklearn.__version__

    elif reducer == "kpca_rbf":
        import sklearn
        from sklearn.decomposition import KernelPCA

        if effective.get("gamma") is None:
            # 1 / (2 * total variance) puts the typical squared distance at
            # order 1 in the exponent, whatever scale the metric works in.
            spread = float(np.var(X, axis=0).sum())
            effective["gamma"] = 1.0 / (2.0 * spread) if spread > 0 else 1.0
        model = KernelPCA(n_components=n_components, **effective)
        coords = model.fit_transform(X)
        info["versions"]["sklearn"] = sklearn.__version__

    elif reducer == "ica":
        import sklearn
        from sklearn.decomposition import FastICA

        model = FastICA(
            n_components=n_components, random_state=random_state, **effective
        )
        coords = model.fit_transform(X)
        info["n_iter"] = int(getattr(model, "n_iter_", 0))
        info["versions"]["sklearn"] = sklearn.__version__

    elif reducer == "random":
        import sklearn
        from sklearn.random_projection import GaussianRandomProjection

        model = GaussianRandomProjection(
            n_components=n_components, random_state=random_state, **effective
        )
        coords = model.fit_transform(X)
        info["versions"]["sklearn"] = sklearn.__version__

    else:  # pragma: no cover - guarded by the REDUCERS membership check above
        raise KeyError(reducer)

    info["params"] = effective
    return np.asarray(coords, dtype=np.float64), info
