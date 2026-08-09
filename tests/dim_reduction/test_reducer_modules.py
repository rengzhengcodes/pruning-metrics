"""Unit tests for the pruning_metrics.dim_reduction package.

Companion to ``tests/test_embedding.py``, which exercises the
backwards-compatible facade at :mod:`pruning_metrics.embedding` and already
covers :func:`~pruning_metrics.dim_reduction.classical_mds_coords`,
:func:`~pruning_metrics.dim_reduction.mds_spectrum`,
:func:`~pruning_metrics.dim_reduction.complete_submatrix_indices`, and
``spectral_affinity`` in depth. This file instead targets the
``dim_reduction`` package directly -- the registry contract each of the
fourteen per-algorithm modules must satisfy (see
:mod:`pruning_metrics.dim_reduction.base`), a shared smoke test parametrized
over every registered reducer so a fifteenth module is covered automatically,
seeded-determinism for the reducers whose fitters actually consume
``random_state``, and the two input-validation error paths ``embed_2d`` must
raise before dispatching to any fitter at all.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from pruning_metrics.dim_reduction import REDUCERS, ReducerSpec, embed_2d

pytest.importorskip("sklearn")
pytest.importorskip("umap")


# ---------------------------------------------------------------------------
# Shared fixture -- one small point cloud, built once for the whole module
# ---------------------------------------------------------------------------

#: Design: a single seeded Generator, constructed once here at import time
#: (never an argless ``np.random.default_rng()``), is the sole source of
#: randomness behind the point cloud every test in this module shares. Fixing
#: the seed makes every test byte-for-byte reproducible across runs.
_SEED = 100
_N_POINTS = 22  # > UMAP's default n_neighbors=15, with margin to spare
_DIM = 3
_RNG = np.random.default_rng(_SEED)


def _euclidean_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Computes the exact pairwise Euclidean distance matrix of a point set.

    Parameters
    ----------
    points:
        ``(n, d)`` array of coordinates.

    Returns
    -------
    np.ndarray
        Symmetric ``(n, n)`` distance matrix with a zero diagonal.
    """
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff**2).sum(-1))


@pytest.fixture(scope="module")
def distance_matrix() -> np.ndarray:
    """The one distance matrix every reducer in this module is run against.

    Module-scoped so the ``(22, 22)`` matrix is built exactly once per test
    session no matter how many reducer keys get parametrized over it --
    fitting fourteen manifold-learning algorithms already dominates the
    suite's runtime, and there is no reason to also pay for rebuilding the
    (cheap, but non-zero) input fourteen-plus times over.

    Returns
    -------
    np.ndarray
        Symmetric, finite ``(22, 22)`` Euclidean distance matrix of 22 fixed
        3-D points drawn from :data:`_RNG`.
    """
    points = _RNG.normal(size=(_N_POINTS, _DIM))
    return _euclidean_distance_matrix(points)


# ---------------------------------------------------------------------------
# Registry contract (base.py / registry.py)
# ---------------------------------------------------------------------------

#: The fourteen reducer modules under pruning_metrics/dim_reduction/, by
#: filename. Distinct from their registry *keys* -- e.g. the module
#: ``kernel_pca`` registers under the key ``"kpca_rbf"`` and
#: ``random_projection`` under ``"random"`` -- which is exactly what
#: test_module_spec_key_matches_its_registry_entry below checks.
_REDUCER_MODULE_NAMES = (
    "pca",
    "tsne",
    "umap",
    "isomap",
    "lle",
    "mds",
    "nmds",
    "spectral",
    "kernel_pca",
    "lle_modified",
    "lle_hessian",
    "ltsa",
    "ica",
    "random_projection",
)


def test_registry_has_exactly_the_fourteen_documented_keys() -> None:
    """The registry key set is the project's one source of truth for "which
    reducers exist"; notebooks and sweep scripts iterate it directly."""
    assert set(REDUCERS) == {
        "pca",
        "tsne",
        "umap",
        "isomap",
        "lle",
        "mds",
        "nmds",
        "spectral",
        "kpca_rbf",
        "lle_modified",
        "lle_hessian",
        "ltsa",
        "ica",
        "random",
    }
    assert len(REDUCERS) == 14


@pytest.mark.parametrize("reducer", sorted(REDUCERS))
def test_registry_values_are_reducer_specs(reducer: str) -> None:
    assert isinstance(REDUCERS[reducer], ReducerSpec)


@pytest.mark.parametrize("module_name", _REDUCER_MODULE_NAMES)
def test_module_spec_key_matches_its_registry_entry(module_name: str) -> None:
    """Every module contributes exactly one SPEC (base.py's contract); this
    checks the registry files it under the same key the module declares,
    rather than under a typo'd or stale one."""
    module = importlib.import_module(f"pruning_metrics.dim_reduction.{module_name}")
    assert module.SPEC.key in REDUCERS
    # Identity, not just equality: registry.py builds REDUCERS as
    # {key: m.SPEC for key, m in _MODULES_BY_KEY.items()}, so the registry
    # entry must be the very same object the module defines, not a copy.
    assert REDUCERS[module.SPEC.key] is module.SPEC


@pytest.mark.parametrize("reducer", sorted(REDUCERS))
def test_spec_fields_match_the_base_contract_types(reducer: str) -> None:
    """Type-checks every field of :class:`ReducerSpec` against the contract
    documented in ``base.py`` rather than trusting NamedTuple's positional
    construction to have been called correctly everywhere."""
    spec = REDUCERS[reducer]
    assert isinstance(spec.key, str) and spec.key
    assert isinstance(spec.title, str) and spec.title
    assert isinstance(spec.needs_coords, bool)
    assert isinstance(spec.defaults, dict)


# ---------------------------------------------------------------------------
# embed_2d smoke test -- shared contract, parametrized over every reducer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reducer", sorted(REDUCERS))
def test_embed_2d_smoke(reducer: str, distance_matrix: np.ndarray) -> None:
    """Every registered reducer must produce a finite 2-D embedding and the
    provenance keys the registry docstring promises for every reducer
    (``mds_dims`` is conditional on ``needs_coords`` and is intentionally not
    checked here -- see test_embed_records_mds_dims_only_when_coordinates_
    were_derived in tests/test_embedding.py). Parametrizing over
    ``sorted(REDUCERS)`` rather than a hard-coded key list means a
    fifteenth reducer module is exercised automatically the moment it is
    registered, with no edit to this file."""
    coords, info = embed_2d(distance_matrix, reducer, random_state=42)

    assert coords.shape == (_N_POINTS, 2)
    assert np.isfinite(coords).all()

    for key in ("reducer", "needs_coords", "params", "versions"):
        assert key in info
    assert info["reducer"] == reducer
    assert info["needs_coords"] is REDUCERS[reducer].needs_coords


# ---------------------------------------------------------------------------
# Seeded determinism -- stochastic reducers only
# ---------------------------------------------------------------------------

#: Reducers whose fitters explicitly document `random_state` as unused: pca
#: (deterministic PCA on classical-MDS coordinates), isomap (deterministic
#: geodesic kNN + dense eigendecomposition), and kpca_rbf (deterministic
#: kernel PCA). Each of the three fit() functions in dim_reduction/ marks its
#: random_state parameter `# pylint: disable=unused-argument  # random_state:
#: uniform contract` for exactly this reason. Asserting seed-stability on
#: these three would be tautological: they return the same coordinates for
#: *any* two random_state values, seeded the same or not, so the test would
#: not actually be exercising seeding at all.
_DETERMINISTIC_REGARDLESS_OF_SEED = frozenset({"pca", "isomap", "kpca_rbf"})

#: The rest of the registry -- t-SNE, UMAP, the SMACOF-based mds/nmds, the
#: whole LLE family, FastICA, and the random-projection control -- all pass
#: `random_state` into a stochastic scikit-learn/umap-learn estimator, so
#: reproducibility across two calls with the same seed is a real property of
#: the fitter, not a given. Verified empirically while writing this suite
#: (see the module's fixture-sized point cloud): every one of these eleven
#: reducers is exactly reproducible at `atol=1e-10` here, so none needed to
#: be excluded as "not seed-stable."
_STOCHASTIC_REDUCERS = sorted(set(REDUCERS) - _DETERMINISTIC_REGARDLESS_OF_SEED)


@pytest.mark.parametrize("reducer", _STOCHASTIC_REDUCERS)
def test_embed_2d_is_seed_stable(reducer: str, distance_matrix: np.ndarray) -> None:
    """Two calls with the same random_state must return identical coords, or
    every figure built from a stochastic reducer would be unreproducible."""
    first, _ = embed_2d(distance_matrix, reducer, random_state=42)
    second, _ = embed_2d(distance_matrix, reducer, random_state=42)
    assert np.allclose(first, second, atol=1e-10, rtol=1e-8)


# ---------------------------------------------------------------------------
# Validation -- errors embed_2d must raise before dispatching to any fitter
# ---------------------------------------------------------------------------


def test_unknown_reducer_raises_key_error(distance_matrix: np.ndarray) -> None:
    with pytest.raises(KeyError, match="unknown reducer"):
        embed_2d(distance_matrix, "not_a_reducer")


def test_non_square_matrix_raises_value_error() -> None:
    with pytest.raises(ValueError, match="square"):
        embed_2d(np.zeros((4, 5)), "pca")


def test_non_finite_matrix_raises_value_error(distance_matrix: np.ndarray) -> None:
    bad = distance_matrix.copy()
    bad[0, 1] = bad[1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        embed_2d(bad, "pca")
