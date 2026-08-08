"""Unit tests for pruning_metrics.embedding."""

from __future__ import annotations

import numpy as np
import pytest

from pruning_metrics.embedding import (
    REDUCERS,
    ReducerSpec,
    classical_mds_coords,
    complete_submatrix_indices,
    embed_2d,
    mds_spectrum,
)

pytest.importorskip("sklearn")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _euclidean_D(X: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance matrix of a coordinate array."""
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff**2).sum(-1))


def _gaussian_D(n: int = 60, dim: int = 5, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _euclidean_D(rng.normal(size=(n, dim)))


def _pairwise(Y: np.ndarray) -> np.ndarray:
    return _euclidean_D(Y)


# ---------------------------------------------------------------------------
# classical_mds_coords
# ---------------------------------------------------------------------------


def test_mds_reproduces_euclidean_distances_exactly() -> None:
    """For Euclidean D, classical MDS is lossless: distances come back exactly."""
    D = _gaussian_D(n=40, dim=4, seed=1)
    X = classical_mds_coords(D)
    assert np.allclose(_pairwise(X), D, atol=1e-8)


def test_mds_recovers_intrinsic_dimension() -> None:
    """Points drawn in 3-D need only 3 MDS dimensions, not n-1."""
    D = _gaussian_D(n=50, dim=3, seed=2)
    assert classical_mds_coords(D).shape == (50, 3)


def test_mds_pads_degenerate_matrix_to_two_columns() -> None:
    """An all-zero D has no positive eigenvalues; 2-D reducers still need 2 columns."""
    X = classical_mds_coords(np.zeros((5, 5)))
    assert X.shape == (5, 2)
    assert np.allclose(X, 0.0)


def test_mds_drops_negative_eigenvalue_dimensions() -> None:
    """Non-Euclidean D loses dimensions -- the effect that distorts PCA/LLE."""
    D = _gaussian_D(n=30, dim=6, seed=3) ** 2  # squaring breaks the triangle inequality
    np.fill_diagonal(D, 0.0)
    assert classical_mds_coords(D).shape[1] < 29


# ---------------------------------------------------------------------------
# mds_spectrum
# ---------------------------------------------------------------------------


def test_spectrum_euclidean_has_no_negative_mass() -> None:
    spec = mds_spectrum(_gaussian_D(n=40, dim=4, seed=4))
    assert spec["neg_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert spec["n_pos"] == 4


def test_spectrum_flags_non_euclidean_matrix() -> None:
    D = _gaussian_D(n=30, dim=6, seed=5) ** 2
    np.fill_diagonal(D, 0.0)
    assert mds_spectrum(D)["neg_ratio"] > 0.01


def test_spectrum_var_2d_is_one_for_planar_data() -> None:
    """Data that genuinely lives in a plane loses nothing in a 2-D picture."""
    spec = mds_spectrum(_gaussian_D(n=40, dim=2, seed=6))
    assert spec["var_2d"] == pytest.approx(1.0)
    assert spec["n_pos"] == 2


def test_spectrum_var_2d_below_one_for_high_dimensional_data() -> None:
    assert mds_spectrum(_gaussian_D(n=60, dim=8, seed=7))["var_2d"] < 0.6


def test_spectrum_handles_all_zero_matrix() -> None:
    """No positive mass to divide by; must return 0 rather than raise or NaN."""
    spec = mds_spectrum(np.zeros((4, 4)))
    assert spec["neg_ratio"] == 0.0
    assert spec["var_2d"] == 0.0


# ---------------------------------------------------------------------------
# complete_submatrix_indices
# ---------------------------------------------------------------------------


def test_complete_submatrix_keeps_everything_when_fully_observed() -> None:
    D = _gaussian_D(n=20, dim=3, seed=30)
    assert np.array_equal(complete_submatrix_indices(D), np.arange(20))


def test_complete_submatrix_drops_an_all_zero_row() -> None:
    """The exact failure in the v2 data: one variant with no data at all."""
    D = _gaussian_D(n=20, dim=3, seed=31)
    D[7, :] = 0.0
    D[:, 7] = 0.0
    assert np.array_equal(complete_submatrix_indices(D), np.delete(np.arange(20), 7))


def test_complete_submatrix_drops_multiple_dead_rows() -> None:
    D = _gaussian_D(n=20, dim=3, seed=32)
    for row in (3, 11, 15):
        D[row, :] = 0.0
        D[:, row] = 0.0
    assert np.array_equal(
        complete_submatrix_indices(D), np.delete(np.arange(20), [3, 11, 15])
    )


def test_complete_submatrix_uses_counts_when_given() -> None:
    """With counts, a genuinely zero distance is not mistaken for missing data."""
    D = _gaussian_D(n=10, dim=3, seed=33)
    D[2, 5] = D[5, 2] = 0.0  # two identical models, legitimately at distance 0
    counts = np.ones((10, 10), dtype=int)
    assert np.array_equal(complete_submatrix_indices(D, counts=counts), np.arange(10))
    # Without counts the same matrix looks like it has a missing pair.
    assert complete_submatrix_indices(D).size < 10


def test_complete_submatrix_result_has_no_missing_pairs() -> None:
    rng = np.random.default_rng(34)
    D = _gaussian_D(n=30, dim=3, seed=35)
    for i, j in rng.integers(0, 30, size=(12, 2)):
        if i != j:
            D[i, j] = D[j, i] = 0.0
    idx = complete_submatrix_indices(D)
    sub = D[np.ix_(idx, idx)]
    np.fill_diagonal(sub, 1.0)
    assert not (sub == 0.0).any()


def test_complete_submatrix_of_all_zeros_is_empty_or_singleton() -> None:
    assert complete_submatrix_indices(np.zeros((5, 5))).size <= 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_covers_the_five_documented_reducers() -> None:
    assert set(REDUCERS) == {"pca", "tsne", "umap", "isomap", "lle"}


def test_registry_entries_are_self_consistent() -> None:
    for key, spec in REDUCERS.items():
        assert isinstance(spec, ReducerSpec)
        assert spec.key == key
        assert spec.title


def test_only_pca_and_lle_need_coordinates() -> None:
    """The distinction that decides whether classical MDS is applied first."""
    needs = {k for k, s in REDUCERS.items() if s.needs_coords}
    assert needs == {"pca", "lle"}


# ---------------------------------------------------------------------------
# embed_2d — contract shared by every reducer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reducer", sorted(REDUCERS))
def test_embed_returns_two_columns(reducer: str) -> None:
    D = _gaussian_D(n=60, dim=4, seed=8)
    coords, info = embed_2d(D, reducer)
    assert coords.shape == (60, 2)
    assert np.isfinite(coords).all()
    assert info["reducer"] == reducer
    assert info["n"] == 60
    assert "params" in info and "versions" in info


@pytest.mark.parametrize("reducer", sorted(REDUCERS))
def test_embed_is_deterministic(reducer: str) -> None:
    """Every reducer must be reproducible, or figures change between runs."""
    D = _gaussian_D(n=50, dim=3, seed=9)
    first, _ = embed_2d(D, reducer)
    second, _ = embed_2d(D, reducer)
    assert np.allclose(first, second)


@pytest.mark.parametrize("reducer", sorted(REDUCERS))
def test_embed_records_mds_dims_only_when_coordinates_were_derived(
    reducer: str,
) -> None:
    D = _gaussian_D(n=40, dim=3, seed=10)
    _, info = embed_2d(D, reducer)
    assert ("mds_dims" in info) is REDUCERS[reducer].needs_coords


# ---------------------------------------------------------------------------
# embed_2d — per-reducer specifics
# ---------------------------------------------------------------------------


def test_pca_preserves_euclidean_geometry_of_planar_data() -> None:
    """PCA on MDS coords of planar data is a rigid motion: distances survive."""
    rng = np.random.default_rng(11)
    D = _euclidean_D(rng.normal(size=(40, 2)))
    coords, info = embed_2d(D, "pca")
    assert np.allclose(_pairwise(coords), D, atol=1e-8)
    assert sum(info["explained_variance_ratio"]) == pytest.approx(1.0)


def test_tsne_reports_kl_divergence() -> None:
    _, info = embed_2d(_gaussian_D(n=50, dim=3, seed=12), "tsne")
    assert info["kl_divergence"] >= 0.0


def test_tsne_perplexity_is_clamped_below_n() -> None:
    """The default perplexity of 30 exceeds n here; sklearn would raise."""
    _, info = embed_2d(_gaussian_D(n=15, dim=3, seed=13), "tsne")
    assert info["params"]["perplexity"] < 15


def test_lle_reports_reconstruction_error() -> None:
    _, info = embed_2d(_gaussian_D(n=50, dim=3, seed=14), "lle")
    assert info["reconstruction_error"] >= 0.0


def test_isomap_records_the_neighbourhood_it_actually_used() -> None:
    """n_neighbors may be bumped to connect the kNN graph; info must show it."""
    D = _gaussian_D(n=60, dim=3, seed=15)
    _, info = embed_2d(D, "isomap")
    assert info["params"]["n_neighbors"] >= 15


def test_isomap_grows_neighbourhood_to_connect_split_clusters() -> None:
    """Two far-apart clusters need a wider neighbourhood than the default."""
    rng = np.random.default_rng(16)
    X = np.vstack([rng.normal(size=(30, 2)), rng.normal(size=(30, 2)) + 500.0])
    _, info = embed_2d(_euclidean_D(X), "isomap", n_neighbors=3)
    # A 3-NN graph cannot bridge the gap; the connectivity check must widen it.
    assert info["params"]["n_neighbors"] > 3


def test_overrides_take_precedence_over_defaults() -> None:
    _, info = embed_2d(_gaussian_D(n=50, dim=3, seed=17), "lle", n_neighbors=7)
    assert info["params"]["n_neighbors"] == 7


# ---------------------------------------------------------------------------
# embed_2d — validation
# ---------------------------------------------------------------------------


def test_unknown_reducer_raises() -> None:
    with pytest.raises(KeyError, match="unknown reducer"):
        embed_2d(_gaussian_D(n=10, seed=18), "mds")


def test_non_square_matrix_raises() -> None:
    with pytest.raises(ValueError, match="square"):
        embed_2d(np.zeros((4, 5)), "pca")


def test_non_symmetric_matrix_raises() -> None:
    D = _gaussian_D(n=10, seed=19)
    D[0, 1] += 1.0
    with pytest.raises(ValueError, match="symmetric"):
        embed_2d(D, "pca")


def test_non_finite_matrix_raises() -> None:
    D = _gaussian_D(n=10, seed=20)
    D[0, 1] = D[1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        embed_2d(D, "pca")
