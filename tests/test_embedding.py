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


def test_registry_covers_every_documented_reducer() -> None:
    assert set(REDUCERS) == {
        # distance-native
        "tsne",
        "umap",
        "isomap",
        "mds",
        "nmds",
        "spectral",
        # coordinate-requiring (classical MDS runs first)
        "pca",
        "lle",
        "lle_modified",
        "lle_hessian",
        "ltsa",
        "kpca_rbf",
        "ica",
        "random",
    }


def test_registry_entries_are_self_consistent() -> None:
    for key, spec in REDUCERS.items():
        assert isinstance(spec, ReducerSpec)
        assert spec.key == key
        assert spec.title


def test_needs_coords_matches_the_precomputed_capable_algorithms() -> None:
    """The distinction that decides whether classical MDS is applied first.

    Anything with a ``metric="precomputed"`` (or ``dissimilarity``/``affinity``)
    mode consumes D directly; everything else is fed coordinates, and pays the
    lossy double-centering step for it.
    """
    needs = {k for k, s in REDUCERS.items() if s.needs_coords}
    assert needs == {
        "pca",
        "lle",
        "lle_modified",
        "lle_hessian",
        "ltsa",
        "kpca_rbf",
        "ica",
        "random",
    }


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
        embed_2d(_gaussian_D(n=10, seed=18), "not_a_reducer")


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


# ---------------------------------------------------------------------------
# The nine reducers added for the full metric x reducer sweep
# ---------------------------------------------------------------------------


def test_spectral_affinity_inverts_distance() -> None:
    """The trap this helper exists to avoid: SpectralEmbedding wants a
    similarity, and handing it a distance matrix silently inverts the geometry
    so that the nearest points get the smallest weights."""
    from pruning_metrics.embedding import _spectral_affinity

    D = _gaussian_D(n=12, seed=31)
    affinity, sigma = _spectral_affinity(D, None)

    assert np.allclose(np.diag(affinity), 1.0)
    assert ((affinity >= 0.0) & (affinity <= 1.0)).all()
    # Monotone decreasing in distance: the closest pair outranks the furthest.
    iu = np.triu_indices_from(D, k=1)
    near, far = np.argmin(D[iu]), np.argmax(D[iu])
    assert affinity[iu][near] > affinity[iu][far]
    assert sigma > 0.0


def test_spectral_bandwidth_defaults_to_the_median_distance() -> None:
    from pruning_metrics.embedding import _spectral_affinity

    D = _gaussian_D(n=12, seed=32)
    off = D[~np.eye(12, dtype=bool)]
    _, sigma = _spectral_affinity(D, None)
    assert sigma == pytest.approx(float(np.median(off[off > 0])))


def test_spectral_records_the_effective_bandwidth() -> None:
    D = _gaussian_D(n=14, seed=33)
    _, info = embed_2d(D, "spectral")
    assert info["params"]["bandwidth"] > 0.0


def test_spectral_bandwidth_is_scale_adaptive() -> None:
    """These metrics differ by twelve orders of magnitude, so a fixed sigma
    would disconnect the graph on some of them and saturate it on others.
    Scaling D must leave the embedding's rank structure alone."""
    D = _gaussian_D(n=14, seed=34)
    from pruning_metrics.metrics.embedding_quality import shepard_rho

    plain, _ = embed_2d(D, "spectral")
    scaled, _ = embed_2d(D * 1e6, "spectral")
    assert shepard_rho(D, plain) == pytest.approx(shepard_rho(D, scaled), abs=0.02)


def test_hessian_lle_neighbourhood_floor_is_enforced() -> None:
    """sklearn requires n_neighbors > n_components*(n_components+3)/2 and
    raises otherwise; a caller passing a smaller value should get a working
    embedding and an honest record of what was actually used."""
    D = _gaussian_D(n=20, seed=35)
    coords, info = embed_2d(D, "lle_hessian", n_neighbors=3)
    assert coords.shape == (20, 2)
    assert info["params"]["n_neighbors"] > 2 * (2 + 3) // 2


def test_mds_variants_record_their_stress() -> None:
    D = _gaussian_D(n=16, seed=36)
    for reducer in ("mds", "nmds"):
        _, info = embed_2d(D, reducer, n_init=1, max_iter=60)
        assert info["stress"] >= 0.0
        assert info["n_iter"] >= 1


def test_metric_mds_beats_classical_mds_on_stress() -> None:
    """The reason to carry SMACOF alongside PCA-on-classical-MDS: it minimises
    Kruskal stress directly instead of discarding negative eigenvalues."""
    from pruning_metrics.metrics.embedding_quality import kruskal_stress1

    D = _gaussian_D(n=24, seed=37)
    smacof, _ = embed_2d(D, "mds")
    classical, _ = embed_2d(D, "pca")
    assert kruskal_stress1(D, smacof)[0] <= kruskal_stress1(D, classical)[0] + 1e-9


def test_kernel_pca_gamma_is_set_from_the_data_when_unspecified() -> None:
    D = _gaussian_D(n=15, seed=38)
    _, info = embed_2d(D, "kpca_rbf")
    assert info["params"]["gamma"] > 0.0
    _, pinned = embed_2d(D, "kpca_rbf", gamma=0.5)
    assert pinned["params"]["gamma"] == 0.5


def test_ica_whitens_to_unit_variance() -> None:
    """What actually distinguishes the ica row from the pca row. A rotation
    cannot change pairwise distances, so ICA's unmixing is invisible to every
    quality score in this repository; the whitening is not."""
    D = _gaussian_D(n=30, seed=39)
    coords, _ = embed_2d(D, "ica")
    assert np.allclose(coords.var(axis=0), 1.0, atol=0.15)


def test_random_projection_is_the_control_and_is_reproducible() -> None:
    D = _gaussian_D(n=25, seed=40)
    first, _ = embed_2d(D, "random", random_state=7)
    second, _ = embed_2d(D, "random", random_state=7)
    other, _ = embed_2d(D, "random", random_state=8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_random_projection_roughly_preserves_distances() -> None:
    """Johnson-Lindenstrauss in miniature. This is why the control is not a
    straw man: a random projection is already a mediocre embedding, so a
    reducer only earns its place by beating it."""
    from pruning_metrics.metrics.embedding_quality import shepard_rho

    D = _gaussian_D(n=40, seed=41)
    coords, _ = embed_2d(D, "random", random_state=3)
    assert shepard_rho(D, coords) > 0.3


def test_every_reducer_survives_a_wildly_scaled_matrix() -> None:
    """The chi-squared matrices run to ~1e13 while total variation stays below
    1. A sweep over all sixteen measures hits both, so no reducer may depend on
    D being order 1."""
    D = _gaussian_D(n=18, seed=42) * 1e12
    for reducer in REDUCERS:
        coords, _ = embed_2d(D, reducer)
        assert coords.shape == (18, 2), reducer
        assert np.isfinite(coords).all(), reducer
