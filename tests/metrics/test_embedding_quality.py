"""Unit tests for pruning_metrics.metrics.embedding_quality.

The load-bearing test here is :func:`test_matches_sklearn_trustworthiness`:
scikit-learn ships a reference implementation of trustworthiness, so our
rank conventions can be pinned against it exactly. :func:`continuity` then
rides on the same verified kernel with its two arguments swapped.
"""

from __future__ import annotations

import numpy as np
import pytest

from pruning_metrics.metrics.embedding_quality import (
    baseline_distances,
    continuity,
    effective_k,
    embedding_quality,
    kruskal_stress1,
    linear_r2,
    shepard_rho,
    trustworthiness,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _euclidean_D(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff**2).sum(-1))


def _blobs(n: int = 60, dim: int = 5, *, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, dim))


# ---------------------------------------------------------------------------
# Agreement with scikit-learn's reference implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 3, 5, 12, 25])
def test_matches_sklearn_trustworthiness(k: int) -> None:
    """Our rank conventions must be scikit-learn's, exactly."""
    sklearn_manifold = pytest.importorskip("sklearn.manifold")
    X = _blobs(n=70, dim=6, seed=1)
    D = _euclidean_D(X)
    Y = np.random.default_rng(2).normal(size=(70, 2))

    expected = sklearn_manifold.trustworthiness(
        D, Y, n_neighbors=k, metric="precomputed"
    )
    assert trustworthiness(D, Y, k) == pytest.approx(expected, rel=0, abs=1e-12)


def test_matches_sklearn_on_a_structured_embedding() -> None:
    """Also pin the agreement when the embedding is genuinely good, not random."""
    sklearn_manifold = pytest.importorskip("sklearn.manifold")
    X = _blobs(n=80, dim=2, seed=3)
    D = _euclidean_D(X)
    Y = X + np.random.default_rng(4).normal(scale=0.01, size=X.shape)

    expected = sklearn_manifold.trustworthiness(
        D, Y, n_neighbors=10, metric="precomputed"
    )
    assert trustworthiness(D, Y, 10) == pytest.approx(expected, rel=0, abs=1e-12)


# ---------------------------------------------------------------------------
# Sanity poles: perfect and random embeddings
# ---------------------------------------------------------------------------


def test_identity_embedding_is_perfect() -> None:
    """Embedding planar data as itself preserves everything."""
    X = _blobs(n=50, dim=2, seed=5)
    D = _euclidean_D(X)
    assert trustworthiness(D, X, 12) == pytest.approx(1.0)
    assert continuity(D, X, 12) == pytest.approx(1.0)
    stress, alpha = kruskal_stress1(D, X)
    assert stress == pytest.approx(0.0, abs=1e-12)
    assert alpha == pytest.approx(1.0)
    assert shepard_rho(D, X) == pytest.approx(1.0)


def test_random_embedding_scores_near_chance() -> None:
    X = _blobs(n=120, dim=6, seed=6)
    D = _euclidean_D(X)
    Y = np.random.default_rng(7).normal(size=(120, 2))
    assert 0.35 < trustworthiness(D, Y, 12) < 0.65
    assert abs(shepard_rho(D, Y)) < 0.2


def test_good_embedding_beats_random_embedding() -> None:
    """The metrics must actually discriminate, not just return plausible numbers."""
    X = _blobs(n=100, dim=2, seed=8)
    D = _euclidean_D(X)
    good = X + np.random.default_rng(9).normal(scale=0.02, size=X.shape)
    bad = np.random.default_rng(10).normal(size=(100, 2))

    assert trustworthiness(D, good, 12) > trustworthiness(D, bad, 12)
    assert continuity(D, good, 12) > continuity(D, bad, 12)
    assert shepard_rho(D, good) > shepard_rho(D, bad)
    assert kruskal_stress1(D, good)[0] < kruskal_stress1(D, bad)[0]


# ---------------------------------------------------------------------------
# Trustworthiness vs continuity are genuinely different statistics
# ---------------------------------------------------------------------------


def test_trustworthiness_and_continuity_differ_on_asymmetric_distortion() -> None:
    """Collapsing one cluster tears true neighbourhoods without inventing many."""
    rng = np.random.default_rng(11)
    X = np.vstack([rng.normal(size=(40, 2)), rng.normal(size=(40, 2)) + 20.0])
    D = _euclidean_D(X)
    Y = X.copy()
    Y[40:] = Y[40:].mean(axis=0)  # squash the second cluster to a single point

    assert trustworthiness(D, Y, 5) != pytest.approx(continuity(D, Y, 5))


def test_continuity_is_not_symmetric_in_its_arguments() -> None:
    X = _blobs(n=60, dim=4, seed=12)
    D = _euclidean_D(X)
    Y = np.random.default_rng(13).normal(size=(60, 2))
    assert trustworthiness(D, Y, 8) != pytest.approx(continuity(D, Y, 8))


# ---------------------------------------------------------------------------
# Stress-1 and its optimal scaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [0.001, 0.5, 3.0, 1000.0])
def test_stress_is_invariant_to_embedding_scale(scale: float) -> None:
    """The reason alpha is fitted: t-SNE and LLE output arbitrary units."""
    X = _blobs(n=50, dim=2, seed=14)
    D = _euclidean_D(X)
    stress, alpha = kruskal_stress1(D, X * scale)
    assert stress == pytest.approx(0.0, abs=1e-10)
    assert alpha == pytest.approx(1.0 / scale, rel=1e-9)


def test_stress_without_scaling_would_have_been_fooled() -> None:
    """Guards the design choice: unscaled residuals blow up with the units."""
    X = _blobs(n=40, dim=2, seed=15)
    D = _euclidean_D(X)
    Y = X * 100.0
    unscaled = np.sqrt(
        np.sum(
            (D[np.triu_indices(40, 1)] - _euclidean_D(Y)[np.triu_indices(40, 1)]) ** 2
        )
        / np.sum(D[np.triu_indices(40, 1)] ** 2)
    )
    assert unscaled > 50.0
    assert kruskal_stress1(D, Y)[0] == pytest.approx(0.0, abs=1e-10)


def test_collapsed_embedding_has_maximal_stress() -> None:
    D = _euclidean_D(_blobs(n=30, dim=3, seed=16))
    stress, alpha = kruskal_stress1(D, np.zeros((30, 2)))
    assert stress == 1.0
    assert alpha == 0.0


def test_stress_of_all_zero_distances_is_nan() -> None:
    stress, alpha = kruskal_stress1(np.zeros((5, 5)), _blobs(n=5, dim=2, seed=17))
    assert np.isnan(stress) and np.isnan(alpha)


# ---------------------------------------------------------------------------
# Shepard rho
# ---------------------------------------------------------------------------


def test_shepard_rho_is_invariant_to_monotone_rescaling() -> None:
    """Rank-based by design, so squaring the embedding's distances changes nothing."""
    X = _blobs(n=50, dim=2, seed=18)
    D = _euclidean_D(X)
    assert shepard_rho(D, X) == pytest.approx(shepard_rho(D, X * 7.0))


def test_shepard_rho_of_constant_distances_is_nan() -> None:
    assert np.isnan(shepard_rho(np.zeros((6, 6)), _blobs(n=6, dim=2, seed=19)))
    D = _euclidean_D(_blobs(n=6, dim=3, seed=20))
    assert np.isnan(shepard_rho(D, np.zeros((6, 2))))


# ---------------------------------------------------------------------------
# effective_k clamping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "n", "expected"),
    [(12, 232, 12), (12, 20, 9), (100, 50, 24), (1, 100, 1), (12, 2, 0)],
)
def test_effective_k(k: int, n: int, expected: int) -> None:
    assert effective_k(k, n) == expected


def test_large_k_is_clamped_instead_of_raising() -> None:
    """sklearn raises for k >= n/2; we clamp, because n varies across benchmarks."""
    X = _blobs(n=20, dim=3, seed=21)
    D = _euclidean_D(X)
    assert np.isfinite(trustworthiness(D, X, 100))
    assert embedding_quality(D, X, k=100)["k"] == 9


def test_tiny_input_returns_nan_rather_than_raising() -> None:
    D = _euclidean_D(_blobs(n=2, dim=2, seed=22))
    assert np.isnan(trustworthiness(D, np.zeros((2, 2)), 12))
    assert np.isnan(continuity(D, np.zeros((2, 2)), 12))


# ---------------------------------------------------------------------------
# embedding_quality aggregation
# ---------------------------------------------------------------------------


def test_embedding_quality_agrees_with_individual_functions() -> None:
    X = _blobs(n=60, dim=4, seed=23)
    D = _euclidean_D(X)
    Y = np.random.default_rng(24).normal(size=(60, 2))

    result = embedding_quality(D, Y, k=12)
    assert result["trustworthiness"] == pytest.approx(trustworthiness(D, Y, 12))
    assert result["continuity"] == pytest.approx(continuity(D, Y, 12))
    assert result["stress1"] == pytest.approx(kruskal_stress1(D, Y)[0])
    assert result["alpha"] == pytest.approx(kruskal_stress1(D, Y)[1])
    assert result["shepard_rho"] == pytest.approx(shepard_rho(D, Y))
    assert result["n"] == 60 and result["k"] == 12


# ---------------------------------------------------------------------------
# Predictive validity: baseline_distances + linear_r2
# ---------------------------------------------------------------------------


def test_baseline_distances_are_zero_at_the_reference() -> None:
    Y = _blobs(n=20, dim=2, seed=30)
    d = baseline_distances(Y, 0)
    assert d[0] == 0.0
    assert d.shape == (20,)
    assert (d[1:] > 0).all()


def test_baseline_distances_match_manual_norm() -> None:
    Y = _blobs(n=15, dim=2, seed=31)
    expected = np.sqrt(((Y - Y[3]) ** 2).sum(axis=1))
    assert np.allclose(baseline_distances(Y, 3), expected)


def test_baseline_distances_are_translation_invariant() -> None:
    """Embeddings are only defined up to position; the measure must not care."""
    Y = _blobs(n=20, dim=2, seed=32)
    assert np.allclose(baseline_distances(Y, 0), baseline_distances(Y + 17.0, 0))


def test_baseline_distances_rejects_bad_index() -> None:
    with pytest.raises(IndexError):
        baseline_distances(_blobs(n=5, dim=2, seed=33), 99)


def test_linear_r2_is_one_for_an_exact_line() -> None:
    x = np.arange(20, dtype=float)
    out = linear_r2(x, 3.0 * x + 5.0)
    assert out["r2"] == pytest.approx(1.0)
    assert out["slope"] == pytest.approx(3.0)
    assert out["intercept"] == pytest.approx(5.0)
    assert out["n"] == 20


def test_linear_r2_matches_squared_pearson() -> None:
    """Same convention as 04_metric_spaces so the numbers are comparable."""
    rng = np.random.default_rng(34)
    x = rng.normal(size=50)
    y = 0.4 * x + rng.normal(scale=0.8, size=50)
    r = np.corrcoef(x, y)[0, 1]
    assert linear_r2(x, y)["r2"] == pytest.approx(r**2)


def test_linear_r2_is_sign_blind_but_r_is_not() -> None:
    x = np.arange(20, dtype=float)
    down = linear_r2(x, -2.0 * x)
    assert down["r2"] == pytest.approx(1.0)
    assert down["r"] == pytest.approx(-1.0)
    assert down["slope"] == pytest.approx(-2.0)


def test_linear_r2_drops_non_finite_pairs() -> None:
    """The v2 grid has variants with no performance record; they must not poison the fit."""
    x = np.arange(10, dtype=float)
    y = 2.0 * x
    y[3] = np.nan
    x_bad = x.copy()
    x_bad[7] = np.inf
    out = linear_r2(x_bad, y)
    assert out["n"] == 8
    assert out["r2"] == pytest.approx(1.0)


def test_linear_r2_needs_three_points() -> None:
    out = linear_r2([1.0, 2.0], [1.0, 2.0])
    assert out["n"] == 2
    assert np.isnan(out["r2"])


def test_linear_r2_of_constant_input_is_nan() -> None:
    assert np.isnan(linear_r2(np.ones(10), np.arange(10.0))["r2"])
    assert np.isnan(linear_r2(np.arange(10.0), np.ones(10))["r2"])


def test_linear_r2_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        linear_r2(np.arange(5.0), np.arange(6.0))


def test_radial_fit_recovers_a_planted_relationship() -> None:
    """End-to-end: degradation proportional to embedding radius scores near 1."""
    rng = np.random.default_rng(35)
    Y = rng.normal(size=(60, 2))
    Y[0] = 0.0  # baseline at the origin
    radius = baseline_distances(Y, 0)
    degradation = 1.5 * radius + rng.normal(scale=0.01, size=60)
    assert linear_r2(radius, degradation)["r2"] > 0.99


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_non_square_distance_matrix_raises() -> None:
    with pytest.raises(ValueError, match="square"):
        trustworthiness(np.zeros((3, 4)), np.zeros((3, 2)), 1)


def test_mismatched_embedding_length_raises() -> None:
    D = _euclidean_D(_blobs(n=10, dim=2, seed=25))
    with pytest.raises(ValueError, match="must have shape"):
        trustworthiness(D, np.zeros((9, 2)), 3)


def test_non_finite_inputs_raise() -> None:
    D = _euclidean_D(_blobs(n=10, dim=2, seed=26))
    bad_D = D.copy()
    bad_D[0, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        trustworthiness(bad_D, np.zeros((10, 2)), 3)

    Y = np.zeros((10, 2))
    Y[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        trustworthiness(D, Y, 3)
