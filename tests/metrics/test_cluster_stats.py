"""Unit tests for pruning_metrics.metrics.cluster_stats."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import squareform, pdist

from pruning_metrics.metrics.cluster_stats import (
    ari_vs_labels,
    label_permutation_pvalue,
    mantel,
    partial_mantel,
    silhouette_by_label,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dist_from_coords(coords: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix from an (n, d) coordinate array."""
    return squareform(pdist(coords, metric="euclidean"))


def _two_gaussian_blobs(
    n_per_blob: int = 20,
    dim: int = 4,
    separation: float = 12.0,
    noise: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated Gaussian blobs; returns (dist_matrix, labels)."""
    rng = np.random.default_rng(seed)
    blob_a = rng.normal(loc=0.0, scale=noise, size=(n_per_blob, dim))
    blob_b = rng.normal(loc=separation, scale=noise, size=(n_per_blob, dim))
    coords = np.vstack([blob_a, blob_b])
    labels = np.array([0] * n_per_blob + [1] * n_per_blob)
    return _dist_from_coords(coords), labels


# ---------------------------------------------------------------------------
# mantel
# ---------------------------------------------------------------------------


class TestMantel:
    def test_recovers_planted_correlation(self) -> None:
        rng = np.random.default_rng(1)
        n = 30
        latent = rng.normal(size=(n, 3))
        d1 = _dist_from_coords(latent + rng.normal(scale=0.05, size=(n, 3)))
        d2 = _dist_from_coords(latent + rng.normal(scale=0.05, size=(n, 3)))

        r, p = mantel(d1, d2, permutations=999, seed=0)

        assert r > 0.8
        assert p < 0.01

    def test_independent_matrices_not_significant(self) -> None:
        rng = np.random.default_rng(2)
        n = 25
        d1 = _dist_from_coords(rng.normal(size=(n, 3)))
        d2 = _dist_from_coords(rng.normal(size=(n, 3)))

        r, p = mantel(d1, d2, permutations=999, seed=0)

        assert p > 0.05

    def test_rejects_shape_mismatch(self) -> None:
        d1 = np.zeros((4, 4))
        d2 = np.zeros((5, 5))
        with pytest.raises(ValueError):
            mantel(d1, d2)

    def test_rejects_asymmetric(self) -> None:
        d1 = np.array([[0.0, 1.0], [0.5, 0.0]])
        d2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(ValueError):
            mantel(d1, d2)

    def test_tolerates_tiny_asymmetry(self) -> None:
        rng = np.random.default_rng(3)
        n = 10
        base = _dist_from_coords(rng.normal(size=(n, 2)))
        noisy = base.copy()
        noisy[0, 1] += 1e-10  # within tolerance, should not raise
        r, p = mantel(base, noisy, permutations=99, seed=0)
        assert r > 0.99


# ---------------------------------------------------------------------------
# partial_mantel
# ---------------------------------------------------------------------------


class TestPartialMantel:
    def test_partial_r_near_zero_when_fully_explained_by_control(self) -> None:
        rng = np.random.default_rng(4)
        n = 30
        # control drives both d1 and d2 with independent noise on top;
        # once control is regressed out there should be ~no residual
        # relationship between d1 and d2.
        #
        # NOTE: the noise scale (0.5) is deliberately a substantial
        # fraction of control's own scale. Near-zero noise here would
        # leave d1/d2's rank order almost identical to control's, so the
        # spearman rank-transform produces near-tied ranks whose
        # residuals are dominated by tie-breaking artifacts rather than
        # genuine signal — an artifact of the rank transform on
        # near-degenerate inputs, not of partial_mantel itself.
        control_coords = rng.normal(size=(n, 3))
        control = _dist_from_coords(control_coords)
        d1 = control + rng.normal(scale=0.5, size=control.shape)
        d1 = (d1 + d1.T) / 2
        d2 = control + rng.normal(scale=0.5, size=control.shape)
        d2 = (d2 + d2.T) / 2

        r, p = partial_mantel(d1, d2, control, permutations=499, seed=0)

        assert abs(r) < 0.3
        assert p > 0.05

    def test_partial_r_significant_with_shared_component_beyond_control(self) -> None:
        rng = np.random.default_rng(5)
        n = 30
        control_coords = rng.normal(size=(n, 3))
        control = _dist_from_coords(control_coords)

        shared_coords = rng.normal(size=(n, 3))
        shared = _dist_from_coords(shared_coords)

        d1 = control + 3.0 * shared + rng.normal(scale=0.01, size=control.shape)
        d1 = (d1 + d1.T) / 2
        d2 = control + 3.0 * shared + rng.normal(scale=0.01, size=control.shape)
        d2 = (d2 + d2.T) / 2

        r, p = partial_mantel(d1, d2, control, permutations=499, seed=0)

        assert r > 0.5
        assert p < 0.01

    def test_rejects_shape_mismatch(self) -> None:
        d1 = np.zeros((4, 4))
        d2 = np.zeros((4, 4))
        control = np.zeros((5, 5))
        with pytest.raises(ValueError):
            partial_mantel(d1, d2, control)


# ---------------------------------------------------------------------------
# silhouette_by_label / ari_vs_labels / label_permutation_pvalue
# ---------------------------------------------------------------------------


class TestLabelStats:
    def test_silhouette_high_for_separated_blobs(self) -> None:
        dist, labels = _two_gaussian_blobs()
        s = silhouette_by_label(dist, labels)
        assert s > 0.5

    def test_ari_perfect_for_separated_blobs(self) -> None:
        dist, labels = _two_gaussian_blobs()
        ari = ari_vs_labels(dist, labels)
        assert ari == pytest.approx(1.0)

    def test_ari_respects_linkage_param(self) -> None:
        dist, labels = _two_gaussian_blobs()
        ari = ari_vs_labels(dist, labels, linkage="complete")
        assert ari == pytest.approx(1.0)

    def test_permutation_pvalue_significant_for_separated_blobs(self) -> None:
        dist, labels = _two_gaussian_blobs()
        observed, p = label_permutation_pvalue(
            dist, labels, stat="silhouette", permutations=499, seed=0
        )
        assert observed > 0.5
        assert p < 0.01

    def test_permutation_pvalue_ari_stat(self) -> None:
        dist, labels = _two_gaussian_blobs()
        observed, p = label_permutation_pvalue(
            dist, labels, stat="ari", permutations=499, seed=0
        )
        assert observed == pytest.approx(1.0)
        assert p < 0.01

    def test_permutation_pvalue_not_significant_for_random_labels(self) -> None:
        rng = np.random.default_rng(6)
        n = 30
        dist = _dist_from_coords(rng.normal(size=(n, 4)))
        labels = rng.integers(0, 2, size=n)
        # Guard against the (rare) degenerate draw of a single label.
        if len(np.unique(labels)) < 2:
            labels[0] = 1 - labels[0]

        _, p = label_permutation_pvalue(
            dist, labels, stat="silhouette", permutations=499, seed=0
        )
        assert p > 0.05

    def test_invalid_stat_raises(self) -> None:
        dist, labels = _two_gaussian_blobs()
        with pytest.raises(ValueError):
            label_permutation_pvalue(dist, labels, stat="bogus")

    def test_label_length_mismatch_raises(self) -> None:
        dist, _ = _two_gaussian_blobs()
        with pytest.raises(ValueError):
            silhouette_by_label(dist, np.array([0, 1, 0]))
        with pytest.raises(ValueError):
            ari_vs_labels(dist, np.array([0, 1, 0]))
