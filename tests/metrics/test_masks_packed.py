"""Unit tests for the bit-packed Jaccard path in pruning_metrics.metrics.masks.

The packed path exists purely as a memory/speed optimisation over
:func:`jaccard_distance`, so almost every test here is an *equivalence* test:
whatever the unpacked implementation returns, the packed one must return
bit-for-bit the same float. These tests deliberately avoid ``torch`` (unlike
``test_masks.py``, which is gated on it) since nothing in this path needs a
model.
"""

from __future__ import annotations

import numpy as np
import pytest

from pruning_metrics.metrics.masks import (
    jaccard_distance,
    jaccard_distance_packed,
    jaccard_matrix_packed,
    load_digest,
    load_digest_packed,
    save_digest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_digest(
    layers: dict[str, int], *, seed: int, p_true: float = 0.5
) -> dict[str, np.ndarray]:
    """Build a ``{layer: bool_array}`` digest with the given per-layer lengths."""
    rng = np.random.default_rng(seed)
    return {name: rng.random(n) < p_true for name, n in layers.items()}


def _write(digest: dict[str, np.ndarray], path) -> str:
    save_digest(digest, path)
    return str(path)


# Lengths chosen so some layers are byte-aligned and others are not; the
# non-aligned ones exercise packbits' zero padding, which the packed Jaccard
# must not count as retained positions.
_LAYERS = {
    "model.layers.0.q_proj": 1000,
    "model.layers.0.k_proj": 7,
    "model.layers.1.v_proj": 64,
}


# ---------------------------------------------------------------------------
# Equivalence with the unpacked implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed_b", [1, 2, 3, 4, 5])
def test_packed_matches_unpacked(tmp_path, seed_b: int) -> None:
    """The whole point: identical distances, including the padding bits."""
    a = _random_digest(_LAYERS, seed=0)
    b = _random_digest(_LAYERS, seed=seed_b)
    path_a = _write(a, tmp_path / "a.npz")
    path_b = _write(b, tmp_path / f"b{seed_b}.npz")

    expected = jaccard_distance(a, b)
    actual = jaccard_distance_packed(
        load_digest_packed(path_a), load_digest_packed(path_b)
    )
    assert actual == expected


def test_padding_bits_are_not_counted(tmp_path) -> None:
    """An all-True digest of non-multiple-of-8 length is still distance 0 to itself.

    If ``packbits`` padding leaked into the union count, identical digests
    would report a non-zero distance here.
    """
    digest = {"layer": np.ones(7, dtype=bool)}
    path = _write(digest, tmp_path / "d.npz")
    packed = load_digest_packed(path)
    assert jaccard_distance_packed(packed, packed) == pytest.approx(0.0)


def test_packed_identical_is_zero(tmp_path) -> None:
    path = _write(_random_digest(_LAYERS, seed=40), tmp_path / "d.npz")
    packed = load_digest_packed(path)
    assert jaccard_distance_packed(packed, packed) == pytest.approx(0.0)


def test_packed_disjoint_is_one(tmp_path) -> None:
    mask = _random_digest({"layer": 1000}, seed=41)
    inverse = {"layer": ~mask["layer"]}
    a = load_digest_packed(_write(mask, tmp_path / "a.npz"))
    b = load_digest_packed(_write(inverse, tmp_path / "b.npz"))
    assert jaccard_distance_packed(a, b) == pytest.approx(1.0)


def test_packed_empty_union_is_zero(tmp_path) -> None:
    """Matches jaccard_distance's convention rather than dividing by zero."""
    empty = {"layer": np.zeros(100, dtype=bool)}
    a = load_digest_packed(_write(empty, tmp_path / "a.npz"))
    b = load_digest_packed(_write(empty, tmp_path / "b.npz"))
    assert jaccard_distance_packed(a, b) == 0.0


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


def test_packed_is_eight_times_smaller(tmp_path) -> None:
    """The memory win that motivates this path at all."""
    digest = _random_digest({"layer": 80_000}, seed=3)
    packed = load_digest_packed(_write(digest, tmp_path / "d.npz"))
    unpacked_bytes = sum(v.nbytes for v in load_digest(tmp_path / "d.npz").values())
    assert packed.bits.nbytes <= unpacked_bytes // 8 + 8


def test_packed_buffer_is_uint64_viewable(tmp_path) -> None:
    """jaccard_distance_packed views the buffer as uint64; padding must allow it."""
    digest = _random_digest({"a": 3, "b": 5}, seed=4)
    packed = load_digest_packed(_write(digest, tmp_path / "d.npz"))
    assert packed.bits.dtype == np.uint8
    assert packed.bits.size % 8 == 0
    assert packed.bits.view(np.uint64).size == packed.bits.size // 8


def test_layout_records_layer_lengths(tmp_path) -> None:
    packed = load_digest_packed(
        _write(_random_digest(_LAYERS, seed=5), tmp_path / "d.npz")
    )
    assert dict(packed.layout) == _LAYERS
    # Sorted-name order is what makes two variants' buffers positionally comparable.
    assert [name for name, _ in packed.layout] == sorted(_LAYERS)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_packed_mismatched_keys_raises(tmp_path) -> None:
    a = load_digest_packed(_write({"layer.a": np.ones(10, bool)}, tmp_path / "a.npz"))
    b = load_digest_packed(_write({"layer.b": np.ones(10, bool)}, tmp_path / "b.npz"))
    with pytest.raises(ValueError, match="key sets differ"):
        jaccard_distance_packed(a, b)


def test_packed_mismatched_lengths_raises(tmp_path) -> None:
    a = load_digest_packed(_write({"layer": np.ones(10, bool)}, tmp_path / "a.npz"))
    b = load_digest_packed(_write({"layer": np.ones(11, bool)}, tmp_path / "b.npz"))
    with pytest.raises(ValueError, match="length mismatch"):
        jaccard_distance_packed(a, b)


# ---------------------------------------------------------------------------
# Tiled matrix construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tile", [1, 2, 3, 7, 100])
def test_matrix_matches_pairwise_regardless_of_tile(tmp_path, tile: int) -> None:
    """Tiling is a memory strategy only; it must not change any entry."""
    digests = [_random_digest(_LAYERS, seed=100 + i) for i in range(7)]
    paths = [_write(d, tmp_path / f"d{i}.npz") for i, d in enumerate(digests)]

    matrix = jaccard_matrix_packed(paths, tile=tile)

    expected = np.zeros((len(digests), len(digests)))
    for i in range(len(digests)):
        for j in range(i + 1, len(digests)):
            expected[i, j] = expected[j, i] = jaccard_distance(digests[i], digests[j])
    assert np.array_equal(matrix, expected)


def test_matrix_is_symmetric_with_zero_diagonal(tmp_path) -> None:
    paths = [
        _write(_random_digest(_LAYERS, seed=200 + i), tmp_path / f"d{i}.npz")
        for i in range(5)
    ]
    matrix = jaccard_matrix_packed(paths, tile=2)
    assert np.array_equal(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0.0)


def test_matrix_empty_and_single(tmp_path) -> None:
    assert jaccard_matrix_packed([]).shape == (0, 0)
    path = _write(_random_digest(_LAYERS, seed=300), tmp_path / "d.npz")
    assert jaccard_matrix_packed([path]).shape == (1, 1)


def test_matrix_rejects_bad_tile(tmp_path) -> None:
    with pytest.raises(ValueError, match="tile must be >= 1"):
        jaccard_matrix_packed([], tile=0)
