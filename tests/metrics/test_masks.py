"""Unit tests for pruning_metrics.metrics.masks."""

from __future__ import annotations

import numpy as np
import pytest

from pruning_metrics.metrics.masks import (
    extract_pruning_masks,
    jaccard_distance,
    load_digest,
    load_packed_masks,
    make_mask_digest,
    save_digest,
    save_packed_masks,
)

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_mask(n: int, *, seed: int, p_true: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(n) < p_true


# ---------------------------------------------------------------------------
# Pack / unpack roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 7, 8, 9, 65, 1000])
def test_pack_unpack_roundtrip_odd_lengths(tmp_path, n: int) -> None:
    """Odd lengths exercise packbits' padding-to-a-multiple-of-8 behavior."""
    mask = _random_mask(n, seed=n)
    masks = {"layer.a": mask}
    # Use an irregular 2-D shape too, to check shape restoration.
    shaped = _random_mask(12, seed=100).reshape(3, 4)
    masks["layer.b"] = shaped

    path = tmp_path / "masks.npz"
    save_packed_masks(masks, path)
    restored = load_packed_masks(path)

    assert set(restored) == set(masks)
    for name in masks:
        assert restored[name].shape == masks[name].shape
        assert restored[name].dtype == bool
        np.testing.assert_array_equal(restored[name], masks[name])


def test_pack_unpack_roundtrip_flat_random(tmp_path) -> None:
    mask = _random_mask(9_999, seed=1).reshape(101, 99)
    masks = {"model.layers.0.self_attn.q_proj": mask}
    path = tmp_path / "masks.npz"
    save_packed_masks(masks, path)
    restored = load_packed_masks(path)
    np.testing.assert_array_equal(restored["model.layers.0.self_attn.q_proj"], mask)


# ---------------------------------------------------------------------------
# Digest determinism / positional comparability
# ---------------------------------------------------------------------------


def test_digest_determinism_across_independent_calls() -> None:
    mask = _random_mask(10_000, seed=2)
    masks = {"model.layers.0.mlp.gate_proj": mask}

    digest_1 = make_mask_digest(masks, fraction=1 / 32, seed=42)
    digest_2 = make_mask_digest(masks, fraction=1 / 32, seed=42)

    assert set(digest_1) == set(digest_2)
    for name in digest_1:
        np.testing.assert_array_equal(digest_1[name], digest_2[name])


def test_digest_length_matches_fraction() -> None:
    n = 10_000
    mask = _random_mask(n, seed=3)
    digest = make_mask_digest({"layer": mask}, fraction=1 / 32, seed=0)
    import math

    assert digest["layer"].shape[0] == math.ceil(n / 32)


def test_digest_jaccard_approximates_full_jaccard() -> None:
    """Digest jaccard should be close to full-mask jaccard for large layers."""
    n = 100_000
    mask_a = {"layer": _random_mask(n, seed=10, p_true=0.5)}
    # Flip ~20% of bits to build a related-but-different mask.
    rng = np.random.default_rng(11)
    flips = rng.random(n) < 0.2
    mask_b = {"layer": mask_a["layer"] ^ flips}

    full_jaccard = jaccard_distance(mask_a, mask_b)

    digest_a = make_mask_digest(mask_a, fraction=1 / 32, seed=7)
    digest_b = make_mask_digest(mask_b, fraction=1 / 32, seed=7)
    digest_jaccard = jaccard_distance(digest_a, digest_b)

    assert abs(full_jaccard - digest_jaccard) < 0.02


def test_digest_positions_are_shared_across_variants() -> None:
    """Same (shapes, fraction, seed) -> same indices, regardless of values."""
    n = 5_000
    mask_a = {"layer": _random_mask(n, seed=20)}
    mask_b = {"layer": _random_mask(n, seed=21)}  # unrelated values, same shape

    digest_a = make_mask_digest(mask_a, fraction=1 / 16, seed=5)
    digest_b = make_mask_digest(mask_b, fraction=1 / 16, seed=5)

    assert digest_a["layer"].shape == digest_b["layer"].shape


# ---------------------------------------------------------------------------
# Digest save/load roundtrip
# ---------------------------------------------------------------------------


def test_digest_save_load_roundtrip(tmp_path) -> None:
    mask = _random_mask(777, seed=30)
    digest = make_mask_digest({"layer.x": mask}, fraction=1 / 8, seed=1)
    path = tmp_path / "digest.npz"
    save_digest(digest, path)
    restored = load_digest(path)

    assert set(restored) == set(digest)
    for name in digest:
        assert restored[name].shape == digest[name].shape
        np.testing.assert_array_equal(restored[name], digest[name])


# ---------------------------------------------------------------------------
# Jaccard edge cases
# ---------------------------------------------------------------------------


def test_jaccard_identical_is_zero() -> None:
    masks = {"layer": _random_mask(1000, seed=40)}
    assert jaccard_distance(masks, masks) == pytest.approx(0.0)


def test_jaccard_disjoint_is_one() -> None:
    mask = _random_mask(1000, seed=41, p_true=0.5)
    a = {"layer": mask}
    b = {"layer": ~mask}
    assert jaccard_distance(a, b) == pytest.approx(1.0)


def test_jaccard_mismatched_keys_raises() -> None:
    a = {"layer.a": _random_mask(10, seed=1)}
    b = {"layer.b": _random_mask(10, seed=2)}
    with pytest.raises(ValueError):
        jaccard_distance(a, b)


def test_jaccard_mismatched_lengths_raises() -> None:
    a = {"layer": _random_mask(10, seed=1)}
    b = {"layer": _random_mask(20, seed=2)}
    with pytest.raises(ValueError):
        jaccard_distance(a, b)


def test_jaccard_empty_union_is_zero() -> None:
    a = {"layer": np.zeros(10, dtype=bool)}
    b = {"layer": np.zeros(10, dtype=bool)}
    assert jaccard_distance(a, b) == 0.0


# ---------------------------------------------------------------------------
# extract_pruning_masks scope/exclude rules
# ---------------------------------------------------------------------------


class _ToyDecoderLayer(nn.Module):
    """One decoder layer: a prunable Linear plus a norm layer with a weight."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Linear(8, 8, bias=False)
        self.input_layernorm = nn.LayerNorm(8)


class _ToyModelInner(nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_ToyDecoderLayer() for _ in range(n_layers))
        self.embed_tokens = nn.Embedding(16, 8)


class _ToyModel(nn.Module):
    """Mimics the HF decoder-only shape: model.model.layers[i], model.lm_head."""

    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.model = _ToyModelInner(n_layers)
        self.lm_head = nn.Linear(8, 16, bias=False)


@pytest.fixture
def toy_model() -> _ToyModel:
    model = _ToyModel(n_layers=2)
    with torch.no_grad():
        # Zero out some entries so masks are non-trivial.
        model.model.layers[0].self_attn.weight[0, :] = 0.0
        model.model.layers[1].self_attn.weight[:, 0] = 0.0
    return model


def test_extract_pruning_masks_scope_and_exclude(toy_model: _ToyModel) -> None:
    masks = extract_pruning_masks(toy_model)

    # Only in-scope (".layers.") Linear leaves are present.
    assert set(masks) == {
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
    }

    # lm_head excluded via default `exclude`, embed_tokens/layernorm out of
    # scope (not under ".layers." or not 2-D / not Linear-shaped).
    assert "lm_head" not in masks
    assert not any("embed_tokens" in name for name in masks)
    assert not any("layernorm" in name for name in masks)

    # Shapes and zeroed entries match what the fixture set up.
    assert masks["model.layers.0.self_attn"].shape == (8, 8)
    assert not masks["model.layers.0.self_attn"][0, :].any()
    assert masks["model.layers.0.self_attn"][1:, :].all()
    assert not masks["model.layers.1.self_attn"][:, 0].any()


def test_extract_pruning_masks_custom_exclude(toy_model: _ToyModel) -> None:
    masks = extract_pruning_masks(toy_model, exclude=("layers.0",))
    assert set(masks) == {"model.layers.1.self_attn"}


def test_extract_pruning_masks_no_exclude_still_scopes_to_layers(
    toy_model: _ToyModel,
) -> None:
    masks = extract_pruning_masks(toy_model, exclude=())
    # lm_head is outside ".layers." scope regardless of `exclude`.
    assert set(masks) == {
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
    }


def test_extract_pruning_masks_bfloat16_weights() -> None:
    """bf16 weights (numpy-unrepresentable) must still yield bool masks."""

    torch = pytest.importorskip("torch")
    from torch import nn

    class _Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 3, bias=False)

    class _Wrap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_Layer()])

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _Wrap()

    model = _Model().to(torch.bfloat16)
    with torch.no_grad():
        model.model.layers[0].proj.weight[0, :2] = 0.0

    masks = extract_pruning_masks(model)
    ((name, mask),) = masks.items()
    assert "proj" in name
    assert mask.dtype == np.bool_
    assert mask.shape == (3, 4)
    assert not mask[0, 0] and not mask[0, 1]
    assert mask[1:].all()
