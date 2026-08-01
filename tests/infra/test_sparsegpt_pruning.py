"""Tests for the canonical SparseGPT pruning routines.

Covers :func:`infra.runners._runner_common.sparsegpt_prune_linear` (the
per-Linear OBS-corrected pruning update) and
:func:`infra.runners._runner_common.apply_sparsegpt_pruning` (the sequential
whole-model driver). Per the v2 spec (package P1):

1. Identity Hessian degenerates to per-block-global magnitude pruning with
   surviving weights left bit-identical.
2. Achieved sparsity for a random (W, PSD H) pair lands within 2% of the
   requested ``prune_ratio`` when the block sweep spans multiple blocks.
3. The OBS correction reconstructs the calibration-data output at least as
   well as a same-sparsity pure-magnitude baseline.
4. The sequential driver reaches the target sparsity in every decoder-layer
   ``nn.Linear`` while leaving ``lm_head``-style Linears outside ``.layers``
   untouched.

These tests use tiny tensors and run in milliseconds on CPU; the whole
module is skipped if ``torch`` is unavailable (this project does not run
torch-dependent tests on the operator workstation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from infra.runners._runner_common import (  # noqa: E402
    apply_sparsegpt_pruning,
    sparsegpt_prune_linear,
)


def _block_global_magnitude_mask(
    weight: torch.Tensor, prune_ratio: float, blocksize: int
) -> torch.Tensor:
    """Reference "per-block-global magnitude" retained-mask for comparison.

    Mirrors the exact threshold-selection rule ``sparsegpt_prune_linear``
    uses (``scores <= sorted(scores)[int(n * prune_ratio)]``) but with
    ``scores = weight**2`` directly, i.e. what the SparseGPT block sweep
    degenerates to when the Hessian is a (scaled) identity -- there is no
    Hessian-weighting and no error propagation to distinguish it from plain
    magnitude pruning. Used only as an independent oracle in the identity-H
    test, not imported from the implementation under test.
    """

    in_features = weight.shape[1]
    retained = torch.ones_like(weight, dtype=torch.bool)
    for i1 in range(0, in_features, blocksize):
        i2 = min(i1 + blocksize, in_features)
        block = weight[:, i1:i2]
        scores = block**2
        threshold_index = int(scores.numel() * prune_ratio)
        threshold = torch.sort(scores.flatten())[0][threshold_index]
        retained[:, i1:i2] = scores > threshold
    return retained


def test_identity_hessian_matches_block_global_magnitude_pruning() -> None:
    """H = I ⇒ same zero pattern as block-global magnitude pruning, and
    every surviving weight is left bit-identical (no error correction is
    possible when the Hessian is isotropic: off-diagonal Hinv entries
    within and across blocks are all zero)."""

    torch.manual_seed(0)
    out_features, in_features, blocksize = 5, 20, 6
    weight = torch.randn(out_features, in_features)
    hessian = torch.eye(in_features)

    new_weight, retained_mask = sparsegpt_prune_linear(
        weight, hessian, prune_ratio=0.3, blocksize=blocksize
    )

    expected_mask = _block_global_magnitude_mask(weight, 0.3, blocksize)
    assert torch.equal(retained_mask, expected_mask)

    # Surviving weights must be exactly the original values (no correction
    # term survives when H is isotropic).
    assert torch.equal(new_weight[retained_mask], weight[retained_mask])
    assert torch.all(new_weight[~retained_mask] == 0.0)


def test_achieved_sparsity_within_two_percent_of_target() -> None:
    """Random W + random PSD H, multi-block sweep ⇒ overall sparsity is
    within 2% of ``prune_ratio``."""

    torch.manual_seed(1)
    out_features, in_features, blocksize = 16, 37, 8  # 5 blocks, uneven last one
    weight = torch.randn(out_features, in_features)

    # A random PSD Hessian via X^T X over more samples than in_features,
    # which is (almost surely) full rank and thus safely invertible.
    x = torch.randn(4 * in_features, in_features)
    hessian = x.T @ x

    prune_ratio = 0.35
    _, retained_mask = sparsegpt_prune_linear(
        weight, hessian, prune_ratio=prune_ratio, blocksize=blocksize
    )

    achieved_sparsity = 1.0 - retained_mask.float().mean().item()
    assert abs(achieved_sparsity - prune_ratio) < 0.02


def test_obs_correction_reconstructs_better_than_magnitude_baseline() -> None:
    """At equal sparsity, the Hessian-corrected weight reconstructs the
    calibration-data output at least as well as a pure-magnitude baseline
    (margin factor 1.0: no slack, since OBS correction can only help)."""

    torch.manual_seed(2)
    out_features, in_features, n_samples = 12, 24, 200

    # Correlated inputs: a random low-rank mixing matrix creates genuine
    # cross-feature correlation for the Hessian to exploit, unlike i.i.d.
    # inputs (where OBS correction would have nothing to work with).
    mixing = torch.randn(in_features, in_features // 3)
    latent = torch.randn(n_samples, in_features // 3)
    x = latent @ mixing.T + 0.1 * torch.randn(n_samples, in_features)
    hessian = (x.T @ x) / n_samples

    weight = torch.randn(out_features, in_features)
    prune_ratio = 0.4
    new_weight, retained_mask = sparsegpt_prune_linear(
        weight, hessian, prune_ratio=prune_ratio, blocksize=8
    )

    # Build a pure-magnitude baseline at the SAME achieved sparsity (not the
    # nominal prune_ratio) so the comparison is apples-to-apples.
    num_pruned = int((~retained_mask).sum())
    magnitude_baseline = weight.clone()
    flat_abs = weight.abs().flatten()
    prune_indices = torch.topk(flat_abs, k=num_pruned, largest=False).indices
    magnitude_baseline.view(-1)[prune_indices] = 0.0

    sparsegpt_error = torch.linalg.norm((new_weight - weight) @ x.T)
    magnitude_error = torch.linalg.norm((magnitude_baseline - weight) @ x.T)

    assert sparsegpt_error <= magnitude_error * 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Sequential driver (apply_sparsegpt_pruning)
# ---------------------------------------------------------------------------


class _ToyDecoderLayer(nn.Module):
    """One Linear + ReLU, returning a 1-tuple like an HF decoder layer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.act = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor, attention_mask=None):
        del attention_mask  # unused by the toy layer, threaded through unchanged
        return (self.act(self.linear(hidden_states)),)


class _ToyCausalLM(nn.Module):
    """Plain ``nn.Module`` with a top-level ``.layers`` ModuleList and a
    ``lm_head`` Linear that lives OUTSIDE it, mimicking the parts of an HF
    causal LM that :func:`apply_sparsegpt_pruning` relies on."""

    def __init__(self, dim: int, vocab: int, num_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            [_ToyDecoderLayer(dim) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            out = layer(hidden, attention_mask=attention_mask)
            hidden = out[0] if isinstance(out, tuple) else out
        return self.lm_head(hidden)


class _StubTokenizer:
    """Maps text deterministically to a fixed sequence of int token ids."""

    def __init__(self, vocab: int) -> None:
        self._vocab = vocab

    def __call__(
        self, text: str, return_tensors: str = "pt", truncation: bool = True,
        max_length: int = 2048,
    ):
        del return_tensors, truncation  # only "pt" tensors are ever needed here
        ids = [ord(ch) % self._vocab for ch in text][:max_length] or [0]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def test_sequential_driver_prunes_layers_but_not_lm_head() -> None:
    """Every decoder-layer Linear reaches ~target sparsity; lm_head (outside
    ``.layers``) is left bit-identical."""

    torch.manual_seed(3)
    dim, vocab, num_layers = 6, 24, 2
    model = _ToyCausalLM(dim=dim, vocab=vocab, num_layers=num_layers)
    model.eval()
    tokenizer = _StubTokenizer(vocab=vocab)

    lm_head_before = model.lm_head.weight.detach().clone()

    calibration_texts = [
        "the quick brown fox",
        "jumps over lazy dogs",
        "sparsegpt prunes weights",
        "hessian correction helps",
        "calibration data matters",
        "pruning ratio target half",
        "sequential layer by layer",
        "activation capture catcher",
    ]
    prune_ratio = 0.5

    apply_sparsegpt_pruning(
        model,
        tokenizer,
        calibration_texts,
        prune_ratio=prune_ratio,
        max_tokens=16,
        blocksize=3,
    )

    for index, layer in enumerate(model.layers):
        weight = layer.linear.weight.data
        sparsity = (weight == 0).float().mean().item()
        assert abs(sparsity - prune_ratio) < 0.2, (
            f"layer {index} sparsity {sparsity} far from target {prune_ratio}"
        )

    assert torch.equal(model.lm_head.weight.detach(), lm_head_before)


def test_sequential_driver_zero_ratio_is_noop() -> None:
    """``prune_ratio <= 0`` must leave every weight unchanged (the level-0
    baseline case; see :func:`sparsegpt_prune_linear`'s Notes on why 0.0
    cannot simply be forwarded to the per-Linear routine)."""

    torch.manual_seed(4)
    dim, vocab, num_layers = 4, 10, 2
    model = _ToyCausalLM(dim=dim, vocab=vocab, num_layers=num_layers)
    tokenizer = _StubTokenizer(vocab=vocab)

    before = [layer.linear.weight.detach().clone() for layer in model.layers]

    apply_sparsegpt_pruning(
        model, tokenizer, ["hello world"], prune_ratio=0.0, max_tokens=8
    )

    for layer, snapshot in zip(model.layers, before):
        assert torch.equal(layer.linear.weight.detach(), snapshot)


class _ToyCache:
    """Minimal stand-in for ``transformers.cache_utils.DynamicCache``.

    Only tracks how many times ``update`` has been called, which is all
    this regression test needs: it lets us detect the exact failure mode
    the reviewed bug produced on a real HF model -- the SAME captured
    ``(args, kwargs)`` (cache object included) being replayed through a
    layer more than once. A real ``DynamicCache.update`` would instead
    silently grow the key/value length past what the captured
    ``attention_mask``/``cache_position`` describe; raising here makes the
    test fail loudly instead of relying on shape-mismatch luck.
    """

    def __init__(self) -> None:
        self.calls = 0

    def update(self) -> None:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError(
                "cache reused across replay: the same captured (args, "
                "kwargs) was forwarded through a layer more than once "
                "with a live KV cache -- apply_sparsegpt_pruning must "
                "disable use_cache during calibration to prevent this."
            )


class _CacheAwareDecoderLayer(nn.Module):
    """Decoder layer whose signature includes ``past_key_value``/
    ``use_cache``, mirroring the real HF decoder-layer calling convention
    that the reviewed KV-cache bug depends on."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        past_key_value=None,
        use_cache: bool = False,
    ):
        del attention_mask
        if use_cache and past_key_value is not None:
            past_key_value.update()
        return (self.linear(hidden_states),)


class _CacheAwareCausalLM(nn.Module):
    """Toy causal LM whose top-level forward constructs a fresh KV cache
    and threads it (plus ``use_cache``) into every decoder layer whenever
    ``self.config.use_cache`` is truthy -- mirroring the real
    ``Qwen2Model.forward`` convention (``use_cache = ... or
    self.config.use_cache``, then ``DynamicCache()`` construction) that the
    reviewed bug exploits."""

    class _Config:
        def __init__(self) -> None:
            self.use_cache = True  # Qwen2's real default

    def __init__(self, dim: int, vocab: int, num_layers: int) -> None:
        super().__init__()
        self.config = self._Config()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            [_CacheAwareDecoderLayer(dim) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask=None, use_cache=None):
        hidden = self.embed(input_ids)
        use_cache = self.config.use_cache if use_cache is None else use_cache
        cache = _ToyCache() if use_cache else None
        for layer in self.layers:
            out = layer(
                hidden,
                attention_mask=attention_mask,
                past_key_value=cache,
                use_cache=use_cache,
            )
            hidden = out[0] if isinstance(out, tuple) else out
        return self.lm_head(hidden)


def test_sequential_driver_disables_kv_cache_during_calibration() -> None:
    """Regression test for the reviewed KV-cache bug: a model whose
    ``config.use_cache`` defaults to ``True`` (like Qwen2) must not crash
    when its decoder layers are replayed twice per level with the SAME
    captured ``(args, kwargs)`` -- once for Hessian collection, once to
    advance to the next layer's inputs -- because a live cache object would
    have its update path invoked twice (see ``_ToyCache.update``'s
    ``RuntimeError`` above, which fires precisely on that second call).
    ``config.use_cache`` must also be restored to its original value once
    pruning completes, since callers keep using the same in-memory model
    for inference afterwards."""

    torch.manual_seed(5)
    dim, vocab, num_layers = 6, 24, 2
    model = _CacheAwareCausalLM(dim=dim, vocab=vocab, num_layers=num_layers)
    model.eval()
    tokenizer = _StubTokenizer(vocab=vocab)

    assert model.config.use_cache is True  # sanity: real Qwen2-like default

    apply_sparsegpt_pruning(
        model,
        tokenizer,
        [
            "the quick brown fox",
            "jumps over lazy dogs",
            "hessian correction helps",
        ],
        prune_ratio=0.5,
        max_tokens=16,
        blocksize=3,
    )

    assert model.config.use_cache is True

    for layer in model.layers:
        sparsity = (layer.linear.weight.data == 0).float().mean().item()
        assert abs(sparsity - 0.5) < 0.2


def test_sparsegpt_prune_linear_rejects_bad_inputs() -> None:
    """Shape / range validation raises ValueError, not an opaque crash
    deep inside the Cholesky call."""

    weight = torch.randn(4, 8)
    hessian = torch.eye(8)

    with pytest.raises(ValueError):
        sparsegpt_prune_linear(weight, torch.eye(7), prune_ratio=0.5)  # wrong side
    with pytest.raises(ValueError):
        sparsegpt_prune_linear(weight, hessian, prune_ratio=1.0)  # out of range
    with pytest.raises(ValueError):
        sparsegpt_prune_linear(weight, hessian, prune_ratio=0.5, blocksize=0)
    with pytest.raises(ValueError):
        sparsegpt_prune_linear(weight.reshape(-1), hessian, prune_ratio=0.5)  # not 2D
