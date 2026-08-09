"""One-shot SparseGPT unstructured pruning for the GPU runners.

Implements Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be
Accurately Pruned in One-Shot" (2023): a per-Linear blockwise
Hessian-aware pruning kernel (:func:`sparsegpt_prune_linear`) and the
sequential layer-by-layer driver (:func:`apply_sparsegpt_pruning`).

Extracted verbatim from ``_runner_common.py``, which re-exports the public
names so runners and tests keep importing every pruning entry point from
there. Like the rest of the runner helpers, this module is torch-free at
import time: ``torch`` is imported inside each function.
"""

from __future__ import annotations

# NOTE: pylint's cross-file similarity checker flags the standard HF
# tokenizer(...) calibration call, which WANDA (in _runner_common) and
# SparseGPT both legitimately make; the duplication predates the split
# and is not worth a shared helper for six lines.
# pylint: disable=duplicate-code

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - annotation-only import
    import torch

#: Same named logger as ``_runner_common`` — ``logging.getLogger`` returns
#: the singleton, so log output is byte-identical to the pre-split module.
LOGGER = logging.getLogger("pruning_runner")


def sparsegpt_prune_linear(
    weight: "torch.Tensor",
    hessian: "torch.Tensor",
    prune_ratio: float,
    *,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """One-shot SparseGPT unstructured pruning for a single Linear weight.

    Implements the sequential, Hessian-corrected pruning update from
    Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be Accurately
    Pruned in One-Shot" (2023), mirroring the reference ``fasterprune``
    routine's unstructured branch (``prune_n = prune_m = 0``). Column blocks
    of width ``blocksize`` are swept left to right; within each block,
    weights are zeroed by a single block-global magnitude score (weighted by
    the inverse-Hessian diagonal, i.e. Optimal Brain Surgeon) and the
    resulting reconstruction error is propagated onto not-yet-processed
    columns via the Cholesky factor of the damped Hessian inverse, so
    surviving weights partially compensate for the removed ones.

    Design: selection is **block-global**, not per-row -- every weight in
    the ``(out_features, blocksize)`` block competes on one shared quantile
    threshold. This is a deliberate mismatch with WANDA's per-row rule
    (:func:`apply_wanda_pruning`); it is the reference SparseGPT behavior,
    not a simplification of it, and must not be "corrected" to per-row.

    Parameters
    ----------
    weight : torch.Tensor
        ``(out_features, in_features)`` weight matrix, any float dtype. Not
        mutated: an independent fp32 copy is pruned and returned.
    hessian : torch.Tensor
        ``(in_features, in_features)`` layer-input second-moment matrix
        (e.g. ``X^T X`` accumulated over calibration activations), assumed
        symmetric positive semi-definite. Not mutated: an independent fp32
        copy is damped internally. Columns whose diagonal entry is exactly
        zero ("dead" input channels that never fired during calibration)
        are treated as already fully pruned -- their Hessian diagonal is set
        to 1 before damping and the corresponding weight column is zeroed,
        matching the reference implementation's handling of unused channels
        (leaving it untouched would make the Hessian singular).
    prune_ratio : float
        Target fraction of each block's weights to zero, in ``[0, 1)``.
    blocksize : int, default=128
        Column-block width for the OBS sweep. Smaller values approach
        per-weight-optimal reconstruction at higher runtime cost; 128 is
        the reference default. Values >= ``in_features`` collapse to a
        single block.
    percdamp : float, default=0.01
        Fraction of the mean Hessian diagonal added to every diagonal entry
        before inversion (Levenberg-Marquardt-style damping), which keeps
        the Cholesky factorization numerically stable when the Hessian is
        near-singular (e.g. too few calibration tokens).

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor)
        ``(new_weight, retained_mask)``. ``new_weight`` is the pruned,
        Hessian-corrected weight: fp32, same shape as ``weight``.
        ``retained_mask`` is a bool tensor of the same shape, ``True``
        where the weight was kept (not zeroed by the pruning selection).
        Note ``retained_mask`` reflects the *selection* decision, not a
        post-hoc ``new_weight != 0`` check -- a kept weight can, in rare
        cases, be corrected to a value very close to (though essentially
        never exactly) zero.

    Raises
    ------
    ValueError
        If ``weight`` is not 2-D, ``hessian`` is not square with side
        length equal to ``weight.shape[1]``, ``prune_ratio`` is outside
        ``[0, 1)``, or ``blocksize`` is not a positive integer.

    Notes
    -----
    Time complexity is dominated by the ``O(in_features**3)`` Cholesky
    factorization plus ``O(out_features * in_features**2 / blocksize)`` for
    the block sweep; space is ``O(in_features**2 + out_features *
    in_features)``. All arithmetic runs in fp32 regardless of the input
    dtypes, matching the reference implementation's use of fp32 for
    numerical stability.

    A ``prune_ratio`` of exactly ``0.0`` is **not** a safe no-op. The
    reference threshold index is ``int(n * prune_ratio)`` into the
    ascending-sorted score array, which at ``prune_ratio=0.0`` still
    selects (and zeros, via ``scores <= threshold``) the single
    lowest-scoring weight per block. This mirrors the reference
    implementation exactly rather than special-casing it away. Callers
    that need a true identity pass at level 0 (as
    :func:`apply_sparsegpt_pruning` and this project's runner both do) must
    skip calling this function entirely rather than call it with
    ``prune_ratio=0.0``.

    Examples
    --------
    >>> import torch
    >>> w = torch.arange(1.0, 9.0).reshape(2, 4)
    >>> h = torch.eye(4)
    >>> new_w, mask = sparsegpt_prune_linear(w, h, prune_ratio=0.5, blocksize=4)
    >>> int(mask.sum())  # scores <= the block-global threshold are pruned,
    ...                  # so the tied threshold element goes too: 3 retained
    3
    """

    import torch

    if weight.dim() != 2:
        raise ValueError(
            "weight must be 2-D (out_features, in_features), got shape "
            f"{tuple(weight.shape)}."
        )
    out_features, in_features = weight.shape
    if (
        hessian.dim() != 2
        or hessian.shape[0] != hessian.shape[1]
        or hessian.shape[0] != in_features
    ):
        raise ValueError(
            f"hessian must be square with side length {in_features} "
            f"(weight.shape[1]), got shape {tuple(hessian.shape)}."
        )
    if not 0.0 <= prune_ratio < 1.0:
        raise ValueError(f"prune_ratio must be in [0, 1), got {prune_ratio}.")
    if blocksize < 1:
        raise ValueError(f"blocksize must be a positive integer, got {blocksize}.")
    del out_features  # only used for the shape check above

    W = weight.detach().clone().to(dtype=torch.float32)
    H = hessian.detach().clone().to(dtype=torch.float32)

    # Design: a "dead" input channel (zero Hessian diagonal) never fired
    # during calibration, so its weight column contributes nothing to the
    # layer's output but would leave H singular. Force it out of the
    # weight and neutralize its Hessian row/column, exactly as the
    # reference implementation does, before damping.
    dead = torch.diag(H) == 0
    if torch.any(dead):
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

    # Damping (Levenberg-Marquardt-style) keeps the Cholesky factorization
    # well-conditioned even when the calibration Hessian is near-singular.
    damp = percdamp * torch.mean(torch.diag(H))
    diag_idx = torch.arange(in_features, device=H.device)
    H[diag_idx, diag_idx] += damp

    # Hinv is the UPPER Cholesky factor of H^-1: Hinv[j, j]**2 gives the
    # per-weight OBS reconstruction-error scale, and Hinv[j, j+1:] carries
    # the correction coefficients propagated onto later columns.
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)
    del H

    pruned = torch.zeros_like(W, dtype=torch.bool)

    for i1 in range(0, in_features, blocksize):
        i2 = min(i1 + blocksize, in_features)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Hinv1 = Hinv[i1:i2, i1:i2]
        Err1 = torch.zeros_like(W1)

        # Design: block-global threshold (reference behavior) -- every
        # weight in this (out_features, count) block competes on ONE
        # quantile cutoff, unlike WANDA's per-row rule. Using a manual
        # sort+index (rather than torch.quantile) also sidesteps
        # quantile's 2**24-element ceiling for large blocks.
        scores = W1**2 / torch.diag(Hinv1).reshape(1, -1) ** 2
        threshold_index = int(scores.numel() * prune_ratio)
        threshold = torch.sort(scores.flatten())[0][threshold_index]
        mask1 = scores <= threshold

        for j in range(count):
            # Design: `w = W1[:, j]` is a VIEW into W1, not a copy. `err`
            # MUST be computed from it before `W1[:, j] = q` writes through
            # that same view -- computing them in the other order (as a
            # naive transcription of "q = ...; W1[:, j] = q; err = (w - q)"
            # might suggest) silently reads back the just-zeroed value for
            # every unmasked column too, making err == 0 always and
            # disabling the Hessian error-correction entirely while still
            # *looking* correct (mask and shapes all check out). Caught by
            # cross-checking sparsegpt_prune_linear's output against the
            # closed-form OBS-optimal correction for a fixed mask.
            w = W1[:, j]
            d = Hinv1[j, j]
            q = w.clone()
            q[mask1[:, j]] = 0.0
            err = (w - q) / d
            W1[:, j] = q
            if j + 1 < count:
                W1[:, j + 1 :] -= err.unsqueeze(1) * Hinv1[j, j + 1 :].unsqueeze(0)
            Err1[:, j] = err

        W[:, i1:i2] = W1
        pruned[:, i1:i2] = mask1
        if i2 < in_features:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    retained_mask = ~pruned
    return W, retained_mask


def _decoder_layers(model: Any) -> Any:
    """Return the model's decoder-layer container, duck-typed.

    Prefers ``model.model.layers`` (the HF ``*ForCausalLM`` wrapper
    convention, e.g. ``Qwen2ForCausalLM.model.layers``) and falls back to
    ``model.layers`` for callers that pass the inner transformer stack
    directly (used by this package's toy-model tests).

    Raises
    ------
    AttributeError
        If neither ``model.model.layers`` nor ``model.layers`` exists.
    """

    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(
        "Model exposes neither `model.layers` nor `layers`; cannot locate "
        "decoder layers for SparseGPT pruning."
    )


def _advance_sparsegpt_layer_inputs(
    layer: Any,
    captured: list[tuple[tuple[Any, ...], dict[str, Any]]],
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Replay each captured ``(args, kwargs)`` through *layer* to build the
    next layer's inputs.

    Assumes -- per the HF decoder-layer calling convention -- that the
    hidden-states tensor is always the first positional argument; the
    ``out[0] if isinstance(out, tuple) else out`` idiom below handles both
    HF layers (which return ``(hidden_states, ...)`` tuples) and plain
    ``nn.Module`` layers that just return the tensor.

    Parameters
    ----------
    layer:
        The (already pruned) decoder layer to replay samples through.
    captured:
        Per-sample ``(args, kwargs)`` pairs captured for the layer this
        replay is currently feeding, i.e. the *previous* layer's output.

    Returns
    -------
    list of (tuple, dict)
        One entry per input sample, with the hidden-states positional
        argument replaced by this layer's output; all other args/kwargs
        (e.g. attention masks, position ids, rotary embeddings) pass
        through unchanged.

    Notes
    -----
    Runs under ``torch.no_grad()`` -- this replay is only used to produce
    the next layer's calibration inputs, never for gradient computation.
    """

    import torch

    advanced: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    with torch.no_grad():
        for args, kwargs in captured:
            output = layer(*args, **kwargs)
            hidden = output[0] if isinstance(output, tuple) else output
            if args:
                new_args = (hidden,) + tuple(args[1:])
                new_kwargs = kwargs
            else:
                # Defensive fallback for architectures that pass
                # hidden_states as a keyword only; the spec's HF-derived
                # convention (positional arg 0) is the expected path.
                new_args = ()
                new_kwargs = dict(kwargs)
                new_kwargs["hidden_states"] = hidden
            advanced.append((new_args, new_kwargs))
    return advanced


def apply_sparsegpt_pruning(
    model: Any,
    tokenizer: Any,
    calibration_texts: list[str],
    prune_ratio: float,
    *,
    max_tokens: int = 2048,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> None:
    """In-place sequential SparseGPT pruning over a causal LM's decoder layers.

    Runs the standard SparseGPT calibration procedure layer by layer: capture
    every calibration sample's input to decoder layer 0, then for each layer
    in order accumulate an input Hessian per ``nn.Linear`` from a replay of
    those samples, prune every Linear in the layer with
    :func:`sparsegpt_prune_linear`, and replay the samples once more through
    the now-pruned layer to obtain the next layer's inputs. Only one layer's
    Hessians are held in memory at a time.

    Decoder layers are located via :func:`_decoder_layers` (``model.model.layers``
    if present, else ``model.layers``); ``lm_head`` and the embedding layers
    live outside that container and are therefore never pruned.

    Parameters
    ----------
    model : Any
        A causal LM (or a duck-typed stand-in for tests) whose decoder
        layers are discoverable per :func:`_decoder_layers`. Mutated in
        place: every in-scope ``nn.Linear.weight`` is overwritten.
    tokenizer : Any
        Callable tokenizer used as ``tokenizer(text, return_tensors="pt",
        truncation=True, max_length=max_tokens)``, matching
        :func:`collect_wanda_activation_stats`'s calling convention.
    calibration_texts : list of str
        Calibration prompts. Must be non-empty. The SAME texts should be
        used across pruning levels for a given run so the calibration
        Hessians (and thus the produced masks) are comparable.
    prune_ratio : float
        Target fraction of each Linear's weights to zero, in ``[0, 1)``.
        ``prune_ratio <= 0.0`` is a documented no-op (the whole-model
        equivalent of skipping :func:`sparsegpt_prune_linear` entirely --
        see that function's Notes on why ``0.0`` cannot simply be forwarded).
    max_tokens : int, default=2048
        Per-prompt truncation length for the calibration forward passes.
    blocksize : int, default=128
        Forwarded to :func:`sparsegpt_prune_linear` for every Linear.
    percdamp : float, default=0.01
        Forwarded to :func:`sparsegpt_prune_linear` for every Linear.

    Raises
    ------
    ValueError
        If ``calibration_texts`` is empty, or if ``prune_ratio`` is not in
        ``[0, 1)``.
    RuntimeError
        If no decoder-layer-0 inputs could be captured (e.g. the model's
        forward pass never reaches ``layers[0]``), which would otherwise
        silently leave every Linear unpruned.
    AttributeError
        Propagated from :func:`_decoder_layers` if the model does not
        expose a recognizable decoder-layer container.

    Notes
    -----
    Peak memory is bounded to one layer's Hessians (``num_linears_per_layer
    * in_features**2`` floats) rather than the whole model's, which is what
    makes this tractable for a 7B-72B parameter model on a single GPU. When
    CUDA is available, ``torch.cuda.empty_cache()`` runs between layers so
    freed Hessian memory is actually returned to the allocator's free pool
    before the next layer's hooks start accumulating.

    This function performs no I/O and returns ``None``; all effects are the
    in-place weight mutation described above.

    ``model.config.use_cache`` (when present) is forced to ``False`` for the
    duration of the capture-and-replay procedure and restored to its
    original value on exit, including on exceptions raised from within the
    procedure. This is required for correctness, not just performance: a
    real HF causal LM's decoder layer accepts a mutable KV cache as
    ``past_key_value``/``use_cache`` kwargs, and every captured sample here
    is replayed through its layer twice per level (once for Hessian
    collection, once to advance to the next layer's inputs) with the SAME
    captured kwargs -- a live cache object would be mutated twice, producing
    a key/value length inconsistent with the also-captured attention mask
    and either raising deep inside attention or silently corrupting every
    layer past the first.
    """

    import torch
    from torch import nn

    if not calibration_texts:
        raise ValueError("calibration_texts must be non-empty for SparseGPT pruning.")
    if not 0.0 <= prune_ratio < 1.0:
        raise ValueError(f"prune_ratio must be in [0, 1), got {prune_ratio}.")
    if prune_ratio <= 0.0:
        return

    # Imported lazily from _runner_common at call time: that module
    # re-exports this one's public names, so a top-level import back into
    # it would be circular.
    from infra.runners._runner_common import ensure_src_on_path

    ensure_src_on_path()
    from pruning_metrics.evals.coding.teacher_forcing import resolve_input_device

    layers = _decoder_layers(model)
    LOGGER.info(
        "SparseGPT: pruning %d decoder layers to %.1f%% sparsity over %d "
        "calibration texts.",
        len(layers),
        prune_ratio * 100.0,
        len(calibration_texts),
    )

    # Design: Qwen2 (and most HF causal LMs) default `config.use_cache` to
    # True, which makes the plain `model(**encoded)` forward used below to
    # capture layer-0's inputs construct a live KV cache (e.g. a
    # `DynamicCache`) and thread it into the captured kwargs as
    # `past_key_value`. Every captured sample is later replayed through its
    # layer TWICE with that SAME (args, kwargs) pair -- once for Hessian
    # collection, once to produce the next layer's inputs (see
    # `_advance_sparsegpt_layer_inputs`) -- so a live cache object would
    # have its update path invoked twice, doubling its key/value length
    # while the captured attention_mask/cache_position still describe the
    # original sequence length (RuntimeError, or silently wrong hidden
    # states fed into every subsequent layer). Forcing `use_cache=False`
    # for the whole capture+replay procedure means the captured kwargs
    # never contain a live cache in the first place, so replaying them any
    # number of times is side-effect-free. The prior value is saved and
    # restored (rather than left False) because callers keep using the
    # same in-memory model object for inference after pruning completes;
    # duck-typed callers without a `config.use_cache` attribute (e.g. this
    # module's own toy-model tests) are left untouched via the sentinel
    # below.
    _NO_CONFIG_USE_CACHE = object()
    model_config = getattr(model, "config", None)
    original_use_cache = (
        getattr(model_config, "use_cache", _NO_CONFIG_USE_CACHE)
        if model_config is not None
        else _NO_CONFIG_USE_CACHE
    )
    if original_use_cache is not _NO_CONFIG_USE_CACHE:
        model_config.use_cache = False

    try:
        # Design: a private exception (rather than a sentinel return value) is
        # the standard "activation capture" idiom for one-shot calibration --
        # it lets us abort the model's forward pass the instant layer 0 has
        # been called, without computing the remaining N-1 layers and the
        # (often huge) lm_head projection for every calibration sample.
        class _CatcherAbort(Exception):
            """Private control-flow signal: layer-0 input captured, abort forward."""

        captured: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        class _Catcher(nn.Module):
            """Temporary stand-in for ``layers[0]`` that records its call and bails."""

            def __init__(self, wrapped: nn.Module) -> None:
                super().__init__()
                self.wrapped = wrapped  # kept only so state_dict/repr stay sane

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                captured.append((args, kwargs))
                raise _CatcherAbort()

        # Phase 1: capture every calibration sample's input to decoder layer 0.
        original_layer0 = layers[0]
        layers[0] = _Catcher(original_layer0)
        target_device = resolve_input_device(model)
        try:
            with torch.no_grad():
                for text in calibration_texts:
                    encoded = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_tokens,
                    )
                    encoded = {
                        key: value.to(target_device) for key, value in encoded.items()
                    }
                    try:
                        model(**encoded)
                    except _CatcherAbort:
                        pass
        finally:
            layers[0] = original_layer0

        if not captured:
            raise RuntimeError(
                "Failed to capture any decoder-layer-0 inputs; check that "
                "calibration_texts is non-empty and that the model's forward "
                "pass actually calls layers[0]."
            )
        LOGGER.info(
            "Captured %d calibration samples' inputs to decoder layer 0.",
            len(captured),
        )

        cuda_available = torch.cuda.is_available()

        # Phase 2: sequential per-layer Hessian collection + pruning + replay.
        for layer_index, layer in enumerate(layers):
            linears = {
                name: module
                for name, module in layer.named_modules()
                if isinstance(module, nn.Linear)
            }
            if not linears:
                LOGGER.warning(
                    "Decoder layer %d has no nn.Linear submodules; skipping.",
                    layer_index,
                )
                continue

            hessians: dict[str, "torch.Tensor"] = {}
            counts: dict[str, int] = {}
            hooks = []

            def _build_hook(layer_name: str, target_module: nn.Module):
                def _hook(module, inputs, output):  # noqa: ANN001 - nn.Module signature
                    del module, output
                    if not inputs or not isinstance(inputs[0], torch.Tensor):
                        return
                    x = inputs[0].detach()
                    flat = x.reshape(-1, x.shape[-1]).to(
                        dtype=torch.float32, device=target_module.weight.device
                    )
                    local_h = flat.T @ flat
                    if layer_name in hessians:
                        hessians[layer_name] += local_h
                    else:
                        hessians[layer_name] = local_h
                    counts[layer_name] = counts.get(layer_name, 0) + int(flat.shape[0])

                return _hook

            for name, module in linears.items():
                hooks.append(module.register_forward_hook(_build_hook(name, module)))

            try:
                with torch.no_grad():
                    for args, kwargs in captured:
                        layer(*args, **kwargs)
            finally:
                for handle in hooks:
                    handle.remove()

            for name in linears:
                count = counts.get(name, 0)
                if count == 0:
                    LOGGER.warning(
                        "Linear %r in layer %d received no calibration "
                        "activations; leaving it unpruned.",
                        name,
                        layer_index,
                    )
                    continue
                hessians[name] /= float(count)

            LOGGER.info(
                "Layer %d: pruning %d Linear module(s).", layer_index, len(linears)
            )
            for name, module in linears.items():
                if name not in hessians:
                    continue
                original_dtype = module.weight.data.dtype
                new_weight, _retained_mask = sparsegpt_prune_linear(
                    module.weight.data.float(),
                    hessians[name],
                    prune_ratio,
                    blocksize=blocksize,
                    percdamp=percdamp,
                )
                module.weight.data.copy_(
                    new_weight.to(dtype=original_dtype, device=module.weight.device)
                )
                # Free this Linear's Hessian as soon as it's consumed so peak
                # memory stays at "one layer's worth", not "one layer's worth
                # plus everything already processed".
                del hessians[name]

            del hessians
            if cuda_available:
                torch.cuda.empty_cache()

            # Phase 3 (folded into the same loop iteration): replay through the
            # now-pruned layer to produce layer_index + 1's captured inputs.
            if layer_index + 1 < len(layers):
                captured = _advance_sparsegpt_layer_inputs(layer, captured)

        LOGGER.info("SparseGPT pruning complete.")
    finally:
        if original_use_cache is not _NO_CONFIG_USE_CACHE:
            model_config.use_cache = original_use_cache
