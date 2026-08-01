"""Pruning-mask extraction, compact on-disk packing, and mask-space distance.

This module lets the v2 experiment ask "how similar are two pruned models at
the *parameter* level" (as opposed to :mod:`pruning_metrics.metrics.distributions`,
which asks the same question at the *behavioral* level). The pipeline is:

1. :func:`extract_pruning_masks` -- pull a ``{module_name: bool_array}`` mask
   (``True`` = weight retained / non-zero) out of a live pruned model.
2. :func:`save_packed_masks` / :func:`load_packed_masks` -- persist the full
   mask bit-for-bit using ``np.packbits`` so a 7B-parameter model's masks fit
   in a manageable ``.npz`` (1 bit/weight instead of numpy's 1 byte/bool).
3. :func:`make_mask_digest` / :func:`save_digest` / :func:`load_digest` --
   because comparing *every* weight across 241 model variants is expensive
   and unnecessary for a diagnostic correlation, draw a small deterministic
   subsample of each layer's flat mask. Digests from different model variants
   remain positionally comparable (same seed + fraction + layer shapes select
   the same coordinates), so they can be Jaccard-compared directly, standing
   in for the full mask at a fraction of the size/compute.
4. :func:`jaccard_distance` -- 1 minus Jaccard similarity over retained
   positions; works identically on full masks and digests since both are just
   ``{name: bool_array}`` dicts.

Notes
-----
- Only :func:`extract_pruning_masks` touches ``torch`` (or anything
  model-shaped), and only via duck-typing: it never imports ``torch`` or
  checks ``isinstance``. Any module tree exposing ``named_modules()`` whose
  leaves carry a ``.weight`` that supports ``.detach().cpu().numpy()`` will
  work, so this module has no hard torch dependency.
- Every other function is pure numpy and operates on plain
  ``dict[str, np.ndarray]`` mask/digest representations.
"""

from __future__ import annotations

import math
import zlib
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_pruning_masks(
    model: Any,
    *,
    exclude: tuple[str, ...] = ("lm_head",),
) -> dict[str, np.ndarray]:
    """Extract a boolean retained-weight mask per prunable leaf module.

    Walks ``model.named_modules()`` and, for every leaf whose ``.weight``
    duck-types as a tensor (i.e. supports ``.detach().cpu().numpy()``),
    records a boolean array that is ``True`` where the weight is non-zero
    ("retained") and ``False`` where it was pruned to zero.

    Parameters
    ----------
    model:
        Any object exposing a ``named_modules()`` method that yields
        ``(name, module)`` pairs (the standard ``torch.nn.Module`` API, or a
        duck-typed equivalent). Not required to actually be a ``torch``
        module -- this function never imports ``torch``.
    exclude:
        Substrings; a module is skipped if its dotted name contains any of
        them. Defaults to excluding the LM head, which is never pruned by
        this project's pruning routines (see C1's ``apply_sparsegpt_pruning``
        docstring) and is out of scope for mask-Jaccard comparisons.

    Returns
    -------
    dict[str, np.ndarray]
        Maps the module's ``named_modules()`` path to a bool array with the
        same shape as its weight. Only modules whose dotted name contains
        ``".layers."`` are included -- this restricts extraction to the
        decoder-layer scope shared by both pruners (C1, C4), so masks from a
        WANDA-pruned run and a SparseGPT-pruned run are directly comparable.

    Notes
    -----
    - A leaf is only included if its extracted weight array is exactly 2-D.
      This is a deliberate duck-typed stand-in for "is an ``nn.Linear``":
      norm layers (``LayerNorm``/``RMSNorm``) also expose a ``.weight``
      tensor, but theirs is 1-D (a per-channel gain), so this shape check
      excludes them without ever importing ``torch`` to do an ``isinstance``
      check.
    - Modules whose ``.weight`` is ``None`` or does not support the
      ``.detach().cpu().numpy()`` chain are silently skipped (e.g. bias-only
      or weight-free container modules) rather than raising, since
      ``named_modules()`` yields many non-leaf containers that legitimately
      have no weight of their own.
    """
    masks: dict[str, np.ndarray] = {}
    for name, module in model.named_modules():
        # Design: scope restriction to decoder layers first (cheap string
        # check) before touching `.weight`, so containers we will never keep
        # (e.g. `model.embed_tokens`) never pay the duck-typing attempt cost.
        if ".layers." not in name:
            continue
        if any(substr in name for substr in exclude):
            continue

        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        try:
            # Compare in torch, convert only the bool result: numpy cannot
            # represent bfloat16, so `.numpy()` on the raw weight raises
            # TypeError for bf16 models (e.g. Qwen2 loaded in bf16).
            mask_np = (weight.detach() != 0).cpu().numpy()
        except (AttributeError, TypeError):
            # Not tensor-like (duck-typing failed) -- not a prunable leaf.
            continue

        if mask_np.ndim != 2:
            # Excludes norm-layer gains (1-D) etc.; see Notes above.
            continue

        masks[name] = mask_np

    return masks


# ---------------------------------------------------------------------------
# Bit-packed persistence (full masks)
# ---------------------------------------------------------------------------

_BITS_SUFFIX = "__bits"
_SHAPE_SUFFIX = "__shape"
_N_SUFFIX = "__n"


def save_packed_masks(masks: dict[str, np.ndarray], path: str | Path) -> None:
    """Persist a full mask dict to a bit-packed ``.npz`` file.

    Each layer's flattened boolean mask is stored via ``np.packbits`` (1 bit
    per weight instead of numpy bool's 1 byte per weight -- an 8x reduction
    before compression), alongside its original shape so it can be restored
    exactly by :func:`load_packed_masks`.

    Parameters
    ----------
    masks:
        Mapping from layer name to boolean array, as returned by
        :func:`extract_pruning_masks`.
    path:
        Destination ``.npz`` path (passed through to
        ``np.savez_compressed``, which appends ``.npz`` if the given path
        lacks that suffix).

    Notes
    -----
    Two npz entries are written per layer: ``"<name>__bits"`` (the packed
    bits) and ``"<name>__shape"`` (an int64 array of the original shape),
    exactly reversed by :func:`load_packed_masks`.
    """
    payload: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        arr = np.asarray(mask, dtype=bool)
        payload[f"{name}{_BITS_SUFFIX}"] = np.packbits(arr.ravel())
        payload[f"{name}{_SHAPE_SUFFIX}"] = np.array(arr.shape, dtype=np.int64)
    np.savez_compressed(path, **payload)


def load_packed_masks(path: str | Path) -> dict[str, np.ndarray]:
    """Load a mask dict previously written by :func:`save_packed_masks`.

    Parameters
    ----------
    path:
        Path to a ``.npz`` file written by :func:`save_packed_masks`.

    Returns
    -------
    dict[str, np.ndarray]
        Exact inverse of :func:`save_packed_masks`: each layer's bool array
        is restored to its original shape (``np.unpackbits`` pads to a
        multiple of 8 bits, so the trailing pad bits are trimmed using the
        stored shape before reshaping).
    """
    masks: dict[str, np.ndarray] = {}
    with np.load(path) as data:
        names = sorted(
            key[: -len(_BITS_SUFFIX)]
            for key in data.files
            if key.endswith(_BITS_SUFFIX)
        )
        for name in names:
            bits = data[f"{name}{_BITS_SUFFIX}"]
            shape = tuple(int(x) for x in data[f"{name}{_SHAPE_SUFFIX}"])
            n_elements = int(np.prod(shape)) if shape else 1
            flat = np.unpackbits(bits)[:n_elements].astype(bool)
            masks[name] = flat.reshape(shape)
    return masks


# ---------------------------------------------------------------------------
# Deterministic digests (subsampled masks)
# ---------------------------------------------------------------------------


def _layer_seed(name: str, seed: int) -> int:
    """Combine a global seed with a stable per-layer hash.

    Design: `zlib.crc32` is used (rather than Python's built-in `hash`,
    which is salted per-process for strings) so the same layer name always
    maps to the same seed across processes/runs -- required for digests
    produced by independent calls (even in different Python invocations) to
    select identical coordinates.
    """
    return seed ^ zlib.crc32(name.encode("utf-8"))


def make_mask_digest(
    masks: dict[str, np.ndarray],
    *,
    fraction: float = 1 / 32,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Draw a deterministic pseudorandom subsample of each layer's flat mask.

    For each layer, a per-layer RNG is seeded from ``seed`` combined with a
    stable hash of the layer name (see :func:`_layer_seed`), then
    ``ceil(n * fraction)`` indices are drawn without replacement from the
    flattened mask and sorted. Because the index selection depends only on
    ``(layer name, layer size, fraction, seed)`` -- never on the mask's
    *values* -- two model variants that share layer shapes (e.g. two
    different pruning runs of the same base model) yield digests that are
    **positionally comparable**: index ``i`` of variant A's digest for a
    layer refers to the same flat coordinate as index ``i`` of variant B's
    digest for that layer.

    Parameters
    ----------
    masks:
        Mapping from layer name to boolean array (a full mask, as returned
        by :func:`extract_pruning_masks`, or another digest).
    fraction:
        Fraction of each layer's flattened elements to keep, in
        ``(0, 1]``. Default ``1/32`` (~3.1%).
    seed:
        Global seed combined with each layer's name hash. Using the same
        ``seed`` (with the same layer shapes and ``fraction``) reproduces
        the exact same subsample.

    Returns
    -------
    dict[str, np.ndarray]
        Maps layer name to a 1-D bool subarray of length
        ``ceil(n * fraction)`` (clipped to the layer size ``n``).

    Raises
    ------
    ValueError
        If ``fraction`` is not in ``(0, 1]``.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    digest: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        flat = np.asarray(mask, dtype=bool).ravel()
        n = flat.size
        rng = np.random.default_rng(_layer_seed(name, seed))
        # Clip to n: guards tiny layers where ceil(n * fraction) could
        # otherwise exceed n (e.g. a 1-element layer at fraction > 0).
        size = min(math.ceil(n * fraction), n)
        indices = rng.choice(n, size=size, replace=False, shuffle=False)
        indices.sort()
        digest[name] = flat[indices]
    return digest


def save_digest(digest: dict[str, np.ndarray], path: str | Path) -> None:
    """Persist a digest (see :func:`make_mask_digest`) to a bit-packed ``.npz``.

    Uses the same ``np.packbits`` scheme as :func:`save_packed_masks`, but
    stores each layer's flat *length* (``"<name>__n"``) instead of a full
    shape, since digests are always 1-D.

    Parameters
    ----------
    digest:
        Mapping from layer name to 1-D bool array, as returned by
        :func:`make_mask_digest`.
    path:
        Destination ``.npz`` path.
    """
    payload: dict[str, np.ndarray] = {}
    for name, arr in digest.items():
        flat = np.asarray(arr, dtype=bool).ravel()
        payload[f"{name}{_BITS_SUFFIX}"] = np.packbits(flat)
        payload[f"{name}{_N_SUFFIX}"] = np.array(flat.size, dtype=np.int64)
    np.savez_compressed(path, **payload)


def load_digest(path: str | Path) -> dict[str, np.ndarray]:
    """Load a digest previously written by :func:`save_digest`.

    Parameters
    ----------
    path:
        Path to a ``.npz`` file written by :func:`save_digest`.

    Returns
    -------
    dict[str, np.ndarray]
        Exact inverse of :func:`save_digest`: each layer's 1-D bool array,
        trimmed to its stored length (``np.unpackbits`` pads to a multiple
        of 8 bits).
    """
    digest: dict[str, np.ndarray] = {}
    with np.load(path) as data:
        names = sorted(
            key[: -len(_BITS_SUFFIX)]
            for key in data.files
            if key.endswith(_BITS_SUFFIX)
        )
        for name in names:
            bits = data[f"{name}{_BITS_SUFFIX}"]
            n = int(data[f"{name}{_N_SUFFIX}"])
            digest[name] = np.unpackbits(bits)[:n].astype(bool)
    return digest


# ---------------------------------------------------------------------------
# Mask-space distance
# ---------------------------------------------------------------------------


def jaccard_distance(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    """Jaccard distance between two masks (or digests) over retained weights.

    Computes ``1 - |A ∩ B| / |A ∪ B|`` where ``A`` and ``B`` are the sets of
    "retained" (``True``) positions across the union of all layers,
    identifying positions by ``(layer name, flat index)``. Because
    :func:`make_mask_digest` produces positionally comparable subsamples,
    this function works identically on full masks and on digests.

    Parameters
    ----------
    a, b:
        Mask or digest dicts as returned by :func:`extract_pruning_masks` or
        :func:`make_mask_digest` (equivalently, loaded via
        :func:`load_packed_masks` / :func:`load_digest`).

    Returns
    -------
    float
        ``0.0`` if ``a`` and ``b`` retain exactly the same positions,
        ``1.0`` if they share no retained position, ``0.0`` (by convention)
        if neither has any retained position anywhere (empty union).

    Raises
    ------
    ValueError
        If ``a`` and ``b`` do not have exactly the same set of layer names,
        or if a shared layer name has mismatched flattened lengths between
        ``a`` and ``b``.

    Notes
    -----
    Intersection/union counts are accumulated one layer at a time
    (layer-streamed) rather than concatenating every layer into one giant
    array first, so peak memory stays bounded by the largest single layer
    even when ``a``/``b`` span an entire 7B-parameter model's decoder.
    """
    keys_a, keys_b = set(a), set(b)
    if keys_a != keys_b:
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        raise ValueError(
            "mask/digest key sets differ: "
            f"only in a={sorted(only_a)}, only in b={sorted(only_b)}"
        )

    intersection = 0
    union = 0
    for name in keys_a:
        flat_a = np.asarray(a[name], dtype=bool).ravel()
        flat_b = np.asarray(b[name], dtype=bool).ravel()
        if flat_a.shape != flat_b.shape:
            raise ValueError(
                f"length mismatch for layer {name!r}: "
                f"{flat_a.shape[0]} vs {flat_b.shape[0]}"
            )
        intersection += int(np.count_nonzero(flat_a & flat_b))
        union += int(np.count_nonzero(flat_a | flat_b))

    if union == 0:
        # Neither mask retains anything anywhere; define distance as 0
        # (vacuously identical) rather than dividing by zero.
        return 0.0
    return 1.0 - (intersection / union)
