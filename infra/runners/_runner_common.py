"""Shared GPU-runner helpers.

Used by:

* :mod:`infra.runners.run_pruning_calibration` (notebook 2),
* :mod:`infra.runners.run_freeform_eval` (notebook 3),
* :mod:`infra.runners.run_teacher_forced` (notebook 4),
* :mod:`infra.provisioning.launch_gpu_instance` (run-id generation).

All helpers are torch-free at import time so the launcher / notebook helpers
can ``import infra.runners._runner_common`` for type hints and S3 utilities
without having ``torch`` installed locally.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("pruning_runner")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"


def configure_logging(level: int = logging.INFO) -> None:
    """Single, consistent logging format for all three runners."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def ensure_src_on_path() -> None:
    """Inject ``<repo>/src`` so runners can import ``pruning_metrics`` without
    requiring an editable install on the GPU box."""

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


def default_run_id() -> str:
    """UTC timestamp + random suffix, e.g. ``20260718T120000Z-a1b2c3``."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def split_csv(raw: str) -> tuple[str, ...] | None:
    """Parse a comma-separated id list (empty / "None" -> ``None``)."""

    if not raw or raw.strip().lower() in ("", "none"):
        return None
    return tuple(token.strip() for token in raw.split(",") if token.strip())


# ---------------------------------------------------------------------------
# Shared CLI arguments
# ---------------------------------------------------------------------------


def add_common_runner_args(
    parser: argparse.ArgumentParser, *, default_results_prefix: str
) -> None:
    """Register the staging/S3 arguments every runner accepts.

    Defaults are pulled from the environment so the same invocation works
    when the user-data script exports ``RESULTS_BUCKET`` etc.
    """

    parser.add_argument(
        "--output-dir",
        default=env_or("RESULTS_LOCAL_DIR", default="/opt/results"),
    )
    parser.add_argument(
        "--results-bucket",
        default=env_or("RESULTS_BUCKET", default=""),
    )
    parser.add_argument(
        "--results-prefix",
        default=env_or("RESULTS_PREFIX", default=default_results_prefix),
    )
    parser.add_argument(
        "--run-id", default=env_or("RUN_ID", default=default_run_id())
    )


def add_eval_artifact_args(parser: argparse.ArgumentParser) -> None:
    """Register the arguments shared by the two artifact-consuming eval runners."""

    parser.add_argument(
        "--artifact-uri",
        default=env_or("PRUNING_ARTIFACT_URI", default=""),
        help="s3://<bucket>/pruning_artifacts/<run_id>/ produced by calibration.",
    )
    parser.add_argument(
        "--eval-dataset-spec",
        default=env_or("EVAL_DATASET_SPEC", "DATASET_SPEC", default=""),
        help="Adapter spec for the eval dataset (defaults to artifact's).",
    )
    parser.add_argument(
        "--eval-levels",
        default=env_or("EVAL_LEVELS", default=""),
        help="Comma-separated levels to evaluate (defaults to artifact's).",
    )


# ---------------------------------------------------------------------------
# S3 helpers (boto3, no torch)
# ---------------------------------------------------------------------------


def boto_client_config() -> Any:
    """Shared botocore client config for all infra scripts.

    Adaptive retries replace the per-call-site retry loops and fixed
    sleeps that scripts previously hand-rolled; the bounded connect
    timeout keeps a wedged endpoint from blocking a run for the default
    60 seconds.
    """

    from botocore.config import Config  # local import keeps unit-test imports cheap

    return Config(
        retries={"mode": "adaptive", "total_max_attempts": 5},
        connect_timeout=10,
    )


# Client construction re-runs the credential chain and rebuilds the
# connection pool, so the eval runners' per-level sync loop must not pay
# that cost on every call.
_S3_CLIENT: Any = None


def s3_client() -> Any:
    """Process-wide cached S3 client used by all S3 helpers below."""

    global _S3_CLIENT
    if _S3_CLIENT is None:
        import boto3  # local import keeps unit-test imports cheap

        _S3_CLIENT = boto3.client("s3", config=boto_client_config())
    return _S3_CLIENT


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key/path`` into ``(bucket, key)`` (key may be empty)."""

    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Not an s3 URI: {s3_uri!r}")
    cleaned = s3_uri[5:]
    if "/" not in cleaned:
        return cleaned, ""
    bucket, key = cleaned.split("/", 1)
    return bucket, key


# (bucket, key) -> (size, mtime_ns) at last successful upload. The eval
# runners call s3_sync after every pruning level, and most of the tree
# (the downloaded calibration artifact, earlier levels' records) never
# changes between calls -- without this cache each level re-uploads the
# multi-hundred-MB ``wanda_stats.pt`` while the GPU sits idle.
_UPLOADED: dict[tuple[str, str], tuple[int, int]] = {}


def s3_sync(local_dir: Path, bucket: str, key_prefix: str) -> None:
    """Mirror ``local_dir`` to ``s3://bucket/key_prefix``.

    Uses boto3 directly (no AWS CLI) so the runner works on minimal AMIs.
    Files already uploaded by an earlier call in this process are skipped
    unless their size or mtime has changed since.
    """

    if not bucket:
        LOGGER.warning(
            "Results bucket is empty; skipping S3 sync (results stay in %s).",
            local_dir,
        )
        return

    client = s3_client()
    prefix = key_prefix.strip("/")
    uploaded = 0
    skipped = 0
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(local_dir).as_posix()
        key = f"{prefix}/{relative}" if prefix else relative
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if _UPLOADED.get((bucket, key)) == signature:
            skipped += 1
            continue
        client.upload_file(str(path), bucket, key)
        _UPLOADED[(bucket, key)] = signature
        uploaded += 1
    LOGGER.info(
        "Synced %d files (%d unchanged skipped) from %s to s3://%s/%s/",
        uploaded,
        skipped,
        local_dir,
        bucket,
        prefix,
    )


def s3_download_file(s3_uri: str, local_path: Path) -> None:
    """Download a single S3 object to ``local_path`` (parents created)."""

    bucket, key = parse_s3_uri(s3_uri)
    if not key:
        raise ValueError(f"S3 URI must include a key: {s3_uri!r}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client().download_file(bucket, key, str(local_path))


def download_calibration_artifact(
    artifact_uri: str, output_dir: Path
) -> tuple[dict[str, Any], Path]:
    """Fetch ``manifest.json`` + ``wanda_stats.pt`` into ``output_dir/_artifact``.

    Returns the parsed manifest and the local artifact directory.
    """

    artifact_dir = output_dir / "_artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base = artifact_uri.rstrip("/")
    LOGGER.info("Downloading manifest %s", f"{base}/manifest.json")
    s3_download_file(f"{base}/manifest.json", artifact_dir / "manifest.json")
    s3_download_file(f"{base}/wanda_stats.pt", artifact_dir / "wanda_stats.pt")
    manifest = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    return manifest, artifact_dir


# ---------------------------------------------------------------------------
# Shared config / metadata plumbing
# ---------------------------------------------------------------------------


def resolve_eval_defaults(
    manifest: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Eval-runner config fields where the CLI wins and the manifest is the fallback.

    Returns the keyword arguments shared by both eval runners' config
    dataclasses (dataset spec, levels, and the calibration split parameters).
    """

    eval_levels_raw = args.eval_levels or ",".join(
        str(level) for level in manifest["pruning_levels"]
    )
    return {
        "eval_dataset_spec": args.eval_dataset_spec or manifest["dataset_spec"],
        "eval_levels": parse_pruning_levels(eval_levels_raw),
        "split_seed": int(manifest["split_seed"]),
        "train_frac": float(manifest["train_frac"]),
        "explicit_train_ids": tuple(manifest.get("explicit_train_ids") or ())
        or None,
        "explicit_test_ids": tuple(manifest.get("explicit_test_ids") or ())
        or None,
    }


def serialise_config(config: Any) -> dict[str, Any]:
    """Best-effort JSON-serialisable copy of a runner config dataclass for logs.

    Paths become strings; tuples become lists (an empty tuple becomes
    ``None``, mirroring how the optional id lists are parsed).
    """

    payload: dict[str, Any] = {}
    for key, value in asdict(config).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value) if value else None
        else:
            payload[key] = value
    return payload


def s3_destination(bucket: str, prefix: str) -> str:
    """Human-readable destination string used in run metadata."""

    return f"s3://{bucket}/{prefix}/" if bucket else "(local only)"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as indented JSON (non-serialisable values via ``str``)."""

    path.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def eval_run_metadata(
    config: Any, manifest: dict[str, Any], mode: str
) -> dict[str, Any]:
    """``run_metadata.json`` payload shared by the two eval runners."""

    return {
        "host": socket.gethostname(),
        "run_id": config.run_id,
        "mode": mode,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_uri": config.artifact_uri,
        "base_model_id": manifest["base_model_id"],
        "calibration_dataset_spec": manifest["dataset_spec"],
        "eval_dataset_spec": config.eval_dataset_spec,
        "destination": s3_destination(
            config.results_bucket, config.results_prefix
        ),
    }


def eval_summary_payload(
    config: Any,
    manifest: dict[str, Any],
    mode: str,
    extra: dict[str, Any],
    *,
    ended: bool,
    elapsed_seconds: float | None,
) -> dict[str, Any]:
    """``summary.json`` payload shared by the two eval runners.

    The common header (identity + provenance) and tail (timing) wrap the
    runner-specific ``extra`` section.
    """

    return {
        "mode": mode,
        "run_id": config.run_id,
        "artifact_uri": config.artifact_uri,
        "base_model_id": manifest["base_model_id"],
        "calibration_dataset_spec": manifest["dataset_spec"],
        "eval_dataset_spec": config.eval_dataset_spec,
        **extra,
        "ended_at_utc": (
            datetime.now(timezone.utc).isoformat() if ended else None
        ),
        "elapsed_seconds": elapsed_seconds,
    }


def sync_results(config: Any) -> None:
    """Sync an eval runner's staging dir to its S3 destination."""

    s3_sync(
        local_dir=config.output_dir,
        bucket=config.results_bucket,
        key_prefix=config.results_prefix,
    )


def run_level_sweep(
    config: Any,
    *,
    model: Any,
    stats: dict[str, Any],
    snapshot: dict[str, Any],
    mode_label: str,
    per_level: Any,
    write_summary: Any,
) -> None:
    """Per-level sweep scaffold shared by the two eval runners.

    Each level: restore + prune, run ``per_level(level)``, persist the
    rolling summary, and sync to S3 so a spot interruption never loses a
    completed level. The ``finally`` block writes the final summary (with
    elapsed time) and syncs once more.
    """

    started = time.monotonic()
    try:
        for level in config.eval_levels:
            LOGGER.info("=== Pruning level %s%% ===", level_label(level))
            prune_to_level(model, stats, snapshot, level)
            per_level(level)
            write_summary(ended=False, elapsed_seconds=None)
            sync_results(config)
    finally:
        write_summary(
            ended=True, elapsed_seconds=time.monotonic() - started
        )
        sync_results(config)
        LOGGER.info(
            "%s finished. Results: s3://%s/%s/",
            mode_label,
            config.results_bucket,
            config.results_prefix,
        )


# ---------------------------------------------------------------------------
# Torch-dependent helpers (imported lazily by callers)
# ---------------------------------------------------------------------------


def collect_wanda_activation_stats(
    model: Any,
    tokenizer: Any,
    calibration_texts: list[str],
    max_tokens: int,
) -> dict[str, Any]:
    """Per-channel input RMS for every ``nn.Linear`` in ``model``.

    Mirrors the implementation that proved out on Qwen2-72B in the
    monolithic runner (run id ``20260504T001802Z-f041ba``). Hooks are
    attached to every ``Linear`` module, calibration texts are tokenized
    individually with truncation, and we accumulate ``sum(x**2)`` per input
    channel. Accumulation happens in float64 on each layer's own device; a
    per-hook ``.cpu()`` would force a CUDA sync for every Linear on every
    calibration text, so the single device-to-host copy is deferred until
    all texts are processed.
    """

    import torch
    from torch import nn

    ensure_src_on_path()
    from pruning_metrics.evals.coding.teacher_forcing import (
        resolve_input_device,
    )

    LOGGER.info(
        "Collecting WANDA activation stats over %d calibration texts (max_tokens=%d)",
        len(calibration_texts),
        max_tokens,
    )

    sumsq: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []

    def _build_hook(layer_name: str):
        def _hook(module, inputs, output):  # noqa: ANN001 - nn.Module signature
            del module, output
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            x = inputs[0].detach()
            flat = x.reshape(-1, x.shape[-1]).to(dtype=torch.float32)
            local_sumsq = torch.sum(flat * flat, dim=0).to(dtype=torch.float64)
            if layer_name in sumsq:
                sumsq[layer_name] += local_sumsq
            else:
                sumsq[layer_name] = local_sumsq
            counts[layer_name] = counts.get(layer_name, 0) + int(flat.shape[0])

        return _hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(_build_hook(name)))

    try:
        with torch.no_grad():
            target_device = resolve_input_device(model)
            for index, text in enumerate(calibration_texts, start=1):
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_tokens,
                )
                encoded = {k: v.to(target_device) for k, v in encoded.items()}
                model(**encoded)
                if index % 10 == 0 or index == len(calibration_texts):
                    LOGGER.info(
                        "Activation stats: processed %d/%d texts",
                        index,
                        len(calibration_texts),
                    )
    finally:
        for handle in hooks:
            handle.remove()

    rms_by_layer: dict[str, torch.Tensor] = {}
    for name, accumulated in sumsq.items():
        accumulated = accumulated.cpu()
        count = counts.get(name, 0)
        if count == 0:
            rms_by_layer[name] = torch.ones_like(accumulated, dtype=torch.float32)
            continue
        mean_square = accumulated / float(count)
        rms_by_layer[name] = torch.sqrt(mean_square).to(dtype=torch.float32)

    LOGGER.info("Collected stats for %d Linear layers.", len(rms_by_layer))
    return rms_by_layer


def snapshot_linear_weights(model: Any) -> dict[str, Any]:
    """Clone every ``nn.Linear.weight`` to host RAM for restore-between-levels.

    Memory cost is approximately the model size in bf16 (~145 GiB for
    Qwen2-72B). Both ``p4d.24xlarge`` (1.1 TiB) and ``p5.48xlarge`` (2 TiB)
    have ample host RAM.
    """

    from torch import nn

    LOGGER.info("Snapshotting original Linear weights to host RAM...")
    snapshot: dict[str, Any] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            snapshot[name] = module.weight.detach().clone().cpu()
    LOGGER.info("Snapshot covers %d Linear layers.", len(snapshot))
    return snapshot


def restore_linear_weights(model: Any, snapshot: dict[str, Any]) -> None:
    """Copy host snapshot tensors back into each ``nn.Linear`` weight."""

    from torch import nn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in snapshot:
            module.weight.data.copy_(
                snapshot[name].to(module.weight.device, non_blocking=True)
            )


def apply_wanda_pruning(
    model: Any,
    stats: dict[str, Any],
    prune_ratio: float,
) -> None:
    """In-place per-output-row WANDA pruning (Sun et al. 2023).

    For each ``nn.Linear`` weight ``W (out_features, in_features)`` with
    per-input-channel RMS ``rms (in_features,)``:

        score[i,j] = |W[i,j]| * rms[j]

    Then for each output row ``i`` independently zero the lowest-scoring
    ``prune_ratio`` fraction of entries.

    Per-row (rather than per-layer-global) thresholding is required because
    ``torch.quantile`` rejects tensors larger than ``2**24`` elements; the
    Qwen2-72B ``lm_head`` score matrix has ~1.2 B entries.
    """

    import torch
    from torch import nn

    if prune_ratio <= 0.0:
        return
    if prune_ratio >= 1.0:
        raise ValueError("prune_ratio must be in [0, 1).")

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in stats:
            continue
        channel_rms = stats[name].to(module.weight.device).float()
        weight = module.weight.data
        score = weight.abs().float() * channel_rms.unsqueeze(0)
        in_features = weight.shape[1]
        num_to_prune = int(round(prune_ratio * in_features))
        if num_to_prune <= 0:
            continue
        _, prune_indices = torch.topk(
            score,
            k=num_to_prune,
            dim=1,
            largest=False,
            sorted=False,
        )
        # topk indices are distinct within each row, so scattering zeros is
        # equivalent to a boolean-mask assignment while touching only the
        # pruned entries (no full-size mask allocation).
        weight.scatter_(1, prune_indices, 0.0)


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
        raise ValueError(
            "calibration_texts must be non-empty for SparseGPT pruning."
        )
    if not 0.0 <= prune_ratio < 1.0:
        raise ValueError(f"prune_ratio must be in [0, 1), got {prune_ratio}.")
    if prune_ratio <= 0.0:
        return

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


def load_base_model(
    model_id: str,
    *,
    dtype: str | None = None,
    trust_remote_code: bool = True,
) -> tuple[Any, Any]:
    """Load tokenizer + bf16 sharded causal LM, log device count.

    Parameters
    ----------
    model_id:
        Hugging Face model id (e.g. ``Qwen/Qwen2-72B``).
    dtype:
        Override the default bf16 precision (rarely useful; bf16 fits
        Qwen2-72B on a p4de while bringing fast TensorCore matmul).
    trust_remote_code:
        Forwarded to ``from_pretrained`` for models with custom
        configuration classes.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype, torch.bfloat16) if dtype else torch.bfloat16

    LOGGER.info("Loading tokenizer for %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )

    LOGGER.info("Loading model %s in %s (device_map=auto)", model_id, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    LOGGER.info(
        "Model loaded. CUDA visible devices: %s",
        torch.cuda.device_count(),
    )
    return tokenizer, model


def load_model_stats_snapshot(
    manifest: dict[str, Any], artifact_dir: Path
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Load the manifest's base model, the WANDA stats, and a weight snapshot.

    Returns ``(tokenizer, model, stats, snapshot)`` -- everything the eval
    runners need before entering their per-level loop.
    """

    tokenizer, model = load_base_model(manifest["base_model_id"])

    import torch

    stats_path = artifact_dir / "wanda_stats.pt"
    LOGGER.info("Loading WANDA stats from %s", stats_path)
    stats = torch.load(stats_path, map_location="cpu")
    snapshot = snapshot_linear_weights(model)
    return tokenizer, model, stats, snapshot


def prune_to_level(
    model: Any,
    stats: dict[str, Any],
    snapshot: dict[str, Any],
    level: float,
) -> None:
    """Restore pristine weights, then apply per-row WANDA at ``level`` percent."""

    restore_linear_weights(model, snapshot)
    apply_wanda_pruning(model, stats, prune_ratio=float(level) / 100.0)


def level_label(level: float) -> str:
    """Render a pruning level as ``"20"`` instead of ``"20.0"`` when whole."""

    return str(int(level)) if float(level).is_integer() else str(level)


def parse_pruning_levels(raw: str) -> tuple[float, ...]:
    """Parse comma-separated levels accepting integer or float tokens."""

    levels: list[float] = []
    for token in raw.split(","):
        if not token.strip():
            continue
        value = float(token.strip())
        if not 0.0 <= value < 100.0:
            raise ValueError(f"Pruning level {value} not in [0, 100).")
        levels.append(value)
    if not levels:
        raise ValueError("Need at least one pruning level.")
    return tuple(sorted(set(levels)))


def env_or(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment variable from ``names``."""

    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default
