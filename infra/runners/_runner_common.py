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

    import boto3  # local import keeps unit-test imports cheap

    client = boto3.client("s3")
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

    import boto3

    bucket, key = parse_s3_uri(s3_uri)
    if not key:
        raise ValueError(f"S3 URI must include a key: {s3_uri!r}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(local_path))


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
