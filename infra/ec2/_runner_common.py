"""Shared GPU-runner helpers.

Used by:

* :mod:`infra.ec2.run_pruning_calibration` (notebook 2),
* :mod:`infra.ec2.run_freeform_eval` (notebook 3),
* :mod:`infra.ec2.run_teacher_forced` (notebook 4).

All helpers are torch-free at import time so the launcher / notebook helpers
can ``import infra.ec2._runner_common`` for type hints and S3 utilities
without having ``torch`` installed locally.
"""

from __future__ import annotations

import logging
import os
import sys
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


def s3_sync(local_dir: Path, bucket: str, key_prefix: str) -> None:
    """Mirror ``local_dir`` to ``s3://bucket/key_prefix``.

    Uses boto3 directly (no AWS CLI) so the runner works on minimal AMIs.
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
    files = [path for path in local_dir.rglob("*") if path.is_file()]
    LOGGER.info(
        "Syncing %d files from %s to s3://%s/%s/",
        len(files),
        local_dir,
        bucket,
        prefix,
    )
    for path in files:
        relative = path.relative_to(local_dir).as_posix()
        key = f"{prefix}/{relative}" if prefix else relative
        client.upload_file(str(path), bucket, key)


def s3_download_file(s3_uri: str, local_path: Path) -> None:
    """Download a single S3 object to ``local_path`` (parents created)."""

    import boto3

    bucket, key = parse_s3_uri(s3_uri)
    if not key:
        raise ValueError(f"S3 URI must include a key: {s3_uri!r}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(local_path))


# ---------------------------------------------------------------------------
# Torch-dependent helpers (imported lazily by callers)
# ---------------------------------------------------------------------------


def embedding_device(model: Any) -> Any:
    """Return the device hosting the input embedding for ``model``.

    Sharded ``device_map='auto'`` models expose ``hf_device_map`` keyed by
    submodule name; the embedding key varies across architectures so we try a
    short list. Falls back to the first parameter's device, then CPU.
    """

    import torch

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for key in (
            "model.embed_tokens",
            "transformer.wte",
            "model.tok_embeddings",
        ):
            if key in device_map:
                value = device_map[key]
                return torch.device(value) if isinstance(value, str) else value
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


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
    channel. The CPU-side accumulator avoids GPU memory spikes for the
    biggest layers.
    """

    import torch
    from torch import nn

    device_map = getattr(model, "hf_device_map", None) or {}
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
            local_sumsq = torch.sum(flat * flat, dim=0).to(dtype=torch.float64).cpu()
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
            for index, text in enumerate(calibration_texts, start=1):
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_tokens,
                )
                target_device = embedding_device(model)
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
        mask = torch.zeros_like(weight, dtype=torch.bool)
        mask.scatter_(1, prune_indices, True)
        weight[mask] = 0


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
