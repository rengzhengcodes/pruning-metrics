"""Run WANDA-style pruning levels and register model artifacts in S3."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from infra.aws.sagemaker.calibration_datasets import load_calibration_records
from infra.aws.sagemaker.config import SageMakerInfraConfig, parse_pruning_levels


@dataclass
class LinearActivationStats:
    """Running activation statistics for one linear layer."""

    sumsq: torch.Tensor
    count: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Prune Qwen and register levels in S3."
    )
    parser.add_argument("--base-model-id", required=True, help="Base model id or path.")
    parser.add_argument(
        "--calibration-source",
        required=True,
        help=(
            "Calibration dataset source. Use 'humanevalplus_prompts' or "
            "'hf:<dataset_name>:<split>:<text_field>'."
        ),
    )
    parser.add_argument(
        "--pruning-levels",
        default=None,
        help="Comma-separated pruning levels. Defaults to 0,20,40,60,80.",
    )
    parser.add_argument("--max-calibration-samples", type=int, default=64)
    parser.add_argument("--max-calibration-tokens", type=int, default=1024)
    parser.add_argument("--artifact-bucket", default=None)
    parser.add_argument("--artifact-prefix", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--manifest-key", default="model_manifest.json")
    return parser.parse_args()


def main() -> None:
    """Execute pruning workflow and emit model manifest."""

    args = parse_args()
    config = SageMakerInfraConfig.from_env()
    region = args.region or config.region
    artifact_bucket = args.artifact_bucket or config.artifact_bucket
    artifact_prefix = args.artifact_prefix or config.artifact_prefix
    if not region or not artifact_bucket:
        raise ValueError("AWS region and artifact bucket must be configured.")

    pruning_levels = parse_pruning_levels(args.pruning_levels)
    calibration_records = load_calibration_records(
        source=args.calibration_source,
        max_samples=args.max_calibration_samples,
    )
    if not calibration_records:
        raise ValueError("Calibration dataset produced zero samples.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_id, trust_remote_code=True
    )
    stats = collect_wanda_activation_stats(
        model_id=args.base_model_id,
        tokenizer=tokenizer,
        calibration_texts=[record.text for record in calibration_records],
        max_tokens=args.max_calibration_tokens,
    )

    s3_client = boto3.client("s3", region_name=region)
    manifest_payload: dict[str, Any] = {
        "base_model_id": args.base_model_id,
        "calibration_source": args.calibration_source,
        "pruning_levels": {},
        "dtype": "bfloat16",
        "method": "wanda_layerwise_unstructured",
    }

    with tempfile.TemporaryDirectory(prefix="qwen-pruning-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for level in pruning_levels:
            # Keep "20" rather than "20.0" in S3 keys when the level is a whole percent.
            level_label = (
                str(int(level)) if float(level).is_integer() else str(level)
            )
            level_dir = tmp_root / f"pruning_{level_label}"
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if float(level) > 0.0:
                apply_wanda_pruning(
                    model=model,
                    stats=stats,
                    prune_ratio=float(level) / 100.0,
                )
            model.save_pretrained(level_dir)
            tokenizer.save_pretrained(level_dir)

            level_prefix = f"{artifact_prefix}/pruning_{level_label}"
            upload_directory(
                s3_client=s3_client,
                bucket=artifact_bucket,
                prefix=level_prefix,
                local_dir=level_dir,
            )
            manifest_payload["pruning_levels"][
                level_label
            ] = f"s3://{artifact_bucket}/{level_prefix}"
            shutil.rmtree(level_dir)

        manifest_key = f"{artifact_prefix.rstrip('/')}/{args.manifest_key}"
        s3_client.put_object(
            Bucket=artifact_bucket,
            Key=manifest_key,
            Body=json.dumps(manifest_payload, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    print(
        json.dumps(
            {"manifest_s3_uri": f"s3://{artifact_bucket}/{manifest_key}"}, indent=2
        )
    )


def collect_wanda_activation_stats(
    model_id: str,
    tokenizer: AutoTokenizer,
    calibration_texts: list[str],
    max_tokens: int,
) -> dict[str, torch.Tensor]:
    """Collect activation RMS values for every linear layer input channel."""

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    stats: dict[str, LinearActivationStats] = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            stats[name] = LinearActivationStats(
                sumsq=torch.zeros(
                    module.in_features, dtype=torch.float64, device=device
                ),
                count=0,
            )
            hooks.append(module.register_forward_hook(_build_hook(name, stats)))

    with torch.no_grad():
        for text in calibration_texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            model(**encoded)

    for hook in hooks:
        hook.remove()

    rms_by_layer: dict[str, torch.Tensor] = {}
    for name, layer_stats in stats.items():
        if layer_stats.count == 0:
            rms_by_layer[name] = torch.ones_like(layer_stats.sumsq, dtype=torch.float32)
            continue
        mean_square = layer_stats.sumsq / float(layer_stats.count)
        rms_by_layer[name] = torch.sqrt(mean_square).to(dtype=torch.float32).cpu()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rms_by_layer


def apply_wanda_pruning(
    model: AutoModelForCausalLM,
    stats: dict[str, torch.Tensor],
    prune_ratio: float,
) -> None:
    """Apply layer-wise WANDA pruning to linear weights.

    Notes
    -----
    This implementation uses layer-wise thresholding for practicality. Each linear
    layer computes the WANDA score ``abs(weight) * rms(input_channel)`` and prunes
    the bottom ``prune_ratio`` fraction.
    """

    if prune_ratio <= 0.0:
        return
    if prune_ratio >= 1.0:
        raise ValueError("Prune ratio must be in [0, 1).")

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in stats:
            continue

        channel_rms = stats[name].to(module.weight.device)
        weight = module.weight.data
        score = weight.abs() * channel_rms.unsqueeze(0)
        threshold = torch.quantile(score.float().flatten(), prune_ratio)
        mask = score <= threshold
        weight[mask] = 0


def upload_directory(
    s3_client: Any,
    bucket: str,
    prefix: str,
    local_dir: Path,
) -> None:
    """Upload a local directory recursively to S3."""

    cleaned_prefix = prefix.rstrip("/")
    for file_path in local_dir.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(local_dir).as_posix()
        key = f"{cleaned_prefix}/{relative}"
        s3_client.upload_file(str(file_path), bucket, key)


def _build_hook(
    layer_name: str,
    stats: dict[str, LinearActivationStats],
):
    """Create forward hook collecting linear input activation power."""

    def _hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        del module, output
        if not inputs:
            return
        x = inputs[0]
        if not isinstance(x, torch.Tensor):
            return
        flattened = x.detach().reshape(-1, x.shape[-1]).to(dtype=torch.float32)
        layer_stats = stats[layer_name]
        layer_stats.sumsq += torch.sum(flattened * flattened, dim=0).to(
            dtype=torch.float64
        )
        layer_stats.count += int(flattened.shape[0])

    return _hook


if __name__ == "__main__":
    main()
