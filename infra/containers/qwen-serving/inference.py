"""SageMaker inference handler for Qwen pruned checkpoints."""

from __future__ import annotations

import json
import os
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_CACHE: dict[int, AutoModelForCausalLM] = {}
TOKENIZER: AutoTokenizer | None = None
MODEL_MANIFEST: dict[int, str] = {}
S3_CLIENT = boto3.client("s3")


@dataclass(frozen=True)
class RequestConfig:
    """Validated request payload for model generation.

    Parameters
    ----------
    prompt:
        Input prompt.
    task_id:
        Task identifier passed through to metadata.
    pruning_level:
        Requested pruning level key in manifest.
    seed:
        Global random seed for deterministic generation.
    max_new_tokens:
        Max generated tokens.
    temperature:
        Temperature for sampling.
    top_p:
        Top-p nucleus sampling value.
    """

    prompt: str
    task_id: str
    pruning_level: int
    seed: int
    max_new_tokens: int
    temperature: float
    top_p: float


def model_fn(model_dir: str) -> dict[str, str]:
    """Load tokenizer and pruning manifest at model startup.

    Parameters
    ----------
    model_dir:
        SageMaker model mount directory.

    Returns
    -------
    dict[str, str]
        Context with model directory metadata.
    """

    global TOKENIZER, MODEL_MANIFEST
    base_model_path = os.environ.get("BASE_MODEL_PATH", model_dir)
    TOKENIZER = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    manifest_path = os.environ.get(
        "MODEL_MANIFEST_PATH", os.path.join(model_dir, "model_manifest.json")
    )
    if not os.path.exists(manifest_path):
        manifest_s3_uri = os.environ.get("MODEL_MANIFEST_S3_URI", "")
        if not manifest_s3_uri:
            raise ValueError(
                "Manifest file does not exist and MODEL_MANIFEST_S3_URI is unset."
            )
        manifest_bucket, manifest_key = _parse_s3_uri(manifest_s3_uri)
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        S3_CLIENT.download_file(manifest_bucket, manifest_key, manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        raw_manifest = json.load(handle)
    MODEL_MANIFEST = {int(k): str(v) for k, v in raw_manifest["pruning_levels"].items()}
    return {"model_dir": model_dir, "manifest_path": manifest_path}


def input_fn(request_body: bytes, request_content_type: str) -> dict[str, Any]:
    """Deserialize request payload."""

    if request_content_type != "application/json":
        raise ValueError("Only application/json content type is supported.")
    return json.loads(request_body.decode("utf-8"))


def predict_fn(input_data: dict[str, Any], model: dict[str, str]) -> dict[str, Any]:
    """Run generation, capture logits, and upload logits artifact to S3."""

    del model
    request = _parse_request(input_data)
    model_instance = _get_model_for_level(request.pruning_level)

    _set_seed(request.seed)
    tokenizer = _require_tokenizer()
    encoded = tokenizer(request.prompt, return_tensors="pt")
    encoded = {key: value.to(model_instance.device) for key, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[1])

    generate_output = model_instance.generate(
        **encoded,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=request.temperature > 0.0,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    all_token_ids = generate_output.sequences[0]
    new_token_ids = all_token_ids[prompt_tokens:]
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    score_tensors = generate_output.scores
    logits = _score_tensors_to_numpy(score_tensors)
    request_id = str(uuid.uuid4())
    logits_s3_uri = _persist_logits_to_s3(
        logits=logits,
        generated_token_ids=new_token_ids.tolist(),
        request=request,
        request_id=request_id,
    )

    return {
        "generated_text": generated_text,
        "task_id": request.task_id,
        "pruning_level": request.pruning_level,
        "seed": request.seed,
        "token_count": int(new_token_ids.shape[0]),
        "request_id": request_id,
        "logits_s3_uri": logits_s3_uri,
    }


def output_fn(prediction: dict[str, Any], accept: str) -> bytes:
    """Serialize response payload."""

    if accept not in ("application/json", "*/*"):
        raise ValueError("Only JSON responses are supported.")
    return json.dumps(prediction).encode("utf-8")


def _parse_request(payload: dict[str, Any]) -> RequestConfig:
    """Validate inbound request payload."""

    try:
        prompt = str(payload["prompt"])
        task_id = str(payload["task_id"])
        pruning_level = int(payload["pruning_level"])
        seed = int(payload["seed"])
    except KeyError as exc:
        raise ValueError(f"Missing required request field: {exc}") from exc

    max_new_tokens = int(payload.get("max_new_tokens", 256))
    temperature = float(payload.get("temperature", 0.0))
    top_p = float(payload.get("top_p", 1.0))
    return RequestConfig(
        prompt=prompt,
        task_id=task_id,
        pruning_level=pruning_level,
        seed=seed,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def _get_model_for_level(pruning_level: int) -> AutoModelForCausalLM:
    """Load or reuse model instance for pruning level."""

    if pruning_level not in MODEL_CACHE:
        if pruning_level not in MODEL_MANIFEST:
            available = sorted(MODEL_MANIFEST.keys())
            raise ValueError(
                f"Unsupported pruning level {pruning_level}. Available levels: {available}"
            )

        model_path = _ensure_local_model_path(MODEL_MANIFEST[pruning_level])
        MODEL_CACHE[pruning_level] = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

    return MODEL_CACHE[pruning_level]


def _score_tensors_to_numpy(score_tensors: list[torch.Tensor]) -> np.ndarray:
    """Convert generated token scores to float32 NumPy array."""

    if not score_tensors:
        return np.zeros((0, 0), dtype=np.float32)

    row_arrays = [
        score_tensor[0].detach().to(dtype=torch.float32).cpu().numpy()
        for score_tensor in score_tensors
    ]
    return np.stack(row_arrays, axis=0)


def _persist_logits_to_s3(
    logits: np.ndarray,
    generated_token_ids: list[int],
    request: RequestConfig,
    request_id: str,
) -> str:
    """Persist per-token full-vocab logits as JSONL to S3."""

    bucket = os.environ["LOGITS_S3_BUCKET"]
    prefix = os.environ.get("LOGITS_S3_PREFIX", "logits")
    timestamp = datetime.now(timezone.utc)
    date_prefix = timestamp.strftime("%Y-%m-%d")
    key = (
        f"{prefix}/model=qwen2_5_72b/"
        f"pruning_level={request.pruning_level}/"
        f"date={date_prefix}/"
        f"task_id={request.task_id}/"
        f"request_id={request_id}/"
        "tokens.jsonl"
    )

    lines: list[str] = []
    for token_index, token_id in enumerate(generated_token_ids):
        row = {
            "token_index": token_index,
            "token_id": int(token_id),
            "seed": request.seed,
            "pruning_level": request.pruning_level,
            "logits": (
                logits[token_index].tolist() if token_index < logits.shape[0] else []
            ),
        }
        lines.append(json.dumps(row))

    body = ("\n".join(lines)).encode("utf-8")
    S3_CLIENT.put_object(
        Bucket=bucket, Key=key, Body=body, ContentType="application/json"
    )
    return f"s3://{bucket}/{key}"


def _set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_local_model_path(model_path: str) -> str:
    """Resolve local model path from local or S3 URI location."""

    if not model_path.startswith("s3://"):
        return model_path

    base_dir = Path(os.environ.get("PRUNED_MODELS_CACHE_DIR", "/tmp/pruned_models"))
    cache_dir = base_dir / model_path.replace("s3://", "").replace("/", "_")
    if cache_dir.exists():
        return str(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    bucket, prefix = _parse_s3_uri(model_path)
    continuation_token: str | None = None
    while True:
        params: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token is not None:
            params["ContinuationToken"] = continuation_token
        response = S3_CLIENT.list_objects_v2(**params)
        for item in response.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue
            relative_path = key[len(prefix) :].lstrip("/")
            destination = cache_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            S3_CLIENT.download_file(bucket, key, str(destination))
        if not response.get("IsTruncated"):
            break
        continuation_token = response.get("NextContinuationToken")
    return str(cache_dir)


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Split S3 URI into bucket and key prefix."""

    stripped = s3_uri.replace("s3://", "", 1)
    if "/" not in stripped:
        return stripped, ""
    bucket, key = stripped.split("/", 1)
    return bucket, key


def _require_tokenizer() -> AutoTokenizer:
    """Get loaded tokenizer or fail fast if not initialized."""

    if TOKENIZER is None:
        raise RuntimeError(
            "Tokenizer has not been initialized. model_fn must run first."
        )
    return TOKENIZER
