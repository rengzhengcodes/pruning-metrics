"""End-to-end Qwen2-72B WANDA pruning + HumanEval+ + teacher-forced log-probs.

This script is the single source of truth for what runs on the EC2 GPU box. It
deliberately bundles everything that needs to share the loaded 72B model into
one process so the model is materialized **once**:

1. Load HumanEval+, deterministically split 80/20 train/test (seed 65320),
   save ``split.json`` so the partition is auditable.
2. Load ``Qwen/Qwen2-72B`` in bf16 across all GPUs (``device_map='auto'``).
3. Collect WANDA-style activation RMS statistics on the **train** prompts.
4. Snapshot original ``nn.Linear`` weights to host RAM so we can iterate over
   pruning levels without reloading the model from disk.
5. For each pruning level in ``[0, 20, 40, 60, 80]``:
   - Restore the original linear weights, apply WANDA at ``level / 100``.
   - Run autoregressive HumanEval+ generation on the **test** split (no
     teacher forcing) and verify with the existing subprocess harness.
   - Compute teacher-forced next-token log-probs on the seeded ``(prompt,
     canonical_solution)`` pair for that level.
6. Sync per-level artifacts to S3 incrementally so a spot interruption does
   not lose completed levels.
7. Write a final ``summary.json`` aggregating eval metrics and teacher-forced
   statistics across all levels.

The script is designed to run unattended from EC2 user-data: errors are
written to stderr but never silently swallowed, and the final S3 sync happens
inside a ``finally`` block so partial results are never abandoned.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
LOGGER = logging.getLogger("pruning_experiment")

# Allow ``python infra/ec2/run_qwen_pruning_experiment.py`` to import the
# in-tree package without an editable install (the ``src/`` layout otherwise
# requires ``pip install -e`` first). Order matters: insert before importing
# project modules.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# pylint: disable=wrong-import-position
from pruning_metrics.evals.coding.humaneval_plus_dataset import (  # noqa: E402
    HumanEvalPlusDatasetLoader,
    HumanEvalPlusTask,
)
from pruning_metrics.evals.coding.pipeline import (  # noqa: E402
    build_coding_prompt,
    run_pipeline,
)
from pruning_metrics.evals.coding.teacher_forcing import (  # noqa: E402
    compute_teacher_forced_logprobs,
    teacher_forced_records_to_summary,
    write_teacher_forced_record,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """Static configuration for one experiment run.

    Parameters
    ----------
    base_model_id:
        HuggingFace model id loaded once and pruned in-place per level.
    pruning_levels:
        Percent-sparsity levels to sweep (``0`` is "no pruning", baseline).
    split_seed:
        Seed for the deterministic 80/20 HumanEval+ split.
    train_frac:
        Fraction of HumanEval+ tasks used for calibration.
    teacher_forcing_seed:
        Seed used to deterministically pick the single ``(prompt, answer)``
        pair from the test split for teacher-forced scoring.
    max_calibration_tokens:
        Per-prompt truncation when collecting WANDA activation statistics.
    max_new_tokens:
        Generation budget for autoregressive HumanEval+ rollouts.
    timeout_seconds:
        Per-task subprocess verifier timeout.
    output_dir:
        Local directory where artifacts are written before syncing to S3.
    s3_results_bucket:
        Destination bucket for the final and incremental results.
    s3_results_prefix:
        Object key prefix under the destination bucket.
    """

    base_model_id: str
    pruning_levels: tuple[float, ...]
    split_seed: int
    train_frac: float
    teacher_forcing_seed: int
    max_calibration_tokens: int
    max_new_tokens: int
    timeout_seconds: float
    output_dir: Path
    s3_results_bucket: str
    s3_results_prefix: str


def parse_args() -> argparse.Namespace:
    """Parse runner CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(
        description="Run Qwen2-72B WANDA pruning + HumanEval+ + TF log-probs."
    )
    parser.add_argument(
        "--base-model-id",
        default=os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2-72B"),
        help="HuggingFace model id; defaults to BASE_MODEL_ID env or Qwen2-72B.",
    )
    parser.add_argument(
        "--pruning-levels",
        default=os.environ.get("PRUNING_LEVELS", "0,20,40,60,80"),
        help="Comma-separated percent levels (e.g. 0,20,40,60,80).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=int(os.environ.get("HUMANEVAL_SPLIT_SEED", "65320")),
        help="Seed for HumanEval+ 80/20 split.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=float(os.environ.get("HUMANEVAL_TRAIN_FRAC", "0.8")),
        help="Train fraction for the calibration split.",
    )
    parser.add_argument(
        "--teacher-forcing-seed",
        type=int,
        default=int(os.environ.get("HUMANEVAL_SPLIT_SEED", "65320")),
        help="Seed used to pick the single TF pair from the test split.",
    )
    parser.add_argument(
        "--max-calibration-tokens",
        type=int,
        default=int(os.environ.get("MAX_CALIBRATION_TOKENS", "512")),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "RESULTS_LOCAL_DIR", "/opt/ml/output"
        ),
        help="Local directory for artifacts before syncing to S3.",
    )
    parser.add_argument(
        "--results-bucket",
        default=os.environ.get("RESULTS_BUCKET", ""),
        help="S3 bucket for results.",
    )
    parser.add_argument(
        "--results-prefix",
        default=os.environ.get(
            "RESULTS_PREFIX",
            "qwen2_72b_pruning",
        ),
        help="S3 key prefix; a run id is appended automatically.",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("RUN_ID", _default_run_id()),
        help="Run identifier appended to the results prefix.",
    )
    return parser.parse_args()


def _default_run_id() -> str:
    """Build a default run id from the current UTC timestamp."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_pruning_levels(raw: str) -> tuple[float, ...]:
    """Parse comma-separated pruning levels accepting int or float tokens."""

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split(config: ExperimentConfig) -> tuple[
    list[HumanEvalPlusTask], list[HumanEvalPlusTask]
]:
    """Load HumanEval+ and produce the deterministic train/test split."""

    LOGGER.info(
        "Loading HumanEval+ and splitting train/test (seed=%d, train_frac=%.3f)",
        config.split_seed,
        config.train_frac,
    )
    loader = HumanEvalPlusDatasetLoader()
    return loader.split_train_test(
        seed=config.split_seed,
        train_frac=config.train_frac,
    )


def write_split_json(
    train: list[HumanEvalPlusTask],
    test: list[HumanEvalPlusTask],
    seed: int,
    train_frac: float,
    output_path: Path,
) -> None:
    """Persist the audit trail of the split (counts + ordered task ids)."""

    payload = {
        "seed": seed,
        "train_frac": train_frac,
        "num_train": len(train),
        "num_test": len(test),
        "train_task_ids": [task.task_id for task in train],
        "test_task_ids": [task.task_id for task in test],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info(
        "Wrote split: %d train + %d test -> %s", len(train), len(test), output_path
    )


def select_teacher_forced_pair(
    test_tasks: list[HumanEvalPlusTask], seed: int
) -> HumanEvalPlusTask:
    """Pick one ``(prompt, canonical_solution)`` pair deterministically.

    The test list is already shuffled by seed, but to make the choice depend
    on ``seed`` alone (not on the train/test cut), we sort by ``task_id`` then
    index by ``seed % N``.
    """

    sorted_tasks = sorted(test_tasks, key=lambda task: task.task_id)
    chosen = sorted_tasks[seed % len(sorted_tasks)]
    if not chosen.canonical_solution:
        raise RuntimeError(
            f"Teacher-forced sample {chosen.task_id} has no canonical_solution."
        )
    return chosen


# ---------------------------------------------------------------------------
# Model loading + pruning
# ---------------------------------------------------------------------------


def load_base_model(model_id: str) -> tuple[Any, Any]:
    """Load tokenizer and model in bf16 sharded across all GPUs."""

    import torch  # local: keep CLI usable on import-only inspection
    from transformers import AutoModelForCausalLM, AutoTokenizer

    LOGGER.info("Loading tokenizer for %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    LOGGER.info("Loading model %s in bf16 (device_map=auto)", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    LOGGER.info(
        "Model loaded. CUDA visible devices: %s", torch.cuda.device_count()
    )
    return tokenizer, model


def collect_activation_stats(
    model: Any,
    tokenizer: Any,
    calibration_texts: list[str],
    max_tokens: int,
) -> dict[str, Any]:
    """Collect mean-square (RMS) activations per Linear input channel.

    This is the in-process replacement for
    ``infra/aws/sagemaker/prune_and_register.collect_wanda_activation_stats``,
    written so we can reuse the already-loaded model rather than reloading it
    from disk for stats collection.
    """

    import torch
    from torch import nn

    device_map = getattr(model, "hf_device_map", None) or {}
    LOGGER.info(
        "Collecting WANDA activation stats over %d calibration texts (max_tokens=%d)",
        len(calibration_texts),
        max_tokens,
    )

    stats: dict[str, Any] = {}
    hooks = []
    counts: dict[str, int] = {}

    def make_hook(layer_name: str):
        def _hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            del module, output
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            x = inputs[0].detach()
            flat = x.reshape(-1, x.shape[-1]).to(dtype=torch.float32)
            sumsq = torch.sum(flat * flat, dim=0).to(dtype=torch.float64)
            if layer_name in stats:
                stats[layer_name] += sumsq.cpu()
            else:
                stats[layer_name] = sumsq.cpu()
            counts[layer_name] = counts.get(layer_name, 0) + int(flat.shape[0])

        return _hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            for index, text in enumerate(calibration_texts, start=1):
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_tokens,
                )
                # Place input ids on the embedding device for sharded models.
                target_device = _embedding_device(model, device_map)
                encoded = {key: val.to(target_device) for key, val in encoded.items()}
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

    rms_by_layer: dict[str, Any] = {}
    for name, sumsq in stats.items():
        count = counts.get(name, 0)
        if count == 0:
            rms_by_layer[name] = torch.ones_like(sumsq, dtype=torch.float32)
            continue
        mean_square = sumsq / float(count)
        rms_by_layer[name] = torch.sqrt(mean_square).to(dtype=torch.float32)

    LOGGER.info("Collected stats for %d Linear layers.", len(rms_by_layer))
    return rms_by_layer


def _embedding_device(model: Any, device_map: dict[str, Any]):
    """Return the device hosting the embedding layer (or first model device)."""

    import torch

    for key in ("model.embed_tokens", "transformer.wte", "model.tok_embeddings"):
        if key in device_map:
            value = device_map[key]
            return torch.device(value) if isinstance(value, str) else value
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def snapshot_linear_weights(model: Any) -> dict[str, Any]:
    """Clone every ``nn.Linear.weight`` to host RAM so we can restore later.

    Necessary because we want to apply WANDA at multiple sparsity levels on
    the same model object without re-downloading and re-loading 72B weights
    five times. p4d.24xlarge has 1.1 TiB of host RAM; 72B bf16 ~= 145 GiB.
    """

    from torch import nn

    LOGGER.info("Snapshotting original Linear weights to host RAM ...")
    snapshot: dict[str, Any] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            snapshot[name] = module.weight.detach().clone().cpu()
    LOGGER.info("Snapshot covers %d Linear layers.", len(snapshot))
    return snapshot


def restore_linear_weights(model: Any, snapshot: dict[str, Any]) -> None:
    """Copy the host snapshot back into each ``nn.Linear`` weight in-place."""

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
    """In-place WANDA per-output-row unstructured pruning.

    Implements the comparison group used by Sun et al., "A Simple and
    Effective Pruning Approach for Large Language Models" (2023): for each
    Linear layer's weight ``W (out_features, in_features)`` and per-input
    activation RMS ``rms (in_features,)``, score ``S = |W| * rms`` per entry,
    then for **each output row independently** prune the lowest-scoring
    ``prune_ratio`` fraction of entries.

    Why per-row rather than per-layer global thresholding:

    * It matches the canonical WANDA algorithm (groups along the input axis).
    * ``torch.quantile`` cannot ingest the full Qwen2-72B ``lm_head`` score
      matrix (~1.2 B elements) in one call — it raises
      ``RuntimeError: quantile() input tensor is too large``. The per-row
      formulation uses ``torch.topk(..., dim=1)`` which has no such limit.
    """

    import torch
    from torch import nn

    if prune_ratio <= 0.0:
        return
    if prune_ratio >= 1.0:
        raise ValueError("Prune ratio must be in [0, 1).")

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in stats:
            continue
        channel_rms = stats[name].to(module.weight.device)
        weight = module.weight.data
        # Score lives in float32 to avoid bf16 ties at zero magnitude.
        score = (weight.abs().float()) * channel_rms.float().unsqueeze(0)
        in_features = weight.shape[1]
        num_to_prune = int(round(prune_ratio * in_features))
        if num_to_prune <= 0:
            continue
        # Indices (per row) of the bottom-k scoring weights to zero out.
        _, prune_indices = torch.topk(
            score, k=num_to_prune, dim=1, largest=False, sorted=False
        )
        mask = torch.zeros_like(weight, dtype=torch.bool)
        mask.scatter_(1, prune_indices, True)
        weight[mask] = 0


# ---------------------------------------------------------------------------
# Inference adapters
# ---------------------------------------------------------------------------


class InProcessLLMClient:
    """LLMClient adapter calling the in-memory model via ``model.generate``.

    Conforms to the ``LLMClient`` Protocol used by ``run_pipeline`` (see
    ``src/pruning_metrics/evals/coding/llm_client.py``). We deliberately use
    deterministic decoding (``do_sample=False``) to make the autoregressive
    HumanEval+ scores reproducible.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_new_tokens: int,
        seed: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.last_response_metadata: dict[str, Any] | None = None

    def generate_code(self, prompt: str, task_id: str) -> str:
        """Generate one completion (greedy, no teacher forcing) for ``task_id``.

        Parameters
        ----------
        prompt:
            Pre-formatted coding prompt (see ``build_coding_prompt``).
        task_id:
            HumanEval+ task id (used for log lines and response metadata).

        Returns
        -------
        str
            Decoded completion text (excluding the prompt).
        """

        import torch

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        encoded = self.tokenizer(prompt, return_tensors="pt")
        device_map = getattr(self.model, "hf_device_map", None) or {}
        target_device = _embedding_device(self.model, device_map)
        encoded = {key: val.to(target_device) for key, val in encoded.items()}
        prompt_len = int(encoded["input_ids"].shape[1])

        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        self.last_response_metadata = {
            "task_id": task_id,
            "token_count": int(new_tokens.shape[0]),
            "seed": self.seed,
        }
        return text


# ---------------------------------------------------------------------------
# Per-level orchestration
# ---------------------------------------------------------------------------


def run_one_level(
    *,
    config: ExperimentConfig,
    level: float,
    model: Any,
    tokenizer: Any,
    stats: dict[str, Any],
    snapshot: dict[str, Any],
    test_tasks: list[HumanEvalPlusTask],
    teacher_forced_pair: HumanEvalPlusTask,
    level_output_dir: Path,
) -> dict[str, Any]:
    """Restore weights, prune at ``level``, then evaluate + score."""

    LOGGER.info(
        "=== Pruning level %.2f%% — restoring base weights ===", level
    )
    restore_linear_weights(model, snapshot)
    apply_wanda_pruning(model, stats, prune_ratio=float(level) / 100.0)

    level_output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Running HumanEval+ test split (n=%d) on level %.2f%% ...",
        len(test_tasks),
        level,
    )
    eval_records_path = level_output_dir / "eval_records.jsonl"
    llm_client = InProcessLLMClient(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.max_new_tokens,
        seed=config.teacher_forcing_seed,
    )
    pipeline_result = run_pipeline(
        tasks=test_tasks,
        llm_client=llm_client,
        timeout_seconds=config.timeout_seconds,
        output_jsonl_path=str(eval_records_path),
    )
    LOGGER.info(
        "Level %.2f%%: pass@1 = %.4f over %d tasks",
        level,
        pipeline_result.pass_at_1,
        pipeline_result.num_tasks,
    )

    tf_record_path = level_output_dir / "teacher_forced.json"
    LOGGER.info(
        "Computing teacher-forced log-probs on %s for level %.2f%%",
        teacher_forced_pair.task_id,
        level,
    )
    tf_record = compute_teacher_forced_logprobs(
        model=model,
        tokenizer=tokenizer,
        prompt=teacher_forced_pair.prompt,
        answer=teacher_forced_pair.canonical_solution or "",
        model_id=f"{config.base_model_id}@prune={level}",
        task_id=teacher_forced_pair.task_id,
        seed=config.teacher_forcing_seed,
        top_k=5,
    )
    write_teacher_forced_record(tf_record, tf_record_path)
    LOGGER.info(
        "Level %.2f%%: TF avg logprob=%.4f perplexity=%.4f over %d tokens",
        level,
        tf_record.average_logprob,
        tf_record.perplexity,
        tf_record.num_answer_tokens,
    )

    return {
        "pruning_level": level,
        "num_test_tasks": pipeline_result.num_tasks,
        "num_passed": pipeline_result.num_passed,
        "pass_at_1": pipeline_result.pass_at_1,
        "status_breakdown": pipeline_result.status_breakdown,
        "eval_records_path": str(eval_records_path),
        "teacher_forced_path": str(tf_record_path),
        "teacher_forced_summary": {
            "task_id": tf_record.task_id,
            "num_answer_tokens": tf_record.num_answer_tokens,
            "average_logprob": tf_record.average_logprob,
            "perplexity": tf_record.perplexity,
        },
    }, tf_record


# ---------------------------------------------------------------------------
# S3 sync
# ---------------------------------------------------------------------------


def s3_sync(local_dir: Path, bucket: str, key_prefix: str) -> None:
    """Upload everything under ``local_dir`` to ``s3://bucket/key_prefix``.

    Implemented with boto3 to avoid relying on the AWS CLI being installed
    inside the deep-learning AMI's default conda env.
    """

    if not bucket:
        LOGGER.warning(
            "Results bucket is empty; skipping S3 sync (results stay in %s).",
            local_dir,
        )
        return

    import boto3

    client = boto3.client("s3")
    prefix = key_prefix.strip("/")
    files = [path for path in local_dir.rglob("*") if path.is_file()]
    LOGGER.info(
        "Syncing %d files from %s to s3://%s/%s/", len(files), local_dir, bucket, prefix
    )
    for path in files:
        relative = path.relative_to(local_dir).as_posix()
        key = f"{prefix}/{relative}" if prefix else relative
        client.upload_file(str(path), bucket, key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the full pruning experiment. Returns process exit code."""

    args = parse_args()
    pruning_levels = _parse_pruning_levels(args.pruning_levels)
    config = ExperimentConfig(
        base_model_id=args.base_model_id,
        pruning_levels=pruning_levels,
        split_seed=args.split_seed,
        train_frac=args.train_frac,
        teacher_forcing_seed=args.teacher_forcing_seed,
        max_calibration_tokens=args.max_calibration_tokens,
        max_new_tokens=args.max_new_tokens,
        timeout_seconds=args.timeout_seconds,
        output_dir=Path(args.output_dir),
        s3_results_bucket=args.results_bucket,
        s3_results_prefix=f"{args.results_prefix.strip('/')}/{args.run_id}",
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "host": socket.gethostname(),
        "run_id": args.run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_id": config.base_model_id,
        "pruning_levels": list(config.pruning_levels),
        "split_seed": config.split_seed,
        "train_frac": config.train_frac,
        "teacher_forcing_seed": config.teacher_forcing_seed,
        "results_destination": (
            f"s3://{config.s3_results_bucket}/{config.s3_results_prefix}/"
            if config.s3_results_bucket
            else "(local only)"
        ),
    }
    (config.output_dir / "run_metadata.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )
    LOGGER.info("Run metadata: %s", json.dumps(run_meta))

    train_tasks, test_tasks = load_split(config)
    write_split_json(
        train=train_tasks,
        test=test_tasks,
        seed=config.split_seed,
        train_frac=config.train_frac,
        output_path=config.output_dir / "split.json",
    )
    teacher_forced_pair = select_teacher_forced_pair(
        test_tasks, config.teacher_forcing_seed
    )
    LOGGER.info(
        "Teacher-forced pair: %s (canonical solution length %d chars)",
        teacher_forced_pair.task_id,
        len(teacher_forced_pair.canonical_solution or ""),
    )

    tokenizer, model = load_base_model(config.base_model_id)

    calibration_texts = [build_coding_prompt(task) for task in train_tasks]
    stats = collect_activation_stats(
        model=model,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        max_tokens=config.max_calibration_tokens,
    )

    snapshot = snapshot_linear_weights(model)

    level_summaries: list[dict[str, Any]] = []
    teacher_forced_records: list[Any] = []
    started = time.monotonic()

    try:
        for level in config.pruning_levels:
            level_label = (
                str(int(level)) if float(level).is_integer() else str(level)
            )
            level_dir = config.output_dir / f"pruning_level={level_label}"
            level_summary, tf_record = run_one_level(
                config=config,
                level=level,
                model=model,
                tokenizer=tokenizer,
                stats=stats,
                snapshot=snapshot,
                test_tasks=test_tasks,
                teacher_forced_pair=teacher_forced_pair,
                level_output_dir=level_dir,
            )
            level_summaries.append(level_summary)
            teacher_forced_records.append(tf_record)

            partial_summary = {
                "completed_levels": [s["pruning_level"] for s in level_summaries],
                "levels": level_summaries,
            }
            (config.output_dir / "summary.json").write_text(
                json.dumps(partial_summary, indent=2), encoding="utf-8"
            )

            # Incremental S3 sync per level so spot interruption is recoverable.
            s3_sync(
                local_dir=config.output_dir,
                bucket=config.s3_results_bucket,
                key_prefix=config.s3_results_prefix,
            )
    finally:
        elapsed = time.monotonic() - started
        final_summary = {
            "run_id": args.run_id,
            "base_model_id": config.base_model_id,
            "pruning_levels": list(config.pruning_levels),
            "split_seed": config.split_seed,
            "teacher_forcing_seed": config.teacher_forcing_seed,
            "teacher_forced_task_id": teacher_forced_pair.task_id,
            "elapsed_seconds": elapsed,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "levels": level_summaries,
            "teacher_forced_summary": teacher_forced_records_to_summary(
                teacher_forced_records
            ),
        }
        (config.output_dir / "summary.json").write_text(
            json.dumps(final_summary, indent=2, default=str), encoding="utf-8"
        )
        s3_sync(
            local_dir=config.output_dir,
            bucket=config.s3_results_bucket,
            key_prefix=config.s3_results_prefix,
        )
        LOGGER.info(
            "Run finished in %.1fs. Output dir: %s. S3: %s",
            elapsed,
            config.output_dir,
            run_meta["results_destination"],
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
