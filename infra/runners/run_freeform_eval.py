"""Free-form (no teacher forcing) per-level evaluation runner.

Worker behind notebook 3 (``03_freeform_eval.ipynb``). Given a pruning
calibration artifact (produced by ``run_pruning_calibration.py``) and a
task-adapter spec, this runner:

1. Downloads ``manifest.json`` + ``split.json`` + ``wanda_stats.pt`` from
   ``s3://<bucket>/pruning_artifacts/<run_id>/``.
2. Loads the matching base model once (``device_map='auto'``, bf16).
3. Snapshots the original ``Linear`` weights to host RAM.
4. For each requested pruning level: restore -> apply per-row WANDA
   (matching ``run_pruning_calibration.py``'s scoring) -> generate the test
   split greedily -> run the adapter's ``verify`` -> compute teacher-forced
   perplexity on the ground-truth answer -> persist
   ``level=NN/eval_records.jsonl`` and update ``summary.json`` -> incremental
   S3 sync so a spot interruption never loses a completed level.

The free-form generation seed (``--generation-seed``) controls
``torch.manual_seed`` so repeated runs are bit-exact deterministic.

Perplexity is computed via a single teacher-forced forward pass on
``(record.prompt, record.target_text)`` after greedy generation for each
test record. This adds <25% overhead versus the generation pass alone and
produces calibrated ``exp(-mean_logprob)`` scores that are meaningful to
compare across pruning levels.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from infra.runners._runner_common import (  # noqa: E402
    LOGGER,
    add_common_runner_args,
    add_eval_artifact_args,
    configure_logging,
    download_calibration_artifact,
    ensure_src_on_path,
    env_or,
    eval_run_metadata,
    eval_summary_payload,
    level_label,
    load_model_stats_snapshot,
    resolve_eval_defaults,
    run_level_sweep,
    serialise_config,
    write_json,
)

ensure_src_on_path()

from pruning_metrics.evals.coding.teacher_forcing import (  # noqa: E402
    compute_teacher_forced_logprobs,
    resolve_input_device,
)
from pruning_metrics.evals.tasks.base import TaskRecord  # noqa: E402
from pruning_metrics.evals.tasks.registry import (  # noqa: E402
    build_adapter_from_spec,
)


@dataclass(frozen=True)
class FreeformConfig:
    """Static config for a free-form evaluation run.

    Parameters
    ----------
    artifact_uri:
        ``s3://<bucket>/pruning_artifacts/<run_id>/`` produced by the
        calibration runner.
    eval_dataset_spec:
        Adapter spec used to *load eval records*. May differ from the
        calibration dataset (e.g. calibrate on HumanEval+, evaluate on
        GSM8K).
    eval_levels:
        Subset of pruning levels (in percent) to evaluate.
    generation_seed:
        Seed for ``torch.manual_seed`` so greedy decoding is reproducible.
    max_new_tokens:
        Generation budget per task.
    timeout_seconds:
        Per-task verifier timeout.
    output_dir, run_id, results_bucket, results_prefix:
        Local + S3 staging. Output goes to
        ``s3://<bucket>/freeform_eval/<run_id>/``.
    """

    artifact_uri: str
    eval_dataset_spec: str
    eval_levels: tuple[float, ...]
    generation_seed: int
    max_new_tokens: int
    timeout_seconds: float
    split_seed: int
    train_frac: float
    explicit_train_ids: tuple[str, ...] | None
    explicit_test_ids: tuple[str, ...] | None
    max_test_samples: int | None
    output_dir: Path
    run_id: str
    results_bucket: str
    results_prefix: str


def parse_args() -> argparse.Namespace:
    """CLI for the free-form eval runner."""

    parser = argparse.ArgumentParser(
        description=(
            "Free-form evaluation against a pruning calibration artifact."
        )
    )
    add_eval_artifact_args(parser)
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=int(env_or("GENERATION_SEED", default="65320")),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(env_or("MAX_NEW_TOKENS", default="512")),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(env_or("TIMEOUT_SECONDS", default="10.0")),
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=int(env_or("MAX_TEST_SAMPLES", default="0")),
        help="Cap on number of test records evaluated (0 = use all).",
    )
    add_common_runner_args(parser, default_results_prefix="freeform_eval")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    configure_logging()
    args = parse_args()
    if not args.artifact_uri:
        raise SystemExit("--artifact-uri is required")

    # Pull the manifest first so eval defaults align with the artifact.
    manifest, artifact_dir = download_calibration_artifact(
        args.artifact_uri, Path(args.output_dir)
    )

    config = FreeformConfig(
        artifact_uri=args.artifact_uri,
        generation_seed=args.generation_seed,
        max_new_tokens=args.max_new_tokens,
        timeout_seconds=args.timeout_seconds,
        max_test_samples=args.max_test_samples or None,
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        results_bucket=args.results_bucket,
        results_prefix=f"{args.results_prefix.strip('/')}/{args.run_id}",
        **resolve_eval_defaults(manifest, args),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Free-form eval config: %s", json.dumps(serialise_config(config)))
    write_json(
        config.output_dir / "run_metadata.json",
        eval_run_metadata(config, manifest, "freeform_eval"),
    )

    LOGGER.info("Building eval adapter from spec %r", config.eval_dataset_spec)
    adapter = build_adapter_from_spec(config.eval_dataset_spec)
    _, test_records = adapter.train_test_split(
        seed=config.split_seed,
        train_frac=config.train_frac,
        explicit_train_ids=config.explicit_train_ids,
        explicit_test_ids=config.explicit_test_ids,
    )
    if config.max_test_samples is not None:
        test_records = test_records[: config.max_test_samples]
    LOGGER.info(
        "Eval set has %d records (after max_test_samples=%s)",
        len(test_records),
        config.max_test_samples,
    )
    if not test_records:
        raise RuntimeError("Eval test split is empty.")

    tokenizer, model, stats, snapshot = load_model_stats_snapshot(
        manifest, artifact_dir
    )

    level_summaries: list[dict[str, Any]] = []

    def _per_level(level: float) -> None:
        level_dir = config.output_dir / f"level={level_label(level)}"
        level_dir.mkdir(parents=True, exist_ok=True)
        level_summaries.append(
            _evaluate_level(
                level=level,
                model=model,
                tokenizer=tokenizer,
                adapter=adapter,
                test_records=test_records,
                config=config,
                records_path=level_dir / "eval_records.jsonl",
                base_model_id=manifest["base_model_id"],
            )
        )

    run_level_sweep(
        config,
        model=model,
        stats=stats,
        snapshot=snapshot,
        mode_label="Free-form eval",
        per_level=_per_level,
        write_summary=lambda **kw: _write_summary(
            config, level_summaries, manifest, **kw
        ),
    )

    return 0


def _evaluate_level(
    *,
    level: float,
    model: Any,
    tokenizer: Any,
    adapter: Any,
    test_records: list[TaskRecord],
    config: FreeformConfig,
    records_path: Path,
    base_model_id: str,
) -> dict[str, Any]:
    """Generate + verify every test record for one pruning level.

    For each record this function performs two passes:

    1. **Greedy generation**: produces the model's free-form completion for
       accuracy measurement (``pass@1`` for coding, exact-match for math/MCQ).
    2. **Teacher-forced forward pass**: computes per-token log-probabilities of
       the ground-truth ``record.target_text`` given ``record.prompt``, yielding
       calibrated perplexity without a second model snapshot.

    Parameters
    ----------
    level:
        Sparsity level in percent (e.g. 20.0 means 20% of weights zeroed).
    model:
        The pruned causal LM (weights already modified in-place by the caller).
    tokenizer:
        Matching HuggingFace tokenizer.
    adapter:
        Task adapter providing ``build_inference_prompt`` and ``verify``.
    test_records:
        Records drawn from the adapter's test split.
    config:
        Run-level configuration (seeds, token limits, etc.).
    records_path:
        Output JSONL path for per-task records.
    base_model_id:
        Base model identifier used to label the teacher-forced record.

    Returns
    -------
    dict[str, Any]
        Level-level summary including ``pass_at_1``, ``average_perplexity``,
        and status breakdowns.
    """

    import torch

    target_device = resolve_input_device(model)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    statuses: dict[str, int] = {}
    perplexities: list[float] = []
    written = 0
    started = time.monotonic()

    with records_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(test_records, start=1):
            torch.manual_seed(config.generation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.generation_seed)
            random.seed(config.generation_seed)

            # --- greedy generation pass ---
            inference_prompt = adapter.build_inference_prompt(record)
            encoded = tokenizer(inference_prompt, return_tensors="pt")
            encoded = {k: v.to(target_device) for k, v in encoded.items()}
            prompt_len = int(encoded["input_ids"].shape[1])
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=pad_token_id,
                )
            new_tokens = outputs[0, prompt_len:]
            generated_text = tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )
            outcome = adapter.verify(
                record,
                generated_text,
                timeout_seconds=config.timeout_seconds,
            )
            statuses[outcome.status] = statuses.get(outcome.status, 0) + 1

            # --- teacher-forced perplexity pass on ground-truth answer ---
            # Uses record.prompt (not the instruction-wrapped inference_prompt)
            # to stay consistent with the standalone teacher-forced runner.
            perplexity: float | None = None
            average_logprob: float | None = None
            num_target_tokens: int = 0
            if record.target_text:
                try:
                    tf_record = compute_teacher_forced_logprobs(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=record.prompt,
                        answer=record.target_text,
                        model_id=f"{base_model_id}@prune={level_label(level)}",
                        task_id=record.task_id,
                        seed=config.generation_seed,
                        top_k=1,
                    )
                    perplexity = tf_record.perplexity
                    average_logprob = tf_record.average_logprob
                    num_target_tokens = tf_record.num_answer_tokens
                    perplexities.append(perplexity)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    LOGGER.warning(
                        "TF perplexity failed for %s at level %s%%: %s",
                        record.task_id,
                        level_label(level),
                        exc,
                    )

            written += 1
            handle.write(
                json.dumps(
                    {
                        "task_id": record.task_id,
                        "pruning_level": level,
                        "tf_prompt": record.prompt,
                        "inference_prompt": inference_prompt,
                        "target_text": record.target_text,
                        "generated_text": generated_text,
                        "num_generated_tokens": int(new_tokens.shape[0]),
                        "verification_status": outcome.status,
                        "verification_detail": outcome.detail,
                        "perplexity": perplexity,
                        "average_logprob": average_logprob,
                        "num_target_tokens": num_target_tokens,
                    }
                )
                + "\n"
            )
            if index % 5 == 0 or index == len(test_records):
                LOGGER.info(
                    "Level %s%%: %d/%d evaluated; running statuses=%s",
                    level_label(level),
                    index,
                    len(test_records),
                    statuses,
                )

    elapsed = time.monotonic() - started
    num_passed = statuses.get("pass", 0)
    pass_at_1 = num_passed / written if written else 0.0
    avg_perplexity = (
        float(sum(perplexities) / len(perplexities)) if perplexities else None
    )
    LOGGER.info(
        "Level %s%%: pass@1=%.4f avg_perplexity=%s over %d in %.1fs",
        level_label(level),
        pass_at_1,
        f"{avg_perplexity:.4f}" if avg_perplexity is not None else "N/A",
        written,
        elapsed,
    )
    return {
        "pruning_level": level,
        "num_test_tasks": written,
        "num_passed": num_passed,
        "pass_at_1": pass_at_1,
        "average_perplexity": avg_perplexity,
        "status_breakdown": statuses,
        "elapsed_seconds": elapsed,
        "eval_records_path": str(records_path),
    }


def _write_summary(
    config: FreeformConfig,
    level_summaries: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    elapsed_seconds: float | None = None,
    ended: bool = False,
) -> None:
    payload = eval_summary_payload(
        config,
        manifest,
        "freeform_eval",
        {
            "split_seed": config.split_seed,
            "train_frac": config.train_frac,
            "generation_seed": config.generation_seed,
            "max_new_tokens": config.max_new_tokens,
            "timeout_seconds": config.timeout_seconds,
            "completed_levels": [s["pruning_level"] for s in level_summaries],
            "levels": level_summaries,
        },
        ended=ended,
        elapsed_seconds=elapsed_seconds,
    )
    write_json(config.output_dir / "summary.json", payload)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
