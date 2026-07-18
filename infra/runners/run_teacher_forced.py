"""Teacher-forced next-token log-prob runner.

Worker behind notebook 4 (``04_teacher_forced.ipynb``). Given a pruning
calibration artifact + a task-adapter spec, this runner picks
``NUM_TF_SAMPLES`` test records deterministically using ``TF_SEED`` (default
65320), and for each requested pruning level computes the per-token log
probability of the ground-truth answer under perfect teacher forcing.

Per-level outputs land in
``s3://<bucket>/teacher_forced/<run_id>/level=NN/sample=KKK/per_token.json``
together with a rolling ``summary.json`` aggregating average log-probability
and perplexity per (sample, level).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# pylint: disable=wrong-import-position
from infra.runners._runner_common import (  # noqa: E402
    LOGGER,
    add_common_runner_args,
    add_eval_artifact_args,
    configure_logging,
    download_calibration_artifact,
    env_or,
    eval_run_metadata,
    level_label,
    load_model_stats_snapshot,
    prune_to_level,
    resolve_eval_defaults,
    s3_sync,
    serialise_config,
    split_csv,
    write_json,
)
from pruning_metrics.evals.coding.teacher_forcing import (  # noqa: E402
    compute_teacher_forced_logprobs,
    write_teacher_forced_record,
)
from pruning_metrics.evals.tasks.base import TaskRecord  # noqa: E402
from pruning_metrics.evals.tasks.registry import (  # noqa: E402
    build_adapter_from_spec,
)


@dataclass(frozen=True)
class TeacherForcedConfig:
    """Static config for one teacher-forced run.

    Parameters
    ----------
    artifact_uri:
        Calibration artifact URI (output of the calibration runner).
    eval_dataset_spec:
        Adapter spec for selecting ``(prompt, target_text)`` pairs.
    eval_levels:
        Subset of pruning levels to evaluate.
    tf_seed:
        Seed used to pick the sampled records from the test split.
    num_tf_samples:
        Number of distinct records scored. Use 0 to score all test records
        (sorted deterministically by task_id). Use >= 1 for a seeded random
        sample.
    explicit_sample_task_ids:
        Optional task-id list overriding the seeded sample selection. Must
        be a subset of the test split.
    top_k:
        Number of alternative tokens recorded per teacher-forced position.
    """

    artifact_uri: str
    eval_dataset_spec: str
    eval_levels: tuple[float, ...]
    tf_seed: int
    num_tf_samples: int
    explicit_sample_task_ids: tuple[str, ...] | None
    top_k: int
    split_seed: int
    train_frac: float
    explicit_train_ids: tuple[str, ...] | None
    explicit_test_ids: tuple[str, ...] | None
    output_dir: Path
    run_id: str
    results_bucket: str
    results_prefix: str


def parse_args() -> argparse.Namespace:
    """CLI for the teacher-forced runner."""

    parser = argparse.ArgumentParser(
        description=(
            "Teacher-forced next-token log-probabilities per pruning level."
        )
    )
    add_eval_artifact_args(parser)
    parser.add_argument(
        "--tf-seed",
        type=int,
        default=int(env_or("TF_SEED", default="65320")),
    )
    parser.add_argument(
        "--num-tf-samples",
        type=int,
        default=int(env_or("NUM_TF_SAMPLES", default="1")),
        help=(
            "Number of test records to score with teacher forcing. "
            "0 = all test records (sorted by task_id for determinism); "
            ">= 1 = seeded random sample."
        ),
    )
    parser.add_argument(
        "--explicit-sample-task-ids",
        default=env_or("EXPLICIT_SAMPLE_TASK_IDS", default=""),
        help="Comma-separated task ids to TF (overrides seeded sampling).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(env_or("TF_TOP_K", default="5")),
    )
    add_common_runner_args(parser, default_results_prefix="teacher_forced")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    configure_logging()
    args = parse_args()
    if not args.artifact_uri:
        raise SystemExit("--artifact-uri is required")
    if args.num_tf_samples < 0:
        raise SystemExit("--num-tf-samples must be >= 0 (0 = all test records)")

    manifest, artifact_dir = download_calibration_artifact(
        args.artifact_uri, Path(args.output_dir)
    )

    config = TeacherForcedConfig(
        artifact_uri=args.artifact_uri,
        tf_seed=args.tf_seed,
        num_tf_samples=args.num_tf_samples,
        explicit_sample_task_ids=split_csv(args.explicit_sample_task_ids),
        top_k=args.top_k,
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        results_bucket=args.results_bucket,
        results_prefix=f"{args.results_prefix.strip('/')}/{args.run_id}",
        **resolve_eval_defaults(manifest, args),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Teacher-forced config: %s", json.dumps(serialise_config(config)))
    _write_run_metadata(config, manifest)

    LOGGER.info("Building eval adapter from spec %r", config.eval_dataset_spec)
    adapter = build_adapter_from_spec(config.eval_dataset_spec)
    _, test_records = adapter.train_test_split(
        seed=config.split_seed,
        train_frac=config.train_frac,
        explicit_train_ids=config.explicit_train_ids,
        explicit_test_ids=config.explicit_test_ids,
    )
    if not test_records:
        raise RuntimeError("Eval test split is empty.")

    sampled_records = _select_tf_samples(test_records, config)
    LOGGER.info(
        "Selected %d teacher-forcing sample(s): %s",
        len(sampled_records),
        [r.task_id for r in sampled_records],
    )
    _write_sample_selection(config, sampled_records)

    tokenizer, model, stats, snapshot = load_model_stats_snapshot(
        manifest, artifact_dir
    )

    sample_summaries: dict[str, list[dict[str, Any]]] = {
        record.task_id: [] for record in sampled_records
    }
    started = time.monotonic()

    try:
        for level in config.eval_levels:
            LOGGER.info("=== Pruning level %s%% ===", level_label(level))
            prune_to_level(model, stats, snapshot, level)

            for sample_idx, record in enumerate(sampled_records):
                if not record.target_text:
                    LOGGER.warning(
                        "Skipping %s at level %s: empty target_text.",
                        record.task_id,
                        level_label(level),
                    )
                    continue
                try:
                    tf_record = compute_teacher_forced_logprobs(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=record.prompt,
                        answer=record.target_text,
                        model_id=f"{manifest['base_model_id']}@prune={level}",
                        task_id=record.task_id,
                        seed=config.tf_seed,
                        top_k=config.top_k,
                    )
                except ValueError as exc:
                    LOGGER.warning(
                        "Skipping sample %d task=%s at level=%s — tokenisation error: %s",
                        sample_idx,
                        record.task_id,
                        level_label(level),
                        exc,
                    )
                    continue
                level_dir = (
                    config.output_dir
                    / f"level={level_label(level)}"
                    / f"sample={sample_idx:03d}_task={_safe_filename(record.task_id)}"
                )
                level_dir.mkdir(parents=True, exist_ok=True)
                write_teacher_forced_record(tf_record, level_dir / "per_token.json")

                LOGGER.info(
                    "level=%s sample=%s task=%s avg_logp=%.4f ppl=%.4f over %d tokens",
                    level_label(level),
                    sample_idx,
                    record.task_id,
                    tf_record.average_logprob,
                    tf_record.perplexity,
                    tf_record.num_answer_tokens,
                )
                sample_summaries[record.task_id].append(
                    {
                        "pruning_level": level,
                        "average_logprob": tf_record.average_logprob,
                        "perplexity": tf_record.perplexity,
                        "num_answer_tokens": tf_record.num_answer_tokens,
                        "per_token_path": str(level_dir / "per_token.json"),
                    }
                )

            _write_summary(config, sample_summaries, manifest)
            s3_sync(
                local_dir=config.output_dir,
                bucket=config.results_bucket,
                key_prefix=config.results_prefix,
            )
    finally:
        _write_summary(
            config,
            sample_summaries,
            manifest,
            elapsed_seconds=time.monotonic() - started,
            ended=True,
        )
        s3_sync(
            local_dir=config.output_dir,
            bucket=config.results_bucket,
            key_prefix=config.results_prefix,
        )
        LOGGER.info(
            "Teacher-forced run finished. Results: s3://%s/%s/",
            config.results_bucket,
            config.results_prefix,
        )

    return 0


def _select_tf_samples(
    test_records: list[TaskRecord],
    config: TeacherForcedConfig,
) -> list[TaskRecord]:
    """Pick records for teacher-forced scoring.

    Selection priority:

    1. ``explicit_sample_task_ids`` — caller-specified task ids (must be a
       subset of the test split).
    2. ``num_tf_samples == 0`` — return all test records sorted by task_id
       for determinism.
    3. ``num_tf_samples >= 1`` — seeded random sample of that size.

    Parameters
    ----------
    test_records:
        Full test split produced by the adapter's ``train_test_split``.
    config:
        Run config carrying ``num_tf_samples``, ``tf_seed``, and optional
        ``explicit_sample_task_ids``.

    Returns
    -------
    list[TaskRecord]
        Selected records in a deterministic order.
    """

    if config.explicit_sample_task_ids:
        id_set = set(config.explicit_sample_task_ids)
        selected = [r for r in test_records if r.task_id in id_set]
        missing = id_set - {r.task_id for r in selected}
        if missing:
            raise RuntimeError(
                "Explicit sample task ids not found in test split: "
                + ", ".join(sorted(missing))
            )
        return selected

    # 0 means "score every test record" — sorted for determinism.
    if config.num_tf_samples == 0:
        return sorted(test_records, key=lambda r: r.task_id)

    sorted_records = sorted(test_records, key=lambda r: r.task_id)
    rng = random.Random(config.tf_seed)
    indices = list(range(len(sorted_records)))
    rng.shuffle(indices)
    chosen = [sorted_records[i] for i in indices[: config.num_tf_samples]]
    if not chosen:
        raise RuntimeError("Teacher-forced sample selection produced 0 records.")
    return chosen


def _safe_filename(task_id: str) -> str:
    """Produce a filesystem-safe slug for a task id."""

    return task_id.replace("/", "_").replace(" ", "_")


def _write_run_metadata(
    config: TeacherForcedConfig, manifest: dict[str, Any]
) -> None:
    payload = eval_run_metadata(config, manifest, "teacher_forced")
    payload["tf_seed"] = config.tf_seed
    payload["num_tf_samples"] = config.num_tf_samples
    write_json(config.output_dir / "run_metadata.json", payload)


def _write_sample_selection(
    config: TeacherForcedConfig, sampled_records: list[TaskRecord]
) -> None:
    payload = {
        "tf_seed": config.tf_seed,
        "num_tf_samples": config.num_tf_samples,
        "explicit_sample_task_ids": (
            list(config.explicit_sample_task_ids)
            if config.explicit_sample_task_ids
            else None
        ),
        "selected_task_ids": [r.task_id for r in sampled_records],
        "selected_records": [
            {
                "task_id": r.task_id,
                "prompt_chars": len(r.prompt),
                "target_text_chars": len(r.target_text),
            }
            for r in sampled_records
        ],
    }
    write_json(config.output_dir / "sample_selection.json", payload)


def _write_summary(
    config: TeacherForcedConfig,
    sample_summaries: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
    *,
    elapsed_seconds: float | None = None,
    ended: bool = False,
) -> None:
    payload = {
        "mode": "teacher_forced",
        "run_id": config.run_id,
        "artifact_uri": config.artifact_uri,
        "base_model_id": manifest["base_model_id"],
        "calibration_dataset_spec": manifest["dataset_spec"],
        "eval_dataset_spec": config.eval_dataset_spec,
        "tf_seed": config.tf_seed,
        "num_tf_samples": config.num_tf_samples,
        "samples": {
            task_id: {
                "completed_levels": [s["pruning_level"] for s in entries],
                "by_level": entries,
            }
            for task_id, entries in sample_summaries.items()
        },
        "ended_at_utc": (
            datetime.now(timezone.utc).isoformat() if ended else None
        ),
        "elapsed_seconds": elapsed_seconds,
    }
    write_json(config.output_dir / "summary.json", payload)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
