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
import socket
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# pylint: disable=wrong-import-position
from infra.ec2._runner_common import (  # noqa: E402
    LOGGER,
    apply_wanda_pruning,
    configure_logging,
    env_or,
    level_label,
    load_base_model,
    parse_pruning_levels,
    restore_linear_weights,
    s3_download_file,
    s3_sync,
    snapshot_linear_weights,
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


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def parse_args() -> argparse.Namespace:
    """CLI for the teacher-forced runner."""

    parser = argparse.ArgumentParser(
        description=(
            "Teacher-forced next-token log-probabilities per pruning level."
        )
    )
    parser.add_argument(
        "--artifact-uri",
        default=env_or("PRUNING_ARTIFACT_URI", default=""),
    )
    parser.add_argument(
        "--eval-dataset-spec",
        default=env_or("EVAL_DATASET_SPEC", "DATASET_SPEC", default=""),
    )
    parser.add_argument(
        "--eval-levels",
        default=env_or("EVAL_LEVELS", default=""),
    )
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
        default=env_or("RESULTS_PREFIX", default="teacher_forced"),
    )
    parser.add_argument("--run-id", default=env_or("RUN_ID", default=_default_run_id()))
    return parser.parse_args()


def _split_csv(raw: str) -> tuple[str, ...] | None:
    if not raw or raw.strip().lower() in ("", "none"):
        return None
    return tuple(token.strip() for token in raw.split(",") if token.strip())


def main() -> int:
    """CLI entry point."""

    configure_logging()
    args = parse_args()
    if not args.artifact_uri:
        raise SystemExit("--artifact-uri is required")
    if args.num_tf_samples < 0:
        raise SystemExit("--num-tf-samples must be >= 0 (0 = all test records)")

    artifact_dir = Path(args.output_dir) / "_artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest_uri = args.artifact_uri.rstrip("/") + "/manifest.json"
    stats_uri = args.artifact_uri.rstrip("/") + "/wanda_stats.pt"
    LOGGER.info("Downloading manifest %s", manifest_uri)
    s3_download_file(manifest_uri, artifact_dir / "manifest.json")
    s3_download_file(stats_uri, artifact_dir / "wanda_stats.pt")
    manifest = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )

    eval_dataset_spec = args.eval_dataset_spec or manifest["dataset_spec"]
    eval_levels_raw = args.eval_levels or ",".join(
        str(level) for level in manifest["pruning_levels"]
    )
    eval_levels = parse_pruning_levels(eval_levels_raw)

    config = TeacherForcedConfig(
        artifact_uri=args.artifact_uri,
        eval_dataset_spec=eval_dataset_spec,
        eval_levels=eval_levels,
        tf_seed=args.tf_seed,
        num_tf_samples=args.num_tf_samples,
        explicit_sample_task_ids=_split_csv(args.explicit_sample_task_ids),
        top_k=args.top_k,
        split_seed=int(manifest["split_seed"]),
        train_frac=float(manifest["train_frac"]),
        explicit_train_ids=tuple(manifest.get("explicit_train_ids") or ()) or None,
        explicit_test_ids=tuple(manifest.get("explicit_test_ids") or ()) or None,
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        results_bucket=args.results_bucket,
        results_prefix=f"{args.results_prefix.strip('/')}/{args.run_id}",
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Teacher-forced config: %s", json.dumps(_serialise_config(config)))
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

    tokenizer, model = load_base_model(manifest["base_model_id"])
    import torch

    LOGGER.info("Loading WANDA stats")
    stats = torch.load(artifact_dir / "wanda_stats.pt", map_location="cpu")
    snapshot = snapshot_linear_weights(model)

    sample_summaries: dict[str, list[dict[str, Any]]] = {
        record.task_id: [] for record in sampled_records
    }
    started = time.monotonic()

    try:
        for level in config.eval_levels:
            LOGGER.info("=== Pruning level %s%% ===", level_label(level))
            restore_linear_weights(model, snapshot)
            apply_wanda_pruning(model, stats, prune_ratio=float(level) / 100.0)

            for sample_idx, record in enumerate(sampled_records):
                if not record.target_text:
                    LOGGER.warning(
                        "Skipping %s at level %s: empty target_text.",
                        record.task_id,
                        level_label(level),
                    )
                    continue
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


def _serialise_config(config: TeacherForcedConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["eval_levels"] = list(config.eval_levels)
    payload["explicit_train_ids"] = (
        list(config.explicit_train_ids) if config.explicit_train_ids else None
    )
    payload["explicit_test_ids"] = (
        list(config.explicit_test_ids) if config.explicit_test_ids else None
    )
    payload["explicit_sample_task_ids"] = (
        list(config.explicit_sample_task_ids)
        if config.explicit_sample_task_ids
        else None
    )
    return payload


def _write_run_metadata(
    config: TeacherForcedConfig, manifest: dict[str, Any]
) -> None:
    payload = {
        "host": socket.gethostname(),
        "run_id": config.run_id,
        "mode": "teacher_forced",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_uri": config.artifact_uri,
        "base_model_id": manifest["base_model_id"],
        "calibration_dataset_spec": manifest["dataset_spec"],
        "eval_dataset_spec": config.eval_dataset_spec,
        "tf_seed": config.tf_seed,
        "num_tf_samples": config.num_tf_samples,
        "destination": (
            f"s3://{config.results_bucket}/{config.results_prefix}/"
            if config.results_bucket
            else "(local only)"
        ),
    }
    (config.output_dir / "run_metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


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
    (config.output_dir / "sample_selection.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


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
    (config.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
