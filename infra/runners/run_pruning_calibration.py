"""Compute WANDA calibration stats and a manifest, then upload to S3.

This runner is the worker behind notebook 2 (``02_prune_llm.ipynb``). It is
deliberately *narrow*: it does NOT do any autoregressive generation or
teacher forcing. The deliverable is a tiny artifact that downstream runners
(``run_freeform_eval.py`` and ``run_teacher_forced.py``) can fetch and use
to deterministically derive pruned weights at any level.

Outputs at ``s3://<results-bucket>/pruning_artifacts/<run_id>/``:

* ``wanda_stats.pt`` - dict of ``{layer_name: float32 input-channel RMS}``;
  ~few hundred MB for Qwen2-72B (one ``in_features``-sized vector per
  ``Linear``). Re-applying per-row WANDA from this matches the exact
  pruning behaviour of the monolithic runner.
* ``manifest.json`` - configuration the artifact was produced under, plus
  package versions, host metadata, and the calibration dataset spec.
* ``split.json`` - full audit of the seeded train/test partition.
* ``run_metadata.json`` - host + start-time bookkeeping.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from infra.runners._runner_common import (  # noqa: E402
    LOGGER,
    add_common_runner_args,
    collect_wanda_activation_stats,
    configure_logging,
    ensure_src_on_path,
    env_or,
    load_base_model,
    parse_pruning_levels,
    s3_destination,
    s3_sync,
    serialise_config,
    split_csv,
    write_json,
)

ensure_src_on_path()

from pruning_metrics.evals.tasks.registry import (  # noqa: E402
    build_adapter_from_spec,
)


@dataclass(frozen=True)
class CalibrationConfig:
    """Static config for one calibration run.

    Parameters
    ----------
    base_model_id:
        Hugging Face model id (any causal LM).
    dataset_spec:
        Spec passed to :func:`pruning_metrics.evals.tasks.registry.build_adapter_from_spec`,
        e.g. ``coding:evalplus/humanevalplus:test``.
    pruning_levels:
        Percent levels recorded in the manifest. The actual pruning is done
        in downstream runners, but storing the levels here documents the
        intended sweep.
    split_seed:
        Seed for the train/test partition.
    train_frac:
        Fraction routed to the train (calibration) split.
    explicit_train_ids, explicit_test_ids:
        Optional task-id overrides for the partition.
    max_calibration_samples:
        Cap on the number of train-split prompts fed to WANDA hooks. ``None``
        passes everything.
    max_calibration_tokens:
        Per-prompt truncation length when running the calibration forward
        pass.
    output_dir:
        Local directory used as the staging area before S3 sync.
    s3_results_bucket:
        Destination bucket for the artifact.
    s3_results_prefix:
        Object prefix (the run id is appended automatically).
    run_id:
        Identifier appended to the prefix and embedded in the manifest.
    """

    base_model_id: str
    dataset_spec: str
    pruning_levels: tuple[float, ...]
    split_seed: int
    train_frac: float
    explicit_train_ids: tuple[str, ...] | None
    explicit_test_ids: tuple[str, ...] | None
    max_calibration_samples: int | None
    max_calibration_tokens: int
    output_dir: Path
    s3_results_bucket: str
    s3_results_prefix: str
    run_id: str


def parse_args() -> argparse.Namespace:
    """CLI for the calibration runner.

    Defaults are pulled from the environment so the same invocation works
    when the user-data script exports ``BASE_MODEL_ID`` etc.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compute WANDA calibration stats for an LLM and a dataset; "
            "upload the artifact to S3."
        )
    )
    parser.add_argument(
        "--base-model-id",
        default=env_or("BASE_MODEL_ID", default="Qwen/Qwen2-72B"),
    )
    parser.add_argument(
        "--dataset-spec",
        default=env_or(
            "CALIBRATION_DATASET_SPEC",
            "DATASET_SPEC",
            default="coding:evalplus/humanevalplus:test",
        ),
        help="Adapter spec, e.g. 'coding:evalplus/humanevalplus:test'.",
    )
    parser.add_argument(
        "--pruning-levels",
        default=env_or("PRUNING_LEVELS", default="0,20,40,60,80"),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=int(env_or("SPLIT_SEED", default="65320")),
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=float(env_or("TRAIN_FRAC", default="0.8")),
    )
    parser.add_argument(
        "--explicit-train-ids",
        default=env_or("EXPLICIT_TRAIN_IDS", default=""),
        help="Comma-separated task ids forced into the train partition.",
    )
    parser.add_argument(
        "--explicit-test-ids",
        default=env_or("EXPLICIT_TEST_IDS", default=""),
        help="Comma-separated task ids forced into the test partition.",
    )
    parser.add_argument(
        "--max-calibration-samples",
        type=int,
        default=int(env_or("MAX_CALIBRATION_SAMPLES", default="0")),
        help="Cap on number of train prompts used (0 = use all).",
    )
    parser.add_argument(
        "--max-calibration-tokens",
        type=int,
        default=int(env_or("MAX_CALIBRATION_TOKENS", default="512")),
    )
    add_common_runner_args(parser, default_results_prefix="pruning_artifacts")
    return parser.parse_args()


def main() -> int:
    """CLI entry point. Returns shell exit code."""

    configure_logging()
    args = parse_args()

    pruning_levels = parse_pruning_levels(args.pruning_levels)
    config = CalibrationConfig(
        base_model_id=args.base_model_id,
        dataset_spec=args.dataset_spec,
        pruning_levels=pruning_levels,
        split_seed=args.split_seed,
        train_frac=args.train_frac,
        explicit_train_ids=split_csv(args.explicit_train_ids),
        explicit_test_ids=split_csv(args.explicit_test_ids),
        max_calibration_samples=(
            args.max_calibration_samples
            if args.max_calibration_samples > 0
            else None
        ),
        max_calibration_tokens=args.max_calibration_tokens,
        output_dir=Path(args.output_dir),
        s3_results_bucket=args.results_bucket,
        s3_results_prefix=f"{args.results_prefix.strip('/')}/{args.run_id}",
        run_id=args.run_id,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Calibration config: %s", json.dumps(serialise_config(config)))
    _write_run_metadata(config)

    LOGGER.info("Loading task adapter from spec %r", config.dataset_spec)
    adapter = build_adapter_from_spec(config.dataset_spec)
    train_records, test_records = adapter.train_test_split(
        seed=config.split_seed,
        train_frac=config.train_frac,
        explicit_train_ids=config.explicit_train_ids,
        explicit_test_ids=config.explicit_test_ids,
    )
    if not train_records:
        raise RuntimeError("Train split is empty; cannot collect calibration stats.")

    if (
        config.max_calibration_samples is not None
        and len(train_records) > config.max_calibration_samples
    ):
        # Native splits (e.g. GSM8K's 7473-row train) are returned in dataset
        # order, so a naive slice would always pick the first N rows. Apply a
        # seeded shuffle keyed on ``SPLIT_SEED`` so the cap is reproducible
        # regardless of HF row order.
        import random  # local import to keep the module light at import time

        rng = random.Random(config.split_seed)
        shuffled = list(train_records)
        rng.shuffle(shuffled)
        train_records = shuffled[: config.max_calibration_samples]

    LOGGER.info(
        "Split: %d train (calibration) / %d test, dataset=%s",
        len(train_records),
        len(test_records),
        adapter.dataset_spec,
    )

    _write_split_json(adapter, train_records, test_records, config)

    tokenizer, model = load_base_model(config.base_model_id)
    started = time.monotonic()
    calibration_texts = [record.prompt for record in train_records]
    stats = collect_wanda_activation_stats(
        model=model,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        max_tokens=config.max_calibration_tokens,
    )
    LOGGER.info(
        "Stats collected in %.1fs over %d layers.",
        time.monotonic() - started,
        len(stats),
    )

    _write_stats_artifact(stats, config)
    _write_manifest(adapter, train_records, test_records, config)

    s3_sync(
        local_dir=config.output_dir,
        bucket=config.s3_results_bucket,
        key_prefix=config.s3_results_prefix,
    )

    LOGGER.info(
        "Calibration artifact ready: s3://%s/%s/",
        config.s3_results_bucket,
        config.s3_results_prefix,
    )
    return 0


def _write_run_metadata(config: CalibrationConfig) -> None:
    payload = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "run_id": config.run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_id": config.base_model_id,
        "dataset_spec": config.dataset_spec,
        "destination": s3_destination(
            config.s3_results_bucket, config.s3_results_prefix
        ),
    }
    write_json(config.output_dir / "run_metadata.json", payload)


def _write_split_json(
    adapter: Any,
    train_records: list[Any],
    test_records: list[Any],
    config: CalibrationConfig,
) -> None:
    payload = {
        "dataset_spec": adapter.dataset_spec,
        "task_adapter": adapter.name,
        "seed": config.split_seed,
        "train_frac": config.train_frac,
        "num_train": len(train_records),
        "num_test": len(test_records),
        "max_calibration_samples": config.max_calibration_samples,
        "explicit_train_ids": (
            list(config.explicit_train_ids) if config.explicit_train_ids else None
        ),
        "explicit_test_ids": (
            list(config.explicit_test_ids) if config.explicit_test_ids else None
        ),
        "train_task_ids": [r.task_id for r in train_records],
        "test_task_ids": [r.task_id for r in test_records],
    }
    write_json(config.output_dir / "split.json", payload)
    LOGGER.info("Split written to %s", config.output_dir / "split.json")


def _write_stats_artifact(stats: dict[str, Any], config: CalibrationConfig) -> None:
    """Save the per-channel RMS dict as a torch tensor archive."""

    import torch  # local import; tests don't need torch loaded just to import this module

    target = config.output_dir / "wanda_stats.pt"
    torch.save(stats, target)
    LOGGER.info("WANDA stats saved to %s (%d layers)", target, len(stats))


def _write_manifest(
    adapter: Any,
    train_records: list[Any],
    test_records: list[Any],
    config: CalibrationConfig,
) -> None:
    """Persist a JSON manifest that fully describes how to reproduce
    the artifact (dataset, seeds, package versions)."""

    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for module_name in ("torch", "transformers", "datasets", "boto3"):
        try:  # pragma: no cover - thin wrapper
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:  # pylint: disable=broad-exception-caught
            versions[module_name] = "missing"

    payload = {
        "schema_version": "1",
        "run_id": config.run_id,
        "base_model_id": config.base_model_id,
        "dataset_spec": adapter.dataset_spec,
        "task_adapter": adapter.name,
        "pruning_levels": list(config.pruning_levels),
        "split_seed": config.split_seed,
        "train_frac": config.train_frac,
        "max_calibration_samples": config.max_calibration_samples,
        "max_calibration_tokens": config.max_calibration_tokens,
        "num_train": len(train_records),
        "num_test": len(test_records),
        "explicit_train_ids": (
            list(config.explicit_train_ids) if config.explicit_train_ids else None
        ),
        "explicit_test_ids": (
            list(config.explicit_test_ids) if config.explicit_test_ids else None
        ),
        "package_versions": versions,
        "host": socket.gethostname(),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_paths": {
            "wanda_stats": "wanda_stats.pt",
            "split": "split.json",
            "manifest": "manifest.json",
            "run_metadata": "run_metadata.json",
        },
    }
    write_json(config.output_dir / "manifest.json", payload)
    LOGGER.info("Manifest written to %s", config.output_dir / "manifest.json")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
