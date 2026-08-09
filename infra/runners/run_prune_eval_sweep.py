"""Prune + teacher-forced eval sweep runner for the v2 experiment.

Worker behind orchestration notebook 6 (``06_prune_eval_v2.ipynb``). Unlike
the three v1 runners (``run_pruning_calibration.py`` /
``run_freeform_eval.py`` / ``run_teacher_forced.py``), which split
"collect calibration stats" and "evaluate against a calibration artifact"
across two GPU jobs joined by an S3 artifact, this runner is a single
self-contained job: for one ``(pruner, calibration_domain, seed)`` grid
point it

1. loads the base model once,
2. builds a chunk of calibration texts from one domain's train split,
3. prunes to every requested level with either WANDA (row-wise, activation
   RMS-scaled magnitude) or SparseGPT (Hessian-aware, sequential per layer),
4. extracts + saves the resulting pruning mask (full + a packed digest) for
   every non-zero level, and
5. teacher-forces the SAME seeded sample of test records from every
   requested eval benchmark at every level, writing ``per_token.json``
   records byte-for-byte compatible with ``run_teacher_forced.py``'s output
   so :mod:`pruning_metrics.metrics` can consume either interchangeably.

Design: this runner intentionally does not reuse
``infra.runners._runner_common.run_level_sweep`` / ``prune_to_level``. Those
helpers hard-code the WANDA "restore -> apply_wanda_pruning(stats, ratio)"
step and sync once per level; this runner needs to dispatch between two
different pruning algorithms with different call signatures (WANDA takes
precomputed per-channel stats, SparseGPT takes calibration texts and the
tokenizer directly) and the C4 contract calls for syncing after each
``(level, bench)`` pair rather than once per level (so a spot interruption
mid-sweep loses at most one benchmark's worth of teacher-forced records).
The per-level "restore, maybe prune, maybe extract masks" shape and the
top-level try/finally (so a rolling ``summary.json`` and a final S3 sync
always happen even under ``SIGTERM``) are still deliberately mirrored from
``run_level_sweep`` -- see :func:`main`.

Output layout under ``<output-dir>`` (synced to
``s3://<bucket>/prune_eval_v2/<run_id>/``):

* ``manifest.json`` -- full config (pruner, calibration spec/seed/chunk,
  levels, eval specs, package versions).
* ``run_metadata.json`` -- host + start-time bookkeeping.
* ``masks/level=NN.npz`` / ``masks/level=NN.digest.npz`` -- packed pruning
  masks for every level > 0.
* ``level=NN/bench=<safe_spec>/sample=KKK_task=<safe_task>/per_token.json``
  -- per-sample teacher-forced records.
* ``summary.json`` -- rolling per-(level, bench) aggregate stats.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from infra.runners._runner_common import (  # noqa: E402
    LOGGER,
    add_common_runner_args,
    apply_wanda_pruning,
    collect_wanda_activation_stats,
    configure_logging,
    ensure_src_on_path,
    env_or,
    level_label,
    load_base_model,
    parse_pruning_levels,
    restore_linear_weights,
    s3_destination,
    safe_filename,
    seeded_task_sample,
    serialise_config,
    snapshot_linear_weights,
    split_csv,
    sync_results,
    write_json,
)

ensure_src_on_path()

from pruning_metrics.evals.coding.teacher_forcing import (  # noqa: E402
    compute_teacher_forced_logprobs,
    write_teacher_forced_record,
)
from pruning_metrics.evals.tasks.base import TaskRecord  # noqa: E402
from pruning_metrics.evals.tasks.registry import (  # noqa: E402
    build_adapter_from_spec,
)

_VALID_PRUNERS = ("wanda", "sparsegpt")


@dataclass(frozen=True)
class PruneEvalSweepConfig:
    """Static config for one ``(pruner, calibration_domain, seed)`` sweep job.

    Parameters
    ----------
    base_model_id:
        Hugging Face model id (any causal LM), e.g. ``Qwen/Qwen2-7B``.
    pruner:
        ``"wanda"`` or ``"sparsegpt"`` -- which unstructured pruning
        algorithm to apply at every non-zero level.
    calibration_dataset_spec:
        Adapter spec (see
        :func:`pruning_metrics.evals.tasks.registry.build_adapter_from_spec`)
        selecting the domain whose train split supplies calibration texts.
    split_seed:
        Seed for every adapter's train/test partition (calibration domain
        AND every eval benchmark), matching the project-wide convention.
    train_frac:
        Fraction routed to each adapter's train (calibration) split.
    calibration_seed:
        Chunk index ``k`` into the calibration domain's seeded train split.
        Distinct seeds (0, 1, 2, ...) pick disjoint chunks of the same
        domain, giving three independent calibration draws per domain.
    calibration_chunk_size:
        Number of consecutive train-split records assigned to chunk
        ``calibration_seed``.
    max_calibration_tokens:
        Per-prompt truncation length for both WANDA stat collection and
        SparseGPT's calibration forward passes.
    pruning_levels:
        Sparsity levels in percent, ascending, deduplicated. ``0`` means
        "baseline" -- restore weights and evaluate, but skip pruning and
        mask extraction/upload for that level.
    eval_dataset_specs:
        Adapter specs for every benchmark evaluated at every level.
    tf_top_k:
        Number of alternative tokens recorded per teacher-forced position.
    num_tf_samples:
        Number of test records teacher-forced per benchmark. ``0`` means
        "every test record, sorted by task_id"; a benchmark with fewer test
        records than requested naturally uses all of them (see
        :func:`select_tf_samples`).
    tf_seed:
        Seed for the per-benchmark sample selection. Selection depends only
        on ``(benchmark, tf_seed, num_tf_samples)`` so every variant in the
        sweep (every pruner/domain/seed/level) scores the exact same
        records for a given benchmark.
    output_dir:
        Local staging directory synced to S3 after every ``(level, bench)``.
    run_id:
        Identifier appended to the S3 prefix and embedded in the manifest.
    results_bucket, results_prefix:
        S3 destination (``results_prefix`` already has ``run_id`` appended
        by :func:`_build_config`, matching the other runners' convention).
    """

    base_model_id: str
    pruner: str
    calibration_dataset_spec: str
    split_seed: int
    train_frac: float
    calibration_seed: int
    calibration_chunk_size: int
    max_calibration_tokens: int
    pruning_levels: tuple[float, ...]
    eval_dataset_specs: tuple[str, ...]
    tf_top_k: int
    num_tf_samples: int
    tf_seed: int
    output_dir: Path
    run_id: str
    results_bucket: str
    results_prefix: str


def parse_args() -> argparse.Namespace:
    """CLI for the prune + eval sweep runner.

    Defaults are pulled from the environment (per the C4 contract) so the
    same invocation works when the user-data script exports
    ``BASE_MODEL_ID`` etc. ``--pruner`` and ``--calibration-dataset-spec``
    default to empty strings and are rejected by :func:`_build_config` with
    a descriptive error rather than argparse's generic "invalid choice" --
    there is no reasonable default pruner or calibration domain.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Prune (WANDA or SparseGPT) a base model to every requested "
            "level and teacher-force every requested eval benchmark at "
            "each level."
        )
    )
    parser.add_argument(
        "--base-model-id",
        default=env_or("BASE_MODEL_ID", default="Qwen/Qwen2-7B"),
    )
    parser.add_argument(
        "--pruner",
        default=env_or("PRUNER", default=""),
        help="'wanda' or 'sparsegpt'.",
    )
    parser.add_argument(
        "--calibration-dataset-spec",
        default=env_or("CALIBRATION_DATASET_SPEC", default=""),
        help="Adapter spec selecting the calibration domain, e.g. 'math'.",
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
        "--calibration-seed",
        type=int,
        default=int(env_or("CALIBRATION_SEED", default="0")),
        help="Chunk index k into the calibration domain's train split.",
    )
    parser.add_argument(
        "--calibration-chunk-size",
        type=int,
        default=int(env_or("CALIBRATION_CHUNK_SIZE", default="128")),
    )
    parser.add_argument(
        "--max-calibration-tokens",
        type=int,
        default=int(env_or("MAX_CALIBRATION_TOKENS", default="2048")),
    )
    parser.add_argument(
        "--pruning-levels",
        default=env_or("PRUNING_LEVELS", default="10,20,30,40,50,60,70,80"),
        help="Comma-separated levels; 0 means 'baseline, no pruning'.",
    )
    parser.add_argument(
        "--eval-dataset-specs",
        default=env_or("EVAL_DATASET_SPECS", default=""),
        help="Comma-separated adapter specs; ALL are evaluated per level.",
    )
    parser.add_argument(
        "--tf-top-k",
        type=int,
        default=int(env_or("TF_TOP_K", default="10")),
    )
    parser.add_argument(
        "--num-tf-samples",
        type=int,
        default=int(env_or("NUM_TF_SAMPLES", default="200")),
        help=(
            "Number of test records teacher-forced per benchmark. "
            "0 = all test records (sorted by task_id for determinism); "
            ">= 1 = seeded random sample (a benchmark with fewer test "
            "records than this simply uses all of them)."
        ),
    )
    parser.add_argument(
        "--tf-seed",
        type=int,
        default=int(env_or("TF_SEED", default="65320")),
    )
    add_common_runner_args(parser, default_results_prefix="prune_eval_v2")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> PruneEvalSweepConfig:
    """Validate + assemble :class:`PruneEvalSweepConfig` from parsed args.

    Kept separate from :func:`main` so the pure validation/parsing logic is
    unit-testable without a GPU, ``torch``, or network access.

    Raises
    ------
    SystemExit
        If a required field is missing/invalid (mirrors the
        ``--artifact-uri is required`` style used by the other runners).
    ValueError
        Propagated from :func:`infra.runners._runner_common.parse_pruning_levels`
        for malformed level lists.
    """

    if not args.base_model_id:
        raise SystemExit("--base-model-id is required")
    if args.pruner not in _VALID_PRUNERS:
        raise SystemExit(
            f"--pruner must be one of {_VALID_PRUNERS}, got {args.pruner!r}"
        )
    if not args.calibration_dataset_spec:
        raise SystemExit("--calibration-dataset-spec is required")
    eval_dataset_specs = split_csv(args.eval_dataset_specs)
    if not eval_dataset_specs:
        raise SystemExit(
            "--eval-dataset-specs is required (comma-separated, >= 1 spec)"
        )
    if args.calibration_seed < 0:
        raise SystemExit("--calibration-seed (chunk index) must be >= 0")
    if args.calibration_chunk_size < 1:
        raise SystemExit("--calibration-chunk-size must be >= 1")
    if args.num_tf_samples < 0:
        raise SystemExit("--num-tf-samples must be >= 0 (0 = all test records)")

    pruning_levels = parse_pruning_levels(args.pruning_levels)

    return PruneEvalSweepConfig(
        base_model_id=args.base_model_id,
        pruner=args.pruner,
        calibration_dataset_spec=args.calibration_dataset_spec,
        split_seed=args.split_seed,
        train_frac=args.train_frac,
        calibration_seed=args.calibration_seed,
        calibration_chunk_size=args.calibration_chunk_size,
        max_calibration_tokens=args.max_calibration_tokens,
        pruning_levels=pruning_levels,
        eval_dataset_specs=eval_dataset_specs,
        tf_top_k=args.tf_top_k,
        num_tf_samples=args.num_tf_samples,
        tf_seed=args.tf_seed,
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        results_bucket=args.results_bucket,
        results_prefix=f"{args.results_prefix.strip('/')}/{args.run_id}",
    )


def main() -> int:
    """CLI entry point. Returns the shell exit code."""

    configure_logging()
    config = _build_config(parse_args())
    config.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Prune-eval-sweep config: %s", json.dumps(serialise_config(config)))
    _write_run_metadata(config)

    LOGGER.info(
        "Loading calibration adapter from spec %r", config.calibration_dataset_spec
    )
    calibration_adapter = build_adapter_from_spec(config.calibration_dataset_spec)
    calibration_train_records, _ = calibration_adapter.train_test_split(
        seed=config.split_seed, train_frac=config.train_frac
    )
    calibration_chunk = select_calibration_chunk(
        calibration_train_records,
        config.calibration_seed,
        config.calibration_chunk_size,
    )
    # Design: calibration texts are the chunk records' prompts only (not
    # prompt + answer) -- this mirrors run_pruning_calibration.py's WANDA
    # stat collection exactly (calibration_texts = [r.prompt for r in
    # train_records]) so a shared calibration chunk drives identical WANDA
    # activations and SparseGPT Hessians across pruners for the same
    # (domain, seed) grid point.
    calibration_texts = [record.prompt for record in calibration_chunk]
    LOGGER.info(
        "Calibration chunk k=%d size=%d -> %d text(s) from %r (domain=%r)",
        config.calibration_seed,
        config.calibration_chunk_size,
        len(calibration_texts),
        calibration_adapter.dataset_spec,
        config.calibration_dataset_spec,
    )

    _write_manifest(config, num_calibration_texts=len(calibration_texts))

    LOGGER.info("Building %d eval adapter(s)", len(config.eval_dataset_specs))
    eval_bench_data: dict[str, list[TaskRecord]] = {}
    for spec in config.eval_dataset_specs:
        adapter = build_adapter_from_spec(spec)
        _, test_records = adapter.train_test_split(
            seed=config.split_seed, train_frac=config.train_frac
        )
        sampled = select_tf_samples(test_records, config.tf_seed, config.num_tf_samples)
        eval_bench_data[spec] = sampled
        LOGGER.info(
            "bench=%s: sampled %d/%d test record(s) (tf_seed=%d, num_tf_samples=%d)",
            spec,
            len(sampled),
            len(test_records),
            config.tf_seed,
            config.num_tf_samples,
        )

    tokenizer, model = load_base_model(config.base_model_id)
    snapshot = snapshot_linear_weights(model)

    wanda_stats: dict[str, Any] | None = None
    if config.pruner == "wanda":
        raw_stats = collect_wanda_activation_stats(
            model=model,
            tokenizer=tokenizer,
            calibration_texts=calibration_texts,
            max_tokens=config.max_calibration_tokens,
        )
        # Design: v2 scope excludes lm_head from mask parity with SparseGPT
        # (apply_sparsegpt_pruning never touches lm_head/embeddings either),
        # so behavioral-vs-mask comparisons in the analysis notebook are
        # apples-to-apples regardless of pruner. Filtering here (rather than
        # only in extract_pruning_masks) also means WANDA never zeroes
        # lm_head weights it wasn't scoped to prune in the first place.
        wanda_stats = {
            name: tensor for name, tensor in raw_stats.items() if ".layers." in name
        }
        LOGGER.info(
            "Filtered WANDA stats to %d in-decoder Linear layer(s) (from %d total).",
            len(wanda_stats),
            len(raw_stats),
        )

    by_level: dict[str, dict[str, dict[str, Any]]] = {}

    def _write_summary(*, ended: bool, elapsed_seconds: float | None) -> None:
        write_json(
            config.output_dir / "summary.json",
            build_summary_payload(
                config, by_level, ended=ended, elapsed_seconds=elapsed_seconds
            ),
        )

    started = time.monotonic()
    try:
        for level in config.pruning_levels:
            LOGGER.info(
                "=== Pruning level %s%% (pruner=%s) ===",
                level_label(level),
                config.pruner,
            )
            # Every level starts from pristine weights: WANDA/SparseGPT
            # scores are computed against the ORIGINAL model, never a
            # previously-pruned one, so levels must not compound.
            restore_linear_weights(model, snapshot)

            if level > 0:
                _prune_to_level(
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    calibration_texts=calibration_texts,
                    wanda_stats=wanda_stats,
                    level=level,
                )
                _extract_and_save_masks(config, model, level)
                sync_results(config)

            level_summary: dict[str, dict[str, Any]] = {}
            for eval_spec, sampled_records in eval_bench_data.items():
                level_summary[eval_spec] = _evaluate_bench(
                    config=config,
                    model=model,
                    tokenizer=tokenizer,
                    level=level,
                    eval_spec=eval_spec,
                    sampled_records=sampled_records,
                )
                by_level[level_label(level)] = level_summary
                _write_summary(ended=False, elapsed_seconds=None)
                sync_results(config)
    finally:
        _write_summary(ended=True, elapsed_seconds=time.monotonic() - started)
        sync_results(config)
        LOGGER.info(
            "Prune-eval-sweep finished. Results: s3://%s/%s/",
            config.results_bucket,
            config.results_prefix,
        )

    return 0


# ---------------------------------------------------------------------------
# Pure helpers (no torch import at module scope; unit-testable in isolation)
# ---------------------------------------------------------------------------


def select_calibration_chunk(
    train_records: Sequence[Any],
    chunk_index: int,
    chunk_size: int,
) -> list[Any]:
    """Slice chunk ``chunk_index`` out of a domain's seeded train split.

    Parameters
    ----------
    train_records:
        The FULL train split, already in seeded order (i.e. the first
        element of the tuple returned by an adapter's ``train_test_split``).
        Generic over any sequence element type so tests can pass plain
        strings/ints instead of constructing :class:`TaskRecord` instances.
    chunk_index:
        ``k`` -- 0-indexed chunk number (``CALIBRATION_SEED``).
    chunk_size:
        Number of consecutive records per chunk (``CALIBRATION_CHUNK_SIZE``).

    Returns
    -------
    list[Any]
        ``train_records[chunk_index * chunk_size : (chunk_index + 1) * chunk_size]``,
        always exactly ``chunk_size`` long.

    Raises
    ------
    ValueError
        If ``chunk_index < 0`` or ``chunk_size < 1``.
    RuntimeError
        If ``train_records`` has fewer than ``(chunk_index + 1) * chunk_size``
        elements -- fails fast rather than silently returning a short (or
        empty) chunk, which would make different seeds calibrate on
        overlapping or vanishing data without warning.
    """

    if chunk_index < 0:
        raise ValueError(
            f"chunk_index (CALIBRATION_SEED) must be >= 0, got {chunk_index}."
        )
    if chunk_size < 1:
        raise ValueError(
            f"chunk_size (CALIBRATION_CHUNK_SIZE) must be >= 1, got {chunk_size}."
        )

    start = chunk_index * chunk_size
    end = start + chunk_size
    available = len(train_records)
    if available < end:
        raise RuntimeError(
            f"Calibration train split has {available} record(s); need at "
            f"least {end} for CALIBRATION_SEED={chunk_index}, "
            f"CALIBRATION_CHUNK_SIZE={chunk_size} (records [{start}:{end})). "
            "Reduce CALIBRATION_CHUNK_SIZE, lower CALIBRATION_SEED, or "
            "increase TRAIN_FRAC for this domain."
        )
    return list(train_records[start:end])


def select_tf_samples(
    test_records: Sequence[TaskRecord],
    tf_seed: int,
    num_tf_samples: int,
) -> list[TaskRecord]:
    """Pick the seeded sample of test records teacher-forced per benchmark.

    Shares the selection core with ``run_teacher_forced.py`` via
    :func:`_runner_common.seeded_task_sample` (sort by ``task_id`` for a
    source-order-independent base ordering, then a single
    ``random.Random(tf_seed)`` shuffle) so that every sweep variant --
    every ``(pruner, calibration_domain, calibration_seed, level)``
    combination -- scores the IDENTICAL set of records for a given
    benchmark: selection is a pure function of ``(test_records, tf_seed,
    num_tf_samples)``, never of the model, pruner, or level. The public
    APIs differ (``run_teacher_forced`` also supports explicit task-id
    lists); only the ordering rule is shared.

    Parameters
    ----------
    test_records:
        Full test split from one eval benchmark's adapter.
    tf_seed:
        Seed for the shuffle.
    num_tf_samples:
        ``0`` selects every test record (sorted, no shuffle needed). ``>=
        1`` selects a seeded random sample of that size; if the benchmark
        has fewer test records than requested, Python's slice-past-the-end
        semantics mean every available record is returned (no error) --
        this is the "a benchmark with fewer test records uses all" rule
        from the C4 contract, satisfied without a special case.

    Returns
    -------
    list[TaskRecord]
        Selected records in a deterministic order.

    Raises
    ------
    RuntimeError
        If ``test_records`` is empty, or selection produces zero records.
    """

    if not test_records:
        raise RuntimeError("Eval test split is empty.")

    chosen = seeded_task_sample(test_records, tf_seed, num_tf_samples)
    if not chosen:
        raise RuntimeError("Teacher-forced sample selection produced 0 records.")
    return chosen


def level_dir(output_dir: Path, level: float) -> Path:
    """``<output_dir>/level=NN`` for a given pruning level."""

    return output_dir / f"level={level_label(level)}"


def bench_dir(output_dir: Path, level: float, eval_spec: str) -> Path:
    """``<output_dir>/level=NN/bench=<safe_spec>`` for one benchmark."""

    return level_dir(output_dir, level) / f"bench={safe_filename(eval_spec)}"


def sample_dir(
    output_dir: Path, level: float, eval_spec: str, sample_idx: int, task_id: str
) -> Path:
    """``.../sample=KKK_task=<safe_task>`` for one teacher-forced record."""

    return bench_dir(output_dir, level, eval_spec) / (
        f"sample={sample_idx:03d}_task={safe_filename(task_id)}"
    )


def mask_paths(output_dir: Path, level: float) -> tuple[Path, Path]:
    """``(full_masks_path, digest_path)`` under ``<output_dir>/masks/``."""

    masks_dir = output_dir / "masks"
    label = level_label(level)
    return masks_dir / f"level={label}.npz", masks_dir / f"level={label}.digest.npz"


def _package_versions() -> dict[str, str]:
    """Best-effort ``{package: version}`` snapshot for the manifest."""

    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for module_name in ("torch", "transformers", "datasets", "boto3"):
        try:  # pragma: no cover - thin wrapper
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:  # pylint: disable=broad-exception-caught
            versions[module_name] = "missing"
    return versions


def build_manifest(
    config: PruneEvalSweepConfig, *, num_calibration_texts: int
) -> dict[str, Any]:
    """Full-config ``manifest.json`` payload (mirrors the calibration runner).

    Parameters
    ----------
    config:
        Resolved run configuration.
    num_calibration_texts:
        Length of the calibration chunk actually used (recorded for audit;
        should equal ``config.calibration_chunk_size`` whenever the fail-fast
        check in :func:`select_calibration_chunk` passed).

    Returns
    -------
    dict[str, Any]
        JSON-serialisable manifest payload. Written EARLY (before the GPU
        pruning loop starts, right after the calibration chunk is known) so
        a job that crashes mid-sweep still leaves behind a manifest
        describing what it was trying to do -- unlike
        ``run_pruning_calibration.py``, which writes its manifest only once
        calibration stats finish, this runner has no analogous "cheap first
        phase" to gate on, since the whole sweep is one job.
    """

    return {
        "schema_version": "1",
        "run_id": config.run_id,
        "base_model_id": config.base_model_id,
        "pruner": config.pruner,
        "calibration_dataset_spec": config.calibration_dataset_spec,
        "calibration_seed": config.calibration_seed,
        "calibration_chunk_size": config.calibration_chunk_size,
        "num_calibration_texts": num_calibration_texts,
        "split_seed": config.split_seed,
        "train_frac": config.train_frac,
        "max_calibration_tokens": config.max_calibration_tokens,
        "pruning_levels": list(config.pruning_levels),
        "eval_dataset_specs": list(config.eval_dataset_specs),
        "tf_top_k": config.tf_top_k,
        "num_tf_samples": config.num_tf_samples,
        "tf_seed": config.tf_seed,
        "package_versions": _package_versions(),
        "host": socket.gethostname(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_paths": {
            "manifest": "manifest.json",
            "run_metadata": "run_metadata.json",
            "summary": "summary.json",
            "masks_dir": "masks/",
        },
    }


def build_summary_payload(
    config: PruneEvalSweepConfig,
    by_level: dict[str, dict[str, dict[str, Any]]],
    *,
    ended: bool,
    elapsed_seconds: float | None,
) -> dict[str, Any]:
    """Rolling ``summary.json`` payload: per level, per bench, aggregate stats.

    Parameters
    ----------
    config:
        Resolved run configuration.
    by_level:
        ``{level_label: {eval_spec: {"n_samples", "mean_logprob",
        "perplexity", "elapsed_seconds"}}}`` accumulated so far.
    ended:
        Whether the sweep has fully completed (sets ``ended_at_utc``).
    elapsed_seconds:
        Total wall-clock time; ``None`` until ``ended=True``.

    Returns
    -------
    dict[str, Any]
        JSON-serialisable payload. ``ended_at_utc`` stays ``None`` until the
        run completes, matching the existing eval-runner convention (see
        ``infra.runners._runner_common.eval_summary_payload``) so a partial
        sweep is unambiguously distinguishable from a finished one.
    """

    return {
        "run_id": config.run_id,
        "base_model_id": config.base_model_id,
        "pruner": config.pruner,
        "calibration_dataset_spec": config.calibration_dataset_spec,
        "calibration_seed": config.calibration_seed,
        "eval_dataset_specs": list(config.eval_dataset_specs),
        "tf_seed": config.tf_seed,
        "num_tf_samples": config.num_tf_samples,
        "levels": by_level,
        "ended_at_utc": (datetime.now(timezone.utc).isoformat() if ended else None),
        "elapsed_seconds": elapsed_seconds,
    }


# ---------------------------------------------------------------------------
# Torch-dependent helpers (imported lazily inside these functions so the
# pure helpers above stay importable/testable without torch, and so the
# concurrently-developed P1 (SparseGPT) / P2 (masks) APIs are only resolved
# at call time -- by which point those packages are expected to have landed.
# ---------------------------------------------------------------------------


def _write_run_metadata(config: PruneEvalSweepConfig) -> None:
    payload = {
        "host": socket.gethostname(),
        "run_id": config.run_id,
        "mode": "prune_eval_sweep",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_id": config.base_model_id,
        "pruner": config.pruner,
        "calibration_dataset_spec": config.calibration_dataset_spec,
        "eval_dataset_specs": list(config.eval_dataset_specs),
        "destination": s3_destination(config.results_bucket, config.results_prefix),
    }
    write_json(config.output_dir / "run_metadata.json", payload)


def _write_manifest(
    config: PruneEvalSweepConfig, *, num_calibration_texts: int
) -> None:
    write_json(
        config.output_dir / "manifest.json",
        build_manifest(config, num_calibration_texts=num_calibration_texts),
    )


def _prune_to_level(
    *,
    model: Any,
    tokenizer: Any,
    config: PruneEvalSweepConfig,
    calibration_texts: list[str],
    wanda_stats: dict[str, Any] | None,
    level: float,
) -> None:
    """Dispatch to WANDA or SparseGPT for one pruning level.

    ``model`` must already be restored to its pristine (unpruned) weights
    by the caller -- both algorithms score against whatever weights are
    currently loaded, so pruning a previously-pruned model would compound
    sparsity instead of re-deriving the level-``level`` mask from scratch.
    """

    prune_ratio = float(level) / 100.0

    if config.pruner == "wanda":
        if wanda_stats is None:  # pragma: no cover - guarded by main()'s branch
            raise RuntimeError("WANDA stats were not collected; cannot prune.")
        apply_wanda_pruning(model, wanda_stats, prune_ratio=prune_ratio)
        return

    if config.pruner == "sparsegpt":
        # Lazy import: package P1 adds this function to _runner_common
        # concurrently with this package. Importing it here (rather than at
        # module scope) means this module -- and every pure helper above --
        # stays importable for unit tests even if P1 hasn't landed yet.
        from infra.runners._runner_common import apply_sparsegpt_pruning

        apply_sparsegpt_pruning(
            model,
            tokenizer,
            calibration_texts,
            prune_ratio,
            max_tokens=config.max_calibration_tokens,
        )
        return

    raise ValueError(f"Unknown pruner {config.pruner!r}")  # pragma: no cover


def _extract_and_save_masks(
    config: PruneEvalSweepConfig, model: Any, level: float
) -> None:
    """Extract, pack, and stage the pruning mask + digest for one level."""

    # Lazy import: package P2 adds this module concurrently with this
    # package (see the module-level docstring's design note).
    from pruning_metrics.metrics.masks import (
        extract_pruning_masks,
        make_mask_digest,
        save_digest,
        save_packed_masks,
    )

    masks = extract_pruning_masks(model)
    full_path, digest_path = mask_paths(config.output_dir, level)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    save_packed_masks(masks, full_path)
    digest = make_mask_digest(masks)
    save_digest(digest, digest_path)
    LOGGER.info(
        "Saved masks for level=%s%%: %d Linear layer(s), digest at %s",
        level_label(level),
        len(masks),
        digest_path,
    )


def _evaluate_bench(
    *,
    config: PruneEvalSweepConfig,
    model: Any,
    tokenizer: Any,
    level: float,
    eval_spec: str,
    sampled_records: list[TaskRecord],
) -> dict[str, Any]:
    """Teacher-force every sampled record of one benchmark at one level.

    Writes one ``per_token.json`` per sample under
    ``level=NN/bench=<safe_spec>/sample=KKK_task=<safe_task>/`` (identical
    format to ``run_teacher_forced.py``'s output) and returns the
    aggregate stats folded into that ``(level, bench)`` cell of
    ``summary.json``.

    Returns
    -------
    dict[str, Any]
        ``{"n_samples", "mean_logprob", "perplexity", "elapsed_seconds"}``.
        ``mean_logprob``/``perplexity`` are ``None`` when every sample was
        skipped (empty ``target_text`` or a tokenisation error).
    """

    started = time.monotonic()
    logprobs: list[float] = []
    written = 0

    for sample_idx, record in enumerate(sampled_records):
        if not record.target_text:
            LOGGER.warning(
                "Skipping %s bench=%s level=%s: empty target_text.",
                record.task_id,
                eval_spec,
                level_label(level),
            )
            continue
        try:
            tf_record = compute_teacher_forced_logprobs(
                model=model,
                tokenizer=tokenizer,
                prompt=record.prompt,
                answer=record.target_text,
                model_id=f"{config.base_model_id}@prune={level_label(level)}",
                task_id=record.task_id,
                seed=config.tf_seed,
                top_k=config.tf_top_k,
            )
        except ValueError as exc:
            LOGGER.warning(
                "Skipping sample %d task=%s bench=%s level=%s "
                "-- tokenisation error: %s",
                sample_idx,
                record.task_id,
                eval_spec,
                level_label(level),
                exc,
            )
            continue

        record_dir = sample_dir(
            config.output_dir, level, eval_spec, sample_idx, record.task_id
        )
        record_dir.mkdir(parents=True, exist_ok=True)
        write_teacher_forced_record(tf_record, record_dir / "per_token.json")

        logprobs.append(tf_record.average_logprob)
        written += 1
        LOGGER.info(
            "level=%s bench=%s sample=%d task=%s avg_logp=%.4f ppl=%.4f "
            "over %d tokens",
            level_label(level),
            eval_spec,
            sample_idx,
            record.task_id,
            tf_record.average_logprob,
            tf_record.perplexity,
            tf_record.num_answer_tokens,
        )

    elapsed = time.monotonic() - started
    mean_logprob = float(sum(logprobs) / len(logprobs)) if logprobs else None
    perplexity = float(math.exp(-mean_logprob)) if mean_logprob is not None else None
    LOGGER.info(
        "level=%s bench=%s: %d sample(s), mean_logprob=%s, ppl=%s, %.1fs",
        level_label(level),
        eval_spec,
        written,
        f"{mean_logprob:.4f}" if mean_logprob is not None else "N/A",
        f"{perplexity:.4f}" if perplexity is not None else "N/A",
        elapsed,
    )
    return {
        "n_samples": written,
        "mean_logprob": mean_logprob,
        "perplexity": perplexity,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
