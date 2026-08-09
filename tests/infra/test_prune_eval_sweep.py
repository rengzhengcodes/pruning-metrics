"""Unit tests for the pure (torch-free) pieces of ``run_prune_eval_sweep``.

Per the v2 spec (package P5), only the pure pieces -- chunk selection,
sample selection, config validation, manifest/summary content, and path
layout -- get unit tests here; the torch-dependent pruning/teacher-forcing
control flow is exercised end-to-end on a GPU box, not in this suite. None
of these tests import ``torch``, ``transformers``, or hit the network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from infra.runners._runner_common import safe_filename  # noqa: E402
from infra.runners.run_prune_eval_sweep import (  # noqa: E402
    PruneEvalSweepConfig,
    _build_config,
    bench_dir,
    build_manifest,
    build_summary_payload,
    level_dir,
    mask_paths,
    sample_dir,
    select_calibration_chunk,
    select_tf_samples,
)
from pruning_metrics.evals.tasks.base import TaskRecord  # noqa: E402

# ---------------------------------------------------------------------------
# select_calibration_chunk
# ---------------------------------------------------------------------------


def test_select_calibration_chunk_slices_expected_range() -> None:
    """Chunk k covers records [k*C : (k+1)*C] of the seeded train split."""

    records = list(range(30))  # generic elements; the function is type-agnostic
    assert select_calibration_chunk(records, 0, 10) == list(range(0, 10))
    assert select_calibration_chunk(records, 1, 10) == list(range(10, 20))
    assert select_calibration_chunk(records, 2, 10) == list(range(20, 30))


def test_select_calibration_chunk_fails_fast_when_short() -> None:
    """Fewer than (k+1)*C records must raise, not silently truncate."""

    records = list(range(25))
    with pytest.raises(RuntimeError, match="need at least 30"):
        select_calibration_chunk(records, 2, 10)


def test_select_calibration_chunk_exact_boundary_is_not_short() -> None:
    """Exactly (k+1)*C records is the boundary success case."""

    records = list(range(20))
    assert select_calibration_chunk(records, 1, 10) == list(range(10, 20))


@pytest.mark.parametrize(
    "chunk_index,chunk_size",
    [(-1, 10), (0, 0), (0, -5)],
)
def test_select_calibration_chunk_rejects_bad_params(
    chunk_index: int, chunk_size: int
) -> None:
    with pytest.raises(ValueError):
        select_calibration_chunk(list(range(100)), chunk_index, chunk_size)


# ---------------------------------------------------------------------------
# select_tf_samples
# ---------------------------------------------------------------------------


def _records(n: int) -> list[TaskRecord]:
    return [
        TaskRecord(task_id=f"task_{i:03d}", prompt=f"p{i}", target_text=f"a{i}")
        for i in range(n)
    ]


def test_select_tf_samples_zero_selects_all_sorted() -> None:
    records = _records(5)
    # Shuffle the input order; num_tf_samples=0 must still return sorted-by-id.
    shuffled = [records[3], records[0], records[4], records[1], records[2]]
    selected = select_tf_samples(shuffled, tf_seed=42, num_tf_samples=0)
    assert [r.task_id for r in selected] == [r.task_id for r in records]


def test_select_tf_samples_is_deterministic_given_same_inputs() -> None:
    """Selection is a pure function of (records, tf_seed, num_tf_samples)."""

    records = _records(50)
    first = select_tf_samples(records, tf_seed=65320, num_tf_samples=10)
    second = select_tf_samples(records, tf_seed=65320, num_tf_samples=10)
    assert [r.task_id for r in first] == [r.task_id for r in second]


def test_select_tf_samples_independent_of_input_order() -> None:
    """The (bench, tf_seed, num_tf_samples) contract: source order must not matter."""

    records = _records(50)
    reversed_records = list(reversed(records))
    a = select_tf_samples(records, tf_seed=7, num_tf_samples=10)
    b = select_tf_samples(reversed_records, tf_seed=7, num_tf_samples=10)
    assert [r.task_id for r in a] == [r.task_id for r in b]


def test_select_tf_samples_uses_all_when_fewer_than_requested() -> None:
    """A benchmark with fewer test records than NUM_TF_SAMPLES uses all of them."""

    records = _records(3)
    selected = select_tf_samples(records, tf_seed=1, num_tf_samples=200)
    assert {r.task_id for r in selected} == {r.task_id for r in records}
    assert len(selected) == 3


def test_select_tf_samples_empty_raises() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        select_tf_samples([], tf_seed=1, num_tf_samples=5)


# ---------------------------------------------------------------------------
# _build_config validation
# ---------------------------------------------------------------------------


def _make_args(**overrides: object) -> argparse.Namespace:
    defaults = dict(
        base_model_id="Qwen/Qwen2-7B",
        pruner="wanda",
        calibration_dataset_spec="math",
        split_seed=65320,
        train_frac=0.8,
        calibration_seed=0,
        calibration_chunk_size=128,
        max_calibration_tokens=2048,
        pruning_levels="10,20,30",
        eval_dataset_specs="math,coding,mcq",
        tf_top_k=10,
        num_tf_samples=200,
        tf_seed=65320,
        output_dir="/opt/results",
        run_id="20260101T000000Z-abcdef",
        results_bucket="my-bucket",
        results_prefix="prune_eval_v2",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_config_happy_path() -> None:
    config = _build_config(_make_args())
    assert isinstance(config, PruneEvalSweepConfig)
    assert config.pruner == "wanda"
    assert config.pruning_levels == (10.0, 20.0, 30.0)
    assert config.eval_dataset_specs == ("math", "coding", "mcq")
    assert config.results_prefix == "prune_eval_v2/20260101T000000Z-abcdef"
    assert config.output_dir == Path("/opt/results")


def test_build_config_accepts_level_zero_as_baseline() -> None:
    config = _build_config(_make_args(pruning_levels="0,10,20"))
    assert config.pruning_levels[0] == 0.0


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"base_model_id": ""}, "base-model-id"),
        ({"pruner": ""}, "pruner"),
        ({"pruner": "magnitude"}, "pruner"),
        ({"calibration_dataset_spec": ""}, "calibration-dataset-spec"),
        ({"eval_dataset_specs": ""}, "eval-dataset-specs"),
        ({"calibration_seed": -1}, "calibration-seed"),
        ({"calibration_chunk_size": 0}, "calibration-chunk-size"),
        ({"num_tf_samples": -1}, "num-tf-samples"),
    ],
)
def test_build_config_rejects_invalid_inputs(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(SystemExit, match=match):
        _build_config(_make_args(**overrides))


def test_build_config_rejects_malformed_levels() -> None:
    with pytest.raises(ValueError):
        _build_config(_make_args(pruning_levels="10,100,30"))


# ---------------------------------------------------------------------------
# Manifest / summary content
# ---------------------------------------------------------------------------


def test_build_manifest_content() -> None:
    config = _build_config(_make_args(pruner="sparsegpt", calibration_seed=2))
    manifest = build_manifest(config, num_calibration_texts=128)

    assert manifest["pruner"] == "sparsegpt"
    assert manifest["calibration_dataset_spec"] == "math"
    assert manifest["calibration_seed"] == 2
    assert manifest["calibration_chunk_size"] == 128
    assert manifest["num_calibration_texts"] == 128
    assert manifest["pruning_levels"] == [10.0, 20.0, 30.0]
    assert manifest["eval_dataset_specs"] == ["math", "coding", "mcq"]
    assert manifest["run_id"] == config.run_id
    assert "package_versions" in manifest
    assert manifest["package_versions"]["python"]
    assert manifest["artifact_paths"]["manifest"] == "manifest.json"


def test_build_summary_payload_running_vs_ended() -> None:
    config = _build_config(_make_args())
    by_level = {
        "10": {
            "math": {
                "n_samples": 5,
                "mean_logprob": -1.2,
                "perplexity": 3.3,
                "elapsed_seconds": 4.5,
            }
        }
    }

    running = build_summary_payload(config, by_level, ended=False, elapsed_seconds=None)
    assert running["ended_at_utc"] is None
    assert running["elapsed_seconds"] is None
    assert running["levels"] == by_level

    ended = build_summary_payload(config, by_level, ended=True, elapsed_seconds=12.0)
    assert ended["ended_at_utc"] is not None
    assert ended["elapsed_seconds"] == 12.0


# ---------------------------------------------------------------------------
# Path layout / filename sanitisation
# ---------------------------------------------------------------------------


def test_safe_filename_sanitises_slashes_and_spaces() -> None:
    assert safe_filename("coding:evalplus/humanevalplus:test") == (
        "coding:evalplus_humanevalplus:test"
    )
    assert safe_filename("HumanEval/137 extra") == "HumanEval_137_extra"


def test_level_dir_layout() -> None:
    out = Path("/opt/results")
    assert level_dir(out, 20.0) == out / "level=20"
    assert level_dir(out, 0.0) == out / "level=0"


def test_bench_dir_layout() -> None:
    out = Path("/opt/results")
    assert bench_dir(out, 20.0, "coding:evalplus/humanevalplus:test") == (
        out / "level=20" / "bench=coding:evalplus_humanevalplus:test"
    )


def test_sample_dir_layout() -> None:
    out = Path("/opt/results")
    result = sample_dir(out, 20.0, "math", 3, "HumanEval/137")
    assert result == (out / "level=20" / "bench=math" / "sample=003_task=HumanEval_137")


def test_mask_paths_layout() -> None:
    out = Path("/opt/results")
    full_path, digest_path = mask_paths(out, 40.0)
    assert full_path == out / "masks" / "level=40.npz"
    assert digest_path == out / "masks" / "level=40.digest.npz"
