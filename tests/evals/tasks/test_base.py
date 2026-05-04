"""Determinism + override behaviour for the shared train/test splitter."""

from __future__ import annotations

import pytest

from pruning_metrics.evals.tasks.base import (
    TaskRecord,
    deterministic_split,
    native_or_seeded_split,
    records_to_id_map,
)


def _make_records(count: int) -> list[TaskRecord]:
    return [
        TaskRecord(
            task_id=f"t/{idx:04d}",
            prompt=f"prompt {idx}",
            target_text=f"target {idx}",
            metadata={"index": idx},
        )
        for idx in range(count)
    ]


def test_deterministic_split_is_stable_for_seed() -> None:
    """Two calls with the same seed must return identical task-id lists."""

    records = _make_records(20)
    first_train, first_test = deterministic_split(records, seed=65320)
    second_train, second_test = deterministic_split(records, seed=65320)
    assert [r.task_id for r in first_train] == [r.task_id for r in second_train]
    assert [r.task_id for r in first_test] == [r.task_id for r in second_test]
    assert len(first_train) == 16
    assert len(first_test) == 4


def test_deterministic_split_changes_with_seed() -> None:
    records = _make_records(50)
    train_a, _ = deterministic_split(records, seed=1)
    train_b, _ = deterministic_split(records, seed=2)
    assert [r.task_id for r in train_a] != [r.task_id for r in train_b]


def test_deterministic_split_validates_train_frac() -> None:
    records = _make_records(5)
    with pytest.raises(ValueError, match="train_frac"):
        deterministic_split(records, train_frac=0.0)
    with pytest.raises(ValueError, match="train_frac"):
        deterministic_split(records, train_frac=1.0)


def test_deterministic_split_honours_explicit_overrides() -> None:
    records = _make_records(20)
    train_ids = ["t/0001", "t/0007", "t/0019"]
    test_ids = ["t/0000", "t/0010"]
    train_records, test_records = deterministic_split(
        records,
        seed=42,
        train_frac=0.5,
        explicit_train_ids=train_ids,
        explicit_test_ids=test_ids,
    )
    train_id_set = {r.task_id for r in train_records}
    test_id_set = {r.task_id for r in test_records}
    assert set(train_ids).issubset(train_id_set)
    assert set(test_ids).issubset(test_id_set)
    assert not train_id_set & set(test_ids)
    assert not test_id_set & set(train_ids)


def test_deterministic_split_rejects_overlapping_overrides() -> None:
    records = _make_records(5)
    with pytest.raises(ValueError, match="overlap"):
        deterministic_split(
            records,
            explicit_train_ids=["t/0001"],
            explicit_test_ids=["t/0001"],
        )


def test_records_to_id_map_detects_duplicates() -> None:
    records = _make_records(2) + [
        TaskRecord(task_id="t/0000", prompt="dup", target_text="dup")
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        records_to_id_map(records)


def _make_records_in_namespace(prefix: str, count: int) -> list[TaskRecord]:
    return [
        TaskRecord(
            task_id=f"{prefix}/{idx:04d}",
            prompt=f"prompt {idx}",
            target_text=f"target {idx}",
            metadata={"index": idx},
        )
        for idx in range(count)
    ]


def test_native_or_seeded_split_uses_native_partitions() -> None:
    """When both native partitions are present and no overrides, use them as-is."""

    train = _make_records_in_namespace("train", 4)
    test = _make_records_in_namespace("test", 2)
    out_train, out_test = native_or_seeded_split(train, test, seed=999, train_frac=0.1)
    assert [r.task_id for r in out_train] == [r.task_id for r in train]
    assert [r.task_id for r in out_test] == [r.task_id for r in test]


def test_native_or_seeded_split_falls_back_when_no_train() -> None:
    """``train_records=None`` triggers the seeded fallback over the test split."""

    test = _make_records(10)
    out_train, out_test = native_or_seeded_split(None, test, seed=65320, train_frac=0.8)
    assert len(out_train) + len(out_test) == 10


def test_native_or_seeded_split_overrides_force_seeded_path() -> None:
    """Explicit overrides re-route through the seeded splitter even with native splits."""

    train = _make_records_in_namespace("train", 4)
    test = _make_records_in_namespace("test", 2)
    forced_train_ids = [r.task_id for r in test]  # promote test rows into train
    out_train, _ = native_or_seeded_split(
        train,
        test,
        seed=65320,
        train_frac=0.5,
        explicit_train_ids=forced_train_ids,
    )
    out_train_ids = {r.task_id for r in out_train}
    assert set(forced_train_ids).issubset(out_train_ids)
