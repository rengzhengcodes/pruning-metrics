"""Numeric extraction + verification for the GSM8K math adapter."""

from __future__ import annotations

import pytest

from pruning_metrics.evals.tasks.base import TaskRecord
from pruning_metrics.evals.tasks.math import (
    MathTaskAdapter,
    _extract_numeric,
    _parse_number,
)
from pruning_metrics.evals.tasks import math as math_module


@pytest.mark.parametrize(
    "text, expected",
    [
        ("the final answer is #### 42", 42.0),
        ("blah blah\n#### -7", -7.0),
        ("answer is \\boxed{3.14}", 3.14),
        ("price was $1,250 today", 1250.0),
        ("she earned $42 last week", 42.0),
        ("solution: 5/8 of the pie", 5 / 8),
        ("nothing parseable here", None),
    ],
)
def test_extract_numeric_handles_common_patterns(text, expected) -> None:
    result = _extract_numeric(text)
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert abs(result - expected) < 1e-9


def test_parse_number_handles_decoration() -> None:
    assert _parse_number("$1,234") == 1234.0
    assert _parse_number("3/4") == 0.75
    assert _parse_number("not-a-number") is None
    assert _parse_number("0/0") is None


def test_math_adapter_verify_pass_and_fail() -> None:
    adapter = MathTaskAdapter()
    record = TaskRecord(
        task_id="gsm8k/test/00001",
        prompt="Q",
        target_text="...\n#### 42",
        metadata={"gold_number": 42.0, "raw_answer": "...\n#### 42", "question": "Q"},
    )
    assert adapter.verify(record, "Reasoning... #### 42").status == "pass"
    assert adapter.verify(record, "Reasoning... #### 41").status == "fail"
    assert adapter.verify(record, "no number").status == "parse_error"


def test_math_adapter_verify_uses_gsm_divider_priority() -> None:
    """If multiple numbers appear, the GSM-style ``#### N`` wins over stray digits."""

    adapter = MathTaskAdapter()
    record = TaskRecord(
        task_id="gsm8k/test/00002",
        prompt="Q",
        target_text="long thinking #### 7",
        metadata={"gold_number": 7.0, "raw_answer": "...", "question": "Q"},
    )
    text = "I had 100 things, kept 13, used 80, finally #### 7"
    assert adapter.verify(record, text).status == "pass"


def _fake_gsm8k_rows(prefix: str, count: int) -> list[dict[str, str]]:
    """Mint ``count`` synthetic GSM8K rows with monotonic gold numbers."""

    return [
        {
            "question": f"{prefix} q{i}",
            "answer": f"reasoning ... #### {i}",
        }
        for i in range(count)
    ]


def test_math_adapter_uses_native_splits_in_dataset_order(monkeypatch) -> None:
    """With both native splits configured, ``train_test_split`` returns the
    natively-loaded partitions verbatim (no shuffling, ignores seed/train_frac).
    """

    train_rows = _fake_gsm8k_rows("train", 5)
    test_rows = _fake_gsm8k_rows("test", 3)

    def fake_load_dataset(_name, _config, split):
        if split == "train":
            return train_rows
        if split == "test":
            return test_rows
        raise AssertionError(f"unexpected split {split!r}")

    monkeypatch.setattr(math_module, "load_dataset", fake_load_dataset)

    adapter = MathTaskAdapter()
    train, test = adapter.train_test_split(seed=999, train_frac=0.1)

    assert [r.task_id for r in train] == [
        f"gsm8k/train/{i:05d}" for i in range(len(train_rows))
    ]
    assert [r.task_id for r in test] == [
        f"gsm8k/test/{i:05d}" for i in range(len(test_rows))
    ]
    assert adapter.dataset_spec == "math:openai/gsm8k:main:train+test"


def test_math_adapter_falls_back_to_seeded_split_when_train_split_is_none(
    monkeypatch,
) -> None:
    """When ``train_split=None`` only the test split is loaded and the seeded
    fallback partitions it (~80/20).
    """

    test_rows = _fake_gsm8k_rows("only", 10)

    def fake_load_dataset(_name, _config, split):
        assert split == "test"
        return test_rows

    monkeypatch.setattr(math_module, "load_dataset", fake_load_dataset)

    adapter = MathTaskAdapter(train_split=None)
    train, test = adapter.train_test_split(seed=65320, train_frac=0.8)
    assert len(train) + len(test) == len(test_rows)
    # Reproducibility: same seed -> same partition.
    train_again, test_again = adapter.train_test_split(seed=65320, train_frac=0.8)
    assert [r.task_id for r in train] == [r.task_id for r in train_again]
    assert [r.task_id for r in test] == [r.task_id for r in test_again]
