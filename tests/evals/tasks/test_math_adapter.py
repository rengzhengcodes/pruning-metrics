"""Numeric extraction + verification for the GSM8K math adapter."""

from __future__ import annotations

import pytest

from pruning_metrics.evals.tasks.base import TaskRecord
from pruning_metrics.evals.tasks.math import (
    MathTaskAdapter,
    _extract_numeric,
    _parse_number,
)


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
