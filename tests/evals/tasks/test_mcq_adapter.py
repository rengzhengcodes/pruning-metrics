"""ARC-Challenge MCQ adapter regex + verification."""

from __future__ import annotations

import pytest

from pruning_metrics.evals.tasks.base import TaskRecord
from pruning_metrics.evals.tasks.mcq import MCQTaskAdapter


@pytest.fixture
def sample_record() -> TaskRecord:
    return TaskRecord(
        task_id="arc/ARC-Challenge/Mercury_test",
        prompt="(prompt body)",
        target_text="The right thing.",
        metadata={
            "answer_key": "C",
            "choice_labels": ["A", "B", "C", "D"],
            "choice_texts": ["wrong", "wrong", "The right thing.", "wrong"],
            "question": "Q",
        },
    )


def test_verify_pass_on_first_letter_match(sample_record) -> None:
    adapter = MCQTaskAdapter()
    assert adapter.verify(sample_record, "C").status == "pass"
    assert adapter.verify(sample_record, "Answer: C\nReason: because.").status == "pass"
    # Word-bounded so embedded letters in words don't match.
    assert adapter.verify(sample_record, "The Answer (C) is correct").status == "pass"


def test_verify_fail_on_wrong_letter(sample_record) -> None:
    adapter = MCQTaskAdapter()
    outcome = adapter.verify(sample_record, "A) wrong")
    assert outcome.status == "fail"
    assert "predicted=A" in outcome.detail


def test_verify_parse_error_when_no_letter(sample_record) -> None:
    adapter = MCQTaskAdapter()
    outcome = adapter.verify(sample_record, "I have no idea.")
    assert outcome.status == "parse_error"


def test_verify_picks_first_letter(sample_record) -> None:
    """If the model emits multiple letters we use the first."""

    adapter = MCQTaskAdapter()
    assert adapter.verify(sample_record, "C is right; A is wrong").status == "pass"
    assert adapter.verify(sample_record, "A is wrong; C is right").status == "fail"


@pytest.mark.parametrize("letter", ["A", "B", "C", "D", "E"])
def test_letter_pattern_handles_lowercase(letter: str, sample_record) -> None:
    """Lowercase letters are normalised in the regex search."""

    adapter = MCQTaskAdapter()
    record = TaskRecord(
        task_id="x",
        prompt="",
        target_text="",
        metadata={
            "answer_key": letter,
            "choice_labels": ["A", "B", "C", "D", "E"],
            "choice_texts": ["a", "b", "c", "d", "e"],
            "question": "?",
        },
    )
    assert adapter.verify(record, f"the answer is {letter.lower()}").status == "pass"
