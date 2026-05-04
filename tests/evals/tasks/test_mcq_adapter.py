"""ARC-Challenge MCQ adapter regex + verification."""

from __future__ import annotations

import pytest

from pruning_metrics.evals.tasks.base import TaskRecord
from pruning_metrics.evals.tasks.mcq import MCQTaskAdapter
from pruning_metrics.evals.tasks import mcq as mcq_module


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


def _fake_arc_rows(prefix: str, count: int) -> list[dict[str, object]]:
    """Mint ``count`` synthetic ARC rows with deterministic ids."""

    return [
        {
            "id": f"{prefix}_{i}",
            "question": f"{prefix} q{i}",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": [f"{prefix}-w0", f"{prefix}-w1", f"{prefix}-w2", f"{prefix}-w3"],
            },
            "answerKey": "B",
        }
        for i in range(count)
    ]


def test_mcq_adapter_uses_native_splits_in_dataset_order(monkeypatch) -> None:
    """ARC has native train + test splits; both are loaded in dataset order."""

    train_rows = _fake_arc_rows("train", 4)
    test_rows = _fake_arc_rows("test", 6)

    def fake_load_dataset(_name, _config, split):
        if split == "train":
            return train_rows
        if split == "test":
            return test_rows
        raise AssertionError(f"unexpected split {split!r}")

    monkeypatch.setattr(mcq_module, "load_dataset", fake_load_dataset)

    adapter = MCQTaskAdapter()
    train, test = adapter.train_test_split(seed=999, train_frac=0.1)
    assert [r.task_id for r in train] == [
        f"arc/ARC-Challenge/train_{i}" for i in range(len(train_rows))
    ]
    assert [r.task_id for r in test] == [
        f"arc/ARC-Challenge/test_{i}" for i in range(len(test_rows))
    ]
    assert adapter.dataset_spec == "mcq:allenai/ai2_arc:ARC-Challenge:train+test"


def test_mcq_adapter_falls_back_to_seeded_split_when_no_train(monkeypatch) -> None:
    """``train_split=None`` reverts to the seeded 80/20 fallback."""

    test_rows = _fake_arc_rows("only", 10)

    def fake_load_dataset(_name, _config, split):
        assert split == "test"
        return test_rows

    monkeypatch.setattr(mcq_module, "load_dataset", fake_load_dataset)

    adapter = MCQTaskAdapter(train_split=None)
    train, test = adapter.train_test_split(seed=65320, train_frac=0.8)
    assert len(train) + len(test) == len(test_rows)
