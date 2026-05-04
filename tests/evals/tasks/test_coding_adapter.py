"""Coding adapter wraps HumanEval+ loader + verifier correctly."""

from __future__ import annotations

import pytest

from pruning_metrics.evals.tasks.coding import CodingTaskAdapter


@pytest.fixture
def fake_dataset_records() -> list[dict[str, str]]:
    return [
        {
            "task_id": "HumanEval/0",
            "prompt": "def add_one(x):\n    pass\n",
            "entry_point": "add_one",
            "test": "def check(candidate):\n    assert candidate(1) == 2\n",
            "canonical_solution": "    return x + 1\n",
        },
        {
            "task_id": "HumanEval/1",
            "prompt": "def add_two(x):\n    pass\n",
            "entry_point": "add_two",
            "test": "def check(candidate):\n    assert candidate(1) == 3\n",
            "canonical_solution": "    return x + 2\n",
        },
    ]


def test_coding_adapter_loads_records(monkeypatch, fake_dataset_records) -> None:
    monkeypatch.setattr(
        "pruning_metrics.evals.coding.humaneval_plus_dataset.load_dataset",
        lambda dataset_name, split: fake_dataset_records,
    )

    adapter = CodingTaskAdapter()
    records = adapter.load_records()
    assert [r.task_id for r in records] == ["HumanEval/0", "HumanEval/1"]
    assert records[0].target_text.strip() == "return x + 1"
    assert records[0].metadata["entry_point"] == "add_one"


def test_coding_adapter_train_test_split_seeded(
    monkeypatch, fake_dataset_records
) -> None:
    """Adapter delegates to the shared seeded splitter."""

    monkeypatch.setattr(
        "pruning_metrics.evals.coding.humaneval_plus_dataset.load_dataset",
        lambda dataset_name, split: fake_dataset_records,
    )

    adapter = CodingTaskAdapter()
    train, test = adapter.train_test_split(seed=65320, train_frac=0.5)
    assert len(train) + len(test) == 2
    train_again, test_again = adapter.train_test_split(seed=65320, train_frac=0.5)
    assert [r.task_id for r in train] == [r.task_id for r in train_again]
    assert [r.task_id for r in test] == [r.task_id for r in test_again]


def test_coding_adapter_verify_pass(monkeypatch, fake_dataset_records) -> None:
    monkeypatch.setattr(
        "pruning_metrics.evals.coding.humaneval_plus_dataset.load_dataset",
        lambda dataset_name, split: fake_dataset_records,
    )
    adapter = CodingTaskAdapter()
    records = adapter.load_records()
    # The HumanEval+ verifier expects the model to provide a function body.
    correct_solution = "def add_one(x):\n    return x + 1\n"
    outcome = adapter.verify(records[0], correct_solution, timeout_seconds=5.0)
    assert outcome.status == "pass"


def test_coding_adapter_verify_fail(monkeypatch, fake_dataset_records) -> None:
    monkeypatch.setattr(
        "pruning_metrics.evals.coding.humaneval_plus_dataset.load_dataset",
        lambda dataset_name, split: fake_dataset_records,
    )
    adapter = CodingTaskAdapter()
    records = adapter.load_records()
    wrong = "def add_one(x):\n    return x - 1\n"
    outcome = adapter.verify(records[0], wrong, timeout_seconds=5.0)
    assert outcome.status in ("fail", "runtime_error")


def test_coding_adapter_build_inference_prompt_wraps_with_instruction(
    monkeypatch, fake_dataset_records
) -> None:
    """The free-form prompt must remind the model to produce a complete callable."""

    monkeypatch.setattr(
        "pruning_metrics.evals.coding.humaneval_plus_dataset.load_dataset",
        lambda dataset_name, split: fake_dataset_records,
    )
    adapter = CodingTaskAdapter()
    record = adapter.load_records()[0]
    wrapped = adapter.build_inference_prompt(record)
    # The wrapper must include the entry-point name and the original prompt.
    assert "add_one" in wrapped
    assert record.prompt in wrapped
    assert "Return only valid Python code" in wrapped
    # TF prompt is the raw prompt, unchanged.
    assert record.prompt != wrapped


def test_coding_adapter_default_train_split_is_none(
    monkeypatch, fake_dataset_records
) -> None:
    """HumanEval+ ships only a test split, so train_split defaults to None
    and the seeded fallback partitions the test split into train/test."""

    monkeypatch.setattr(
        "pruning_metrics.evals.coding.humaneval_plus_dataset.load_dataset",
        lambda dataset_name, split: fake_dataset_records,
    )

    adapter = CodingTaskAdapter()
    assert adapter.train_split is None
    assert adapter.test_split == "test"

    train, test = adapter.train_test_split(seed=65320, train_frac=0.5)
    # All records still accounted for after the seeded shuffle.
    assert len(train) + len(test) == len(fake_dataset_records)
