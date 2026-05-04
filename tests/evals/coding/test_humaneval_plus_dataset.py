"""Tests for HumanEval+ dataset loader utilities."""

from __future__ import annotations

import pytest

from pruning_metrics.evals.coding.humaneval_plus_dataset import (
    HumanEvalPlusDatasetLoader,
)


@pytest.fixture
def fake_dataset_records() -> list[dict[str, str]]:
    """Provide minimal fake dataset records.

    Parameters
    ----------
    None

    Returns
    -------
    list[dict[str, str]]
        Fake rows that mimic HumanEval+ schema.

    Preconditions
    -------------
    None

    Postconditions
    --------------
    Returned list has two records with distinct IDs.
    """

    return [
        {
            "task_id": "HumanEval/0",
            "prompt": "def foo(x):\n    pass",
            "entry_point": "foo",
            "test": "def check(candidate):\n    assert candidate(1) == 2",
            "canonical_solution": "def foo(x):\n    return x + 1",
        },
        {
            "task_id": "HumanEval/1",
            "prompt": "def bar(x):\n    pass",
            "entry_point": "bar",
            "test": "def check(candidate):\n    assert candidate(1) == 3",
            "canonical_solution": "def bar(x):\n    return x + 2",
        },
    ]


def test_load_tasks_applies_max_samples(
    monkeypatch: pytest.MonkeyPatch,
    fake_dataset_records: list[dict[str, str]],
) -> None:
    """Validate ``max_samples`` truncates filtered output.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture.
    fake_dataset_records:
        Fake dataset rows.

    Returns
    -------
    None

    Preconditions
    -------------
    ``load_dataset`` is monkeypatched to fake data.

    Postconditions
    --------------
    Exactly one task is returned with the first dataset ID.
    """

    monkeypatch.setattr(
        "pruning_metrics.evals.coding.humaneval_plus_dataset.load_dataset",
        lambda dataset_name, split: fake_dataset_records,
    )

    loader = HumanEvalPlusDatasetLoader()
    tasks = loader.load_tasks(max_samples=1)

    assert len(tasks) == 1
    assert tasks[0].task_id == "HumanEval/0"


def test_load_tasks_filters_on_task_ids(
    monkeypatch: pytest.MonkeyPatch,
    fake_dataset_records: list[dict[str, str]],
) -> None:
    """Validate task ID filtering returns requested records only.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture.
    fake_dataset_records:
        Fake dataset rows.

    Returns
    -------
    None

    Preconditions
    -------------
    Requested ID exists in fake dataset.

    Postconditions
    --------------
    Returned list contains exactly the requested task.
    """

    monkeypatch.setattr(
        "pruning_metrics.evals.coding.humaneval_plus_dataset.load_dataset",
        lambda dataset_name, split: fake_dataset_records,
    )

    loader = HumanEvalPlusDatasetLoader()
    tasks = loader.load_tasks(task_ids=["HumanEval/1"])

    assert len(tasks) == 1
    assert tasks[0].task_id == "HumanEval/1"
    assert tasks[0].entry_point == "bar"


def test_load_tasks_raises_for_missing_task_ids(
    monkeypatch: pytest.MonkeyPatch,
    fake_dataset_records: list[dict[str, str]],
) -> None:
    """Validate loader raises when requested IDs are absent.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture.
    fake_dataset_records:
        Fake dataset rows.

    Returns
    -------
    None

    Preconditions
    -------------
    Requested ID is not present in fake dataset.

    Postconditions
    --------------
    ``ValueError`` is raised with missing ID details.
    """

    monkeypatch.setattr(
        "pruning_metrics.evals.coding.humaneval_plus_dataset.load_dataset",
        lambda dataset_name, split: fake_dataset_records,
    )

    loader = HumanEvalPlusDatasetLoader()
    with pytest.raises(ValueError, match="HumanEval/404"):
        loader.load_tasks(task_ids=["HumanEval/404"])


def _make_records(num_tasks: int) -> list[dict[str, str]]:
    """Build ``num_tasks`` synthetic HumanEval+ rows for split-determinism tests."""

    return [
        {
            "task_id": f"HumanEval/{idx}",
            "prompt": f"def f_{idx}(x):\n    pass",
            "entry_point": f"f_{idx}",
            "test": "def check(candidate):\n    assert True",
            "canonical_solution": f"def f_{idx}(x):\n    return x",
        }
        for idx in range(num_tasks)
    ]


def test_split_train_test_is_deterministic_for_seed_65320(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 80/20 split for seed 65320 must be byte-identical across calls.

    The pruning runner records calibration vs. evaluation IDs in
    ``split.json`` and we must guarantee that re-running the same code on the
    same seed produces the same partition (so a paused job can resume cleanly,
    and so reviewers can reproduce the split).
    """

    records = _make_records(20)
    monkeypatch.setattr(
        "pruning_metrics.evals.coding.humaneval_plus_dataset.load_dataset",
        lambda dataset_name, split: records,
    )

    loader = HumanEvalPlusDatasetLoader()
    train_a, test_a = loader.split_train_test(seed=65320, train_frac=0.8)
    train_b, test_b = loader.split_train_test(seed=65320, train_frac=0.8)

    assert [task.task_id for task in train_a] == [task.task_id for task in train_b]
    assert [task.task_id for task in test_a] == [task.task_id for task in test_b]
    assert len(train_a) + len(test_a) == 20
    # 80 % of 20 -> 16 train, 4 test
    assert len(train_a) == 16
    assert len(test_a) == 4

    overlap = {task.task_id for task in train_a} & {task.task_id for task in test_a}
    assert not overlap

    different_seed = loader.split_train_test(seed=1, train_frac=0.8)[0]
    assert [task.task_id for task in different_seed] != [
        task.task_id for task in train_a
    ], "Different seeds should usually produce different shuffles."


def test_split_train_test_validates_train_frac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``train_frac`` must be strictly between 0 and 1."""

    records = _make_records(5)
    monkeypatch.setattr(
        "pruning_metrics.evals.coding.humaneval_plus_dataset.load_dataset",
        lambda dataset_name, split: records,
    )

    loader = HumanEvalPlusDatasetLoader()
    with pytest.raises(ValueError, match="train_frac"):
        loader.split_train_test(train_frac=0.0)
    with pytest.raises(ValueError, match="train_frac"):
        loader.split_train_test(train_frac=1.0)
