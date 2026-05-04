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
