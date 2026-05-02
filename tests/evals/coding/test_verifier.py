"""Tests for HumanEval+ verifier behavior."""

from __future__ import annotations

from pruning_metrics.evals.coding.humaneval_plus_dataset import HumanEvalPlusTask
from pruning_metrics.evals.coding.verifier import verify_task_solution


def test_verify_task_solution_pass_status() -> None:
    """Verify passing generated code is marked as ``pass``.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Preconditions
    -------------
    Generated solution satisfies test assertions.

    Postconditions
    --------------
    Result status is ``pass`` and timeout flag is false.
    """

    task = HumanEvalPlusTask(
        task_id="HumanEval/test-pass",
        prompt="def add_one(x):\n    pass",
        entry_point="add_one",
        test="def check(candidate):\n    assert candidate(1) == 2",
    )
    generated_solution = "def add_one(x):\n    return x + 1\n"

    result = verify_task_solution(task, generated_solution, timeout_seconds=1.0)

    assert result.status == "pass"
    assert result.timed_out is False
    assert result.return_code == 0


def test_verify_task_solution_fail_status() -> None:
    """Verify failing generated code is marked as ``fail``.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Preconditions
    -------------
    Generated solution violates task assertions.

    Postconditions
    --------------
    Result status is ``fail``.
    """

    task = HumanEvalPlusTask(
        task_id="HumanEval/test-fail",
        prompt="def add_one(x):\n    pass",
        entry_point="add_one",
        test="def check(candidate):\n    assert candidate(1) == 2",
    )
    generated_solution = "def add_one(x):\n    return x\n"

    result = verify_task_solution(task, generated_solution, timeout_seconds=1.0)

    assert result.status == "fail"
    assert result.return_code != 0


def test_verify_task_solution_timeout_status() -> None:
    """Verify timeout is reported for long-running solutions.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Preconditions
    -------------
    Generated solution enters an infinite loop.

    Postconditions
    --------------
    Result status is ``timeout`` and timeout flag is true.
    """

    task = HumanEvalPlusTask(
        task_id="HumanEval/test-timeout",
        prompt="def hang(x):\n    pass",
        entry_point="hang",
        test="def check(candidate):\n    candidate(1)",
    )
    generated_solution = "def hang(x):\n    while True:\n        pass\n"

    result = verify_task_solution(task, generated_solution, timeout_seconds=0.1)

    assert result.status == "timeout"
    assert result.timed_out is True
    assert result.return_code is None
