"""Solution verification against HumanEval+ tests."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pruning_metrics.evals.coding.humaneval_plus_dataset import HumanEvalPlusTask


@dataclass(frozen=True)
class VerificationResult:
    """Verification outcome for a single task solution.

    Parameters
    ----------
    task_id:
        HumanEval+ task identifier.
    status:
        One of ``pass``, ``fail``, ``timeout``, or ``runtime_error``.
    return_code:
        Python process return code, if available.
    stdout:
        Captured standard output (trimmed).
    stderr:
        Captured standard error (trimmed).
    timed_out:
        Whether execution exceeded timeout.
    execution_seconds:
        Timeout budget used for this execution.

    Returns
    -------
    None

    Preconditions
    -------------
    Status is one of the supported values.

    Postconditions
    --------------
    Result metadata is immutable after initialization.
    """

    task_id: str
    status: str
    return_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    execution_seconds: float


def verify_task_solution(
    task: HumanEvalPlusTask,
    generated_solution: str,
    timeout_seconds: float = 5.0,
) -> VerificationResult:
    """Execute HumanEval+ tests against a generated solution.

    Parameters
    ----------
    task:
        Normalized HumanEval+ task.
    generated_solution:
        Candidate Python solution to validate.
    timeout_seconds:
        Maximum wall-clock time allowed for execution.

    Returns
    -------
    VerificationResult
        Structured verification result with status and logs.

    Preconditions
    -------------
    ``generated_solution`` is Python source text.

    Postconditions
    --------------
    A subprocess is executed in an isolated temporary directory.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be strictly positive.")

    test_script = _build_execution_script(task, generated_solution)
    with tempfile.TemporaryDirectory(prefix="humaneval_plus_") as tmp_dir:
        script_path = Path(tmp_dir) / "candidate_eval.py"
        script_path.write_text(test_script, encoding="utf-8")
        try:
            process = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                cwd=tmp_dir,
            )
        except subprocess.TimeoutExpired as error:
            return VerificationResult(
                task_id=task.task_id,
                status="timeout",
                return_code=None,
                stdout=(error.stdout or "").strip(),
                stderr=(error.stderr or "").strip(),
                timed_out=True,
                execution_seconds=timeout_seconds,
            )

    status = _return_code_to_status(process.returncode)
    return VerificationResult(
        task_id=task.task_id,
        status=status,
        return_code=process.returncode,
        stdout=process.stdout.strip(),
        stderr=process.stderr.strip(),
        timed_out=False,
        execution_seconds=timeout_seconds,
    )


def _build_execution_script(task: HumanEvalPlusTask, generated_solution: str) -> str:
    """Construct executable script from model output and HumanEval+ test.

    Parameters
    ----------
    task:
        HumanEval+ task containing test source and entry point.
    generated_solution:
        Candidate solution source.

    Returns
    -------
    str
        Complete Python script to execute in subprocess.

    Preconditions
    -------------
    Task has non-empty ``test`` and ``entry_point`` fields.

    Postconditions
    --------------
    Script triggers HumanEval+ check function against entry point.
    """

    return (
        f"{generated_solution.rstrip()}\n\n"
        f"{task.test.rstrip()}\n\n"
        f"check({task.entry_point})\n"
    )


def _return_code_to_status(return_code: int) -> str:
    """Map Python process return code to evaluation status.

    Parameters
    ----------
    return_code:
        Process exit code from task execution.

    Returns
    -------
    str
        ``pass`` for zero code, otherwise ``fail``.

    Preconditions
    -------------
    ``return_code`` is an integer.

    Postconditions
    --------------
    Status string belongs to supported verifier statuses.
    """

    if return_code == 0:
        return "pass"
    if return_code in (1, 2):
        return "fail"
    return "runtime_error"
