"""End-to-end HumanEval+ inference and verification pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pruning_metrics.evals.coding.humaneval_plus_dataset import HumanEvalPlusTask
from pruning_metrics.evals.coding.llm_client import LLMClient
from pruning_metrics.evals.coding.verifier import (
    VerificationResult,
    verify_task_solution,
)


@dataclass(frozen=True)
class TaskEvaluationRecord:
    """One task evaluation record emitted by pipeline.

    Parameters
    ----------
    task_id:
        HumanEval+ task identifier.
    entry_point:
        Expected function name for this task.
    prompt:
        Prompt sent to the model.
    generated_code:
        Model-generated candidate solution.
    verification:
        Verification metadata from test execution.

    Returns
    -------
    None

    Preconditions
    -------------
    Fields are consistent with the evaluated task.

    Postconditions
    --------------
    Record is immutable and serializable via ``asdict``.
    """

    task_id: str
    entry_point: str
    prompt: str
    generated_code: str
    verification: VerificationResult


@dataclass(frozen=True)
class PipelineResult:
    """Aggregated result from a pipeline run.

    Parameters
    ----------
    records:
        Per-task evaluation records.
    num_tasks:
        Number of tasks processed.
    num_passed:
        Number of tasks with passing verification.
    pass_at_1:
        Fraction of tasks passing with first generated sample.
    status_breakdown:
        Counts by verification status.

    Returns
    -------
    None

    Preconditions
    -------------
    ``num_tasks`` matches ``len(records)``.

    Postconditions
    --------------
    Aggregates are consistent with underlying records.
    """

    records: list[TaskEvaluationRecord]
    num_tasks: int
    num_passed: int
    pass_at_1: float
    status_breakdown: dict[str, int]


def build_coding_prompt(task: HumanEvalPlusTask) -> str:
    """Build strict coding prompt for a HumanEval+ task.

    Parameters
    ----------
    task:
        Task definition from HumanEval+.

    Returns
    -------
    str
        Prompt requesting Python code for the specified entry point.

    Preconditions
    -------------
    Task has non-empty prompt and entry point.

    Postconditions
    --------------
    Returned prompt asks for Python code only.
    """

    return (
        "You are solving a Python programming task.\n"
        "Return only valid Python code, no markdown fences or explanations.\n"
        f"The solution must define a callable `{task.entry_point}`.\n\n"
        f"{task.prompt}"
    )


def run_pipeline(
    tasks: list[HumanEvalPlusTask],
    llm_client: LLMClient,
    timeout_seconds: float = 5.0,
    output_jsonl_path: str | None = None,
) -> PipelineResult:
    """Run full prompt -> generation -> verification workflow.

    Parameters
    ----------
    tasks:
        HumanEval+ tasks to evaluate.
    llm_client:
        Model client implementing ``generate_code``.
    timeout_seconds:
        Verification timeout budget per task.
    output_jsonl_path:
        Optional path for per-task JSONL records.

    Returns
    -------
    PipelineResult
        Aggregated results with per-task details.

    Preconditions
    -------------
    ``tasks`` is non-empty for meaningful evaluation.

    Postconditions
    --------------
    Returns deterministic aggregation for deterministic model outputs.
    """

    records: list[TaskEvaluationRecord] = []
    for task in tasks:
        prompt = build_coding_prompt(task)
        generated_code = llm_client.generate_code(prompt=prompt, task_id=task.task_id)
        verification = verify_task_solution(
            task=task,
            generated_solution=generated_code,
            timeout_seconds=timeout_seconds,
        )
        records.append(
            TaskEvaluationRecord(
                task_id=task.task_id,
                entry_point=task.entry_point,
                prompt=prompt,
                generated_code=generated_code,
                verification=verification,
            )
        )

    if output_jsonl_path is not None:
        _write_jsonl_records(records, output_jsonl_path)

    status_breakdown = _build_status_breakdown(records)
    num_tasks = len(records)
    num_passed = status_breakdown.get("pass", 0)
    pass_at_1 = (num_passed / num_tasks) if num_tasks else 0.0

    return PipelineResult(
        records=records,
        num_tasks=num_tasks,
        num_passed=num_passed,
        pass_at_1=pass_at_1,
        status_breakdown=status_breakdown,
    )


def _build_status_breakdown(records: list[TaskEvaluationRecord]) -> dict[str, int]:
    """Count verification statuses from task records.

    Parameters
    ----------
    records:
        Task-level pipeline records.

    Returns
    -------
    dict[str, int]
        Status count mapping.

    Preconditions
    -------------
    ``records`` entries each contain a verification result.

    Postconditions
    --------------
    Sum of all counts equals ``len(records)``.
    """

    breakdown: dict[str, int] = {}
    for record in records:
        status = record.verification.status
        breakdown[status] = breakdown.get(status, 0) + 1
    return breakdown


def _write_jsonl_records(records: list[TaskEvaluationRecord], output_path: str) -> None:
    """Write per-task records to JSONL output.

    Parameters
    ----------
    records:
        Task-level pipeline records.
    output_path:
        Destination JSONL path.

    Returns
    -------
    None

    Preconditions
    -------------
    ``output_path`` parent directory is writable.

    Postconditions
    --------------
    Output file contains one JSON object per task.
    """

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")
