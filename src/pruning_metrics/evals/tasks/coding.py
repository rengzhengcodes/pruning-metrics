"""Coding-task adapter (HumanEval+, MBPP-shaped data).

Wraps the existing :class:`HumanEvalPlusDatasetLoader` and
:func:`verify_task_solution` so the new task-adapter layer can drive the same
subprocess-based pass@1 pipeline that the legacy notebook relied on.
"""

from __future__ import annotations

from typing import Sequence

from pruning_metrics.evals.coding.humaneval_plus_dataset import (
    HumanEvalPlusDatasetLoader,
    HumanEvalPlusTask,
)
from pruning_metrics.evals.coding.verifier import verify_task_solution
from pruning_metrics.evals.tasks.base import (
    TaskAdapter,
    TaskRecord,
    VerificationOutcome,
    deterministic_split,
)


class CodingTaskAdapter(TaskAdapter):
    """Adapter for HumanEval+-shaped coding datasets.

    Parameters
    ----------
    dataset_name:
        ``datasets.load_dataset`` name. Default ``evalplus/humanevalplus``.
    split:
        Dataset split (default ``test``; HumanEval+ ships only that split).

    Notes
    -----
    Verification reuses the project's existing subprocess test harness which
    expects a ``check`` function defined by the dataset's ``test`` field and
    runs against the model-generated source plus the canonical solution
    fallback. ``target_text`` is the canonical solution so teacher forcing
    can score how well the model reproduces the reference Python code.
    """

    name = "coding"

    def __init__(
        self,
        dataset_name: str = "evalplus/humanevalplus",
        split: str = "test",
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.dataset_spec = f"coding:{dataset_name}:{split}"
        self._records: list[TaskRecord] | None = None

    def load_records(self) -> list[TaskRecord]:
        """Materialise all HumanEval+ tasks as :class:`TaskRecord` objects."""

        if self._records is not None:
            return list(self._records)

        loader = HumanEvalPlusDatasetLoader(
            dataset_name=self.dataset_name, split=self.split
        )
        tasks = loader.load_tasks()
        records = [self._to_record(task) for task in tasks]
        # Cache so repeated calls (notebook + runner) avoid re-downloading.
        self._records = records
        return list(records)

    def train_test_split(
        self,
        seed: int = 65320,
        train_frac: float = 0.8,
        explicit_train_ids: Sequence[str] | None = None,
        explicit_test_ids: Sequence[str] | None = None,
    ) -> tuple[list[TaskRecord], list[TaskRecord]]:
        return deterministic_split(
            self.load_records(),
            seed=seed,
            train_frac=train_frac,
            explicit_train_ids=explicit_train_ids,
            explicit_test_ids=explicit_test_ids,
        )

    def build_inference_prompt(self, record: TaskRecord) -> str:
        """Wrap the raw HumanEval+ prompt with an instruction.

        The raw prompt is just a partial ``def`` plus a docstring; on its own
        a model would tend to continue with a function body and the verifier
        wouldn't find a complete callable. The wrapper mirrors the
        long-running :func:`pruning_metrics.evals.coding.pipeline.build_coding_prompt`
        helper used by the legacy notebook so the smoke-test pass@1 numbers
        line up with prior baselines.
        """

        entry_point = record.metadata.get("entry_point", "solution")
        return (
            "You are solving a Python programming task.\n"
            "Return only valid Python code, no markdown fences or explanations.\n"
            f"The solution must define a callable `{entry_point}`.\n\n"
            f"{record.prompt}"
        )

    def verify(
        self,
        record: TaskRecord,
        generated_text: str,
        timeout_seconds: float = 10.0,
    ) -> VerificationOutcome:
        """Run HumanEval+ subprocess tests on ``generated_text``.

        Reconstructs a temporary :class:`HumanEvalPlusTask` from the metadata
        embedded in ``record`` so the existing verifier (which is typed
        against the legacy class) keeps working.
        """

        task = HumanEvalPlusTask(
            task_id=record.task_id,
            prompt=record.prompt,
            entry_point=str(record.metadata["entry_point"]),
            test=str(record.metadata["test"]),
            canonical_solution=record.target_text,
        )
        result = verify_task_solution(
            task=task,
            generated_solution=generated_text,
            timeout_seconds=timeout_seconds,
        )
        return VerificationOutcome(
            task_id=record.task_id,
            status=result.status,
            detail=str(result.return_code) if result.return_code is not None else "",
        )

    @staticmethod
    def _to_record(task: HumanEvalPlusTask) -> TaskRecord:
        canonical = task.canonical_solution or ""
        if not canonical:
            # Without a canonical solution we cannot teacher-force; fall back to
            # an empty target so the calibration path still works while making
            # downstream TF skip the record loudly.
            canonical = ""
        return TaskRecord(
            task_id=task.task_id,
            prompt=task.prompt,
            target_text=canonical,
            metadata={
                "entry_point": task.entry_point,
                "test": task.test,
            },
        )
