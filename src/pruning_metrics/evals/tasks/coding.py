"""Coding-task adapter (HumanEval+, MBPP-shaped data).

Wraps the existing :class:`HumanEvalPlusDatasetLoader` and
:func:`verify_task_solution` so the task-adapter layer can drive the same
subprocess-based pass@1 verification used by notebook 3.
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
    native_or_seeded_split,
)


class CodingTaskAdapter(TaskAdapter):
    """Adapter for HumanEval+-shaped coding datasets.

    Parameters
    ----------
    dataset_name:
        ``datasets.load_dataset`` name. Default ``evalplus/humanevalplus``.
    train_split:
        Native train split name. Default ``None`` because HumanEval+ ships
        only a ``test`` split; the seeded 80/20 fallback in
        :func:`pruning_metrics.evals.tasks.base.native_or_seeded_split`
        produces the calibration partition.
    test_split:
        Dataset split that holds all records. Default ``"test"``.

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
        train_split: str | None = None,
        test_split: str = "test",
    ) -> None:
        self.dataset_name = dataset_name
        self.train_split = train_split
        self.test_split = test_split
        split_label = (
            f"{train_split}+{test_split}" if train_split else test_split
        )
        self.dataset_spec = f"coding:{dataset_name}:{split_label}"
        self._train_records: list[TaskRecord] | None = None
        self._test_records: list[TaskRecord] | None = None

    def _load_split(self, split: str) -> list[TaskRecord]:
        """Materialise records from a single HumanEval+ split."""

        loader = HumanEvalPlusDatasetLoader(
            dataset_name=self.dataset_name, split=split
        )
        tasks = loader.load_tasks()
        return [self._to_record(task) for task in tasks]

    def load_records(self) -> list[TaskRecord]:
        """Concatenate train + test records (train first when available)."""

        if self._test_records is None:
            self._test_records = self._load_split(self.test_split)
        if self.train_split is not None and self._train_records is None:
            self._train_records = self._load_split(self.train_split)

        if self._train_records is not None:
            return list(self._train_records) + list(self._test_records)
        return list(self._test_records)

    def train_test_split(
        self,
        seed: int = 65320,
        train_frac: float = 0.8,
        explicit_train_ids: Sequence[str] | None = None,
        explicit_test_ids: Sequence[str] | None = None,
    ) -> tuple[list[TaskRecord], list[TaskRecord]]:
        # Trigger lazy load so the (single or paired) splits are populated.
        self.load_records()
        return native_or_seeded_split(
            self._train_records,
            self._test_records,
            seed=seed,
            train_frac=train_frac,
            explicit_train_ids=explicit_train_ids,
            explicit_test_ids=explicit_test_ids,
        )

    def build_inference_prompt(self, record: TaskRecord) -> str:
        """Wrap the raw HumanEval+ prompt with an instruction.

        The raw prompt is just a partial ``def`` plus a docstring; on its
        own a model would tend to continue with a function body and the
        verifier wouldn't find a complete callable. This wrapper asks for
        a complete Python definition so the verifier can resolve the
        named entry point.
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
