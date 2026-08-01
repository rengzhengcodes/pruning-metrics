"""Coding-task adapter (HumanEval+, MBPP-shaped data).

Wraps the existing :class:`HumanEvalPlusDatasetLoader` and
:func:`verify_task_solution` so the task-adapter layer can drive the same
subprocess-based pass@1 verification used by notebook 3.

Two concrete adapters live here:

* :class:`CodingTaskAdapter` -- HumanEval+ (a ``def`` stub plus a ``check``
  test harness in the dataset).
* :class:`MbppTaskAdapter` -- MBPP+ (``evalplus/mbppplus``), whose rows are a
  natural-language description, a canonical ``code`` solution, and a list of
  ``assert`` statements rather than a HumanEval-style ``check`` function. The
  MBPP adapter reshapes each row into the same
  ``(entry_point, test)`` metadata that the shared subprocess verifier
  expects, so verification is inherited unchanged.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from datasets import load_dataset

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


_DEF_PATTERN = re.compile(r"^\s*def\s+(\w+)\s*\(", re.MULTILINE)
_CALL_PATTERN = re.compile(r"(\w+)\s*\(")


class MbppTaskAdapter(CodingTaskAdapter):
    """Adapter for MBPP+-shaped coding datasets (``evalplus/mbppplus``).

    MBPP+ rows differ from HumanEval+ in three ways that this subclass
    reconciles so the rest of the pipeline (teacher forcing, subprocess
    verification, the seeded split) works without modification:

    * There is no ``def`` stub prompt -- only a one-line natural-language
      ``prompt`` describing the function. The teacher-forcing prompt is
      rebuilt from that description plus the sample ``assert`` statements so
      the model has the function signature it must reproduce; the
      ground-truth ``target_text`` is the canonical ``code`` solution.
    * There is no ``entry_point`` field. It is recovered as the function name
      that is both defined in ``code`` and called by the ``test_list``
      assertions (falling back to the last top-level ``def``).
    * There is no HumanEval-style ``check`` function. One is synthesised from
      ``test_imports`` + ``test_list`` (the asserts call the function by its
      literal name, resolved from the module scope where the candidate
      solution defines it), and stored in ``metadata["test"]`` so the
      inherited :meth:`CodingTaskAdapter.verify` runs the shared subprocess
      harness unchanged.

    Parameters
    ----------
    dataset_name:
        ``datasets.load_dataset`` name. Default ``evalplus/mbppplus``.
    train_split:
        Native train split name. Default ``None`` because MBPP+ ships only a
        ``test`` split; the seeded 80/20 fallback produces the calibration
        partition.
    test_split:
        Dataset split that holds all records. Default ``"test"``.
    """

    name = "coding"

    def __init__(
        self,
        dataset_name: str = "evalplus/mbppplus",
        train_split: str | None = None,
        test_split: str = "test",
    ) -> None:
        self.dataset_name = dataset_name
        self.train_split = train_split
        self.test_split = test_split
        split_label = (
            f"{train_split}+{test_split}" if train_split else test_split
        )
        self.dataset_spec = f"mbpp:{dataset_name}:{split_label}"
        self._train_records: list[TaskRecord] | None = None
        self._test_records: list[TaskRecord] | None = None

    def _load_split(self, split: str) -> list[TaskRecord]:
        """Materialise MBPP+ rows from a single Hugging Face split."""

        rows = load_dataset(self.dataset_name, split=split)
        return [self._row_to_record(row, split) for row in rows]

    @staticmethod
    def _row_to_record(row: Mapping[str, object], split: str) -> TaskRecord:
        description = str(row["prompt"]).strip()
        code = str(row["code"])
        test_list = [str(assertion) for assertion in row.get("test_list", [])]
        test_imports = [str(line) for line in row.get("test_imports", [])]
        entry_point = _mbpp_entry_point(code, test_list)
        prompt = _format_mbpp_prompt(description, test_list)
        check_source = _build_mbpp_check(test_imports, test_list)
        return TaskRecord(
            task_id=f"mbpp/{split}/{row['task_id']}",
            prompt=prompt,
            target_text=code,
            metadata={
                "entry_point": entry_point,
                "test": check_source,
                "test_list": test_list,
                "description": description,
            },
        )


def _mbpp_entry_point(code: str, test_list: Sequence[str]) -> str:
    """Recover the function under test from an MBPP+ row.

    Prefer the top-level ``def`` whose name is invoked by the ``test_list``
    assertions (there may be helper definitions too); fall back to the last
    top-level definition, then to ``"solution"`` for degenerate rows.
    """

    defined = _DEF_PATTERN.findall(code)
    called: set[str] = set()
    for assertion in test_list:
        called.update(_CALL_PATTERN.findall(assertion))
    for name in reversed(defined):
        if name in called:
            return name
    return defined[-1] if defined else "solution"


def _build_mbpp_check(
    test_imports: Sequence[str],
    test_list: Sequence[str],
) -> str:
    """Wrap MBPP+ assertions in a HumanEval-style ``check`` function.

    The assertions reference the target function by its literal name, which
    the candidate solution defines at module scope, so the ``candidate``
    parameter is unused but present for parity with the shared verifier's
    ``check(entry_point)`` invocation.
    """

    body_lines = list(test_imports)
    body_lines.append("def check(candidate):")
    if test_list:
        for assertion in test_list:
            for line in assertion.splitlines() or [""]:
                body_lines.append(f"    {line}")
    else:
        body_lines.append("    pass")
    return "\n".join(body_lines) + "\n"


def _format_mbpp_prompt(description: str, test_list: Sequence[str]) -> str:
    """Build the teacher-forcing prompt for an MBPP+ row.

    The canonical ``code`` solution naturally continues this prefix: the
    description states the task and the sample assertions pin down the
    function name and call signature the solution must define.
    """

    lines = [
        "Write a Python function for the following task.",
        description,
    ]
    if test_list:
        lines.append("Your function must satisfy these tests:")
        lines.extend(test_list)
    lines.append("")
    return "\n".join(lines)
