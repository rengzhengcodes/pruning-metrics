"""HumanEval+ dataset loading and normalization utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from datasets import load_dataset


@dataclass(frozen=True)
class HumanEvalPlusTask:
    """Normalized HumanEval+ task record.

    Parameters
    ----------
    task_id:
        Unique task identifier (e.g. ``HumanEval/0``).
    prompt:
        Prompt text given to the model.
    entry_point:
        Expected function name that the model should implement.
    test:
        Python test source supplied by HumanEval+.
    canonical_solution:
        Optional reference implementation used for debugging only.

    Returns
    -------
    None

    Preconditions
    -------------
    All string fields are non-empty.

    Postconditions
    --------------
    Task metadata is immutable after initialization.
    """

    task_id: str
    prompt: str
    entry_point: str
    test: str
    canonical_solution: str | None = None


class HumanEvalPlusDatasetLoader:  # pylint: disable=too-few-public-methods
    """Load and filter HumanEval+ records from Hugging Face datasets.

    Parameters
    ----------
    dataset_name:
        Dataset identifier passed to ``datasets.load_dataset``.
    split:
        Dataset split name to load.

    Returns
    -------
    None

    Preconditions
    -------------
    ``dataset_name`` and ``split`` are valid inputs for ``load_dataset``.

    Postconditions
    --------------
    Loader is configured and can fetch tasks with ``load_tasks``.
    """

    def __init__(
        self,
        dataset_name: str = "evalplus/humanevalplus",
        split: str = "test",
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split

    def load_tasks(
        self,
        max_samples: int | None = None,
        task_ids: Sequence[str] | None = None,
    ) -> list[HumanEvalPlusTask]:
        """Load HumanEval+ tasks with optional filtering and truncation.

        Parameters
        ----------
        max_samples:
            Maximum number of tasks to return after filtering.
        task_ids:
            Optional list of task IDs to include.

        Returns
        -------
        list[HumanEvalPlusTask]
            Normalized task list.

        Preconditions
        -------------
        ``max_samples`` is ``None`` or a positive integer.

        Postconditions
        --------------
        Returned tasks satisfy filter criteria and preserve dataset ordering.
        """

        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive when provided.")

        raw_dataset = load_dataset(self.dataset_name, split=self.split)
        requested_ids = set(task_ids) if task_ids is not None else None

        tasks: list[HumanEvalPlusTask] = []
        for record in raw_dataset:
            task = self._normalize_task(record)
            if requested_ids is not None and task.task_id not in requested_ids:
                continue
            tasks.append(task)
            if max_samples is not None and len(tasks) >= max_samples:
                break

        self._validate_requested_ids(requested_ids, tasks)
        return tasks

    @staticmethod
    def _normalize_task(record: dict[str, str]) -> HumanEvalPlusTask:
        """Normalize one dataset record into a ``HumanEvalPlusTask``.

        Parameters
        ----------
        record:
            Raw dictionary from Hugging Face dataset.

        Returns
        -------
        HumanEvalPlusTask
            Typed task entry.

        Preconditions
        -------------
        Required keys exist in ``record``.

        Postconditions
        --------------
        Returned dataclass contains required task fields.
        """

        required_fields = ("task_id", "prompt", "entry_point", "test")
        missing_fields = [field for field in required_fields if field not in record]
        if missing_fields:
            raise KeyError(
                f"Missing required HumanEval+ fields: {', '.join(missing_fields)}"
            )

        return HumanEvalPlusTask(
            task_id=str(record["task_id"]),
            prompt=str(record["prompt"]),
            entry_point=str(record["entry_point"]),
            test=str(record["test"]),
            canonical_solution=(
                str(record["canonical_solution"])
                if "canonical_solution" in record
                else None
            ),
        )

    def split_train_test(
        self,
        seed: int = 65320,
        train_frac: float = 0.8,
        max_samples: int | None = None,
    ) -> tuple[list[HumanEvalPlusTask], list[HumanEvalPlusTask]]:
        """Deterministically partition HumanEval+ tasks into train/test lists.

        The HumanEval+ release ships only a single ``test`` split. For pruning we
        need a calibration ("train") subset and a held-out evaluation ("test")
        subset. This helper produces a reproducible 80/20 (or any
        ``train_frac``) split using ``random.Random(seed)`` to shuffle a stable
        ordering of all task IDs.

        Parameters
        ----------
        seed:
            Random seed controlling the shuffle. Default mirrors the project
            constant ``65320`` used elsewhere.
        train_frac:
            Fraction of tasks routed to the calibration ("train") subset.
            Must satisfy ``0 < train_frac < 1``.
        max_samples:
            Optional cap on the total number of tasks loaded before splitting.
            Useful for smoke-tests; ``None`` keeps the entire 164-task dataset.

        Returns
        -------
        tuple[list[HumanEvalPlusTask], list[HumanEvalPlusTask]]
            ``(train_tasks, test_tasks)`` lists, each ordered by the random
            shuffle so the partition is independent of dataset ordering.

        Preconditions
        -------------
        ``train_frac`` is in ``(0, 1)``.

        Postconditions
        --------------
        - The two lists are disjoint and their union covers every loaded task
          exactly once.
        - The split is identical across runs for the same ``seed``,
          ``train_frac``, and underlying dataset.
        """

        if not 0.0 < train_frac < 1.0:
            raise ValueError("train_frac must be strictly between 0 and 1.")

        all_tasks = self.load_tasks(max_samples=max_samples)
        # Sort by task_id so the input ordering is independent of HF row order.
        ordered = sorted(all_tasks, key=lambda task: task.task_id)
        rng = random.Random(seed)
        shuffled = ordered[:]
        rng.shuffle(shuffled)

        cut = max(1, min(len(shuffled) - 1, int(round(train_frac * len(shuffled)))))
        return shuffled[:cut], shuffled[cut:]

    @staticmethod
    def _validate_requested_ids(
        requested_ids: set[str] | None,
        tasks: Iterable[HumanEvalPlusTask],
    ) -> None:
        """Validate all explicitly requested task IDs were found.

        Parameters
        ----------
        requested_ids:
            Optional set of task IDs requested by caller.
        tasks:
            Resulting task collection after filtering.

        Returns
        -------
        None

        Preconditions
        -------------
        ``tasks`` is iterable over ``HumanEvalPlusTask`` objects.

        Postconditions
        --------------
        Raises ``ValueError`` if any requested IDs are not present.
        """

        if requested_ids is None:
            return

        found_ids = {task.task_id for task in tasks}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise ValueError(
                "Requested task IDs were not found in dataset split: "
                + ", ".join(missing_ids)
            )
