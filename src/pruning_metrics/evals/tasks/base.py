"""Base types shared by all pluggable task adapters.

A *task adapter* normalizes a Hugging Face dataset (or any other record source)
into a uniform ``(prompt, target_text, metadata)`` schema so the same
calibration / free-form evaluation / teacher-forcing runners work across
coding (HumanEval+), math (GSM8K), and multiple-choice (ARC-Challenge) tasks.

Adapters are intentionally tiny dataclasses with three responsibilities:

1. ``load_records`` -> list of normalized records keyed by ``task_id``.
2. ``train_test_split`` -> deterministic partition driven by a seed (default
   80/20), with optional explicit task-id overrides.
3. ``verify`` -> task-specific scoring (subprocess test execution for coding,
   numeric extraction for math, regex parsing for MCQ).

The interface is intentionally narrower than the existing
``HumanEvalPlusDatasetLoader`` / ``verify_task_solution`` pair so it can scale
to text-only tasks where there is no executable test harness.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class TaskRecord:
    """One normalized evaluation record.

    Parameters
    ----------
    task_id:
        Stable identifier used to anchor the deterministic train/test split
        and to name per-record output files (for example
        ``HumanEval/137`` or ``arc_challenge/Mercury_7220990``).
    prompt:
        Text presented to the model at inference time. For free-form
        evaluation this is appended with adapter-specific decoration
        (such as the HumanEval+ "Return only valid Python code..." preamble);
        for teacher forcing this is the unmodified prefix concatenated with
        ``target_text`` before the forward pass.
    target_text:
        Ground-truth continuation. For coding tasks this is the canonical
        Python solution; for math it is the numeric or boxed answer; for MCQ
        it is the body of the correct choice. Always non-empty.
    metadata:
        Optional adapter-specific payload (entry points, test sources,
        choice keys, etc.). Treat as opaque outside the adapter.
    """

    task_id: str
    prompt: str
    target_text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationOutcome:
    """Verification status for a single ``(record, generated_text)`` pair.

    Parameters
    ----------
    task_id:
        Echo of ``TaskRecord.task_id`` so downstream JSONL records are
        self-describing.
    status:
        One of ``"pass"``, ``"fail"``, ``"timeout"``, ``"runtime_error"``,
        ``"parse_error"``. Adapters are responsible for choosing the
        appropriate status; downstream metrics treat ``"pass"`` as success
        and everything else as a non-pass.
    detail:
        Free-form short explanation (e.g. extracted numeric answer, parsed
        choice letter, subprocess return code). Useful when manually
        auditing low-pass-rate runs.
    """

    task_id: str
    status: str
    detail: str = ""


class TaskAdapter(Protocol):
    """Protocol all task adapters must implement.

    Implementations live in
    :mod:`pruning_metrics.evals.tasks.coding`,
    :mod:`pruning_metrics.evals.tasks.math`, and
    :mod:`pruning_metrics.evals.tasks.mcq`. The matching class names are
    exposed via :data:`pruning_metrics.evals.tasks.registry.TASK_REGISTRY`.

    Two flavours of "prompt" matter to the runners:

    * ``record.prompt`` is the **teacher-forcing prompt** -- it must be the
      string that ``record.target_text`` naturally continues, because TF
      tokenises ``prompt + target_text`` and scores the suffix.
    * :meth:`build_inference_prompt` is the **free-form inference prompt**
      -- the runner feeds this through ``model.generate`` and the model's
      decoded output is what the verifier scores. The default implementation
      returns ``record.prompt`` unchanged; adapters override it when the
      free-form prompt needs additional instruction wrapping (coding tasks
      need an instruction telling the model to define a complete function,
      whereas math and MCQ already include their instruction in
      ``record.prompt``).
    """

    name: str
    """Short identifier (``"coding"``, ``"math"``, ``"mcq"``)."""

    dataset_spec: str
    """Descriptive ``<source>:<config?>:<split>`` spec, recorded in manifests."""

    def load_records(self) -> list[TaskRecord]:
        """Load all evaluation records (no truncation, no shuffling)."""
        ...

    def train_test_split(
        self,
        seed: int = 65320,
        train_frac: float = 0.8,
        explicit_train_ids: Sequence[str] | None = None,
        explicit_test_ids: Sequence[str] | None = None,
    ) -> tuple[list[TaskRecord], list[TaskRecord]]:
        """Deterministic partition into ``(train_records, test_records)``."""
        ...

    def build_inference_prompt(self, record: TaskRecord) -> str:
        """Return the prompt that the free-form runner should feed to ``model.generate``."""
        ...

    def verify(
        self,
        record: TaskRecord,
        generated_text: str,
        timeout_seconds: float = 10.0,
    ) -> VerificationOutcome:
        """Score a free-form generation against ``record``'s ground truth."""
        ...


def native_or_seeded_split(
    train_records: Sequence[TaskRecord] | None,
    test_records: Sequence[TaskRecord] | None,
    *,
    seed: int = 65320,
    train_frac: float = 0.8,
    explicit_train_ids: Sequence[str] | None = None,
    explicit_test_ids: Sequence[str] | None = None,
) -> tuple[list[TaskRecord], list[TaskRecord]]:
    """Route between native train/test splits and the seeded fallback.

    Adapters that load from datasets with native splits (GSM8K, ARC) call
    this helper with both ``train_records`` and ``test_records`` populated;
    HumanEval+ (single split) passes ``train_records=None`` and the records
    are seeded-shuffled into 80/20 partitions.

    Parameters
    ----------
    train_records:
        Records from the adapter's configured **train** Hub split (e.g.
        split name ``"train"``), or ``None`` when the dataset exposes only a
        single split for evaluation (no ``train`` key on the Hub). This is
        unrelated to a ``validation`` split: many benchmarks ship ``train``
        + ``test`` only, and we never require ``validation``. When
        ``train_records`` is ``None``, ``test_records`` is partitioned via
        :func:`deterministic_split`.
    test_records:
        Native (or only) split. Always required.
    seed, train_frac:
        Forwarded to :func:`deterministic_split` when falling back. Ignored
        (logged via the return value) when both native splits are present
        and no explicit overrides are supplied.
    explicit_train_ids, explicit_test_ids:
        Optional task-id overrides. Forces the seeded fallback over the
        union of both partitions so the override semantics stay uniform.

    Returns
    -------
    tuple[list[TaskRecord], list[TaskRecord]]
        ``(train_records, test_records)``. With native splits and no
        overrides, partitions are returned in dataset order.
    """

    if test_records is None:
        raise ValueError("test_records must be provided")

    has_overrides = bool(explicit_train_ids) or bool(explicit_test_ids)
    if train_records is not None and not has_overrides:
        # Native split path: dataset order, ignore seed/train_frac.
        return list(train_records), list(test_records)

    pool: list[TaskRecord] = list(test_records)
    if train_records is not None:
        pool = list(train_records) + pool

    return deterministic_split(
        pool,
        seed=seed,
        train_frac=train_frac,
        explicit_train_ids=explicit_train_ids,
        explicit_test_ids=explicit_test_ids,
    )


def deterministic_split(
    records: Sequence[TaskRecord],
    *,
    seed: int = 65320,
    train_frac: float = 0.8,
    explicit_train_ids: Sequence[str] | None = None,
    explicit_test_ids: Sequence[str] | None = None,
) -> tuple[list[TaskRecord], list[TaskRecord]]:
    """Reusable seeded 80/20-by-default partitioning helper.

    All adapters share the same partitioning policy:

    * If ``explicit_train_ids`` and/or ``explicit_test_ids`` are given, every
      record is routed by membership; missing ids fall through to the seeded
      shuffle (so partial overrides are allowed).
    * Records sorted by ``task_id`` then shuffled with ``random.Random(seed)``
      to remove any source-ordering bias.
    * The first ``round(train_frac * N)`` records (clamped to ``[1, N-1]``)
      land in the train partition.

    Parameters
    ----------
    records:
        All loaded records to partition.
    seed:
        Seed forwarded to ``random.Random``. Default ``65320`` matches the
        project-wide convention.
    train_frac:
        Fraction routed to the calibration partition. Must be strictly
        between ``0`` and ``1``.
    explicit_train_ids, explicit_test_ids:
        Optional task-id collections. When both are supplied, any record
        present in either is honoured; records appearing in neither still
        fall back to the seeded shuffle.

    Returns
    -------
    tuple[list[TaskRecord], list[TaskRecord]]
        ``(train_records, test_records)`` in shuffle order.

    Raises
    ------
    ValueError
        If ``train_frac`` is outside ``(0, 1)`` or if a record is listed in
        both explicit collections.
    """

    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be strictly between 0 and 1.")

    train_set = set(explicit_train_ids or ())
    test_set = set(explicit_test_ids or ())
    overlap = train_set & test_set
    if overlap:
        raise ValueError(
            "explicit_train_ids and explicit_test_ids overlap on: "
            + ", ".join(sorted(overlap))
        )

    explicit_train: list[TaskRecord] = []
    explicit_test: list[TaskRecord] = []
    fallback: list[TaskRecord] = []
    for record in records:
        if record.task_id in train_set:
            explicit_train.append(record)
        elif record.task_id in test_set:
            explicit_test.append(record)
        else:
            fallback.append(record)

    fallback_sorted = sorted(fallback, key=lambda record: record.task_id)
    rng = random.Random(seed)
    shuffled = list(fallback_sorted)
    rng.shuffle(shuffled)

    cut = max(1, min(len(shuffled) - 1, round(train_frac * len(shuffled))))
    fallback_train = shuffled[:cut]
    fallback_test = shuffled[cut:]

    return (
        explicit_train + fallback_train,
        explicit_test + fallback_test,
    )


def records_to_id_map(records: Iterable[TaskRecord]) -> dict[str, TaskRecord]:
    """Build a ``task_id -> record`` lookup, asserting uniqueness of ids."""

    out: dict[str, TaskRecord] = {}
    for record in records:
        if record.task_id in out:
            raise ValueError(f"Duplicate task_id detected: {record.task_id!r}")
        out[record.task_id] = record
    return out
