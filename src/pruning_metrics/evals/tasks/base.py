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


class HFSplitAdapter(TaskAdapter):
    """Shared construction + split-loading machinery for HF-backed adapters.

    :class:`CodingTaskAdapter`, :class:`MbppTaskAdapter`, :class:`MathTaskAdapter`,
    :class:`MCQTaskAdapter`, and :class:`MathQaTaskAdapter` (see
    :mod:`pruning_metrics.evals.tasks.coding`, :mod:`pruning_metrics.evals.tasks.math`,
    and :mod:`pruning_metrics.evals.tasks.mcq`) all load rows from a Hugging
    Face dataset that exposes an optional native ``train`` split plus a
    required ``test`` (or only) split, cache the materialised
    :class:`TaskRecord` lists per split, and hand the pair to
    :func:`native_or_seeded_split` to produce the calibration/evaluation
    partition. Before this class existed, that bookkeeping -- constructor
    state, lazy per-split caching in ``load_records``, and the
    ``train_test_split`` delegation -- was duplicated nearly byte-for-byte
    in every adapter's ``__init__``/``load_records``/``train_test_split``.
    This class factors out exactly that shared behaviour so each subclass
    only supplies what genuinely differs between datasets: the per-row
    parsing logic in :meth:`_load_split`, plus (for coding/MCQ tasks)
    ``build_inference_prompt`` and ``verify``.

    Parameters
    ----------
    dataset_name:
        ``datasets.load_dataset`` name (or dataset id resolved by the
        subclass's own loader).
    spec_prefix:
        Short adapter-family label prepended to :attr:`dataset_spec` (for
        example ``"coding"``, ``"mbpp"``, ``"math"``, ``"mcq"``,
        ``"mathqa"``). Kept as an explicit caller-supplied argument rather
        than reusing ``name`` because two adapter families can share the
        same ``name`` (``CodingTaskAdapter`` and ``MbppTaskAdapter`` are
        both ``name = "coding"``) while still needing distinct spec
        prefixes for the registry's spec-string grammar.
    train_split:
        Native Hugging Face split name used for calibration rows, or
        ``None`` when the dataset ships only one usable split. In that
        case :meth:`train_test_split` falls back to the seeded 80/20
        partition of ``test_split`` (see :func:`native_or_seeded_split`).
    test_split:
        Native split name loaded for evaluation. Always required.
    config:
        Optional Hugging Face dataset config name (``"main"`` for GSM8K,
        ``"ARC-Challenge"`` for ARC). Adapters with no config concept
        (HumanEval+, MBPP+, MathQA) pass ``None`` (the default), which
        also drops the config segment from :attr:`dataset_spec` so the
        formatted spec string matches the pre-refactor format exactly.

    Notes
    -----
    Subclasses MUST override :meth:`_load_split`; the implementation here
    raises :class:`NotImplementedError` so a forgotten override fails
    loudly instead of silently returning no records.
    """

    def __init__(
        self,
        *,
        dataset_name: str,
        spec_prefix: str,
        train_split: str | None,
        test_split: str,
        config: str | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.config = config
        self.train_split = train_split
        self.test_split = test_split
        split_label = f"{train_split}+{test_split}" if train_split else test_split
        # Design: the pre-refactor adapters formatted `dataset_spec` two
        # different ways depending on whether they had a `config` concept
        # (math/MCQ include it in the spec, coding/MBPP/MathQA don't).
        # Branching on `config is None` reproduces both formats exactly
        # instead of forcing every caller through one shape.
        if config is not None:
            self.dataset_spec = f"{spec_prefix}:{dataset_name}:{config}:{split_label}"
        else:
            self.dataset_spec = f"{spec_prefix}:{dataset_name}:{split_label}"
        self._train_records: list[TaskRecord] | None = None
        self._test_records: list[TaskRecord] | None = None

    def _load_split(self, split: str) -> list[TaskRecord]:
        """Materialise records from a single named Hugging Face split.

        This is the one piece of split-loading logic that is genuinely
        adapter-specific (row schema, entry-point recovery, prompt
        formatting, ...), so unlike ``load_records``/``train_test_split``
        it is intentionally NOT shared here -- every concrete subclass
        must override it.

        Parameters
        ----------
        split:
            Hugging Face split name to load (for example ``"train"`` or
            ``"test"``).

        Returns
        -------
        list[TaskRecord]
            Normalized records for that split.

        Raises
        ------
        NotImplementedError
            Always, unless a subclass overrides this method.
        """

        raise NotImplementedError(f"{type(self).__name__} must implement _load_split.")

    def load_records(self) -> list[TaskRecord]:
        """Concatenate train + test records (train first when available).

        Both splits are loaded at most once and cached on ``self`` (via
        :meth:`_load_split`), so repeated calls -- including the internal
        call made by :meth:`train_test_split` -- never re-hit the
        underlying dataset source.

        Returns
        -------
        list[TaskRecord]
            Just the test split when ``self.train_split is None``,
            otherwise the train split followed by the test split.
        """

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
        """Deterministic partition into ``(train_records, test_records)``.

        Parameters
        ----------
        seed, train_frac, explicit_train_ids, explicit_test_ids:
            Forwarded to :func:`native_or_seeded_split` (which in turn
            forwards to :func:`deterministic_split` for the seeded
            fallback path).

        Returns
        -------
        tuple[list[TaskRecord], list[TaskRecord]]
            ``(train_records, test_records)``.
        """

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
