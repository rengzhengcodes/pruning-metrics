"""Multiple-choice task adapter (ARC-Challenge by default).

Targets the ``allenai/ai2_arc`` family on the Hugging Face Hub. ARC-Challenge
ships ``train`` and ``test`` splits; like GSM8K, there is no separate
``validation`` split on the default config, and this adapter never assumes
one—only the configured train and test split names are loaded.

Free-form verification regex-extracts the first ``A|B|C|D|E`` token from the
model's generation and compares it to the gold ``answerKey``.

For teacher forcing, ``target_text`` is the body of the correct choice (not
its letter). This makes the TF score a meaningful signal of how confidently
the model emits the *content* of the right answer rather than whether it
emits the right letter.

:class:`MathQaTaskAdapter` reuses the same machinery for MathQA
(``allenai/math_qa``), which is a math word problem posed as a five-way
multiple-choice question. Its schema differs (the choices arrive as a single
string like ``"a ) 38 , b ) 27.675 , ..."`` and the gold field is a bare
letter), so the adapter only overrides record loading; verification, the
seeded split, and the letter-matching prompt are inherited.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from datasets import load_dataset

from pruning_metrics.evals.tasks.base import (
    TaskAdapter,
    TaskRecord,
    VerificationOutcome,
    native_or_seeded_split,
)

_LETTER_PATTERN = re.compile(r"\b([A-E])\b")


class MCQTaskAdapter(TaskAdapter):
    """Multiple-choice (ARC-Challenge-shaped) tasks.

    Parameters
    ----------
    dataset_name:
        Hugging Face dataset name. Default ``allenai/ai2_arc``.
    config:
        Dataset config (``ARC-Challenge`` or ``ARC-Easy``). Default
        ``ARC-Challenge``.
    train_split:
        Hugging Face split name for calibration rows. Default ``"train"``
        (ARC-Challenge ships 1119 rows there). Pass ``None`` to force the
        seeded 80/20 fallback over ``test_split`` when the dataset exposes
        only one split (no named train partition on the Hub).
    test_split:
        Native test split name. Default ``"test"`` (ARC-Challenge ships
        1172 rows there).
    """

    name = "mcq"

    def __init__(
        self,
        dataset_name: str = "allenai/ai2_arc",
        config: str = "ARC-Challenge",
        train_split: str | None = "train",
        test_split: str = "test",
    ) -> None:
        self.dataset_name = dataset_name
        self.config = config
        self.train_split = train_split
        self.test_split = test_split
        split_label = (
            f"{train_split}+{test_split}" if train_split else test_split
        )
        self.dataset_spec = f"mcq:{dataset_name}:{config}:{split_label}"
        self._train_records: list[TaskRecord] | None = None
        self._test_records: list[TaskRecord] | None = None

    def _load_split(self, split: str) -> list[TaskRecord]:
        """Materialise ARC-style rows from a single Hugging Face split."""

        rows = load_dataset(self.dataset_name, self.config, split=split)
        records: list[TaskRecord] = []
        for row in rows:
            row_id = str(row["id"])
            question = str(row["question"])
            choices = row["choices"]
            choice_texts = [str(text) for text in choices["text"]]
            choice_labels = [str(label).strip() for label in choices["label"]]
            answer_key = str(row["answerKey"]).strip()
            if answer_key not in choice_labels:
                # Skip malformed rows; ARC occasionally has a "1"/"A" mismatch
                # in older splits.
                continue
            answer_index = choice_labels.index(answer_key)
            target_text = choice_texts[answer_index]
            prompt = _format_mcq_prompt(question, choice_labels, choice_texts)
            records.append(
                TaskRecord(
                    task_id=f"arc/{self.config}/{row_id}",
                    prompt=prompt,
                    target_text=target_text,
                    metadata={
                        "answer_key": answer_key,
                        "choice_labels": choice_labels,
                        "choice_texts": choice_texts,
                        "question": question,
                    },
                )
            )
        return records

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
        # Trigger lazy load so both splits are populated.
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
        """MCQ prompts already include the choices + instruction; passthrough."""

        return record.prompt

    def verify(
        self,
        record: TaskRecord,
        generated_text: str,
        timeout_seconds: float = 10.0,
    ) -> VerificationOutcome:
        """Regex-extract the first A-E letter and compare to the gold key."""

        del timeout_seconds  # noqa: F841 -- protocol parity
        match = _LETTER_PATTERN.search(generated_text.upper())
        if match is None:
            return VerificationOutcome(
                task_id=record.task_id,
                status="parse_error",
                detail="no choice letter found",
            )
        predicted = match.group(1)
        gold = str(record.metadata["answer_key"]).upper()
        if predicted == gold:
            return VerificationOutcome(
                task_id=record.task_id,
                status="pass",
                detail=f"predicted={predicted}",
            )
        return VerificationOutcome(
            task_id=record.task_id,
            status="fail",
            detail=f"predicted={predicted} gold={gold}",
        )


# A marker only counts at the start of the string or after an option
# separator (comma, quote, bracket) -- a bare "b )" inside an option body
# such as "( x + b )" must not start a new option.
_MATHQA_MARKER = re.compile(r"(?:^|[\[,'\"])\s*([a-eA-E])\s*\)")
_MATHQA_STRIP = re.compile(r"^[\s\[\]',\"]+|[\s\[\]',\"]+$")
# Some MathQA rows double the letter marker, e.g. "a ) a ) 56 , b ) b ) 35".
# Collapse an immediately-repeated same-letter marker so the option body
# ("56") is recovered instead of parsing as empty.
_MATHQA_DOUBLED = re.compile(r"([a-eA-E])\s*\)\s*\1\s*\)")


class MathQaTaskAdapter(MCQTaskAdapter):
    """MathQA (``allenai/math_qa``) as a five-way multiple-choice task.

    MathQA ships ``train`` / ``validation`` / ``test`` splits. The original
    Hub dataset is a loading *script*, unsupported by ``datasets>=3``; the
    auto-converted parquet export (``revision="refs/convert/parquet"``) holds
    the same records and is what this adapter reads by default.

    Each row exposes ``Problem`` (question text), ``options`` (a single string
    ``"a ) 38 , b ) 27.675 , ..."`` -- occasionally bracket/quote wrapped as
    ``"['a ) 8', 'b ) 16', ...]"``), and ``correct`` (a lowercase letter). The
    options string is parsed into ``(LETTER, text)`` pairs so the record looks
    exactly like an ARC row: uppercase letter labels, the correct choice's
    body as ``target_text``, and the gold letter in ``metadata["answer_key"]``.
    Verification, prompting, and the seeded split are inherited from
    :class:`MCQTaskAdapter`.

    Parameters
    ----------
    dataset_name:
        Hugging Face dataset name. Default ``allenai/math_qa``.
    train_split:
        Native calibration split name. Default ``"train"`` (29837 rows).
    test_split:
        Native evaluation split name. Default ``"test"`` (2985 rows).
    revision:
        Hub revision to load. Default ``"refs/convert/parquet"`` (the
        script-free parquet mirror).
    """

    name = "mcq"

    def __init__(
        self,
        dataset_name: str = "allenai/math_qa",
        train_split: str | None = "train",
        test_split: str = "test",
        revision: str = "refs/convert/parquet",
    ) -> None:
        self.dataset_name = dataset_name
        self.config = None
        self.train_split = train_split
        self.test_split = test_split
        self.revision = revision
        split_label = (
            f"{train_split}+{test_split}" if train_split else test_split
        )
        self.dataset_spec = f"mathqa:{dataset_name}:{split_label}"
        self._train_records: list[TaskRecord] | None = None
        self._test_records: list[TaskRecord] | None = None

    def _load_split(self, split: str) -> list[TaskRecord]:
        """Materialise MathQA rows from a single parquet split."""

        rows = load_dataset(self.dataset_name, split=split, revision=self.revision)
        records: list[TaskRecord] = []
        for index, row in enumerate(rows):
            record = self._row_to_record(row, split, index)
            if record is not None:
                records.append(record)
        return records

    def _row_to_record(
        self,
        row: Mapping[str, object],
        split: str,
        index: int,
    ) -> TaskRecord | None:
        question = str(row["Problem"]).strip()
        pairs = _parse_mathqa_options(row["options"])
        labels = [label for label, _ in pairs]
        texts = [text for _, text in pairs]
        answer_key = str(row["correct"]).strip().upper()
        if answer_key not in labels or len(pairs) < 2:
            # Defensive: skip rows whose gold letter has no parsed option.
            return None
        target_text = texts[labels.index(answer_key)]
        if not target_text:
            # The correct option must have non-empty body for teacher forcing.
            return None
        prompt = _format_mcq_prompt(question, labels, texts)
        return TaskRecord(
            task_id=f"mathqa/{split}/{index:05d}",
            prompt=prompt,
            target_text=target_text,
            metadata={
                "answer_key": answer_key,
                "choice_labels": labels,
                "choice_texts": texts,
                "question": question,
            },
        )


def _parse_mathqa_options(raw: object) -> list[tuple[str, str]]:
    """Parse a MathQA ``options`` value into ``(LETTER, text)`` pairs.

    Handles both the plain ``"a ) 38 , b ) 27.675 , ..."`` form and the
    bracket/quote-wrapped ``"['a ) 8', 'b ) 16', ...]"`` form by locating the
    ``LETTER )`` markers and slicing the text between consecutive markers,
    stripping surrounding whitespace, brackets, quotes, and separators. The
    option body may itself contain commas (``"rs . 400"``), so a naive split
    on commas is avoided.
    """

    text = raw if isinstance(raw, str) else " , ".join(str(item) for item in raw)
    previous = None
    while previous != text:
        previous = text
        text = _MATHQA_DOUBLED.sub(r"\1 )", text)
    markers = list(_MATHQA_MARKER.finditer(text))
    pairs: list[tuple[str, str]] = []
    for position, marker in enumerate(markers):
        start = marker.end()
        end = (
            markers[position + 1].start()
            if position + 1 < len(markers)
            else len(text)
        )
        body = _MATHQA_STRIP.sub("", text[start:end])
        pairs.append((marker.group(1).upper(), body))
    return pairs


def _format_mcq_prompt(
    question: str,
    labels: Sequence[str],
    texts: Sequence[str],
) -> str:
    """Render an ARC-style question with choices ``A) ... B) ...`` and an
    explicit instruction to emit a single letter answer."""

    lines = [f"Question: {question}", "Choices:"]
    for label, text in zip(labels, texts):
        lines.append(f"  {label}) {text}")
    lines.append(
        "Respond with only the letter of the correct choice "
        "(for example ``A``)."
    )
    lines.append("Answer:")
    return "\n".join(lines)
