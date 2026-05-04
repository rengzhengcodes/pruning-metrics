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
"""

from __future__ import annotations

import re
from typing import Sequence

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
