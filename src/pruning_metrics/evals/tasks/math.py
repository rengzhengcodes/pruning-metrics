"""Math-task adapter (GSM8K).

GSM8K (Cobbe et al. 2021, https://huggingface.co/datasets/gsm8k) on the Hub
``main`` config exposes ``train`` and ``test`` splits only (7473 + 1319
rows); there is no ``validation`` split, and this adapter does not require
one for any GSM8K-shaped dataset you point it at—calibration uses the
configured train split name and evaluation uses the test split name.

GSM8K rows use chain-of-thought solutions terminated by a literal ``####``
followed by the final numeric answer. Verification therefore extracts the
model's last plausible numeric token and compares it against the gold final
number.

Notation supported by ``_extract_numeric``:

* GSM8K-style ``#### 42`` (highest priority - matches the gold convention).
* LaTeX boxed answers ``\\boxed{42}`` or ``\\boxed{42.5}``.
* The final number anywhere in the generated text (fallback).

Numbers may be negative, contain commas as thousand separators, or be
fractions written as ``a/b``; the helper normalises all of those to a single
``float``.
"""

from __future__ import annotations

import re

from datasets import load_dataset

from pruning_metrics.evals.tasks.base import (
    HFSplitAdapter,
    TaskRecord,
    VerificationOutcome,
)

_GSM8K_GOLD_DIVIDER = "####"
_BOXED_PATTERN = re.compile(r"\\boxed\{\s*(-?[0-9.,/]+)\s*\}")
_GSM_DIVIDER_PATTERN = re.compile(r"####\s*(-?[0-9.,/]+)")
_NUMBER_PATTERN = re.compile(r"-?\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?(?:/[0-9]+)?")


class MathTaskAdapter(HFSplitAdapter):
    """GSM8K (and GSM8K-shaped) math word problems.

    Parameters
    ----------
    dataset_name:
        Hugging Face dataset name. Default ``gsm8k``.
    config:
        Dataset config (``main`` or ``socratic``). Default ``main``.
    train_split:
        Hugging Face split name for calibration rows. Default ``"train"``
        (GSM8K ``main`` has 7473 rows there). Pass ``None`` to force the
        seeded 80/20 fallback over ``test_split`` when the dataset exposes
        only one split (no named train partition on the Hub).
    test_split:
        Native test split name. Default ``"test"`` (GSM8K ships 1319 rows
        there).
    keep_chain_of_thought:
        When ``True`` (default), ``target_text`` is the full canonical
        solution including the chain-of-thought and ``#### N`` divider so
        teacher forcing measures the full reasoning trace. When ``False``
        only the final numeric answer is used.
    """

    name = "math"

    def __init__(
        self,
        dataset_name: str = "openai/gsm8k",
        config: str = "main",
        train_split: str | None = "train",
        test_split: str = "test",
        keep_chain_of_thought: bool = True,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            spec_prefix="math",
            config=config,
            train_split=train_split,
            test_split=test_split,
        )
        self.keep_chain_of_thought = keep_chain_of_thought

    def _load_split(self, split: str) -> list[TaskRecord]:
        """Materialise rows from a single Hugging Face split."""

        rows = load_dataset(self.dataset_name, self.config, split=split)
        records: list[TaskRecord] = []
        for index, row in enumerate(rows):
            question = str(row["question"])
            answer_full = str(row["answer"])
            gold_number = _extract_numeric(answer_full)
            if gold_number is None:
                # Skip malformed rows with no extractable final answer; rare
                # in GSM8K but defensive.
                continue
            target_text = (
                answer_full
                if self.keep_chain_of_thought
                else f"{_GSM8K_GOLD_DIVIDER} {answer_full.split(_GSM8K_GOLD_DIVIDER)[-1].strip()}"
            )
            records.append(
                TaskRecord(
                    task_id=f"gsm8k/{split}/{index:05d}",
                    prompt=_format_gsm8k_prompt(question),
                    target_text=target_text,
                    metadata={
                        "gold_number": gold_number,
                        "raw_answer": answer_full,
                        "question": question,
                    },
                )
            )
        return records

    def build_inference_prompt(self, record: TaskRecord) -> str:
        """Math prompts already include the GSM-style instruction; passthrough."""

        return record.prompt

    def verify(
        self,
        record: TaskRecord,
        generated_text: str,
        timeout_seconds: float = 10.0,
    ) -> VerificationOutcome:
        """Numeric exact-match against ``record.metadata["gold_number"]``.

        ``timeout_seconds`` is accepted for API uniformity but unused; numeric
        extraction runs in microseconds.
        """

        del timeout_seconds  # noqa: F841 -- protocol parity
        predicted = _extract_numeric(generated_text)
        if predicted is None:
            return VerificationOutcome(
                task_id=record.task_id,
                status="parse_error",
                detail="no number found in generation",
            )
        gold = float(record.metadata["gold_number"])
        if abs(predicted - gold) <= 1e-6:
            return VerificationOutcome(
                task_id=record.task_id,
                status="pass",
                detail=f"predicted={predicted} gold={gold}",
            )
        return VerificationOutcome(
            task_id=record.task_id,
            status="fail",
            detail=f"predicted={predicted} gold={gold}",
        )


def _format_gsm8k_prompt(question: str) -> str:
    """Wrap a raw question with a brief instruction.

    The instruction nudges the model toward the GSM8K convention so the
    numeric extractor finds a clear final answer.
    """

    return (
        "Solve the following math word problem. "
        "Show your reasoning step-by-step, then write the final answer on a "
        "new line as ``#### <number>`` (no units, no commas).\n\n"
        f"Problem: {question}\n\nSolution:"
    )


def _extract_numeric(text: str) -> float | None:
    """Best-effort numeric extraction with GSM8K conventions.

    Priority order:

    1. ``#### <number>`` (the GSM8K gold pattern).
    2. ``\\boxed{<number>}`` (LaTeX convention).
    3. The last number-shaped substring in ``text``.

    Returns ``None`` when nothing parseable is found.
    """

    for pattern in (_GSM_DIVIDER_PATTERN, _BOXED_PATTERN):
        match = None
        for match in pattern.finditer(text):
            pass  # iterate to capture the last match for stability
        if match is not None:
            value = _parse_number(match.group(1))
            if value is not None:
                return value

    last_value: float | None = None
    for match in _NUMBER_PATTERN.finditer(text):
        candidate = _parse_number(match.group(0))
        if candidate is not None:
            last_value = candidate
    return last_value


def _parse_number(token: str) -> float | None:
    """Parse comma- and dollar-decorated numerics, including ``a/b`` fractions."""

    cleaned = token.replace(",", "").replace("$", "").strip()
    if not cleaned:
        return None
    if "/" in cleaned:
        try:
            numerator_str, denominator_str = cleaned.split("/", 1)
            numerator = float(numerator_str)
            denominator = float(denominator_str)
            if denominator == 0:
                return None
            return numerator / denominator
        except ValueError:
            return None
    try:
        return float(cleaned)
    except ValueError:
        return None
