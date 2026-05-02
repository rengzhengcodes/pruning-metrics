"""Calibration dataset adapters for pruning activation collection."""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset


@dataclass(frozen=True)
class CalibrationRecord:
    """One calibration text sample.

    Parameters
    ----------
    text:
        Sample text used for collecting activation statistics.
    source:
        Source identifier for traceability.
    """

    text: str
    source: str


def load_calibration_records(
    source: str,
    max_samples: int,
) -> list[CalibrationRecord]:
    """Load calibration records from a source selector.

    Parameters
    ----------
    source:
        Dataset selector. Supported formats:
        - ``humanevalplus_prompts``
        - ``hf:<dataset_name>:<split>:<text_field>``
    max_samples:
        Maximum samples to load.

    Returns
    -------
    list[CalibrationRecord]
        Loaded records.
    """

    if source == "humanevalplus_prompts":
        dataset = load_dataset("evalplus/humanevalplus", split="test")
        rows = dataset.select(range(min(max_samples, len(dataset))))
        return [
            CalibrationRecord(text=str(row["prompt"]), source=source) for row in rows  # type: ignore[index]
        ]

    if source.startswith("hf:"):
        parts = source.split(":")
        if len(parts) != 4:
            raise ValueError(
                "HF calibration selector must be: hf:<dataset_name>:<split>:<text_field>"
            )
        dataset_name, split, text_field = parts[1], parts[2], parts[3]
        dataset = load_dataset(dataset_name, split=split)
        rows = dataset.select(range(min(max_samples, len(dataset))))
        records: list[CalibrationRecord] = []
        for row in rows:
            if text_field not in row:
                raise ValueError(
                    f"Field '{text_field}' not found in dataset '{dataset_name}' split '{split}'."
                )
            records.append(CalibrationRecord(text=str(row[text_field]), source=source))
        return records

    raise ValueError(
        "Unsupported calibration source. Use 'humanevalplus_prompts' or "
        "'hf:<dataset_name>:<split>:<text_field>'."
    )
