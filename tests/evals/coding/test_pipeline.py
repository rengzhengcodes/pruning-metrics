"""Integration-style smoke tests for HumanEval+ pipeline."""

from __future__ import annotations

import json

from pruning_metrics.evals.coding.humaneval_plus_dataset import HumanEvalPlusTask
from pruning_metrics.evals.coding.llm_client import MockLLMClient
from pruning_metrics.evals.coding.pipeline import run_pipeline


def test_run_pipeline_with_mock_client_writes_artifacts(tmp_path) -> None:
    """Run a small end-to-end flow using deterministic mock completions.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory fixture.

    Returns
    -------
    None

    Preconditions
    -------------
    Mock client returns one correct and one incorrect completion.

    Postconditions
    --------------
    Pipeline summary metrics and JSONL records are written consistently.
    """

    tasks = [
        HumanEvalPlusTask(
            task_id="HumanEval/0",
            prompt="def add_one(x):\n    pass",
            entry_point="add_one",
            test="def check(candidate):\n    assert candidate(1) == 2",
        ),
        HumanEvalPlusTask(
            task_id="HumanEval/1",
            prompt="def add_two(x):\n    pass",
            entry_point="add_two",
            test="def check(candidate):\n    assert candidate(1) == 3",
        ),
    ]
    mock_client = MockLLMClient(
        completions_by_task_id={
            "HumanEval/0": "def add_one(x):\n    return x + 1\n",
            "HumanEval/1": "def add_two(x):\n    return x\n",
        }
    )
    output_file = tmp_path / "records.jsonl"

    result = run_pipeline(
        tasks=tasks,
        llm_client=mock_client,
        timeout_seconds=1.0,
        output_jsonl_path=str(output_file),
    )

    assert result.num_tasks == 2
    assert result.num_passed == 1
    assert result.pass_at_1 == 0.5
    assert result.status_breakdown == {"pass": 1, "fail": 1}
    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first_payload = json.loads(lines[0])
    assert first_payload["task_id"] == "HumanEval/0"


def test_run_pipeline_persists_inference_metadata_when_available(tmp_path) -> None:
    """Ensure provider metadata is serialized when client exposes it."""

    task = HumanEvalPlusTask(
        task_id="HumanEval/meta",
        prompt="def identity(x):\n    pass",
        entry_point="identity",
        test="def check(candidate):\n    assert candidate(2) == 2",
    )

    class MetadataClient:  # pylint: disable=too-few-public-methods
        """Tiny stub client exposing metadata side channel."""

        def __init__(self) -> None:
            self.last_response_metadata: dict[str, object] | None = None

        def generate_code(self, prompt: str, task_id: str) -> str:
            """Return deterministic code and attach metadata."""

            del prompt
            self.last_response_metadata = {
                "task_id": task_id,
                "seed": 3,
                "pruning_level": 20,
                "logits_s3_uri": "s3://bucket/path/tokens.jsonl",
            }
            return "def identity(x):\n    return x\n"

    output_file = tmp_path / "records.jsonl"
    result = run_pipeline(
        tasks=[task],
        llm_client=MetadataClient(),
        timeout_seconds=1.0,
        output_jsonl_path=str(output_file),
    )

    assert result.num_passed == 1
    payload = json.loads(output_file.read_text(encoding="utf-8").strip())
    assert payload["inference_metadata"]["seed"] == 3
    assert payload["inference_metadata"]["pruning_level"] == 20
