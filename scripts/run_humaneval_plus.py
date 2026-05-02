"""CLI entrypoint for running HumanEval+ pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pruning_metrics.evals.coding.humaneval_plus_dataset import (
    HumanEvalPlusDatasetLoader,
)
from pruning_metrics.evals.coding.llm_client import (
    BedrockClient,
    LLMClient,
    MockLLMClient,
    SageMakerClient,
)
from pruning_metrics.evals.coding.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for HumanEval+ runner.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed arguments.

    Preconditions
    -------------
    None

    Postconditions
    --------------
    Returned namespace includes provider and output configuration.
    """

    parser = argparse.ArgumentParser(description="Run HumanEval+ coding evaluation.")
    parser.add_argument(
        "--provider",
        choices=("mock", "bedrock", "sagemaker"),
        default="mock",
        help="LLM backend provider.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Maximum number of HumanEval+ tasks to evaluate.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        default=None,
        help="Specific task ID to evaluate. Repeatable.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5.0,
        help="Per-task timeout for verification subprocess.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/humaneval_plus",
        help="Directory for JSONL records and summary.",
    )
    parser.add_argument(
        "--mock-completions-file",
        default=None,
        help="Path to JSON mapping of task_id -> Python completion for mock mode.",
    )
    parser.add_argument(
        "--bedrock-model-id",
        default="",
        help="Bedrock model ID (required when provider=bedrock).",
    )
    parser.add_argument(
        "--sagemaker-endpoint-name",
        default="",
        help="SageMaker endpoint name (required when provider=sagemaker).",
    )
    return parser.parse_args()


def build_client(args: argparse.Namespace) -> LLMClient:
    """Build configured model client for selected provider.

    Parameters
    ----------
    args:
        Parsed CLI arguments.

    Returns
    -------
    LLMClient
        Configured model client.

    Preconditions
    -------------
    Provider-specific args are valid for selected provider.

    Postconditions
    --------------
    Raises ``ValueError`` if required provider configuration is missing.
    """

    if args.provider == "mock":
        completions_by_task = load_mock_completions(args.mock_completions_file)
        return MockLLMClient(completions_by_task_id=completions_by_task)

    if args.provider == "bedrock":
        if not args.bedrock_model_id:
            raise ValueError("--bedrock-model-id is required with provider=bedrock")
        return BedrockClient(model_id=args.bedrock_model_id)

    if args.provider == "sagemaker":
        if not args.sagemaker_endpoint_name:
            raise ValueError(
                "--sagemaker-endpoint-name is required with provider=sagemaker"
            )
        return SageMakerClient(endpoint_name=args.sagemaker_endpoint_name)

    raise ValueError(f"Unsupported provider: {args.provider}")


def load_mock_completions(mock_file: str | None) -> dict[str, str]:
    """Load mock completion mapping from JSON file.

    Parameters
    ----------
    mock_file:
        Optional path to JSON object mapping task IDs to completions.

    Returns
    -------
    dict[str, str]
        Completion mapping.

    Preconditions
    -------------
    JSON file content is an object if provided.

    Postconditions
    --------------
    Returns empty mapping when file path is not provided.
    """

    if mock_file is None:
        return {}

    with Path(mock_file).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Mock completions file must contain a JSON object.")
    return {str(key): str(value) for key, value in payload.items()}


def main() -> None:
    """Run HumanEval+ inference and verification pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Preconditions
    -------------
    Runtime environment has required dependencies installed.

    Postconditions
    --------------
    Writes per-task artifacts and prints summary JSON to stdout.
    """

    args = parse_args()
    loader = HumanEvalPlusDatasetLoader()
    tasks = loader.load_tasks(max_samples=args.max_samples, task_ids=args.task_ids)
    llm_client = build_client(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"

    result = run_pipeline(
        tasks=tasks,
        llm_client=llm_client,
        timeout_seconds=args.timeout_seconds,
        output_jsonl_path=str(output_jsonl),
    )
    summary = {
        "num_tasks": result.num_tasks,
        "num_passed": result.num_passed,
        "pass_at_1": result.pass_at_1,
        "status_breakdown": result.status_breakdown,
        "records_path": str(output_jsonl),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
