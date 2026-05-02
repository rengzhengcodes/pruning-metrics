"""Coding benchmark integrations for model evaluation."""

from pruning_metrics.evals.coding.humaneval_plus_dataset import (
    HumanEvalPlusDatasetLoader,
    HumanEvalPlusTask,
)
from pruning_metrics.evals.coding.llm_client import (
    BedrockClient,
    LLMClient,
    MockLLMClient,
    SageMakerClient,
)
from pruning_metrics.evals.coding.pipeline import PipelineResult, run_pipeline
from pruning_metrics.evals.coding.verifier import (
    VerificationResult,
    verify_task_solution,
)

__all__ = [
    "BedrockClient",
    "HumanEvalPlusDatasetLoader",
    "HumanEvalPlusTask",
    "LLMClient",
    "MockLLMClient",
    "PipelineResult",
    "SageMakerClient",
    "VerificationResult",
    "run_pipeline",
    "verify_task_solution",
]
