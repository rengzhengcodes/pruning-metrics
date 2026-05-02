"""Tests for HumanEval+ CLI client construction rules."""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest

from pruning_metrics.evals.coding.llm_client import SageMakerClient

_RUNNER_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_humaneval_plus.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_humaneval_plus", _RUNNER_PATH
)
if _RUNNER_SPEC is None or _RUNNER_SPEC.loader is None:
    raise RuntimeError("Failed to resolve scripts/run_humaneval_plus.py")
_RUNNER_MODULE = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_RUNNER_MODULE)
build_client = _RUNNER_MODULE.build_client


def _base_args() -> Namespace:
    """Build minimal argparse namespace used by build_client tests."""

    return Namespace(
        provider="sagemaker",
        mock_completions_file=None,
        bedrock_model_id="",
        sagemaker_endpoint_name="endpoint-a",
        pruning_level=None,
        seed=None,
        sagemaker_region=None,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
    )


def test_build_client_requires_pruning_level_for_sagemaker() -> None:
    """Validate pruning level is mandatory in SageMaker mode."""

    args = _base_args()
    args.seed = 7
    with pytest.raises(ValueError, match="--pruning-level"):
        build_client(args)


def test_build_client_requires_seed_for_sagemaker() -> None:
    """Validate seed is mandatory in SageMaker mode."""

    args = _base_args()
    args.pruning_level = 40
    with pytest.raises(ValueError, match="--seed"):
        build_client(args)


def test_build_client_creates_sagemaker_client_with_parameters() -> None:
    """Ensure SageMaker client receives CLI-driven generation settings."""

    args = _base_args()
    args.pruning_level = 60
    args.seed = 42
    args.temperature = 0.3
    args.top_p = 0.85

    client = build_client(args)
    assert isinstance(client, SageMakerClient)
    assert client.pruning_level == 60
    assert client.seed == 42
    assert client.temperature == 0.3
    assert client.top_p == 0.85
