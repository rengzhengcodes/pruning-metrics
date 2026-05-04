"""Tests for the spec-string parser used by the four notebooks."""

from __future__ import annotations

import pytest

from pruning_metrics.evals.tasks import registry as registry_module
from pruning_metrics.evals.tasks.coding import CodingTaskAdapter
from pruning_metrics.evals.tasks.math import MathTaskAdapter
from pruning_metrics.evals.tasks.mcq import MCQTaskAdapter
from pruning_metrics.evals.tasks.registry import (
    TASK_REGISTRY,
    build_adapter,
    build_adapter_from_spec,
)


def test_registry_keys() -> None:
    assert set(TASK_REGISTRY) == {"coding", "math", "mcq"}


def test_build_adapter_short_name() -> None:
    assert isinstance(build_adapter("coding"), CodingTaskAdapter)
    assert isinstance(build_adapter("math"), MathTaskAdapter)
    assert isinstance(build_adapter("mcq"), MCQTaskAdapter)


def test_build_adapter_unknown_name() -> None:
    with pytest.raises(KeyError):
        build_adapter("unknown")


def test_build_adapter_from_spec_bare_name() -> None:
    adapter = build_adapter_from_spec("coding")
    assert isinstance(adapter, CodingTaskAdapter)
    assert adapter.dataset_spec == "coding:evalplus/humanevalplus:test"


def test_build_adapter_from_spec_full_coding() -> None:
    adapter = build_adapter_from_spec("coding:evalplus/humanevalplus:test")
    assert isinstance(adapter, CodingTaskAdapter)
    assert adapter.dataset_name == "evalplus/humanevalplus"
    assert adapter.split == "test"


def test_build_adapter_from_spec_full_math() -> None:
    adapter = build_adapter_from_spec("math:gsm8k:main:test")
    assert isinstance(adapter, MathTaskAdapter)
    assert adapter.dataset_name == "gsm8k"
    assert adapter.config == "main"
    assert adapter.split == "test"


def test_build_adapter_from_spec_full_mcq() -> None:
    adapter = build_adapter_from_spec("mcq:allenai/ai2_arc:ARC-Challenge:test")
    assert isinstance(adapter, MCQTaskAdapter)
    assert adapter.config == "ARC-Challenge"


def test_build_adapter_from_spec_invalid() -> None:
    with pytest.raises((KeyError, ValueError)):
        build_adapter_from_spec("not-a-real-task")
    with pytest.raises((KeyError, ValueError)):
        build_adapter_from_spec("")
