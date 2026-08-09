"""Pluggable task adapters for pruning evaluation.

Each adapter exposes a uniform interface for loading records, splitting them
into train/test partitions deterministically, and verifying generated text
against a ground-truth target. The registry in
:mod:`pruning_metrics.evals.tasks.registry` lets downstream code select an
adapter by short name (``coding``, ``math``, ``mcq``, ``mbpp``,
``mathqa``).
"""

from pruning_metrics.evals.tasks.base import (
    TaskAdapter,
    TaskRecord,
    VerificationOutcome,
)
from pruning_metrics.evals.tasks.registry import (
    TASK_REGISTRY,
    build_adapter,
    build_adapter_from_spec,
)

__all__ = [
    "TaskAdapter",
    "TaskRecord",
    "VerificationOutcome",
    "TASK_REGISTRY",
    "build_adapter",
    "build_adapter_from_spec",
]
