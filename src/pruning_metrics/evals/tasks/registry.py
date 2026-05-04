"""Task adapter registry and spec parsing.

Notebooks 2-4 select a task adapter by short name (``coding``, ``math``,
``mcq``) or by a more detailed spec string of the form
``<name>[:<dataset_name>[:<config>[:<split>]]]``. This module provides both
shapes plus light validation so the four notebooks have a single import.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from pruning_metrics.evals.tasks.base import TaskAdapter
from pruning_metrics.evals.tasks.coding import CodingTaskAdapter
from pruning_metrics.evals.tasks.math import MathTaskAdapter
from pruning_metrics.evals.tasks.mcq import MCQTaskAdapter


TASK_REGISTRY: Mapping[str, Callable[..., TaskAdapter]] = {
    "coding": CodingTaskAdapter,
    "math": MathTaskAdapter,
    "mcq": MCQTaskAdapter,
}


def build_adapter(name: str, **kwargs: Any) -> TaskAdapter:
    """Instantiate a task adapter by short name.

    Parameters
    ----------
    name:
        One of the keys of :data:`TASK_REGISTRY`.
    **kwargs:
        Forwarded to the adapter's constructor (``dataset_name``, ``split``,
        ``config``, etc.).

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """

    if name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task adapter {name!r}. Known: {sorted(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[name](**kwargs)


def build_adapter_from_spec(spec: str) -> TaskAdapter:
    """Parse a ``<name>[:<dataset_name>[:<config>[:<split>]]]`` spec.

    Examples
    --------
    >>> build_adapter_from_spec("coding")
    >>> build_adapter_from_spec("coding:evalplus/humanevalplus:test")
    >>> build_adapter_from_spec("math:gsm8k:main:test")
    >>> build_adapter_from_spec("mcq:allenai/ai2_arc:ARC-Challenge:test")

    The spec is intentionally permissive about the number of segments so the
    notebooks accept either a bare ``"coding"`` shorthand or a fully-qualified
    descriptor.
    """

    if not spec or ":" not in spec and spec not in TASK_REGISTRY:
        raise ValueError(
            f"Spec must be one of {sorted(TASK_REGISTRY)} or "
            f"'<name>:<dataset>[:<config>[:<split>]]'; got {spec!r}"
        )

    parts = spec.split(":")
    name = parts[0]
    if name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task adapter {name!r}. Known: {sorted(TASK_REGISTRY)}"
        )

    if name == "coding":
        kwargs: dict[str, Any] = {}
        if len(parts) >= 2 and parts[1]:
            kwargs["dataset_name"] = parts[1]
        if len(parts) >= 3 and parts[2]:
            kwargs["split"] = parts[2]
        return build_adapter(name, **kwargs)

    if name in ("math", "mcq"):
        kwargs = {}
        if len(parts) >= 2 and parts[1]:
            kwargs["dataset_name"] = parts[1]
        if len(parts) >= 3 and parts[2]:
            kwargs["config"] = parts[2]
        if len(parts) >= 4 and parts[3]:
            kwargs["split"] = parts[3]
        return build_adapter(name, **kwargs)

    raise KeyError(name)  # pragma: no cover -- guarded above
