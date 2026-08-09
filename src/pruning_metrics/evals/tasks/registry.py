"""Task adapter registry and spec parsing.

Notebooks and runners select a task adapter by short name (``coding``,
``math``, ``mcq``, plus the v2 additions ``mbpp`` and ``mathqa``) or by a
more detailed spec string. The spec grammar varies by adapter so that
math / MCQ users can override either or both native splits:

* ``coding[:<dataset_name>[:<test_split>]]`` -- HumanEval+ ships only a
  ``test`` split; the seeded 80/20 fallback produces calibration data.
* ``math[:<dataset_name>[:<config>[:<train_split>[:<test_split>]]]]`` --
  default ``train_split=train, test_split=test`` (GSM8K ``main`` has both
  Hub splits; there is no ``validation`` split on that config, and nothing
  here assumes one). Pass ``""`` for ``train_split`` to force the seeded
  fallback over the test split.
* ``mcq[:<dataset_name>[:<config>[:<train_split>[:<test_split>]]]]`` --
  default ``train_split=train, test_split=test`` (ARC-Challenge native
  splits; likewise no ``validation`` split required).
* ``mbpp[:<dataset_name>[:<test_split>]]`` and
  ``mathqa[:<dataset_name>[:<train_split>[:<test_split>]]]`` -- the v2
  sweep additions; see :func:`build_adapter_from_spec` for details.

This module provides both shapes plus light validation so the notebooks
and runners have a single import.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from pruning_metrics.evals.tasks.base import TaskAdapter
from pruning_metrics.evals.tasks.coding import CodingTaskAdapter, MbppTaskAdapter
from pruning_metrics.evals.tasks.math import MathTaskAdapter
from pruning_metrics.evals.tasks.mcq import MathQaTaskAdapter, MCQTaskAdapter

TASK_REGISTRY: Mapping[str, Callable[..., TaskAdapter]] = {
    "coding": CodingTaskAdapter,
    "math": MathTaskAdapter,
    "mcq": MCQTaskAdapter,
}

# v2 domain adapters (MBPP, MathQA). Kept separate from :data:`TASK_REGISTRY`
# so the three-way core registry stays stable, while :func:`build_adapter` and
# :func:`build_adapter_from_spec` still resolve these spec names. MBPP is
# coding-shaped (subprocess pass@1); MathQA is MCQ-shaped (letter matching).
_V2_ADAPTERS: Mapping[str, Callable[..., TaskAdapter]] = {
    "mbpp": MbppTaskAdapter,
    "mathqa": MathQaTaskAdapter,
}

# Every spec name resolvable by name, core + v2.
_ALL_ADAPTERS: Mapping[str, Callable[..., TaskAdapter]] = {
    **TASK_REGISTRY,
    **_V2_ADAPTERS,
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

    if name not in _ALL_ADAPTERS:
        raise KeyError(f"Unknown task adapter {name!r}. Known: {sorted(_ALL_ADAPTERS)}")
    return _ALL_ADAPTERS[name](**kwargs)


def build_adapter_from_spec(spec: str) -> TaskAdapter:
    """Parse a task spec string into an adapter instance.

    Spec grammar (each segment is optional, left-to-right):

    * ``coding[:<dataset_name>[:<test_split>]]``
    * ``mbpp[:<dataset_name>[:<test_split>]]``
    * ``math[:<dataset_name>[:<config>[:<train_split>[:<test_split>]]]]``
    * ``mcq[:<dataset_name>[:<config>[:<train_split>[:<test_split>]]]]``
    * ``mathqa[:<dataset_name>[:<train_split>[:<test_split>]]]``

    For math/MCQ, an empty ``train_split`` segment (e.g. ``math:gsm8k:main::test``)
    forces the seeded 80/20 fallback over the test split. MBPP has no native
    train split (like HumanEval+), so ``mbpp`` mirrors the coding grammar and
    always uses the seeded fallback. MathQA has no dataset config, so its
    grammar drops the ``config`` segment.

    Examples
    --------
    >>> build_adapter_from_spec("coding")
    >>> build_adapter_from_spec("coding:evalplus/humanevalplus:test")
    >>> build_adapter_from_spec("mbpp")
    >>> build_adapter_from_spec("math:gsm8k:main")
    >>> build_adapter_from_spec("math:gsm8k:main:train:test")
    >>> build_adapter_from_spec("mcq:allenai/ai2_arc:ARC-Challenge")
    >>> build_adapter_from_spec("mathqa")

    The spec is intentionally permissive about the number of segments so the
    notebooks accept either a bare ``"coding"`` shorthand or a fully-qualified
    descriptor.
    """

    if not spec or ":" not in spec and spec not in _ALL_ADAPTERS:
        raise ValueError(
            f"Spec must be one of {sorted(_ALL_ADAPTERS)} or "
            f"'<name>:<dataset>[:<config>[:<train_split>[:<test_split>]]]'; "
            f"got {spec!r}"
        )

    parts = spec.split(":")
    name = parts[0]
    if name not in _ALL_ADAPTERS:
        raise KeyError(f"Unknown task adapter {name!r}. Known: {sorted(_ALL_ADAPTERS)}")

    if name in ("coding", "mbpp"):
        kwargs: dict[str, Any] = {}
        if len(parts) >= 2 and parts[1]:
            kwargs["dataset_name"] = parts[1]
        if len(parts) >= 3 and parts[2]:
            kwargs["test_split"] = parts[2]
        return build_adapter(name, **kwargs)

    if name == "mathqa":
        kwargs = {}
        if len(parts) >= 2 and parts[1]:
            kwargs["dataset_name"] = parts[1]
        if len(parts) >= 3:
            # Empty 3rd segment (e.g. "mathqa:allenai/math_qa::test") disables
            # the native train split and triggers the seeded fallback.
            kwargs["train_split"] = parts[2] if parts[2] else None
        if len(parts) >= 4 and parts[3]:
            kwargs["test_split"] = parts[3]
        return build_adapter(name, **kwargs)

    if name in ("math", "mcq"):
        kwargs = {}
        if len(parts) >= 2 and parts[1]:
            kwargs["dataset_name"] = parts[1]
        if len(parts) >= 3 and parts[2]:
            kwargs["config"] = parts[2]
        if len(parts) >= 4:
            # An empty 4th segment (e.g. "math:gsm8k:main::test") explicitly
            # disables the native train split and triggers the seeded fallback.
            kwargs["train_split"] = parts[3] if parts[3] else None
        if len(parts) >= 5 and parts[4]:
            kwargs["test_split"] = parts[4]
        return build_adapter(name, **kwargs)

    raise KeyError(name)  # pragma: no cover -- guarded above
