"""Helpers shared by the orchestration notebooks.

Six notebooks import this module: ``notebooks/aws_tutorial/02_prune_llm.ipynb``
through ``04_teacher_forced.ipynb``, plus ``notebooks/experiment/01_prune_all``,
``02_eval_all``, and ``06_prune_eval_v2``. They all share the same shape of
work:

1. find a viable spot capacity candidate for the requested instance types,
2. shell out to ``infra/provisioning/launch_gpu_instance.py`` with the right runner
   + runner-env JSON,
3. poll EC2 + S3 until the run finishes,
4. fetch the artifact directory back to the local notebook for display.

This module wraps each of those steps so the notebooks themselves stay
short and readable. All AWS calls go through ``boto3`` using the credentials
the kernel inherits (typically ``AWS_PROFILE=rengz``); nothing here writes
to the local filesystem outside the notebook's chosen download directory.
"""

from __future__ import annotations

# Design: this package is a pure code-organization split of what used to be
# one ~660-line flat module. Every public name below is re-exported here so
# `from pruning_metrics.notebook_helpers import X` keeps working unchanged
# for the six consumer notebooks and scripts/build_notebooks.py — none of
# them need to learn about the new launch/polling/util submodule boundary.
from pruning_metrics.notebook_helpers.launch import (
    InsufficientCapacityError,
    LaunchedRun,
    QuotaExhaustedError,
    find_capacity,
    launch_runner,
    launch_runner_with_fallback,
)
from pruning_metrics.notebook_helpers.polling import (
    describe_instance,
    wait_for_artifact,
    wait_for_runner_completion,
)
from pruning_metrics.notebook_helpers.util import list_results, render_run_id_default

__all__ = [
    "InsufficientCapacityError",
    "QuotaExhaustedError",
    "LaunchedRun",
    "find_capacity",
    "launch_runner",
    "launch_runner_with_fallback",
    "wait_for_artifact",
    "wait_for_runner_completion",
    "describe_instance",
    "list_results",
    "render_run_id_default",
]
