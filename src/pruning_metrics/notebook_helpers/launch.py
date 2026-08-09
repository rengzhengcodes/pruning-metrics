"""Spot-capacity search and GPU-runner launch helpers.

Holds the exceptions and functions used to find viable spot capacity and
shell out to ``infra/provisioning/launch_gpu_instance.py``:
:class:`InsufficientCapacityError`, :class:`QuotaExhaustedError`,
:class:`LaunchedRun`, :func:`find_capacity`, :func:`launch_runner`, and
:func:`launch_runner_with_fallback`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from pruning_metrics.notebook_helpers.util import REPO_ROOT, _last_json_object


class InsufficientCapacityError(RuntimeError):
    """Raised when EC2 returns InsufficientInstanceCapacity / UnfulfillableCapacity."""


class QuotaExhaustedError(RuntimeError):
    """Raised when EC2 returns MaxSpotInstanceCountExceeded for a region."""

    def __init__(self, message: str, region: str) -> None:
        super().__init__(message)
        self.region = region


@dataclass
class LaunchedRun:
    """Result of a successful ``RunInstances`` call.

    Parameters
    ----------
    instance_id:
        EC2 instance id ("i-..."). Empty string for ``--dry-run``.
    region:
        AWS region used for the launch.
    availability_zone, instance_type:
        Confirmed launch placement.
    run_id:
        Unique run identifier (also used as the S3 prefix suffix).
    runner:
        Logical runner name (``pruning_calibration`` etc.).
    results_uri:
        ``s3://<bucket>/<prefix>/<run_id>/`` where the runner uploads.
    raw_plan:
        Full JSON dict printed by ``launch_gpu_instance.py``.
    """

    instance_id: str
    region: str
    availability_zone: str
    instance_type: str
    run_id: str
    runner: str
    results_uri: str
    raw_plan: dict[str, Any]


def find_capacity(
    *,
    regions: Iterable[str] = ("us-east-1", "us-west-2", "us-east-2"),
    instance_types: Iterable[str] = (
        "p4de.24xlarge",
        "p5.48xlarge",
        "p4d.24xlarge",
    ),
    aws_profile: str | None = None,
) -> list[dict[str, Any]]:
    """Return ``find_capacity.py``'s candidate list, sorted by priority.

    Parameters
    ----------
    regions, instance_types:
        Search lists (priority order). The first viable region/instance pair
        in priority order tends to be the cheapest available.
    aws_profile:
        Override ``AWS_PROFILE``. ``None`` keeps whatever the kernel inherits.

    Returns
    -------
    list[dict]
        Candidates as dicts with ``region``, ``availability_zone``,
        ``instance_type``, ``spot_price_usd_per_hour``,
        ``max_bid_usd_per_hour`` (suggested bid), and tie-breaker indices.

    Notes
    -----
    For p4de or p5 in particular, capacity flips minute-to-minute: the
    notebook should call this twice, picking the first candidate, and only
    fall back to the next candidate if ``RunInstances`` returns
    ``InsufficientInstanceCapacity``.
    """

    cmd = [
        sys.executable,
        str(REPO_ROOT / "infra" / "provisioning" / "find_capacity.py"),
        "--regions",
        ",".join(regions),
        "--instance-types",
        ",".join(instance_types),
    ]
    env = dict(os.environ)
    if aws_profile is not None:
        env["AWS_PROFILE"] = aws_profile
    completed = subprocess.run(
        cmd, env=env, check=False, stdout=subprocess.PIPE, stderr=None, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"find_capacity.py exited {completed.returncode}. "
            f"See stderr output above."
        )
    payload = json.loads(completed.stdout)
    return list(payload.get("candidates", []))


def launch_runner(
    *,
    runner: str,
    runner_env: Mapping[str, Any],
    region: str,
    availability_zone: str,
    instance_type: str,
    max_spot_price: float,
    results_bucket: str,
    results_prefix: str,
    run_id: str | None = None,
    aws_profile: str | None = None,
    instance_profile: str = "pruning-metrics-ec2",
    hf_token: str = "",
    dry_run: bool = False,
    no_shutdown_on_exit: bool = False,
    name_tag: str = "pruning-metrics-runner",
) -> LaunchedRun:
    """Shell out to ``launch_gpu_instance.py`` and parse its JSON output.

    Parameters
    ----------
    runner:
        One of ``pruning_calibration``, ``freeform_eval``, ``teacher_forced``,
        ``full_pipeline``.
    runner_env:
        Dict of runner-specific env vars (passed via ``--runner-env-json``).
    region, availability_zone, instance_type, max_spot_price:
        Launch placement and bid (typically derived from
        :func:`find_capacity`).
    results_bucket, results_prefix:
        Where the runner uploads. The launcher appends ``<run_id>/``.
    run_id:
        Optional pre-chosen id. The launcher generates one when ``None``.
    aws_profile:
        Optional AWS profile override.
    instance_profile:
        IAM instance profile attached to the box. Default matches the
        bootstrap script.
    hf_token:
        Optional HF token for gated models.
    dry_run:
        Skip ``RunInstances`` (still uploads tarball + writes
        ``infra/provisioning/_last_userdata.sh``).
    no_shutdown_on_exit:
        Keep instance alive after the runner finishes (debug).
    name_tag:
        Value for the ``Name`` tag.

    Returns
    -------
    LaunchedRun
        Parsed JSON payload from the launcher.
    """

    cmd = [
        sys.executable,
        str(REPO_ROOT / "infra" / "provisioning" / "launch_gpu_instance.py"),
        "--region",
        region,
        "--availability-zone",
        availability_zone,
        "--instance-type",
        instance_type,
        "--max-spot-price",
        f"{max_spot_price:.4f}",
        "--results-bucket",
        results_bucket,
        "--results-prefix",
        results_prefix,
        "--instance-profile",
        instance_profile,
        "--name-tag",
        name_tag,
        "--runner",
        runner,
        "--runner-env-json",
        json.dumps({str(k): str(v) for k, v in runner_env.items()}),
    ]
    if run_id:
        cmd += ["--run-id", run_id]
    if hf_token:
        cmd += ["--hf-token", hf_token]
    if dry_run:
        cmd.append("--dry-run")
    if no_shutdown_on_exit:
        cmd.append("--no-shutdown-on-exit")

    env = dict(os.environ)
    if aws_profile is not None:
        env["AWS_PROFILE"] = aws_profile

    # stderr is NOT captured — it flows directly to the notebook cell so
    # progress lines and any error tracebacks are visible immediately.
    completed = subprocess.run(
        cmd, env=env, check=False, stdout=subprocess.PIPE, stderr=None, text=True
    )
    if completed.returncode == 2:
        raise InsufficientCapacityError(
            f"No spot capacity for {instance_type} in {availability_zone}."
        )
    if completed.returncode == 3:
        raise QuotaExhaustedError(
            f"Spot vCPU quota exhausted in {region}.", region=region
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"launch_gpu_instance.py exited {completed.returncode}. "
            f"See stderr output above."
        )
    # The launcher prints two JSON blobs to stdout: the early plan and the
    # post-launch enriched plan. The last full JSON object is the canonical one.
    plan = _last_json_object(completed.stdout)
    if plan is None:
        raise RuntimeError("Could not parse launcher JSON output:\n" + completed.stdout)

    instance_id = plan.get("instance_id", "")
    return LaunchedRun(
        instance_id=instance_id,
        region=plan.get("region", region),
        availability_zone=plan.get("availability_zone", availability_zone),
        instance_type=plan.get("instance_type", instance_type),
        run_id=plan["run_id"],
        runner=plan.get("runner", runner),
        results_uri=(
            f"s3://{results_bucket}/{results_prefix.strip('/')}/{plan['run_id']}/"
        ),
        raw_plan=plan,
    )


def launch_runner_with_fallback(
    candidates: list[dict[str, Any]],
    *,
    runner: str,
    runner_env: Mapping[str, Any],
    results_bucket: str,
    results_prefix: str,
    run_id: str | None = None,
    aws_profile: str | None = None,
    instance_profile: str = "pruning-metrics-ec2",
    hf_token: str = "",
    dry_run: bool = False,
    no_shutdown_on_exit: bool = False,
    name_tag: str = "pruning-metrics-runner",
    recheck_regions: list[str] | None = None,
    recheck_instance_types: list[str] | None = None,
) -> LaunchedRun:
    """Try each capacity candidate in order, retrying on InsufficientInstanceCapacity.

    When the initial ``candidates`` list is exhausted, re-probes spot capacity
    using ``recheck_regions`` / ``recheck_instance_types`` (or the union of
    whatever was in ``candidates``) and tries any newly-visible AZs.

    Parameters
    ----------
    candidates:
        Ordered list returned by :func:`find_capacity`. Tried front-to-back;
        the first successful ``RunInstances`` wins.
    recheck_regions:
        Regions to search if the initial candidates are all full.  Defaults to
        the union of regions already in ``candidates``.
    recheck_instance_types:
        Instance types to search on recheck.  Defaults to the union already in
        ``candidates``.
    All other parameters:
        Forwarded verbatim to :func:`launch_runner`.

    Raises
    ------
    InsufficientCapacityError
        When no candidate (including recheck) has available capacity.
    """

    if not candidates:
        raise ValueError("candidates list is empty.")

    attempted: set[str] = set()
    quota_exhausted_regions: set[str] = set()

    def _try_list(clist: list[dict[str, Any]]) -> LaunchedRun | None:
        for candidate in clist:
            az = candidate["availability_zone"]
            itype = candidate["instance_type"]
            region = candidate["region"]
            key = f"{region}/{az}/{itype}"
            if key in attempted:
                continue
            if region in quota_exhausted_regions:
                print(
                    f"  Skipping {az} ({itype}): quota exhausted in {region}.",
                    flush=True,
                )
                attempted.add(key)
                continue
            attempted.add(key)
            try:
                return launch_runner(
                    runner=runner,
                    runner_env=runner_env,
                    region=region,
                    availability_zone=az,
                    instance_type=itype,
                    max_spot_price=float(candidate["max_bid_usd_per_hour"]),
                    results_bucket=results_bucket,
                    results_prefix=results_prefix,
                    run_id=run_id,
                    aws_profile=aws_profile,
                    instance_profile=instance_profile,
                    hf_token=hf_token,
                    dry_run=dry_run,
                    no_shutdown_on_exit=no_shutdown_on_exit,
                    name_tag=name_tag,
                )
            except QuotaExhaustedError as exc:
                quota_exhausted_regions.add(exc.region)
                print(
                    f"  Spot quota exhausted in {exc.region}; "
                    f"skipping all remaining {exc.region} candidates.",
                    flush=True,
                )
            except InsufficientCapacityError:
                print(f"  No capacity at {az} ({itype}); trying next...", flush=True)
        return None

    result = _try_list(candidates)
    if result is not None:
        return result

    # All initial candidates exhausted — re-probe for fresh capacity data.
    regions = recheck_regions or sorted({c["region"] for c in candidates})
    instance_types = recheck_instance_types or list(
        dict.fromkeys(c["instance_type"] for c in candidates)
    )
    print(
        f"  All {len(candidates)} initial candidates exhausted. "
        f"Re-probing capacity across {regions} ...",
        flush=True,
    )
    fresh = find_capacity(
        regions=regions,
        instance_types=instance_types,
        aws_profile=aws_profile,
    )
    result = _try_list(fresh)
    if result is not None:
        return result

    raise InsufficientCapacityError(
        f"No spot capacity found after trying {len(attempted)} candidates "
        f"(initial + recheck) across {regions}: " + ", ".join(sorted(attempted))
    )
