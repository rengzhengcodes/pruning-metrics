"""Helpers shared by the four orchestration notebooks.

The four notebooks (``notebooks/01_setup_aws.ipynb`` through
``notebooks/04_teacher_forced.ipynb``) all share the same shape of work:

1. find a viable spot capacity candidate for the requested instance types,
2. shell out to ``infra/ec2/launch_gpu_instance.py`` with the right runner
   + runner-env JSON,
3. poll EC2 + S3 until the run finishes,
4. fetch the artifact directory back to the local notebook for display.

This module wraps each of those steps so the notebooks themselves stay
short and readable. All AWS calls go through ``boto3`` using the credentials
the kernel inherits (typically ``AWS_PROFILE=rengz``); nothing here writes
to the local filesystem outside the notebook's chosen download directory.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        str(REPO_ROOT / "infra" / "ec2" / "find_capacity.py"),
        "--regions",
        ",".join(regions),
        "--instance-types",
        ",".join(instance_types),
    ]
    env = dict(os.environ)
    if aws_profile is not None:
        env["AWS_PROFILE"] = aws_profile
    completed = subprocess.run(
        cmd, env=env, check=True, capture_output=True, text=True
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
        ``infra/ec2/_last_userdata.sh``).
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
        str(REPO_ROOT / "infra" / "ec2" / "launch_gpu_instance.py"),
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

    completed = subprocess.run(
        cmd, env=env, check=True, capture_output=True, text=True
    )
    # The launcher prints two JSON blobs to stdout: the early plan and the
    # post-launch enriched plan. The last full JSON object is the canonical one.
    plan = _last_json_object(completed.stdout)
    if plan is None:
        raise RuntimeError(
            "Could not parse launcher JSON output:\n" + completed.stdout
        )

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


def wait_for_artifact(
    bucket: str,
    key: str,
    *,
    aws_profile: str | None = None,
    poll_seconds: float = 30.0,
    timeout_seconds: float = 60 * 60 * 6,
) -> dict[str, Any]:
    """Poll S3 every ``poll_seconds`` until the object at ``key`` exists.

    Returns the ``HeadObject`` response so callers can sanity-check size /
    LastModified.
    """

    import boto3

    session_kwargs: dict[str, Any] = {}
    if aws_profile is not None:
        session_kwargs["profile_name"] = aws_profile
    session = boto3.session.Session(**session_kwargs)
    s3 = session.client("s3")

    started = time.monotonic()
    while True:
        try:
            return s3.head_object(Bucket=bucket, Key=key)
        except s3.exceptions.ClientError:  # type: ignore[attr-defined]
            pass
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # 404 manifests as ClientError above; anything else (network,
            # auth) bubbles up as a generic exception that we re-raise.
            raise RuntimeError(f"head_object failed: {exc}") from exc
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Artifact s3://{bucket}/{key} not present after {elapsed:.0f}s"
            )
        time.sleep(poll_seconds)


def list_results(
    bucket: str,
    prefix: str,
    *,
    aws_profile: str | None = None,
) -> list[dict[str, Any]]:
    """List all S3 objects under ``s3://bucket/prefix/`` with sizes / mtimes."""

    import boto3

    session_kwargs: dict[str, Any] = {}
    if aws_profile is not None:
        session_kwargs["profile_name"] = aws_profile
    session = boto3.session.Session(**session_kwargs)
    s3 = session.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    out: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for entry in page.get("Contents", []) or []:
            out.append(
                {
                    "key": entry["Key"],
                    "size": entry["Size"],
                    "last_modified": entry["LastModified"].isoformat(),
                }
            )
    return out


def download_results(
    bucket: str,
    prefix: str,
    target_dir: Path,
    *,
    aws_profile: str | None = None,
    suffixes: tuple[str, ...] | None = None,
) -> list[Path]:
    """Download every object under ``prefix`` into ``target_dir``.

    Parameters
    ----------
    suffixes:
        Optional whitelist of filename suffixes to download (``(".json",
        ".jsonl")`` for example). ``None`` downloads everything.
    """

    import boto3

    session_kwargs: dict[str, Any] = {}
    if aws_profile is not None:
        session_kwargs["profile_name"] = aws_profile
    session = boto3.session.Session(**session_kwargs)
    s3 = session.client("s3")
    target_dir.mkdir(parents=True, exist_ok=True)
    fetched: list[Path] = []
    for entry in list_results(bucket, prefix, aws_profile=aws_profile):
        key = entry["key"]
        if suffixes is not None and not key.endswith(suffixes):
            continue
        relative = Path(key).relative_to(prefix.strip("/"))
        local_path = target_dir / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local_path))
        fetched.append(local_path)
    return fetched


def describe_instance(
    instance_id: str,
    region: str,
    *,
    aws_profile: str | None = None,
) -> dict[str, Any]:
    """Return ``DescribeInstances`` Reservation[0].Instances[0] payload."""

    if not instance_id:
        return {}

    import boto3

    session_kwargs: dict[str, Any] = {}
    if aws_profile is not None:
        session_kwargs["profile_name"] = aws_profile
    session = boto3.session.Session(**session_kwargs)
    ec2 = session.client("ec2", region_name=region)
    response = ec2.describe_instances(InstanceIds=[instance_id])
    reservations = response.get("Reservations") or []
    if not reservations or not reservations[0].get("Instances"):
        return {}
    return reservations[0]["Instances"][0]


def wait_for_instance_terminated(
    instance_id: str,
    region: str,
    *,
    aws_profile: str | None = None,
    poll_seconds: float = 30.0,
    timeout_seconds: float = 60 * 60 * 8,
) -> str:
    """Block until the instance reaches a terminal state ('terminated' or 'stopped')."""

    started = time.monotonic()
    while True:
        info = describe_instance(instance_id, region, aws_profile=aws_profile)
        state = (info.get("State") or {}).get("Name", "unknown")
        if state in ("terminated", "stopped"):
            return state
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(
                f"Instance {instance_id} still in state {state!r} after timeout."
            )
        time.sleep(poll_seconds)


def render_run_id_default() -> str:
    """Generate a notebook-side run id (UTC timestamp + short uuid)."""

    import uuid

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _last_json_object(text: str) -> dict[str, Any] | None:
    """Return the last balanced ``{...}`` JSON object in ``text`` (or None).

    The launcher prints two JSON dicts on stdout - an early plan and the
    final post-launch enriched plan. We always want the last one.
    """

    last_end = text.rfind("}")
    while last_end != -1:
        depth = 0
        for start in range(last_end, -1, -1):
            if text[start] == "}":
                depth += 1
            elif text[start] == "{":
                depth -= 1
                if depth == 0:
                    candidate = text[start : last_end + 1]
                    try:
                        decoded = json.loads(candidate)
                        if isinstance(decoded, dict):
                            return decoded
                    except json.JSONDecodeError:
                        break
        last_end = text.rfind("}", 0, last_end)
    return None


def shlex_join(args: Iterable[str]) -> str:
    """``shlex.join`` polyfill for older Python; passes through otherwise."""

    if hasattr(shlex, "join"):
        return shlex.join(list(args))
    return " ".join(shlex.quote(arg) for arg in args)
