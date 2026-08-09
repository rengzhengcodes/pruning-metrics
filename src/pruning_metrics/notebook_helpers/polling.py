"""S3/EC2 polling helpers used while a notebook waits on a launched runner.

Holds :func:`wait_for_artifact`, :func:`wait_for_runner_completion`, and
:func:`describe_instance`.
"""

from __future__ import annotations

import json
import time
from typing import Any

# NOTE: this pragma was added by the notebook_helpers package split (not
# present in the flat notebook_helpers.py). The boto3-session boilerplate
# in wait_for_artifact/wait_for_runner_completion below was already
# duplicated with list_results() in the original single-file module, but
# pylint's similarity checker only reports duplicate-code (R0801) across
# module boundaries, so the pre-split file never triggered it. Per the
# task's "pure move" constraint we don't extract a shared session helper
# here — that would be a body-level refactor — so we suppress the
# checker instead.
# pylint: disable=duplicate-code


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
        except s3.exceptions.ClientError as exc:  # type: ignore[attr-defined]
            # Only swallow 404/NoSuchKey; re-raise AccessDenied and other errors
            # so they surface immediately rather than looping until timeout.
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in ("404", "NoSuchKey"):
                raise RuntimeError(f"head_object failed ({code}): {exc}") from exc
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise RuntimeError(f"head_object failed: {exc}") from exc
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Artifact s3://{bucket}/{key} not present after {elapsed:.0f}s"
            )
        time.sleep(poll_seconds)


def wait_for_runner_completion(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    bucket: str,
    summary_key: str,
    instance_id: str,
    region: str,
    aws_profile: str | None = None,
    poll_seconds: float = 60.0,
    timeout_seconds: float = 60 * 60 * 6,
    progress_log_interval_seconds: float | None = 300.0,
) -> tuple[str, dict[str, Any] | None]:
    """Block until a GPU runner is done, using S3 plus an EC2 fallback.

    EC2-only polling is a poor match for these notebooks: runners upload an
    incremental ``summary.json`` after **each** pruning level, so the object
    can exist in S3 for a long time while the instance stays ``running`` on
    later levels. Completion is signaled by a non-null ``ended_at_utc`` field
    in that JSON (written in the runner's ``finally`` block).

    This function returns when **either**:

    1. ``summary.json`` parses and ``ended_at_utc`` is not ``None`` — normal
       successful exit; or
    2. The instance reaches ``terminated`` or ``stopped`` — fallback if the
       process died without a final summary (``SIGKILL``, etc.).

    Parameters
    ----------
    bucket:
        Results bucket (e.g. from ``RESULTS_BUCKET`` in the notebooks).
    summary_key:
        Full S3 object key for ``summary.json`` (no ``s3://`` prefix).
    instance_id, region:
        Launched instance; if ``instance_id`` is empty, only the S3 condition
        is used.
    aws_profile:
        Optional boto3 profile.
    poll_seconds:
        Sleep between polls.
    timeout_seconds:
        Raise ``TimeoutError`` if neither completion signal appears in time.
    progress_log_interval_seconds:
        When set and positive, print EC2 state and summary progress on that
        wall-clock interval (and once on the first iteration).

    Returns
    -------
    tuple[str, dict[str, Any] | None]
        ``(reason, summary)`` where ``reason`` is ``\"summary_ended\"`` or
        ``\"instance_terminal\"``, and ``summary`` is the parsed
        ``summary.json`` when available.

    Raises
    ------
    TimeoutError
        If neither signal appears within ``timeout_seconds``.
    """

    import boto3
    from botocore.exceptions import ClientError

    session_kwargs: dict[str, Any] = {}
    if aws_profile is not None:
        session_kwargs["profile_name"] = aws_profile
    session = boto3.session.Session(**session_kwargs)
    s3 = session.client("s3")

    def load_summary() -> dict[str, Any] | None:
        try:
            body = s3.get_object(Bucket=bucket, Key=summary_key)["Body"].read()
            parsed = json.loads(body.decode("utf-8"))
        except ClientError:
            return None
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, KeyError):
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    started = time.monotonic()
    log_progress = (
        progress_log_interval_seconds is not None
        and progress_log_interval_seconds > 0.0
    )
    progress_interval = float(progress_log_interval_seconds) if log_progress else 0.0
    last_progress_log_mono = float("-inf")

    if log_progress:
        print(
            f"Waiting for final summary s3://{bucket}/{summary_key} "
            f"(ended_at_utc set) or instance {instance_id!r} to stop. "
            f"Poll every {poll_seconds:.0f}s, timeout {timeout_seconds / 3600:.1f} h.",
            flush=True,
        )

    while True:
        now = time.monotonic()
        elapsed = now - started
        summary = load_summary()
        ended = bool(summary and summary.get("ended_at_utc"))

        if ended:
            return "summary_ended", summary

        ec2_state = "n/a"
        if instance_id:
            info = describe_instance(instance_id, region, aws_profile=aws_profile)
            ec2_state = (info.get("State") or {}).get("Name", "unknown")
            if ec2_state in ("terminated", "stopped"):
                final_summary = load_summary()
                return "instance_terminal", final_summary

        if log_progress and (now - last_progress_log_mono >= progress_interval):
            levels = summary.get("completed_levels") if summary else None
            ended_show = summary.get("ended_at_utc") if summary else None
            print(
                f"  EC2 state={ec2_state!r} summary_levels={levels!r} "
                f"ended_at_utc={ended_show!r} "
                f"elapsed_min={elapsed / 60.0:.1f}",
                flush=True,
            )
            last_progress_log_mono = now

        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Runner still not finished after {elapsed:.0f}s "
                f"(no ended_at_utc in {summary_key!r}, "
                f"instance {instance_id!r} state {ec2_state!r})."
            )
        time.sleep(poll_seconds)


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
