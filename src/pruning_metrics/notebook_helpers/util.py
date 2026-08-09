"""Shared constants and small utilities for the ``notebook_helpers`` package.

Holds ``REPO_ROOT`` (needed by both :mod:`.launch` and this module) plus the
result-listing, run-id, and JSON-parsing helpers that don't belong to either
the launch or polling flow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Design: this file lives three directories below the repo root
# (src/pruning_metrics/notebook_helpers/util.py), one deeper than the old
# flat src/pruning_metrics/notebook_helpers.py. ``parents[3]`` (not the
# original ``parents[2]``) is what now resolves to the repo root; this is
# the one line in the move that had to change to preserve behavior.
REPO_ROOT = Path(__file__).resolve().parents[3]


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


def render_run_id_default() -> str:
    """Generate a notebook-side run id.

    Delegates to the runners' canonical generator so the format cannot
    drift between notebook-side and launcher-side ids.
    """

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from infra.runners._runner_common import default_run_id

    return default_run_id()


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
