"""Tests for ``pruning_metrics.notebook_helpers``."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

from pruning_metrics.notebook_helpers import wait_for_runner_completion


def test_wait_for_runner_completion_summary_ended_immediate() -> None:
    """Exit on first poll when ``ended_at_utc`` is already set."""

    summary = {
        "ended_at_utc": "2026-01-01T00:00:00+00:00",
        "completed_levels": [0.0, 50.0],
    }
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {
        "Body": io.BytesIO(json.dumps(summary).encode("utf-8")),
    }
    mock_session = MagicMock()
    mock_session.client.return_value = mock_s3

    with patch("boto3.session.Session", return_value=mock_session):
        reason, out = wait_for_runner_completion(
            bucket="b",
            summary_key="pref/summary.json",
            instance_id="i-123",
            region="us-east-1",
            aws_profile=None,
            poll_seconds=0.01,
            timeout_seconds=30.0,
            progress_log_interval_seconds=None,
        )
    assert reason == "summary_ended"
    assert out == summary


def test_wait_for_runner_completion_instance_terminal_fallback() -> None:
    """When summary stays without ``ended_at_utc``, stop once EC2 is terminal."""

    partial = {"ended_at_utc": None, "completed_levels": [0.0]}
    mock_s3 = MagicMock()

    def _get_object(**_: object) -> dict[str, io.BytesIO]:
        # Fresh buffer each call (second read happens after EC2 terminal).
        return {"Body": io.BytesIO(json.dumps(partial).encode("utf-8"))}

    mock_s3.get_object.side_effect = _get_object
    mock_session = MagicMock()
    mock_session.client.return_value = mock_s3

    with patch("boto3.session.Session", return_value=mock_session):
        with patch(
            "pruning_metrics.notebook_helpers.polling.describe_instance",
            return_value={"State": {"Name": "terminated"}},
        ):
            reason, out = wait_for_runner_completion(
                bucket="b",
                summary_key="pref/summary.json",
                instance_id="i-123",
                region="us-east-1",
                aws_profile=None,
                poll_seconds=0.01,
                timeout_seconds=30.0,
                progress_log_interval_seconds=None,
            )
    assert reason == "instance_terminal"
    assert out == partial
