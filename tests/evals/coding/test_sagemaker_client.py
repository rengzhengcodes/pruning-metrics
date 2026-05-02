"""Unit tests for SageMaker client payload and parsing behavior."""

from __future__ import annotations

import json

import pytest

from pruning_metrics.evals.coding.llm_client import SageMakerClient


class _FakeBody:  # pylint: disable=too-few-public-methods
    """Simple boto3 streaming body stand-in for unit tests."""

    def __init__(self, payload: str) -> None:
        """Store encoded body payload."""

        self._payload = payload.encode("utf-8")

    def read(self) -> bytes:
        """Mirror botocore streaming body read API."""

        return self._payload


class _FakeRuntime:  # pylint: disable=too-few-public-methods
    """Capture endpoint invocation payload for assertions."""

    def __init__(self, response_payload: str) -> None:
        """Store canned response and latest invocation kwargs."""

        self.response_payload = response_payload
        self.last_invoke_kwargs: dict[str, object] | None = None

    def invoke_endpoint(self, **kwargs):
        """Return canned response payload for endpoint invocation."""

        self.last_invoke_kwargs = kwargs
        return {"Body": _FakeBody(self.response_payload)}


def test_sagemaker_client_invokes_endpoint_and_tracks_metadata() -> None:
    """Validate invoke payload shape and metadata extraction."""

    runtime = _FakeRuntime(
        response_payload=json.dumps(
            {
                "generated_text": "def foo(x):\n    return x\n",
                "task_id": "HumanEval/0",
                "pruning_level": 40,
                "seed": 7,
                "token_count": 5,
                "request_id": "req-1",
                "logits_s3_uri": "s3://bucket/logits/req-1/tokens.jsonl",
            }
        )
    )
    client = SageMakerClient(
        endpoint_name="test-endpoint",
        pruning_level=40,
        seed=7,
        max_new_tokens=32,
        temperature=0.2,
        top_p=0.9,
        runtime_client=runtime,
    )

    result = client.generate_code(prompt="Write code", task_id="HumanEval/0")

    assert result == "def foo(x):\n    return x\n"
    assert runtime.last_invoke_kwargs is not None
    assert runtime.last_invoke_kwargs["EndpointName"] == "test-endpoint"
    body = json.loads(runtime.last_invoke_kwargs["Body"].decode("utf-8"))
    assert body["task_id"] == "HumanEval/0"
    assert body["pruning_level"] == 40
    assert body["seed"] == 7
    assert body["max_new_tokens"] == 32
    assert client.last_response_metadata is not None
    assert client.last_response_metadata["logits_s3_uri"].startswith(
        "s3://bucket/logits/"
    )


def test_sagemaker_client_rejects_non_string_generated_text() -> None:
    """Ensure malformed endpoint responses raise explicit errors."""

    runtime = _FakeRuntime(response_payload=json.dumps({"generated_text": 3}))
    client = SageMakerClient(
        endpoint_name="test-endpoint",
        pruning_level=20,
        seed=99,
        runtime_client=runtime,
    )

    with pytest.raises(ValueError, match="generated_text"):
        client.generate_code(prompt="Write code", task_id="HumanEval/1")
