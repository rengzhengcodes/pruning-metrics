"""Interfaces and implementations for model code generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


class LLMClient(Protocol):  # pylint: disable=too-few-public-methods
    """Protocol for code-generation model clients.

    Parameters
    ----------
    prompt:
        Prompt text for model completion.
    task_id:
        Optional identifier used for logging or routing.

    Returns
    -------
    str
        Generated Python code.

    Preconditions
    -------------
    Prompt is a valid text input accepted by the model backend.

    Postconditions
    --------------
    Returned string is model-generated code content.
    """

    def generate_code(self, prompt: str, task_id: str) -> str:
        """Generate a candidate code solution for one task."""


@dataclass
class MockLLMClient:  # pylint: disable=too-few-public-methods
    """Deterministic client backed by a task ID -> solution map.

    Parameters
    ----------
    completions_by_task_id:
        Mapping from task IDs to generated code snippets.
    default_completion:
        Fallback completion for unknown IDs.

    Returns
    -------
    None

    Preconditions
    -------------
    Mapping keys are task IDs and values are Python code strings.

    Postconditions
    --------------
    ``generate_code`` behaves deterministically for fixed inputs.
    """

    completions_by_task_id: dict[str, str]
    default_completion: str = ""

    def generate_code(self, prompt: str, task_id: str) -> str:
        """Return predefined completion for a given task ID.

        Parameters
        ----------
        prompt:
            Prompt text (unused in the mock implementation).
        task_id:
            Task identifier.

        Returns
        -------
        str
            Mock completion text.

        Preconditions
        -------------
        None

        Postconditions
        --------------
        Returns mapped completion when present, else default.
        """

        del prompt
        return self.completions_by_task_id.get(task_id, self.default_completion)


class BedrockClient:  # pylint: disable=too-few-public-methods
    """Placeholder adapter for Amazon Bedrock model invocation."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def generate_code(self, prompt: str, task_id: str) -> str:
        """Generate code by invoking Bedrock runtime.

        Parameters
        ----------
        prompt:
            Prompt text.
        task_id:
            Task identifier.

        Returns
        -------
        str
            Generated code text.

        Preconditions
        -------------
        AWS credentials and runtime integration are configured.

        Postconditions
        --------------
        Raises ``NotImplementedError`` until wired to AWS.
        """

        del prompt, task_id
        raise NotImplementedError(
            "BedrockClient is not implemented yet. "
            "Use MockLLMClient or provide a concrete AWS adapter."
        )


class SageMakerClient:  # pylint: disable=too-few-public-methods,too-many-instance-attributes,too-many-arguments,too-many-positional-arguments
    """Adapter for Amazon SageMaker endpoint invocation.

    Parameters
    ----------
    endpoint_name:
        SageMaker endpoint name.
    pruning_level:
        Pruning level routed by endpoint handler.
    seed:
        Deterministic sampling seed required per request.
    max_new_tokens:
        Generation length limit.
    temperature:
        Temperature for generation.
    top_p:
        Nucleus sampling value.
    region_name:
        Optional AWS region override.
    runtime_client:
        Optional boto3 SageMaker runtime client for dependency injection.
    """

    def __init__(
        self,
        endpoint_name: str,
        pruning_level: int,
        seed: int,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        region_name: str | None = None,
        runtime_client: Any | None = None,
    ) -> None:  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self.endpoint_name = endpoint_name
        self.pruning_level = pruning_level
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.region_name = region_name
        self.last_response_metadata: dict[str, Any] | None = None
        self._runtime_client = runtime_client

    def generate_code(self, prompt: str, task_id: str) -> str:
        """Generate code by invoking a SageMaker endpoint.

        Parameters
        ----------
        prompt:
            Prompt text.
        task_id:
            Task identifier.

        Returns
        -------
        str
            Generated code text.

        Preconditions
        -------------
        AWS credentials and endpoint permissions are configured.

        Postconditions
        --------------
        Stores endpoint metadata from the latest response.
        """

        payload = {
            "prompt": prompt,
            "task_id": task_id,
            "pruning_level": self.pruning_level,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        runtime_client = self._get_runtime_client()
        response = runtime_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload).encode("utf-8"),
        )
        body = response["Body"].read().decode("utf-8")
        decoded = json.loads(body)
        generated_text = decoded.get("generated_text", "")
        if not isinstance(generated_text, str):
            raise ValueError("SageMaker response field 'generated_text' must be a string.")

        self.last_response_metadata = {
            "task_id": decoded.get("task_id", task_id),
            "pruning_level": decoded.get("pruning_level", self.pruning_level),
            "seed": decoded.get("seed", self.seed),
            "token_count": decoded.get("token_count"),
            "request_id": decoded.get("request_id"),
            "logits_s3_uri": decoded.get("logits_s3_uri"),
        }
        return generated_text

    def _get_runtime_client(self) -> Any:
        """Resolve boto3 runtime client lazily."""

        if self._runtime_client is None:
            import boto3  # pylint: disable=import-outside-toplevel

            self._runtime_client = boto3.client(
                "sagemaker-runtime",
                region_name=self.region_name,
            )
        return self._runtime_client
