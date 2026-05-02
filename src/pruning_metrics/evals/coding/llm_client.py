"""Interfaces and implementations for model code generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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


class SageMakerClient:  # pylint: disable=too-few-public-methods
    """Placeholder adapter for Amazon SageMaker endpoint invocation."""

    def __init__(self, endpoint_name: str) -> None:
        self.endpoint_name = endpoint_name

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
        AWS credentials and endpoint integration are configured.

        Postconditions
        --------------
        Raises ``NotImplementedError`` until wired to AWS.
        """

        del prompt, task_id
        raise NotImplementedError(
            "SageMakerClient is not implemented yet. "
            "Use MockLLMClient or provide a concrete AWS adapter."
        )
