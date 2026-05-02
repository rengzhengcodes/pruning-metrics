"""Shared configuration models for SageMaker deployment workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_PRUNING_LEVELS = (0, 20, 40, 60, 80)


@dataclass(frozen=True)
class SageMakerInfraConfig:
    """Common SageMaker and S3 configuration.

    Parameters
    ----------
    region:
        AWS region for SageMaker and S3 APIs.
    role_arn:
        SageMaker execution role ARN.
    artifact_bucket:
        S3 bucket for model artifacts and manifests.
    artifact_prefix:
        Prefix under the artifact bucket.
    logits_bucket:
        S3 bucket for per-token logits.
    logits_prefix:
        Prefix under logits bucket.
    endpoint_name:
        Deployed endpoint name.
    instance_type:
        SageMaker instance type.
    instance_count:
        Number of endpoint instances.
    """

    region: str
    role_arn: str
    artifact_bucket: str
    artifact_prefix: str
    logits_bucket: str
    logits_prefix: str
    endpoint_name: str
    instance_type: str = "ml.g5.48xlarge"
    instance_count: int = 1

    @classmethod
    def from_env(cls) -> "SageMakerInfraConfig":
        """Build configuration from environment variables."""

        return cls(
            region=os.environ.get(
                "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "")
            ),
            role_arn=os.environ.get("SAGEMAKER_ROLE_ARN", ""),
            artifact_bucket=os.environ.get("PRUNING_ARTIFACT_BUCKET", ""),
            artifact_prefix=os.environ.get("PRUNING_ARTIFACT_PREFIX", "qwen-pruning"),
            logits_bucket=os.environ.get("PRUNING_LOGITS_BUCKET", ""),
            logits_prefix=os.environ.get("PRUNING_LOGITS_PREFIX", "logits"),
            endpoint_name=os.environ.get(
                "PRUNING_ENDPOINT_NAME", "qwen-pruning-endpoint"
            ),
            instance_type=os.environ.get("PRUNING_INSTANCE_TYPE", "ml.g5.48xlarge"),
            instance_count=int(os.environ.get("PRUNING_INSTANCE_COUNT", "1")),
        )


def parse_pruning_levels(raw_levels: str | None) -> list[int]:
    """Parse comma-separated pruning levels string.

    Parameters
    ----------
    raw_levels:
        Comma-separated levels (for example ``0,20,40,60,80``). ``None`` maps to
        default levels.

    Returns
    -------
    list[int]
        Parsed and sorted pruning levels.
    """

    if raw_levels is None:
        return list(DEFAULT_PRUNING_LEVELS)

    parsed = sorted(
        {int(part.strip()) for part in raw_levels.split(",") if part.strip()}
    )
    if not parsed:
        raise ValueError("At least one pruning level must be provided.")
    return parsed
