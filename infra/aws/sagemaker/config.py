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


def parse_pruning_levels(raw_levels: str | None) -> list[float]:
    """Parse comma-separated pruning levels string.

    Accepts integer- or float-formatted percentages (e.g. ``"0,20,40,60,80"`` or
    ``"0.0,20.0,40.0,60.0,80.0"``). Returned values are sorted floats so callers
    can compute fractional sparsities without re-parsing.

    Parameters
    ----------
    raw_levels:
        Comma-separated levels (for example ``0,20,40,60,80`` or
        ``0.0,20.0,40.0,60.0,80.0``). ``None`` maps to default levels.

    Returns
    -------
    list[float]
        Parsed and sorted pruning levels in percent units.

    Preconditions
    -------------
    Each non-empty token must parse as a finite float in ``[0, 100)``.

    Postconditions
    --------------
    Output is sorted in ascending order with duplicates removed.
    """

    if raw_levels is None:
        return [float(level) for level in DEFAULT_PRUNING_LEVELS]

    parsed_set: set[float] = set()
    for part in raw_levels.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse pruning level token {token!r} as float."
            ) from exc
        if not 0.0 <= value < 100.0:
            raise ValueError(
                f"Pruning level {value} must be in [0, 100) percent."
            )
        parsed_set.add(value)

    if not parsed_set:
        raise ValueError("At least one pruning level must be provided.")
    return sorted(parsed_set)
