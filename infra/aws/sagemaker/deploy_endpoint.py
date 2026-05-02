"""Deploy or update a SageMaker endpoint for Qwen pruning inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

from infra.aws.sagemaker.config import SageMakerInfraConfig


def parse_args() -> argparse.Namespace:
    """Parse endpoint deployment arguments."""

    parser = argparse.ArgumentParser(
        description="Deploy Qwen pruning SageMaker endpoint."
    )
    parser.add_argument("--container-image-uri", required=True)
    parser.add_argument("--manifest-s3-uri", required=True)
    parser.add_argument("--endpoint-name", default=None)
    parser.add_argument("--role-arn", default=None)
    parser.add_argument("--instance-type", default=None)
    parser.add_argument("--instance-count", type=int, default=None)
    parser.add_argument("--model-data-url", default=None)
    parser.add_argument("--model-name-prefix", default="qwen-pruning-model")
    parser.add_argument("--config-name-prefix", default="qwen-pruning-config")
    parser.add_argument("--region", default=None)
    parser.add_argument("--logits-bucket", default=None)
    parser.add_argument("--logits-prefix", default=None)
    return parser.parse_args()


def main() -> None:
    """Create or update model, endpoint config, and endpoint."""

    args = parse_args()
    defaults = SageMakerInfraConfig.from_env()
    region = args.region or defaults.region
    if not region:
        raise ValueError("AWS region is required.")

    endpoint_name = args.endpoint_name or defaults.endpoint_name
    role_arn = args.role_arn or defaults.role_arn
    instance_type = args.instance_type or defaults.instance_type
    instance_count = args.instance_count or defaults.instance_count
    logits_bucket = args.logits_bucket or defaults.logits_bucket
    logits_prefix = args.logits_prefix or defaults.logits_prefix
    if not role_arn:
        raise ValueError("SageMaker role ARN is required.")
    if not logits_bucket:
        raise ValueError("Logits bucket is required.")

    sagemaker = boto3.client("sagemaker", region_name=region)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    model_name = f"{args.model_name_prefix}-{timestamp}"
    endpoint_config_name = f"{args.config_name_prefix}-{timestamp}"

    env = {
        "MODEL_MANIFEST_S3_URI": args.manifest_s3_uri,
        "LOGITS_S3_BUCKET": logits_bucket,
        "LOGITS_S3_PREFIX": logits_prefix,
    }
    container_def = {
        "Image": args.container_image_uri,
        "Environment": env,
    }
    if args.model_data_url:
        container_def["ModelDataUrl"] = args.model_data_url

    sagemaker.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer=container_def,
    )
    sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": instance_count,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    if endpoint_exists(sagemaker, endpoint_name):
        sagemaker.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = "updated"
    else:
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = "created"

    print(
        f"Endpoint {action}: {endpoint_name}\n"
        f"Model: {model_name}\n"
        f"EndpointConfig: {endpoint_config_name}"
    )


def endpoint_exists(sagemaker_client: boto3.client, endpoint_name: str) -> bool:
    """Return whether endpoint currently exists."""

    try:
        sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ValidationException":
            return False
        raise


if __name__ == "__main__":
    main()
