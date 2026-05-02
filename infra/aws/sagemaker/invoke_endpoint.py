"""Invoke Qwen pruning SageMaker endpoint from local machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3

from infra.aws.sagemaker.config import SageMakerInfraConfig


def parse_args() -> argparse.Namespace:
    """Parse local invocation arguments."""

    parser = argparse.ArgumentParser(description="Invoke Qwen pruning endpoint.")
    parser.add_argument("--endpoint-name", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--pruning-level", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--region", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    """Invoke endpoint and print response JSON."""

    args = parse_args()
    defaults = SageMakerInfraConfig.from_env()
    region = args.region or defaults.region
    endpoint_name = args.endpoint_name or defaults.endpoint_name
    if not region:
        raise ValueError("AWS region is required.")
    if not endpoint_name:
        raise ValueError("Endpoint name is required.")

    runtime = boto3.client("sagemaker-runtime", region_name=region)
    payload = {
        "prompt": args.prompt,
        "task_id": args.task_id,
        "pruning_level": args.pruning_level,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    body = response["Body"].read().decode("utf-8")
    parsed = json.loads(body)
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(parsed, indent=2), encoding="utf-8"
        )
    print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()
