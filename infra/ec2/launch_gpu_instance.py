"""Launch an EC2 spot GPU instance to run the Qwen2-72B WANDA experiment.

Workflow
--------
1. Build a tarball of this repository (excluding ``.venv``, ``.git``, caches)
   and upload it to ``s3://<bucket>/<prefix>/<run_id>/code/repo.tar.gz``.
2. Resolve the latest **Deep Learning OSS Nvidia Driver AMI GPU PyTorch
   (Ubuntu 22.04)** AMI id via SSM Parameter Store in the chosen region.
3. Render the user-data shell script with the run-specific variables
   substituted in.
4. Issue ``RunInstances`` with the previously-bootstrapped instance profile
   attached, ``InstanceMarketOptions=spot``, a 1.5 TiB gp3 root volume, and
   the rendered user-data.
5. Print the run plan as JSON for the operator (or a wrapping subagent) to
   pipe into a monitoring loop.

This script is intended to be invoked from the operator workstation while a
short-lived SSO session (``rengz``) is still valid; once the instance is
running it operates entirely under its attached IAM instance profile.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import tarfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError


REPO_ROOT = Path(__file__).resolve().parents[2]
USERDATA_TEMPLATE = Path(__file__).with_name("userdata_bootstrap.sh")

# Items we never need on the GPU box; trimming keeps the tarball small enough
# to round-trip through S3 in seconds rather than minutes. ``.env`` is
# excluded because it carries the operator's short-lived AWS credentials —
# the EC2 box gets its own credentials from the attached instance profile.
TAR_EXCLUDES = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    "artifacts",
    "node_modules",
    "build",
    "dist",
    ".env",
    ".env.local",
}

# AMI parameter names (DLAMI; resolved via SSM Parameter Store).
# Probe order: newest published PyTorch on Ubuntu 24.04, then 22.04 fallback,
# then a pure base GPU image as a last resort. A region that hides one rev
# usually still publishes another in the same channel.
DLAMI_PARAMETERS_PRIORITY = (
    "/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.10-ubuntu-24.04/latest/ami-id",
    "/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.5-ubuntu-22.04/latest/ami-id",
    "/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.4-ubuntu-22.04/latest/ami-id",
    "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id",
)


def parse_args() -> argparse.Namespace:
    """Parse launcher arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(
        description="Launch a spot EC2 GPU box and bootstrap the Qwen experiment."
    )
    parser.add_argument(
        "--region", required=True, help="AWS region for RunInstances."
    )
    parser.add_argument(
        "--availability-zone",
        required=True,
        help="Availability zone (e.g. us-east-1b) — must match the spot price probe.",
    )
    parser.add_argument(
        "--instance-type",
        required=True,
        help="EC2 instance type (e.g. p5.48xlarge, p4d.24xlarge).",
    )
    parser.add_argument(
        "--max-spot-price",
        required=True,
        type=float,
        help="Spot bid ceiling in USD/hour.",
    )
    parser.add_argument(
        "--results-bucket",
        default=os.environ.get("RESULTS_BUCKET", ""),
        help="S3 bucket for code tarball, logs, and results.",
    )
    parser.add_argument(
        "--results-prefix",
        default=os.environ.get("RESULTS_PREFIX", "qwen2_72b_pruning"),
        help="S3 key prefix for the run.",
    )
    parser.add_argument(
        "--instance-profile",
        default=os.environ.get(
            "EC2_INSTANCE_PROFILE_NAME", "pruning-metrics-ec2"
        ),
        help="IAM instance profile name.",
    )
    parser.add_argument(
        "--base-model-id",
        default=os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2-72B"),
    )
    parser.add_argument(
        "--pruning-levels",
        default=os.environ.get("PRUNING_LEVELS", "0,20,40,60,80"),
    )
    parser.add_argument(
        "--split-seed",
        default=os.environ.get("HUMANEVAL_SPLIT_SEED", "65320"),
    )
    parser.add_argument(
        "--train-frac",
        default=os.environ.get("HUMANEVAL_TRAIN_FRAC", "0.8"),
    )
    parser.add_argument(
        "--teacher-forcing-seed",
        default=os.environ.get("HUMANEVAL_SPLIT_SEED", "65320"),
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_HUB_TOKEN", "")),
        help="Optional Hugging Face Hub token for gated model downloads.",
    )
    parser.add_argument(
        "--root-volume-gib",
        type=int,
        default=1500,
        help="Root EBS volume size in GiB.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id (default: timestamp + random suffix).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the tarball + render user-data but do NOT call RunInstances.",
    )
    parser.add_argument(
        "--no-shutdown-on-exit",
        action="store_true",
        help="Keep the instance running after the experiment for debugging.",
    )
    parser.add_argument(
        "--name-tag",
        default="pruning-metrics-qwen2-72b",
        help="Value for the ``Name`` tag.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Repo packaging
# ---------------------------------------------------------------------------


def build_repo_tarball(repo_root: Path) -> bytes:
    """Tar the repository, omitting build/cache directories.

    Returns
    -------
    bytes
        Gzipped tarball as raw bytes ready to upload to S3.
    """

    def _filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(tarinfo.name).parts
        if any(segment in TAR_EXCLUDES for segment in parts):
            return None
        if tarinfo.name.endswith((".pyc", ".pyo", ".log")):
            return None
        return tarinfo

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for entry in sorted(repo_root.iterdir()):
            tar.add(entry, arcname=entry.name, filter=_filter)
    return buffer.getvalue()


def upload_tarball(
    s3_client: Any, bucket: str, key: str, payload: bytes
) -> None:
    """Upload the tarball to S3."""

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="application/gzip",
    )


# ---------------------------------------------------------------------------
# AMI resolution
# ---------------------------------------------------------------------------


def resolve_dlami(ssm_client: Any) -> str:
    """Resolve the Deep Learning AMI id via SSM Parameter Store.

    The parameter list is searched in priority order: newest PyTorch + Ubuntu
    revision first, then progressively older fallbacks, then the base GPU
    image as a last resort.
    """

    last_error: Exception | None = None
    for parameter_name in DLAMI_PARAMETERS_PRIORITY:
        try:
            response = ssm_client.get_parameter(Name=parameter_name)
            ami_id = response["Parameter"]["Value"]
            print(
                f"Resolved {parameter_name} -> {ami_id}",
                file=sys.stderr,
            )
            return ami_id
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ParameterNotFound":
                last_error = exc
                continue
            raise
    raise RuntimeError(
        "No Deep Learning AMI parameter resolved in this region; tried: "
        + ", ".join(DLAMI_PARAMETERS_PRIORITY)
        + (f" (last error: {last_error})" if last_error else "")
    )


# ---------------------------------------------------------------------------
# User-data rendering
# ---------------------------------------------------------------------------


def render_userdata(
    template_path: Path,
    *,
    results_bucket: str,
    repo_tarball_key: str,
    results_prefix: str,
    run_id: str,
    base_model_id: str,
    pruning_levels: str,
    split_seed: str,
    train_frac: str,
    tf_seed: str,
    hf_token: str,
    shutdown_on_exit: bool,
) -> str:
    """Substitute template placeholders in the user-data script."""

    text = template_path.read_text(encoding="utf-8")
    replacements = {
        "__RESULTS_BUCKET__": results_bucket,
        "__REPO_TARBALL_KEY__": repo_tarball_key,
        "__RESULTS_PREFIX__": results_prefix,
        "__RUN_ID__": run_id,
        "__BASE_MODEL_ID__": base_model_id,
        "__PRUNING_LEVELS__": pruning_levels,
        "__SPLIT_SEED__": split_seed,
        "__TRAIN_FRAC__": train_frac,
        "__TF_SEED__": tf_seed,
        "__HF_TOKEN__": hf_token,
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    if not shutdown_on_exit:
        text = "SHUTDOWN_ON_EXIT=no\n" + text
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _default_run_id() -> str:
    """Generate a timestamped, partially-random run id."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def main() -> int:
    """CLI entry point: build tarball, upload, RunInstances."""

    args = parse_args()
    if not args.results_bucket:
        raise SystemExit("--results-bucket (or RESULTS_BUCKET env) is required.")

    run_id = args.run_id or _default_run_id()
    results_prefix = args.results_prefix.strip("/")
    repo_tarball_key = f"{results_prefix}/{run_id}/code/repo.tar.gz"

    print(f"Run ID: {run_id}", file=sys.stderr)
    print(f"Building tarball from {REPO_ROOT}", file=sys.stderr)
    tarball = build_repo_tarball(REPO_ROOT)
    print(f"Tarball size: {len(tarball)/1024/1024:.1f} MiB", file=sys.stderr)

    s3 = boto3.client("s3", region_name=args.region)
    print(
        f"Uploading tarball to s3://{args.results_bucket}/{repo_tarball_key}",
        file=sys.stderr,
    )
    upload_tarball(s3, args.results_bucket, repo_tarball_key, tarball)

    ssm = boto3.client("ssm", region_name=args.region)
    ami_id = resolve_dlami(ssm)
    print(f"DLAMI ID in {args.region}: {ami_id}", file=sys.stderr)

    userdata = render_userdata(
        USERDATA_TEMPLATE,
        results_bucket=args.results_bucket,
        repo_tarball_key=repo_tarball_key,
        results_prefix=results_prefix,
        run_id=run_id,
        base_model_id=args.base_model_id,
        pruning_levels=args.pruning_levels,
        split_seed=str(args.split_seed),
        train_frac=str(args.train_frac),
        tf_seed=str(args.teacher_forcing_seed),
        hf_token=args.hf_token,
        shutdown_on_exit=not args.no_shutdown_on_exit,
    )
    encoded_userdata = base64.b64encode(userdata.encode("utf-8")).decode("ascii")

    plan = {
        "run_id": run_id,
        "region": args.region,
        "availability_zone": args.availability_zone,
        "instance_type": args.instance_type,
        "max_spot_price": args.max_spot_price,
        "results_bucket": args.results_bucket,
        "results_prefix": results_prefix,
        "repo_tarball_key": repo_tarball_key,
        "ami_id": ami_id,
        "instance_profile": args.instance_profile,
        "shutdown_on_exit": not args.no_shutdown_on_exit,
    }
    print(json.dumps(plan, indent=2))

    if args.dry_run:
        print("--dry-run set; skipping RunInstances.", file=sys.stderr)
        (REPO_ROOT / "infra" / "ec2" / "_last_userdata.sh").write_text(
            userdata, encoding="utf-8"
        )
        return 0

    ec2 = boto3.client("ec2", region_name=args.region)
    print("Calling RunInstances ...", file=sys.stderr)
    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=args.instance_type,
        MinCount=1,
        MaxCount=1,
        Placement={"AvailabilityZone": args.availability_zone},
        IamInstanceProfile={"Name": args.instance_profile},
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": args.root_volume_gib,
                    "VolumeType": "gp3",
                    "Iops": 6000,
                    "Throughput": 500,
                    "DeleteOnTermination": True,
                },
            }
        ],
        InstanceMarketOptions={
            "MarketType": "spot",
            "SpotOptions": {
                "MaxPrice": f"{args.max_spot_price:.4f}",
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        },
        InstanceInitiatedShutdownBehavior="terminate",
        UserData=encoded_userdata,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": args.name_tag},
                    {"Key": "RunId", "Value": run_id},
                    {"Key": "Project", "Value": "pruning-metrics"},
                ],
            }
        ],
        MetadataOptions={
            "HttpTokens": "required",
            "HttpEndpoint": "enabled",
            "HttpPutResponseHopLimit": 2,
        },
    )

    instance_id = response["Instances"][0]["InstanceId"]
    plan["instance_id"] = instance_id

    # Quick wait until pending -> running so we can surface fast failures.
    waiter = ec2.get_waiter("instance_running")
    try:
        waiter.wait(
            InstanceIds=[instance_id],
            WaiterConfig={"Delay": 15, "MaxAttempts": 40},
        )
        plan["state"] = "running"
    except Exception as exc:  # pylint: disable=broad-exception-caught
        plan["state"] = f"wait_failed: {exc}"

    description = ec2.describe_instances(InstanceIds=[instance_id])
    instance = description["Reservations"][0]["Instances"][0]
    plan["public_dns"] = instance.get("PublicDnsName")
    plan["public_ip"] = instance.get("PublicIpAddress")
    plan["private_ip"] = instance.get("PrivateIpAddress")
    plan["launched_at_utc"] = datetime.now(timezone.utc).isoformat()

    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
