"""Launch an EC2 spot GPU instance and bootstrap a chosen runner.

The launcher is the same regardless of which of the four notebooks invokes
it: it tars the repo, uploads it to S3, resolves the latest Deep Learning
AMI, renders the user-data shell with the appropriate runner script path
plus runner-specific environment variables, and calls ``RunInstances`` with
the project's instance profile attached.

Supported ``--runner`` values:

* ``pruning_calibration`` -> ``infra/ec2/run_pruning_calibration.py``
* ``freeform_eval``       -> ``infra/ec2/run_freeform_eval.py``
* ``teacher_forced``      -> ``infra/ec2/run_teacher_forced.py``
* ``full_pipeline``       -> ``infra/ec2/run_qwen_pruning_experiment.py`` (legacy)

Runner-specific knobs are passed via ``--runner-env KEY=VALUE`` (repeatable)
or ``--runner-env-json '{"KEY": "VALUE"}'``. Each runner consumes those env
vars through its ``argparse`` defaults / ``env_or`` helper.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shlex
import sys
import tarfile
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
# excluded because it carries the operator's short-lived AWS credentials --
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

# Path of each runner relative to the repo root.
RUNNER_RELPATHS: dict[str, str] = {
    "pruning_calibration": "infra/ec2/run_pruning_calibration.py",
    "freeform_eval": "infra/ec2/run_freeform_eval.py",
    "teacher_forced": "infra/ec2/run_teacher_forced.py",
    "full_pipeline": "infra/ec2/run_qwen_pruning_experiment.py",
}


def parse_args() -> argparse.Namespace:
    """Parse launcher arguments."""

    parser = argparse.ArgumentParser(
        description="Launch a spot EC2 GPU box and bootstrap a runner script.",
    )
    parser.add_argument("--region", required=True)
    parser.add_argument("--availability-zone", required=True)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument(
        "--max-spot-price",
        required=True,
        type=float,
        help="Spot bid ceiling in USD/hour.",
    )
    parser.add_argument(
        "--runner",
        choices=sorted(RUNNER_RELPATHS),
        default="full_pipeline",
        help=(
            "Which runner the user-data should invoke. Default keeps the "
            "monolithic pipeline for back-compat."
        ),
    )
    parser.add_argument(
        "--runner-env",
        action="append",
        default=[],
        help=(
            "Repeatable ``KEY=VALUE`` env var passed to the runner. "
            "Combined with --runner-env-json (the JSON wins on conflict)."
        ),
    )
    parser.add_argument(
        "--runner-env-json",
        default="",
        help="JSON object of runner-specific env vars (overrides --runner-env).",
    )
    parser.add_argument(
        "--runner-cli-args",
        default="",
        help=(
            "Optional argv string appended verbatim to the runner invocation "
            "in user-data. Most callers should rely on --runner-env instead."
        ),
    )
    parser.add_argument(
        "--results-bucket",
        default=os.environ.get("RESULTS_BUCKET", ""),
        help="S3 bucket for code tarball, logs, and results.",
    )
    parser.add_argument(
        "--results-prefix",
        default=os.environ.get("RESULTS_PREFIX", "pruning_metrics"),
        help=(
            "S3 key prefix; the run id is appended automatically. Different "
            "runners use different default prefixes when called via the "
            "notebook helpers."
        ),
    )
    parser.add_argument(
        "--instance-profile",
        default=os.environ.get(
            "EC2_INSTANCE_PROFILE_NAME", "pruning-metrics-ec2"
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get(
            "HF_TOKEN", os.environ.get("HUGGINGFACE_HUB_TOKEN", "")
        ),
        help="Optional Hugging Face Hub token for gated model downloads.",
    )
    parser.add_argument(
        "--root-volume-gib",
        type=int,
        default=1500,
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id (default: UTC timestamp + random suffix).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-shutdown-on-exit", action="store_true")
    parser.add_argument("--name-tag", default="pruning-metrics-runner")

    # ----- back-compat aliases for the legacy ``full_pipeline`` runner -----
    # These are preserved so existing scripts / cells that pre-date the
    # generalisation keep working. They are folded into ``runner_env`` in
    # main() when --runner=full_pipeline is selected.
    parser.add_argument("--base-model-id", default=None)
    parser.add_argument("--pruning-levels", default=None)
    parser.add_argument("--split-seed", default=None)
    parser.add_argument("--train-frac", default=None)
    parser.add_argument("--teacher-forcing-seed", default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Repo packaging
# ---------------------------------------------------------------------------


def build_repo_tarball(repo_root: Path) -> bytes:
    """Tar the repository, omitting build/cache directories."""

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
    """Resolve the Deep Learning AMI id via SSM Parameter Store."""

    last_error: Exception | None = None
    for parameter_name in DLAMI_PARAMETERS_PRIORITY:
        try:
            response = ssm_client.get_parameter(Name=parameter_name)
            ami_id = response["Parameter"]["Value"]
            print(f"Resolved {parameter_name} -> {ami_id}", file=sys.stderr)
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
# Runner env handling
# ---------------------------------------------------------------------------


def collect_runner_env(args: argparse.Namespace) -> dict[str, str]:
    """Combine ``--runner-env`` flags + ``--runner-env-json``.

    ``--runner-env-json`` wins on conflict so notebook callers can build the
    canonical env dict in Python and forward it as a single argument.
    """

    env: dict[str, str] = {}
    for entry in args.runner_env:
        if "=" not in entry:
            raise ValueError(f"--runner-env entry must be KEY=VALUE: {entry!r}")
        key, value = entry.split("=", 1)
        env[key.strip()] = value
    if args.runner_env_json:
        parsed = json.loads(args.runner_env_json)
        if not isinstance(parsed, dict):
            raise ValueError("--runner-env-json must decode to a JSON object.")
        for key, value in parsed.items():
            env[str(key)] = "" if value is None else str(value)

    # Back-compat: fold legacy CLI flags into runner env when present.
    legacy_pairs = [
        ("BASE_MODEL_ID", args.base_model_id),
        ("PRUNING_LEVELS", args.pruning_levels),
        ("HUMANEVAL_SPLIT_SEED", args.split_seed),
        ("HUMANEVAL_TRAIN_FRAC", args.train_frac),
        # Both runner_freeform_eval / teacher_forced read TF_SEED / GENERATION_SEED;
        # the legacy monolithic runner uses HUMANEVAL_SPLIT_SEED for both.
        ("TF_SEED", args.teacher_forcing_seed),
    ]
    for key, value in legacy_pairs:
        if value is not None and key not in env:
            env[key] = str(value)

    return env


def render_runner_env_exports(env: dict[str, str]) -> str:
    """Render ``export KEY="VALUE"`` lines safe for inclusion in user-data."""

    lines: list[str] = []
    for key, value in env.items():
        if not key:
            continue
        lines.append(f"export {key}={shlex.quote(str(value))}")
    return "\n".join(lines)


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
    hf_token: str,
    runner_relpath: str,
    runner_env_exports: str,
    runner_cli_args: str,
    shutdown_on_exit: bool,
) -> str:
    """Substitute template placeholders in the user-data script."""

    text = template_path.read_text(encoding="utf-8")
    replacements = {
        "__RESULTS_BUCKET__": results_bucket,
        "__REPO_TARBALL_KEY__": repo_tarball_key,
        "__RESULTS_PREFIX__": results_prefix,
        "__RUN_ID__": run_id,
        "__HF_TOKEN__": hf_token,
        "__RUNNER_RELPATH__": runner_relpath,
        "__RUNNER_CLI_ARGS__": runner_cli_args,
        "__RUNNER_ENV_EXPORTS__": runner_env_exports,
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def main() -> int:
    """CLI entry point: build tarball, upload, render user-data, RunInstances."""

    args = parse_args()
    if not args.results_bucket:
        raise SystemExit("--results-bucket (or RESULTS_BUCKET env) is required.")

    runner_relpath = RUNNER_RELPATHS[args.runner]
    runner_env = collect_runner_env(args)
    runner_env_exports = render_runner_env_exports(runner_env)

    run_id = args.run_id or _default_run_id()
    results_prefix = args.results_prefix.strip("/")
    repo_tarball_key = f"{results_prefix}/{run_id}/code/repo.tar.gz"

    print(f"Run ID: {run_id}", file=sys.stderr)
    print(f"Runner: {args.runner} ({runner_relpath})", file=sys.stderr)
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
        hf_token=args.hf_token,
        runner_relpath=runner_relpath,
        runner_env_exports=runner_env_exports,
        runner_cli_args=args.runner_cli_args,
        shutdown_on_exit=not args.no_shutdown_on_exit,
    )
    encoded_userdata = base64.b64encode(userdata.encode("utf-8")).decode("ascii")

    plan = {
        "run_id": run_id,
        "runner": args.runner,
        "runner_relpath": runner_relpath,
        "runner_env": runner_env,
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
                    {"Key": "Runner", "Value": args.runner},
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
