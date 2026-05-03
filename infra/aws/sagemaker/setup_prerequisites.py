"""Verify and bootstrap AWS resources for the SageMaker pruning notebook workflow.

Loads ``.env`` from the repository root via python-dotenv. Reads shared fields from
``config.py`` in this directory (no ``PYTHONPATH`` setup required for that module).

Run from the repository root so relative paths in documentation and ``REPO_ROOT``
resolution match your checkout layout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Repository root: .../pruning-metrics/infra/aws/sagemaker/this_file.py -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _sagemaker_infra_config_cls() -> type:
    """Load ``SageMakerInfraConfig`` from sibling ``config.py`` without ``sys.path`` hacks."""

    mod_name = "_pruning_metrics_sm_setup_config"
    config_path = Path(__file__).resolve().parent / "config.py"
    spec = importlib.util.spec_from_file_location(mod_name, config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses (and similar) resolve ``__module__`` on Python 3.12+.
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    cls = getattr(module, "SageMakerInfraConfig", None)
    if cls is None:
        raise RuntimeError("SageMakerInfraConfig missing from config module.")
    return cls


def _load_dotenv() -> None:
    """Load ``.env`` from the repository root if present."""

    env_path = REPO_ROOT / ".env"
    load_dotenv(dotenv_path=env_path, encoding="utf-8")


def _require_config(cfg: Any) -> None:
    """Raise if required fields needed for S3/SageMaker/ECR steps are empty."""

    missing: list[str] = []
    if not (cfg.region or "").strip():
        missing.append("AWS_REGION or AWS_DEFAULT_REGION")
    if not (cfg.role_arn or "").strip():
        missing.append("SAGEMAKER_ROLE_ARN")
    if not (cfg.artifact_bucket or "").strip():
        missing.append("PRUNING_ARTIFACT_BUCKET")
    if not (cfg.logits_bucket or "").strip():
        missing.append("PRUNING_LOGITS_BUCKET")
    if missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy template.env to .env at the repository root and fill values."
        )


def _ecr_repository_name() -> str:
    name = os.environ.get("ECR_REPOSITORY_NAME", "").strip()
    if not name:
        raise SystemExit(
            "ECR_REPOSITORY_NAME is not set. Add it to .env (see template.env)."
        )
    return name


def _account_id_expected() -> str | None:
    raw = os.environ.get("AWS_ACCOUNT_ID", "").strip()
    return raw or None


def _role_name_from_arn(role_arn: str) -> str:
    """Return IAM API ``RoleName`` segment from a role ARN."""

    if ":role/" not in role_arn:
        raise ValueError(f"Not a valid IAM role ARN: {role_arn!r}")
    return role_arn.split(":role/", 1)[-1]


def cmd_verify(_args: argparse.Namespace) -> int:
    """Check STS identity, buckets, ECR repo, and optional SageMaker execution role."""

    _load_dotenv()
    cfg_cls = _sagemaker_infra_config_cls()
    cfg = cfg_cls.from_env()
    _require_config(cfg)
    ecr_name = _ecr_repository_name()

    sts = boto3.client("sts", region_name=cfg.region)
    ident = sts.get_caller_identity()
    print("STS caller identity:")
    print(json.dumps(ident, indent=2, default=str))

    expected_account = _account_id_expected()
    if expected_account and ident.get("Account") != expected_account:
        print(
            f"ERROR: AWS_ACCOUNT_ID in .env ({expected_account}) does not match "
            f"caller Account ({ident.get('Account')}).",
            file=sys.stderr,
        )
        return 1

    s3 = boto3.client("s3", region_name=cfg.region)
    for bucket in sorted({cfg.artifact_bucket, cfg.logits_bucket}):
        try:
            s3.head_bucket(Bucket=bucket)
            print(f"OK S3 head-bucket: {bucket}")
        except ClientError as exc:
            print(f"ERROR S3 head-bucket failed for {bucket}: {exc}", file=sys.stderr)
            return 1

    ecr = boto3.client("ecr", region_name=cfg.region)
    try:
        ecr.describe_repositories(repositoryNames=[ecr_name])
        print(f"OK ECR repository exists: {ecr_name}")
    except ClientError as exc:
        print(
            f"ERROR ECR describe-repositories failed for {ecr_name}: {exc}",
            file=sys.stderr,
        )
        return 1

    iam = boto3.client("iam")
    role_name = _role_name_from_arn(cfg.role_arn)
    try:
        iam.get_role(RoleName=role_name)
        print(f"OK IAM get-role: {role_name}")
    except ClientError as exc:
        print(
            f"WARNING: IAM get-role failed for {role_name!r} ({exc}). "
            "Deploy may still work if your credentials lack iam:GetRole.",
            file=sys.stderr,
        )

    return 0


def _ensure_s3_bucket(s3: Any, bucket: str, region: str) -> None:
    """Create bucket if missing (us-east-1 has no LocationConstraint)."""

    try:
        s3.head_bucket(Bucket=bucket)
        print(f"S3 bucket already exists: {bucket}")
        return
    except ClientError:
        pass

    print(f"Creating S3 bucket: {bucket}")
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"S3 bucket already present (concurrent create or owned): {bucket}")
            return
        raise


def cmd_ensure_s3(_args: argparse.Namespace) -> int:
    """Ensure artifact and logits buckets exist."""

    _load_dotenv()
    cfg_cls = _sagemaker_infra_config_cls()
    cfg = cfg_cls.from_env()
    _require_config(cfg)
    s3 = boto3.client("s3", region_name=cfg.region)
    for bucket in sorted({cfg.artifact_bucket, cfg.logits_bucket}):
        try:
            _ensure_s3_bucket(s3, bucket, cfg.region)
        except ClientError as exc:
            print(f"ERROR ensuring bucket {bucket}: {exc}", file=sys.stderr)
            return 1
    return 0


def cmd_ensure_ecr(_args: argparse.Namespace) -> int:
    """Create ECR repository if it does not exist."""

    _load_dotenv()
    cfg_cls = _sagemaker_infra_config_cls()
    cfg = cfg_cls.from_env()
    _require_config(cfg)
    ecr_name = _ecr_repository_name()
    ecr = boto3.client("ecr", region_name=cfg.region)
    try:
        ecr.describe_repositories(repositoryNames=[ecr_name])
        print(f"ECR repository already exists: {ecr_name}")
        return 0
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "RepositoryNotFoundException":
            print(f"ERROR describing ECR repo: {exc}", file=sys.stderr)
            return 1

    print(f"Creating ECR repository: {ecr_name}")
    try:
        ecr.create_repository(repositoryName=ecr_name)
    except ClientError as exc:
        print(f"ERROR creating ECR repo: {exc}", file=sys.stderr)
        return 1
    return 0


def _policy_template_path(name: str) -> Path:
    return REPO_ROOT / "infra" / "aws" / "iam" / "policies" / name


def _substitute_policy_placeholders(document: str) -> str:
    """Fill placeholders from environment (caller must have called _load_dotenv)."""

    cfg_cls = _sagemaker_infra_config_cls()
    cfg = cfg_cls.from_env()
    _require_config(cfg)
    account = _account_id_expected()
    if not account:
        raise SystemExit(
            "AWS_ACCOUNT_ID must be set in .env to render IAM policy ARNs."
        )
    ecr_name = _ecr_repository_name()

    replacements = {
        "__ACCOUNT_ID__": account,
        "__REGION__": cfg.region,
        "__ARTIFACT_BUCKET__": cfg.artifact_bucket,
        "__LOGITS_BUCKET__": cfg.logits_bucket,
        "__ARTIFACT_PREFIX__": cfg.artifact_prefix.strip("/"),
        "__LOGITS_PREFIX__": cfg.logits_prefix.strip("/"),
        "__SAGEMAKER_ROLE_ARN__": cfg.role_arn,
        "__ECR_REPOSITORY_NAME__": ecr_name,
    }
    out = document
    for key, val in replacements.items():
        out = out.replace(key, val)
    return out


def cmd_iam_print(_args: argparse.Namespace) -> int:
    """Print IAM policy JSON with bucket/role ARNs substituted (no AWS IAM API calls)."""

    _load_dotenv()
    operator_path = _policy_template_path("notebook-operator-policy.json")
    execution_path = _policy_template_path("sagemaker-execution-policy.json")
    for label, path in (
        ("notebook_operator", operator_path),
        ("sagemaker_execution", execution_path),
    ):
        if not path.is_file():
            print(f"ERROR: missing policy template: {path}", file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8")
        rendered = _substitute_policy_placeholders(text)
        json.loads(rendered)  # validate
        print(f"--- {label} ({path.name}) ---")
        print(rendered)
        print()
    return 0


def cmd_setup(_args: argparse.Namespace) -> int:
    """Ensure S3 buckets and ECR repository, then verify."""

    for cmd in (cmd_ensure_s3, cmd_ensure_ecr, cmd_verify):
        code = cmd(_args)
        if code != 0:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="AWS prerequisites for SageMaker pruning notebooks and scripts."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("verify", help="Check STS, S3 buckets, ECR repo, and IAM role.")
    sub.add_parser(
        "ensure-s3-buckets",
        help="Create artifact and logits buckets if they do not exist.",
    )
    sub.add_parser(
        "ensure-ecr-repo",
        help="Create the configured ECR repository if it does not exist.",
    )
    sub.add_parser(
        "iam-print",
        help="Print rendered IAM policy JSON (substitute .env); no AWS calls.",
    )
    sub.add_parser(
        "setup",
        help="Ensure S3 + ECR then verify (common first-time bootstrap).",
    )
    return parser


def main() -> None:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()
    handlers = {
        "verify": cmd_verify,
        "ensure-s3-buckets": cmd_ensure_s3,
        "ensure-ecr-repo": cmd_ensure_ecr,
        "iam-print": cmd_iam_print,
        "setup": cmd_setup,
    }
    code = handlers[args.command](args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
