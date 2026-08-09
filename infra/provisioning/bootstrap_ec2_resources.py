"""Idempotently bootstrap AWS resources for the EC2 GPU pruning workflow.

Creates (when missing):

* A versioned S3 bucket holding the code tarball, model artifacts, and
  experiment results.
* An IAM role trusted by ``ec2.amazonaws.com`` with the policies needed by
  the GPU box (S3 read/write to the bucket and SSM Session Manager).
* The matching IAM instance profile.

This script is safe to run multiple times: existing resources are detected
and updated rather than re-created. It is intended to be called from the
operator workstation while the ``rengz`` SSO session is still active.

Usage
-----
::

    python infra/provisioning/bootstrap_ec2_resources.py \\
        --bucket pruning-metrics-results-414266451290 \\
        --region us-east-1 \\
        --role-name pruning-metrics-ec2

Notes
-----
The IAM principal running this script needs ``s3:CreateBucket``,
``iam:CreateRole``, ``iam:PutRolePolicy``, ``iam:AttachRolePolicy``,
``iam:CreateInstanceProfile``, ``iam:AddRoleToInstanceProfile``, and
``iam:GetInstanceProfile`` permissions. The ``rengz@mit.edu`` SSO role
``IdP-admin-role`` already has admin scope.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

SSM_MANAGED_INSTANCE_CORE = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
CLOUDWATCH_AGENT_SERVER = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the bootstrap script.

    Returns
    -------
    argparse.Namespace
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(
        description="Bootstrap S3 + IAM for the EC2 GPU pruning experiment."
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket name to ensure (must be globally unique).",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region for the S3 bucket and IAM verification.",
    )
    parser.add_argument(
        "--role-name",
        default="pruning-metrics-ec2",
        help="IAM role + instance profile name for the GPU box.",
    )
    return parser.parse_args()


def ensure_bucket(s3_client: Any, bucket: str, region: str) -> None:
    """Create the S3 bucket if it does not already exist.

    Parameters
    ----------
    s3_client:
        Boto3 ``s3`` client tied to ``region``.
    bucket:
        Globally unique S3 bucket name.
    region:
        AWS region for ``CreateBucket`` configuration.
    """

    try:
        s3_client.head_bucket(Bucket=bucket)
        print(f"S3 bucket already exists: {bucket}")
        return
    except ClientError as exc:
        # Anything but "not found" (403 included) means we can't claim the
        # bucket as ours; propagate rather than trying to create over it.
        if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchBucket"):
            raise

    print(f"Creating S3 bucket {bucket} in {region}")
    if region == "us-east-1":
        s3_client.create_bucket(Bucket=bucket)
    else:
        s3_client.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": region},
        )

    s3_client.put_bucket_versioning(
        Bucket=bucket,
        VersioningConfiguration={"Status": "Enabled"},
    )
    s3_client.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    print(f"Bucket {bucket} versioned and locked from public access.")


def ensure_role_and_profile(
    iam_client: Any,
    role_name: str,
    bucket: str,
) -> str:
    """Create or update the EC2 IAM role and matching instance profile.

    Parameters
    ----------
    iam_client:
        Boto3 IAM client.
    role_name:
        IAM role and instance-profile name (kept identical for clarity).
    bucket:
        Bucket name scoped into the inline S3 access policy.

    Returns
    -------
    str
        ARN of the created or existing IAM role.
    """

    # Trust policy: only the EC2 service may assume this role. This is what
    # lets the instance profile hand temporary credentials to the GPU box.
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    mutated = False
    try:
        role = iam_client.get_role(RoleName=role_name)["Role"]
        print(f"IAM role already exists: {role['Arn']}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            raise
        role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="EC2 GPU box for pruning-metrics WANDA pipeline.",
        )["Role"]
        mutated = True
        print(f"Created IAM role {role['Arn']}")

    # Scope: exactly the actions the runner lifecycle needs, on one bucket.
    # - GetObject: pull the uploaded repo tarball at boot.
    # - PutObject: sync results and userdata logs back up.
    # - DeleteObject: clear stale artifacts when a run id is re-synced.
    # - ListBucket + GetBucketLocation: bucket-level calls used by
    #   `aws s3 sync`/`cp` (these apply to the bucket ARN, hence the
    #   two Resource entries).
    # - AbortMultipartUpload: large artifacts upload as multipart; abort
    #   stops interrupted uploads from accruing orphaned-part storage.
    inline_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ResultsBucketAccess",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                    "s3:GetBucketLocation",
                    "s3:AbortMultipartUpload",
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket}",
                    f"arn:aws:s3:::{bucket}/*",
                ],
            }
        ],
    }
    iam_client.put_role_policy(
        RoleName=role_name,
        PolicyName="ResultsBucketAccess",
        PolicyDocument=json.dumps(inline_policy),
    )
    print(f"Attached inline ResultsBucketAccess policy to {role_name}.")

    for managed_arn in (SSM_MANAGED_INSTANCE_CORE, CLOUDWATCH_AGENT_SERVER):
        try:
            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=managed_arn)
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "EntityAlreadyExists":
                raise
        print(f"Attached managed policy {managed_arn} to {role_name}.")

    profile_name = role_name
    try:
        profile_data = iam_client.get_instance_profile(
            InstanceProfileName=profile_name
        )["InstanceProfile"]
        print(f"Instance profile already exists: {profile_name}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            raise
        profile_data = iam_client.create_instance_profile(
            InstanceProfileName=profile_name
        )["InstanceProfile"]
        mutated = True
        print(f"Created instance profile {profile_name}.")

    if not any(
        attached_role["RoleName"] == role_name
        for attached_role in profile_data["Roles"]
    ):
        iam_client.add_role_to_instance_profile(
            InstanceProfileName=profile_name, RoleName=role_name
        )
        mutated = True
        print(f"Added role {role_name} to instance profile {profile_name}.")
    else:
        print(f"Role {role_name} already attached to profile {profile_name}.")

    if mutated:
        # IAM is eventually consistent — give freshly created/attached
        # resources a moment so RunInstances does not fail. No-op re-runs
        # skip the wait.
        time.sleep(5)
    return role["Arn"]


def main() -> int:
    """CLI entry point: bootstrap S3 + IAM resources."""

    args = parse_args()

    s3_client = boto3.client("s3", region_name=args.region)
    iam_client = boto3.client("iam")

    ensure_bucket(s3_client, args.bucket, args.region)
    role_arn = ensure_role_and_profile(iam_client, args.role_name, args.bucket)

    summary = {
        "bucket": args.bucket,
        "region": args.region,
        "role_name": args.role_name,
        "role_arn": role_arn,
        "instance_profile_name": args.role_name,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
