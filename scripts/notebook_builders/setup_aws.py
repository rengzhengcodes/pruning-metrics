"""Builder for 01_setup_aws.ipynb: one-time AWS bootstrap notebook."""

from ._shared import COMMON_BOOTSTRAP_CELL, NOTEBOOKS, _code, _md, write_notebook

# ---------------------------------------------------------------------------
# Notebook 1: 01_setup_aws.ipynb
# ---------------------------------------------------------------------------


def build_notebook_01() -> None:
    cells = [
        _md("""
            # 01 - One-time AWS bootstrap

            Idempotent setup for the pruning-metrics workflow. Run this notebook
            once per AWS account / region. It creates (or refreshes):

            * an S3 bucket for code tarballs, pruning calibration artifacts,
              free-form eval results, and teacher-forced records;
            * an IAM role (`pruning-metrics-ec2`) trusted by `ec2.amazonaws.com`
              with scoped S3 + SSM Session Manager + CloudWatch agent
              permissions;
            * the matching IAM instance profile (same name).

            Re-running is a no-op for existing resources. Notebook 2 onwards
            assume these resources exist.
            """),
        _code(COMMON_BOOTSTRAP_CELL),
        _md("""
            ## Configuration

            All knobs live in this cell. The defaults match what the rest of
            the project expects, so most users only edit `RESULTS_BUCKET`
            (must be globally unique, follow S3 naming rules).
            """),
        _code("""
            AWS_PROFILE = os.environ.get("AWS_PROFILE", "rengz")
            AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
            ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "414266451290")
            RESULTS_BUCKET = os.environ.get(
                "RESULTS_BUCKET", f"pruning-metrics-results-{ACCOUNT_ID}"
            )
            EC2_INSTANCE_ROLE_NAME = os.environ.get(
                "EC2_INSTANCE_ROLE_NAME", "pruning-metrics-ec2"
            )
            print({
                "AWS_PROFILE": AWS_PROFILE,
                "AWS_REGION": AWS_REGION,
                "ACCOUNT_ID": ACCOUNT_ID,
                "RESULTS_BUCKET": RESULTS_BUCKET,
                "EC2_INSTANCE_ROLE_NAME": EC2_INSTANCE_ROLE_NAME,
            })
            """),
        _md("""
            ## Verify AWS credentials

            STS `GetCallerIdentity` confirms the kernel can reach AWS APIs and
            displays the assumed identity (the SSO role for `rengz`).
            """),
        _code("""
            import json
            import boto3

            session = boto3.session.Session(profile_name=AWS_PROFILE)
            sts = session.client("sts", region_name=AWS_REGION)
            identity = sts.get_caller_identity()
            print(json.dumps(identity, indent=2, default=str))
            assert identity["Account"] == ACCOUNT_ID, (
                f"Expected account {ACCOUNT_ID}, got {identity['Account']}"
            )
            """),
        _md("""
            ## Run the bootstrap script

            `infra/provisioning/bootstrap_ec2_resources.py` is idempotent. It
            checks for the bucket / role / profile and creates anything
            missing. Output below is a single JSON summary.
            """),
        _code("""
            import subprocess

            cmd = [
                sys.executable,
                str(REPO_ROOT / "infra" / "provisioning" / "bootstrap_ec2_resources.py"),
                "--bucket", RESULTS_BUCKET,
                "--region", AWS_REGION,
                "--role-name", EC2_INSTANCE_ROLE_NAME,
            ]
            env = dict(os.environ)
            env["AWS_PROFILE"] = AWS_PROFILE
            completed = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
            print(completed.stdout)
            if completed.stderr:
                print("--- stderr ---")
                print(completed.stderr)
            """),
        _md("""
            ## Verify the resources

            Confirms the bucket exists, the role exists and is trusted by EC2,
            and the instance profile carries the role.
            """),
        _code("""
            s3 = session.client("s3", region_name=AWS_REGION)
            iam = session.client("iam")

            head = s3.head_bucket(Bucket=RESULTS_BUCKET)
            print("OK S3 bucket:", RESULTS_BUCKET)

            role = iam.get_role(RoleName=EC2_INSTANCE_ROLE_NAME)["Role"]
            print("OK IAM role:", role["Arn"])

            profile = iam.get_instance_profile(InstanceProfileName=EC2_INSTANCE_ROLE_NAME)["InstanceProfile"]
            attached_roles = [r["RoleName"] for r in profile.get("Roles", [])]
            assert EC2_INSTANCE_ROLE_NAME in attached_roles, attached_roles
            print("OK Instance profile:", profile["Arn"])
            """),
        _md("""
            ## Summary

            Bucket and instance profile are ready. Use the values above as
            inputs to notebooks 2/3/4. The instance profile name is also
            stored in `.env` as `EC2_INSTANCE_PROFILE_NAME` so the launch
            scripts pick it up automatically.
            """),
        _code("""
            print("Setup complete. Use these values in subsequent notebooks:")
            print(json.dumps({
                "AWS_PROFILE": AWS_PROFILE,
                "AWS_REGION": AWS_REGION,
                "RESULTS_BUCKET": RESULTS_BUCKET,
                "EC2_INSTANCE_PROFILE_NAME": EC2_INSTANCE_ROLE_NAME,
            }, indent=2))
            """),
    ]
    write_notebook(NOTEBOOKS / "01_setup_aws.ipynb", cells)
