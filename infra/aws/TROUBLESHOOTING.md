# AWS CLI and credentials (notebook / scripts)

## `NoCredentials` / "Unable to locate credentials"

If `aws sts get-caller-identity` or `make -f infra/aws/Makefile verify` fails with **NoCredentials** or **Unable to locate credentials**, the AWS SDK did not find usable credentials in the environment where the command or Jupyter kernel runs.

**What the SDK checks:** Environment variables (`AWS_ACCESS_KEY_ID`, …), shared files under `~/.aws/credentials` and `~/.aws/config`, SSO cache for named profiles, and on EC2/SageMaker the instance metadata role.

**Fixes (pick what matches your org):**

1. **IAM Identity Center (SSO)** — Configure a named profile: `aws configure sso`. When the session expires: `aws sso login --profile <profile-name>`. Use that profile in Jupyter: set `AWS_PROFILE` in `.env` at the repo root (see `template.env`) or `export AWS_PROFILE=...` before starting Jupyter from the same shell.
2. **Access keys** — `aws configure` (default or named profile). Prefer short-lived keys where possible.
3. **Temporary session keys** — Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` in `.env` or the process environment.
4. **Terminal works, notebook does not** — The kernel often starts without your shell profile. Put `AWS_PROFILE` (and keys if needed) in **`.env`**, then restart the kernel and reload dotenv (this notebook’s imports cell).
5. **EC2 / SageMaker notebook instance** — Use an instance profile with the right policies. If you still see `NoCredentials`, confirm the metadata service is reachable.

After fixing, re-run:

```bash
make -f infra/aws/Makefile verify
```

See also [iam/README.md](iam/README.md) for attaching the operator and execution-role policies.
