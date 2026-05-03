# IAM for the SageMaker pruning workflow

Two principals need permissions:

1. **Notebook operator** — the IAM user or role whose credentials run Jupyter, `setup_prerequisites.py`, and `aws` / `docker`. Attach or inline the JSON from [`policies/notebook-operator-policy.json`](policies/notebook-operator-policy.json) after substituting placeholders (or run `make -f infra/aws/Makefile iam-print` with a filled `.env` to print rendered JSON).
2. **SageMaker execution role** — the role ARN in `SAGEMAKER_ROLE_ARN`. SageMaker assumes this role to pull your container image, read model artifacts from S3, and (via the container) write logits to S3. Attach the JSON from [`policies/sagemaker-execution-policy.json`](policies/sagemaker-execution-policy.json) (rendered the same way).

## Trust policy (execution role only)

The execution role must trust the SageMaker service. In the IAM console, edit the role’s **Trust relationships** so `sagemaker.amazonaws.com` can assume the role. Example trust policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "sagemaker.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

## Applying policies

1. Copy [`template.env`](../../template.env) to `.env` at the repository root and set `AWS_ACCOUNT_ID`, `AWS_REGION`, bucket names, `SAGEMAKER_ROLE_ARN`, `ECR_REPOSITORY_NAME`, and prefixes.
2. Render policies: `make -f infra/aws/Makefile iam-print` (from repo root).
3. As an IAM administrator, create managed or inline policies from the printed JSON and attach them to the operator identity and to the SageMaker execution role respectively.

## `iam:PassRole`

The operator policy includes `iam:PassRole` scoped to `SAGEMAKER_ROLE_ARN` with `iam:PassedToService` = `sagemaker.amazonaws.com`. Without this, `CreateModel` / endpoint deployment fails when passing the execution role to SageMaker.

## Tightening scope

The sample operator policy allows broad SageMaker control-plane actions on `Resource: "*"` for readability. Restrict `sagemaker:InvokeEndpoint` to your endpoint ARN when you no longer need exploratory access.
