# Getting started

A new user on a fresh machine should be able to land on this page and run
their first pruning sweep in under 30 minutes (small model) or ~1 hour
(Qwen2-72B). The four numbered notebooks in [`notebooks/`](../notebooks/)
do all of the heavy orchestration. This page tells you how to get them
ready to run.

## 1. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install nbformat nbclient ipykernel matplotlib pandas
```

`torch` is **not** required on the workstation; everything that touches a
GPU runs on EC2. If you do install `torch` locally the additional unit
tests for teacher forcing and per-row WANDA will be picked up by `pytest`.

## 2. Configure AWS credentials

The notebooks default to `AWS_PROFILE=rengz`; point them at any profile
with the permissions described in
[`infra/aws/iam/README.md`](../infra/aws/iam/README.md). Verify with:

```bash
AWS_PROFILE=<profile> aws sts get-caller-identity --region us-east-1
```

If you use SSO, run `aws sso login --profile <profile>` first. If you
prefer environment variables, set `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` (and `AWS_REGION`) before
launching Jupyter.

## 3. Copy `.env`

```bash
cp template.env .env
$EDITOR .env  # set AWS_ACCOUNT_ID, RESULTS_BUCKET, BASE_MODEL_ID, etc.
```

The notebooks call `python-dotenv`'s `load_dotenv()` so anything you set
in `.env` is visible to the kernel. `.env` is git-ignored.

## 4. Run notebook 1 (one-time)

`notebooks/01_setup_aws.ipynb` is idempotent. It:

* creates the S3 bucket if missing,
* creates the IAM role + instance profile (`pruning-metrics-ec2`),
* runs `STS GetCallerIdentity`, `S3 HeadBucket`, `IAM GetRole`, `IAM
  GetInstanceProfile` to confirm everything resolves.

Re-running it is a no-op for existing resources, so it is safe to leave
in your habit list at the start of every session.

## 5. Run notebook 2 to produce a pruning artifact

```python
# Inside the configuration cell of 02_prune_llm.ipynb
BASE_MODEL_ID = "Qwen/Qwen2-72B"             # or any HF causal LM
CALIBRATION_DATASET_SPEC = "coding:evalplus/humanevalplus:test"
PRUNING_LEVELS = [0, 20, 40, 60, 80]
SPLIT_SEED = 65320
TRAIN_FRAC = 0.8
INSTANCE_TYPE_PRIORITY = ["p4de.24xlarge", "p5.48xlarge", "p4d.24xlarge"]
```

Run all cells in order. The notebook prints `PRUNING_ARTIFACT_URI` at the
bottom -- copy it. The same artifact can drive arbitrarily many downstream
evaluations.

For a smoke test, swap `BASE_MODEL_ID` for `Qwen/Qwen2-1.5B-Instruct`,
set `INSTANCE_TYPE_PRIORITY = ["g5.xlarge"]`, and reduce
`MAX_CALIBRATION_SAMPLES` to `8`. Total spend is well under $1.

## 6. Notebook 3 -- free-form evaluation

Paste the artifact URI into the configuration cell along with an
`EVAL_DATASET_SPEC` (may be different from calibration, e.g. calibrate on
coding, evaluate on math) and `EVAL_LEVELS` (a subset of the manifest's
`pruning_levels`). The notebook displays a `pass@1` (or accuracy) table
and a sparsity-vs-metric line plot.

```python
PRUNING_ARTIFACT_URI = "s3://...<copy-from-notebook-2>"
TASK_TYPE = "coding"          # "coding" | "math" | "mcq"
EVAL_DATASET_SPEC = ""        # empty -> reuse calibration spec
EVAL_LEVELS = [0, 20, 40, 60, 80]
```

## 7. Notebook 4 -- teacher-forced log-probs

Same artifact, same dataset spec shape. Picks `NUM_TF_SAMPLES` records
from the test split using `TF_SEED`, runs one teacher-forced forward pass
per record per level. Output is a per-token table (target token,
log-prob, rank, top-`TF_TOP_K` alternatives) plus a per-level summary.

## When things go wrong

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `InsufficientInstanceCapacity` | spot capacity flips minute-to-minute | re-run the `find_capacity` cell, retry with the next candidate |
| `ExpiredToken` mid-experiment | SSO session lapsed | the EC2 box keeps running on its own instance profile; just refresh `rengz` and re-poll |
| pip install errors on first cell | `torch` not pinned and the workstation conflicts | install nothing locally; the notebooks delegate compute to EC2 |
| `Qwen2-72B is gated` | HF gating | set `HF_TOKEN` in `.env` so the launcher passes it to the GPU box |

## Where to look next

* [`architecture.md`](architecture.md): system overview, repo layout,
  determinism contract.
* [`tasks.md`](tasks.md): how to add a new task type (math word problems
  beyond GSM8K, classification, dialog quality, etc.).
* [`infra/ec2/README.md`](../infra/ec2/README.md): operator runbook for
  monitoring, SSM Session Manager access, spot interruption recovery.
