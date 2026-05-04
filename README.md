# pruning-metrics

Tooling for studying how pruning a large language model degrades its
behaviour, framed as a metric space over pruning scenarios. The project
turns "prune at X% sparsity, evaluate" into a reproducible, seeded,
notebook-driven workflow that runs on EC2 spot GPUs and persists every
artifact to S3.

## What's in here

* **Four orchestration notebooks** in [`notebooks/`](notebooks/) that
  run the experiment end-to-end without writing any boto3 / argparse
  scaffolding yourself:

  | Notebook | Purpose |
  |----------|---------|
  | [`01_setup_aws.ipynb`](notebooks/01_setup_aws.ipynb) | One-time idempotent AWS bootstrap (S3 bucket + IAM role + instance profile). |
  | [`02_prune_llm.ipynb`](notebooks/02_prune_llm.ipynb) | Compute and upload WANDA calibration stats for any HF causal LM and any task adapter. |
  | [`03_freeform_eval.ipynb`](notebooks/03_freeform_eval.ipynb) | Free-form (no teacher forcing) per-pruning-level evaluation on a chosen test set. |
  | [`04_teacher_forced.ipynb`](notebooks/04_teacher_forced.ipynb) | Teacher-forced next-token log-probabilities for seeded test samples per pruning level. |

* **Pluggable task adapters** under
  [`src/pruning_metrics/evals/tasks/`](src/pruning_metrics/evals/tasks/)
  for HumanEval+ (coding subprocess pass@1), GSM8K (numeric-answer math),
  and ARC-Challenge (regex letter MCQ). Adding a new task is one file +
  one registry entry; see [`docs/tasks.md`](docs/tasks.md).

* **EC2 spot GPU runners** under [`infra/ec2/`](infra/ec2/): a tarball
  launcher, a runner-agnostic user-data bootstrap, three narrow runner
  scripts, and shared WANDA / S3 helpers. Each runner runs unattended on
  an EC2 box and self-terminates when finished.

## Documentation entry points

* **[`docs/getting_started.md`](docs/getting_started.md)** -- run your
  first sweep in 30-60 minutes (small or large model).
* **[`docs/architecture.md`](docs/architecture.md)** -- system overview
  with diagrams, repo layout, determinism contract, cost reference.
* **[`docs/tasks.md`](docs/tasks.md)** -- how to add a new task type.
* **[`infra/ec2/README.md`](infra/ec2/README.md)** -- operator runbook
  (capacity probes, SSM Session Manager, spot interruption recovery).

## Quickstart

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install nbformat ipykernel matplotlib pandas

# 2. configure
cp template.env .env
$EDITOR .env  # AWS_PROFILE, AWS_ACCOUNT_ID, RESULTS_BUCKET, BASE_MODEL_ID

# 3. run notebooks 1 -> 4 in order
jupyter lab notebooks/
```

The workstation does not need a GPU or `torch`. Every GPU operation runs
on a spot EC2 instance launched by the notebooks.

## Canonical reference run

A successful Qwen2-72B sweep is preserved at:

```
s3://pruning-metrics-results-414266451290/qwen2_72b_pruning/20260504T001802Z-f041ba/
```

With pass@1 by sparsity (HumanEval+ test split, seed 65320, 33 tasks):

| Sparsity | pass@1 | Teacher-forced perplexity (HumanEval/137, 49 tokens) |
|---------:|-------:|-----------------------------------------------------:|
| 0%       | 0.273  | 1.638 |
| 20%      | 0.242  | 1.631 |
| 40%      | 0.121  | 1.669 |
| 60%      | 0.121  | 1.609 |
| 80%      | 0.000  | 5.135 |

## Native dataset splits

Math (GSM8K) and MCQ (ARC-Challenge) adapters use the dataset's native
``train`` and ``test`` Hub splits by default (GSM8K ``main`` has both; it
does not define a separate ``validation`` split, and nothing in this repo
requires one). The seed (`SPLIT_SEED`) is only used to reproducibly
truncate when `MAX_CALIBRATION_SAMPLES` caps the native train split.
Coding (HumanEval+) ships a single ``test`` split, so its calibration
partition still comes from a seeded 80/20 fallback. See
[`docs/tasks.md`](docs/tasks.md) for the spec grammar.
