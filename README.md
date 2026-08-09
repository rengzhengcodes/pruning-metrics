# pruning-metrics

Tooling for studying how pruning a large language model degrades its
behaviour, framed as a metric space over pruning scenarios: the project
turns "prune at X% sparsity, evaluate" into a reproducible, seeded,
notebook-driven workflow that runs on EC2 spot GPUs and persists every
artifact to S3.

## What's in here

* **Four orchestration notebooks** in
  [`notebooks/aws_tutorial/`](notebooks/aws_tutorial/) that run the
  experiment end-to-end without writing any boto3 / argparse scaffolding
  yourself:

  | Notebook | Purpose |
  |----------|---------|
  | [`01_setup_aws.ipynb`](notebooks/aws_tutorial/01_setup_aws.ipynb) | One-time idempotent AWS bootstrap (S3 bucket + IAM role + instance profile). |
  | [`02_prune_llm.ipynb`](notebooks/aws_tutorial/02_prune_llm.ipynb) | Compute and upload WANDA calibration stats for any HF causal LM and any task adapter. |
  | [`03_freeform_eval.ipynb`](notebooks/aws_tutorial/03_freeform_eval.ipynb) | Free-form (no teacher forcing) per-pruning-level evaluation on a chosen test set. |
  | [`04_teacher_forced.ipynb`](notebooks/aws_tutorial/04_teacher_forced.ipynb) | Teacher-forced next-token log-probabilities for seeded test samples per pruning level. |

* **A nine-notebook local analysis pipeline** in
  [`notebooks/experiment/`](notebooks/experiment/) that consumes the
  artifacts the AWS tutorial notebooks (and the v2 sweep runner) produce --
  pruning + evaluation collection, metric-space construction, t-SNE /
  reducer visualization, and the diagnosticity analysis behind this
  README's "Key finding". Runs locally against cached results; no AWS
  needed once the artifacts exist.

* **Pluggable task adapters** under
  [`src/pruning_metrics/evals/tasks/`](src/pruning_metrics/evals/tasks/)
  for HumanEval+ and MBPP+ (coding subprocess pass@1), GSM8K and MathQA
  (numeric-answer / MCQ math), and ARC-Challenge (regex letter MCQ).
  Adding a new task is one file + one registry entry; see
  [`docs/tasks.md`](docs/tasks.md).

* **Sixteen distributional distances and fourteen dimensionality
  reducers** in [`src/pruning_metrics/prob_measures/`](src/pruning_metrics/prob_measures/)
  and [`src/pruning_metrics/dim_reduction/`](src/pruning_metrics/dim_reduction/)
  respectively, one module per measure / reducer plus a registry each.
  These back notebooks 08 and 09 in `notebooks/experiment/`, which cross
  every distance against every reduction.

* **EC2 spot GPU runners** under [`infra/`](infra/): a tarball launcher
  and user-data bootstrap in [`infra/provisioning/`](infra/provisioning/),
  narrow runner scripts and shared WANDA / SparseGPT / S3 helpers in
  [`infra/runners/`](infra/runners/). Each runner runs unattended on an
  EC2 box and self-terminates when finished.

## Documentation entry points

* **[`docs/getting_started.md`](docs/getting_started.md)** -- run your
  first sweep in 30-60 minutes (small or large model).
* **[`docs/architecture.md`](docs/architecture.md)** -- system overview
  with diagrams, repo layout, determinism contract, cost reference.
* **[`docs/tasks.md`](docs/tasks.md)** -- how to add a new task type.
* **[`docs/methodology_review.md`](docs/methodology_review.md)** --
  adversarial review of the methodology, plus the v2 experiment (§6)
  that empirically tested it.
* **[`infra/README.md`](infra/README.md)** -- operator runbook
  (capacity probes, SSM Session Manager, spot interruption recovery).

## Key finding: behavioral distances are diagnostic of parameter-level structure

An adversarial review
([`docs/methodology_review.md`](docs/methodology_review.md)) initially
judged the v1 pipeline (13 WANDA-pruned Qwen2-72B variants, visual t-SNE
clustering) unable to support any internal-structure claim. The v2
experiment (2 pruners x 5 calibration domains x 3 seeds x 8 sparsity
levels on Qwen2-7B, 232 variants) reversed the review's central finding:
teacher-forced output-distribution distances track pruning-mask Jaccard
overlap at r = +0.75 to +0.83 at matched sparsity (all 20 benchmark x
metric combos at the permutation floor), and WANDA variants cluster by
calibration domain with ARI up to 1.0 within (pruner, level) strata.

Four changes produced the positive finding, in order of contribution:

1. **Parameter-level ground truth.** v2 saved every variant's pruning
   mask (1/32 digests) and computed pairwise mask-Jaccard distances --
   giving the behavioral matrices something measurable to be diagnostic
   *of*. v1 had no ground truth, so the question was untestable.
2. **Seed replicates and more domains.** 5 domains x 3 seeds yields 15
   points per (pruner, level) stratum -- enough for silhouette / ARI /
   permutation tests to have power. v1's ~4 points per domain was below
   any cluster-test floor.
3. **Stratifying by sparsity level (analysis, not design).** The
   sparsity pair explains ~94-97% of both distance matrices, so every
   pooled statistic is uninterpretable -- the pre-registered pooled
   analysis printed "MIXED". Holding (pruner, level) fixed revealed both
   signals. The finer 8-level sweep is what made stratification viable.
4. **Quantitative statistics replacing visual embeddings.** Mantel /
   restricted-permutation / ARI machinery instead of eyeballing t-SNE --
   necessary but not sufficient (the first pass used a suppressor
   control until adversarial verification corrected it).

Notably, the change motivated by the literature -- adding SparseGPT as
the "more calibration-sensitive" pruner -- did **not** contribute:
WANDA's masks turned out to be ~4x more domain-differentiated, and
SparseGPT is the weak, boundary-condition case. Full corrected analysis,
tables, and caveats: [`docs/methodology_review.md`](docs/methodology_review.md) §6.

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
jupyter lab notebooks/aws_tutorial/
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

Math and MCQ adapters default to native Hub train/test splits; coding
(HumanEval+) has only one split, so it falls back to a seeded 80/20
partition. `SPLIT_SEED` also reproducibly truncates the native train
split when `MAX_CALIBRATION_SAMPLES` caps it. See
[`docs/tasks.md`](docs/tasks.md#train-test-and-validation-splits) for
the full split contract and spec grammar.
