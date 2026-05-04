# Qwen2-72B WANDA pruning on EC2 — operator runbook

This directory holds the end-to-end pipeline for the
**Qwen2-72B WANDA prune + HumanEval+ + teacher-forced log-probs** experiment.
The notebook-targeted SageMaker-endpoint workflow at
[`../aws/sagemaker/`](../aws/sagemaker/) cannot host a 72 B-parameter model in
this account (the relevant endpoint quotas are `0`); the EC2 spot-GPU path
documented here is the way the actual run executes.

The pipeline does everything inside **one** GPU process so the model is
materialized exactly once: prune → autoregressive HumanEval+ eval → teacher-
forced log-probs, repeated for each pruning level. Per-level artifacts are
written to S3 incrementally so a spot interruption never loses a completed
level.

## Files

| File | Purpose |
|------|---------|
| [`run_qwen_pruning_experiment.py`](run_qwen_pruning_experiment.py) | Single-GPU-box runner. Loads Qwen2-72B once, sweeps pruning levels, writes per-level results + a final `summary.json`. |
| [`find_capacity.py`](find_capacity.py) | Probes EC2 spot price + availability across regions/AZs for `p5.48xlarge` / `p4de.24xlarge` / `p4d.24xlarge` and prints viable candidates as JSON. |
| [`launch_gpu_instance.py`](launch_gpu_instance.py) | Tars the repo to S3, resolves the latest Deep Learning AMI, and calls `RunInstances` with the instance profile + spot market + user-data bootstrap. |
| [`userdata_bootstrap.sh`](userdata_bootstrap.sh) | Cloud-init script the launcher renders and attaches as user-data. Pulls the tarball, installs deps, runs the experiment, syncs results to S3, and self-shuts. |

The reusable WANDA + teacher-forcing code lives under
[`src/pruning_metrics/evals/coding/`](../../src/pruning_metrics/evals/coding/);
the EC2 runner imports it via `PYTHONPATH` injection so it works without
`pip install -e .`.

## One-time prerequisites

These are already done for the current experiment but listed for
reproducibility / a fresh account.

```bash
AWS_PROFILE=rengz python3 infra/aws/setup/bootstrap_ec2_resources.py \
    --bucket pruning-metrics-results-414266451290 \
    --region us-east-1 \
    --role-name pruning-metrics-ec2
```

This creates:

* an S3 bucket with versioning + public-access-block (`pruning-metrics-results-414266451290`);
* an IAM role + instance profile `pruning-metrics-ec2` trusted by EC2 with:
    * inline `ResultsBucketAccess` (S3 read/write to the bucket only),
    * managed `AmazonSSMManagedInstanceCore` (so SSM Session Manager works
      without an SSH key pair),
    * managed `CloudWatchAgentServerPolicy` (for future CloudWatch logs).

Re-running the script is a no-op for existing resources.

## Running the experiment

### 1. Find capacity

```bash
AWS_PROFILE=rengz python3 infra/ec2/find_capacity.py \
    --regions us-east-1,us-west-2,us-east-2 \
    --instance-types p5.48xlarge,p4de.24xlarge,p4d.24xlarge
```

Top candidate (lowest priority index, then lowest spot price) is what you want.
Spot for `p5.48xlarge` in `us-east-1b` was ~$12.66/hr at the time of writing.

### 2. Launch the spot box

```bash
AWS_PROFILE=rengz python3 infra/ec2/launch_gpu_instance.py \
    --region us-east-1 \
    --availability-zone us-east-1b \
    --instance-type p5.48xlarge \
    --max-spot-price 31.65 \
    --results-bucket pruning-metrics-results-414266451290
```

The launcher:

1. Tars the repository (excluding `.git`, `.venv`, `.env`, caches) and uploads
   it to `s3://<bucket>/<prefix>/<run_id>/code/repo.tar.gz`.
2. Resolves the latest **Deep Learning OSS Nvidia Driver GPU PyTorch** AMI
   via SSM Parameter Store (Ubuntu 24.04 / PyTorch 2.10 first, with older
   Ubuntu 22.04 fallbacks).
3. Calls `RunInstances` with:
    * the `pruning-metrics-ec2` instance profile attached,
    * a 1.5 TiB gp3 root volume (Qwen2-72B is ~145 GiB; we need slack for HF
      cache, the WANDA snapshot on disk, and per-level outputs),
    * `InstanceMarketOptions=spot`, `MaxPrice=<bid>`, `one-time`,
    * `InstanceInitiatedShutdownBehavior=terminate` (so the
      `shutdown -h now` at end-of-run frees the spot reservation),
    * the rendered `userdata_bootstrap.sh` as cloud-init user-data.
4. Waits until `running` and prints the launch plan + instance metadata
   (instance id, IPs, run id, S3 results path).

Use `--dry-run` for a no-network smoke test that builds the tarball, resolves
the AMI, renders the user-data into `infra/ec2/_last_userdata.sh`, and exits
without calling `RunInstances`.

If `RunInstances` fails due to spot capacity, re-run with the next candidate
from step 1. The included subagent automation walks the candidate list
automatically with retry.

### 3. Monitor

The instance has no SSH key by default; access it via **SSM Session
Manager** (the role grants `AmazonSSMManagedInstanceCore`):

```bash
AWS_PROFILE=rengz aws ssm start-session \
    --target <instance-id> --region <region>
```

Inside the session:

```bash
sudo tail -f /var/log/pruning-experiment/userdata.log
sudo nvidia-smi
ls /opt/results
```

Without an SSM session, watch the run progress directly through S3:

```bash
AWS_PROFILE=rengz aws s3 ls --recursive --human-readable \
    s3://pruning-metrics-results-414266451290/qwen2_72b_pruning/<run-id>/
```

Each completed pruning level adds:

* `pruning_level=NN/eval_records.jsonl`  — per-task HumanEval+ verifications
  (autoregressive, **no** teacher forcing)
* `pruning_level=NN/teacher_forced.json` — per-token next-token log-probs for
  the seeded `(prompt, canonical_solution)` pair
* an updated `summary.json` at the run root

When the run finishes (or a spot interruption fires), the EC2 box runs
`aws s3 sync` one more time and `shutdown -h now`. The instance terminates
itself; you do not need to clean it up.

## Cost & wall-clock estimates

* `p5.48xlarge` spot: ~$12.7/hr in `us-east-1b` (Apr 2026 prices).
* `p4d.24xlarge` spot: ~$10–14/hr depending on AZ — adequate for Qwen2-72B
  in bf16 (320 GiB total GPU memory across 8 A100s).
* Expected runtime end-to-end: roughly **2–4 hours** on `p5.48xlarge` for
  Qwen2-72B with the default split: model download (~10–15 min on cold
  cache), WANDA stats over 131 train prompts (~10–30 min), 5 levels × (weight
  restore ~3 min + prune ~1 min + 33-task eval ~10–15 min + TF scoring
  ~10 sec). Expect 4–8 hours on `p4d.24xlarge` due to slower interconnect.

## What gets recorded under the run prefix

```
s3://pruning-metrics-results-414266451290/qwen2_72b_pruning/<run-id>/
├── code/repo.tar.gz                            # snapshot of the repo at launch time
├── run_metadata.json                            # host, run id, model id, seed, ...
├── split.json                                   # full train/test partition (audit)
├── summary.json                                 # per-level metrics aggregated
├── pruning_level=0/eval_records.jsonl
├── pruning_level=0/teacher_forced.json
├── pruning_level=20/...
├── pruning_level=40/...
├── pruning_level=60/...
├── pruning_level=80/...
└── _logs/userdata.log                           # bootstrap stdout/stderr
```

`split.json` records the seed, train fraction, and full ordered task-id lists
for both partitions. The deterministic 80/20 split (seed `65320`) shipped
with this repo is the canonical one for the experiment.

## Recovering from spot interruption

The user-data script also polls the EC2 instance metadata service for the
[spot interruption notice](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-interruptions.html)
and runs `aws s3 sync` immediately when one is announced. If a level was in
progress when interruption fired, that level's `eval_records.jsonl` /
`teacher_forced.json` may be missing or partial; previously-completed levels
are intact in S3.

To resume, re-launch with `--run-id <same-id>` (the current runner re-runs
**all** levels — a future enhancement is to skip levels whose
`teacher_forced.json` is already present in S3).

## Pruning the SageMaker-endpoint legacy

The `infra/aws/sagemaker/` workflow remains in the repo for smaller-model
demonstrations and as a reference of the originally-intended hosting path,
but it is **not** used for the Qwen2-72B run. SageMaker GPU-endpoint quota in
this account (`ml.g5.48xlarge`, `ml.p4d.24xlarge`, `ml.p5.48xlarge`, …) is
`0`; raising it requires an AWS support case.

## Quick reference: model name, split seed, teacher-forced pair

| Item | Value |
|------|-------|
| Base model | `Qwen/Qwen2-72B` |
| Pruning levels | `0, 20, 40, 60, 80` (percent layer-wise WANDA) |
| HumanEval+ split seed | `65320` |
| Train fraction | `0.8` |
| Teacher-forcing pair selector | `sorted(test_tasks, key=task_id)[seed % len(test)]` |

The seeded `(prompt, canonical_solution)` pair is identical for every
pruning level so the per-token probabilities are directly comparable across
sparsity levels.
