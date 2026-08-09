# EC2 spot GPU pipeline -- operator runbook

This directory holds the machinery for the four-notebook pruning workflow,
split into two folders:

* [`provisioning/`](provisioning/) -- operator-side tooling: one-time AWS
  bootstrap, spot-capacity probing, instance launch, and the user-data
  template rendered onto the box.
* [`runners/`](runners/) -- the scripts that execute **on** the GPU
  instance: shared helpers plus one worker per notebook.

Each of notebooks
[`02_prune_llm.ipynb`](../notebooks/aws_tutorial/02_prune_llm.ipynb),
[`03_freeform_eval.ipynb`](../notebooks/aws_tutorial/03_freeform_eval.ipynb), and
[`04_teacher_forced.ipynb`](../notebooks/aws_tutorial/04_teacher_forced.ipynb)
launches one of the runner scripts here on a spot EC2 instance, polls
S3 for results, and self-terminates the box when the job is done.

If you have not yet read it, start with
[`docs/getting_started.md`](../docs/getting_started.md) and
[`docs/architecture.md`](../docs/architecture.md). The notebooks
themselves are the primary interface; this runbook covers the operator-
side concerns: capacity probing, monitoring, debugging, recovery.

## Files

| File | What it does |
|------|--------------|
| [`provisioning/bootstrap_ec2_resources.py`](provisioning/bootstrap_ec2_resources.py) | One-time, idempotent AWS setup: versioned + public-access-blocked S3 results bucket, EC2 IAM role with scoped S3 / SSM / CloudWatch permissions, and the matching instance profile. |
| [`provisioning/find_capacity.py`](provisioning/find_capacity.py) | Scans `describe_spot_price_history` + `describe_instance_type_offerings` across regions/AZs for `p5.48xlarge`, `p4de.24xlarge`, `p4d.24xlarge` (or any user-supplied list) and prints the cheapest currently-fulfillable candidates as JSON. The notebooks call this through `pruning_metrics.notebook_helpers.find_capacity`. |
| [`provisioning/launch_gpu_instance.py`](provisioning/launch_gpu_instance.py) | Tars the repo (excluding `.env`, `.git`, `.venv`, caches), uploads to S3, resolves the latest Deep Learning AMI via SSM Parameter Store, renders [`provisioning/userdata_bootstrap.sh`](provisioning/userdata_bootstrap.sh) with the chosen runner + runner-env, and calls `RunInstances` with the `pruning-metrics-ec2` instance profile + spot market + 1500 GiB gp3 root. |
| [`provisioning/userdata_bootstrap.sh`](provisioning/userdata_bootstrap.sh) | Cloud-init script. Probes DLAMI conda envs for a python with `torch` pre-installed (falls back to system pip if none), pulls the repo tarball from S3, exports runner-specific env vars, runs the chosen runner, and on exit (success, failure, or spot interruption) syncs results to S3 and shuts down. |
| [`runners/_runner_common.py`](runners/_runner_common.py) | Shared helpers reused by all five runners: per-row WANDA pruning, snapshot/restore of `nn.Linear` weights to host RAM, S3 sync / download, model loading. The S3 helpers do not import `torch` so the launcher can use them without a GPU env. |
| [`runners/run_pruning_calibration.py`](runners/run_pruning_calibration.py) | Notebook 2's worker. Loads model once -> WANDA stats over the train split of the chosen calibration dataset -> uploads `wanda_stats.pt` + `manifest.json` + `split.json` + `run_metadata.json`. Fast (~5-25 min). |
| [`runners/run_freeform_eval.py`](runners/run_freeform_eval.py) | Notebook 3's worker. Downloads the calibration artifact, loads the base model, and per requested pruning level: restore -> apply per-row WANDA -> generate the test split greedily -> task-adapter `verify` -> incremental S3 sync of `level=NN/eval_records.jsonl` and a rolling `summary.json`. |
| [`runners/run_teacher_forced.py`](runners/run_teacher_forced.py) | Notebook 4's worker. Same artifact + adapter + level sweep, but instead of free-form generation it runs `compute_teacher_forced_logprobs` for `NUM_TF_SAMPLES` records picked deterministically using `TF_SEED`. Outputs `level=NN/sample=KKK_task=.../per_token.json`. |
| [`runners/run_prune_eval_sweep.py`](runners/run_prune_eval_sweep.py) | Notebook 6's worker (v2 experiment, `06_prune_eval_v2.ipynb`). For one `(pruner, calibration_domain, seed)` grid point in a single self-contained job: loads the model once, prunes to every requested level with WANDA or SparseGPT, saves the resulting pruning mask (full + packed digest) per level, and teacher-forces the same seeded sample of test records from every requested eval benchmark at every level -- output is byte-for-byte compatible with `run_teacher_forced.py`'s `per_token.json`. |
| [`runners/run_v2_analysis.py`](runners/run_v2_analysis.py) | CPU-only worker behind notebook 7 (`07_diagnosticity.ipynb`). Pulls every run's `per_token.json` / `*.digest.npz` / `summary.json` from S3 into the notebook's local cache, executes the notebook headlessly via `jupyter nbconvert`, and uploads the executed notebook plus `results/v2_*` artifacts back to S3. Needs no GPU. |

The launcher's user-data renders the chosen runner via:

```bash
python infra/provisioning/launch_gpu_instance.py \
    --runner {pruning_calibration|freeform_eval|teacher_forced} \
    --runner-env-json '{"BASE_MODEL_ID": "...", "PRUNING_LEVELS": "0,20,40,60,80", ...}'
```

The notebooks build the right `runner_env` dict and call `launch_runner`
in [`pruning_metrics.notebook_helpers`](../src/pruning_metrics/notebook_helpers/),
so most users never call this CLI directly.

## One-time AWS bootstrap

Notebook 1 (`01_setup_aws.ipynb`) calls
[`provisioning/bootstrap_ec2_resources.py`](provisioning/bootstrap_ec2_resources.py).
For a fresh account / region you can also run it from the command line:

```bash
AWS_PROFILE=rengz python3 infra/provisioning/bootstrap_ec2_resources.py \
    --bucket pruning-metrics-results-414266451290 \
    --region us-east-1 \
    --role-name pruning-metrics-ec2
```

That creates a versioned + public-access-blocked S3 bucket, an IAM role
trusted by EC2 with scoped S3 + SSM Session Manager + CloudWatch agent
permissions, and the matching IAM instance profile. It is idempotent.

## Manually launching a runner (debug only)

The notebooks are the recommended interface, but for debugging you can
shell out directly:

```bash
# Find capacity:
AWS_PROFILE=rengz python3 infra/provisioning/find_capacity.py

# Launch the calibration runner with Qwen2-1.5B-Instruct on a g5.xlarge:
AWS_PROFILE=rengz python3 infra/provisioning/launch_gpu_instance.py \
    --region us-east-1 --availability-zone us-east-1b \
    --instance-type g5.xlarge --max-spot-price 1.50 \
    --results-bucket pruning-metrics-results-414266451290 \
    --results-prefix pruning_artifacts \
    --runner pruning_calibration \
    --runner-env BASE_MODEL_ID=Qwen/Qwen2-1.5B-Instruct \
    --runner-env "PRUNING_LEVELS=0,50" \
    --runner-env CALIBRATION_DATASET_SPEC=coding \
    --runner-env MAX_CALIBRATION_SAMPLES=8 \
    --runner-env MAX_CALIBRATION_TOKENS=256 \
    --root-volume-gib 200
```

`--dry-run` skips `RunInstances` (still uploads tarball + writes
`infra/provisioning/_last_userdata.sh` for inspection).

### The `v2_analysis` runner (CPU, no GPU needed)

`--runner v2_analysis` executes `notebooks/experiment/07_diagnosticity.ipynb`
headlessly. It needs no GPU, so launch it on a compute-optimised instance --
the DLAMI boots fine without one, and the bootstrap's `torch.cuda` line simply
reports `cuda False`.

The heavy part is cell 10, which builds a 232x232 distance matrix per
`(benchmark, metric)`. Measured cost is dominated by one benchmark:
`math:openai_gsm8k:main` alone is ~29 of the ~31 core-hours (112-token answers
versus 3-6 for the MCQ sets). Use `--runner-env V2_BENCHES=...` to scope a run;
it filters both the S3 cache sync and the notebook itself.

```bash
AWS_PROFILE=rengz python3 infra/provisioning/launch_gpu_instance.py \
    --region us-east-1 --availability-zone us-east-1d \
    --instance-type c7i.16xlarge --max-spot-price 1.90 \
    --results-bucket pruning-metrics-results-414266451290 \
    --results-prefix prune_eval_v2_analysis \
    --runner v2_analysis \
    --runner-env V2_BENCHES=math:openai_gsm8k:main \
    --runner-env V2_PERMUTATIONS=4999 \
    --root-volume-gib 150
```

Results land in
`s3://<bucket>/prune_eval_v2_analysis/<run_id>/results/v2_pairwise_*`; pull
those into `notebooks/experiment/results/` and re-run the notebook locally to
regenerate figures across every benchmark at once.

**Capacity, not price, is the binding constraint.** The account has a 1152
vCPU Standard-spot quota in us-east-1, yet `c7i.16xlarge` in `us-east-1c`
returned `InsufficientInstanceCapacity` while `us-east-1d` fulfilled
immediately. The launcher returns exit code 2 for that case (and 3 for quota
exhaustion) precisely so a caller can walk a candidate list; be flexible about
both AZ and instance family rather than retrying one pair.

## Monitoring a running job

The instance has no SSH key by default; access it via
**SSM Session Manager** (the role grants `AmazonSSMManagedInstanceCore`):

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

Without an SSM session, watch progress through S3:

```bash
AWS_PROFILE=rengz aws s3 ls --recursive --human-readable \
    s3://pruning-metrics-results-414266451290/<runner-prefix>/<run-id>/
```

The runner uploads incrementally per level, so you can stream progress
as completed levels appear.

## Spot-interruption recovery

The user-data script polls `http://169.254.169.254/latest/meta-data/spot/instance-action`
in a background loop and runs an immediate `aws s3 sync` of `/opt/results`
when AWS announces an interruption. Completed levels are intact in S3;
the in-progress level may be missing or partial.

To resume, re-launch the same runner with the same `--run-id`. The
runners are not yet level-skipping, so they re-run all requested levels;
this is intentionally simple and rare in practice (sub-1% of spot runs
in our testing).

## Cost & wall-clock reference (May 2026 prices)

| Phase | Box | Wall clock (Qwen2-72B) | Wall clock (Qwen2-1.5B-Instruct) | Spot rate |
|------:|-----|-----------------------:|---------------------------------:|----------:|
| `pruning_calibration` (notebook 2) | `p4de.24xlarge` (72B) / `g5.xlarge` (1.5B) | ~25 min | ~6 min | $13/hr / $0.6/hr |
| `freeform_eval` (notebook 3) | same | ~30 min for 33 tasks x 5 levels | ~7 min | same |
| `teacher_forced` (notebook 4) | same | ~5 min for 1 sample x 5 levels | ~5 min | same |

The full Qwen2-72B sweep across all three notebooks is ~$25 of GPU time.
A small-model smoke pass through all three notebooks is well under $0.50.

## Output layout

```
s3://<bucket>/pruning_artifacts/<run_id>/
├── code/repo.tar.gz
├── _logs/userdata.log
├── run_metadata.json
├── manifest.json
├── split.json
└── wanda_stats.pt

s3://<bucket>/freeform_eval/<run_id>/
├── code/repo.tar.gz
├── _logs/userdata.log
├── _artifact/{manifest.json, wanda_stats.pt}   # pulled from the calibration artifact
├── run_metadata.json
├── summary.json
├── level=0/eval_records.jsonl
├── level=20/eval_records.jsonl
└── ...

s3://<bucket>/teacher_forced/<run_id>/
├── code/repo.tar.gz
├── _logs/userdata.log
├── _artifact/{manifest.json, wanda_stats.pt}
├── run_metadata.json
├── summary.json
├── sample_selection.json
├── level=0/sample=000_task=HumanEval_137/per_token.json
├── level=20/sample=000_task=HumanEval_137/per_token.json
└── ...
```

`per_token.json` files contain the full ground-truth log-probability and
top-`TF_TOP_K` alternatives for every answer position; the notebooks
render them as a per-token DataFrame for inline review.
