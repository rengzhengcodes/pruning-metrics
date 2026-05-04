"""Generate the four orchestration notebooks programmatically.

Authoring notebooks via :mod:`nbformat` keeps them under version control as
plain Python and avoids the noise of hand-edited ``.ipynb`` JSON. Run with
``python scripts/build_notebooks.py`` from the repo root; the script
overwrites the four notebook files in ``notebooks/``.

The notebooks themselves stay short: each cell is either a markdown
explanation or a thin orchestration call that delegates to
``pruning_metrics.notebook_helpers``.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat import v4 as nb

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / "notebooks"
NOTEBOOKS.mkdir(parents=True, exist_ok=True)


def _md(text: str) -> nbformat.NotebookNode:
    """Build a markdown cell from a dedented heredoc."""

    return nb.new_markdown_cell(dedent(text).strip() + "\n")


def _code(text: str) -> nbformat.NotebookNode:
    """Build a code cell from a dedented heredoc."""

    return nb.new_code_cell(dedent(text).strip() + "\n")


def write_notebook(path: Path, cells: list[nbformat.NotebookNode]) -> None:
    """Materialise the notebook on disk with python3 metadata."""

    notebook = nb.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    nbformat.write(notebook, path)
    print(f"Wrote {path}")


# Common header used across every notebook so users know the prerequisites.
COMMON_BOOTSTRAP_CELL = """
import os
import sys
from pathlib import Path

# Allow the notebook to be run from anywhere by pinning to the repo root.
REPO_ROOT = Path.cwd()
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "pyproject.toml").is_file():
    REPO_ROOT = REPO_ROOT.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Optional: load .env so AWS_PROFILE etc. surface in the kernel.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:
    pass

print("REPO_ROOT =", REPO_ROOT)
print("AWS_PROFILE =", os.environ.get("AWS_PROFILE"))
print("AWS_REGION  =", os.environ.get("AWS_REGION"))
"""


# ---------------------------------------------------------------------------
# Notebook 1: 01_setup_aws.ipynb
# ---------------------------------------------------------------------------


def build_notebook_01() -> None:
    cells = [
        _md(
            """
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
            """
        ),
        _code(COMMON_BOOTSTRAP_CELL),
        _md(
            """
            ## Configuration

            All knobs live in this cell. The defaults match what the rest of
            the project expects, so most users only edit `RESULTS_BUCKET`
            (must be globally unique, follow S3 naming rules).
            """
        ),
        _code(
            """
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
            """
        ),
        _md(
            """
            ## Verify AWS credentials

            STS `GetCallerIdentity` confirms the kernel can reach AWS APIs and
            displays the assumed identity (the SSO role for `rengz`).
            """
        ),
        _code(
            """
            import json
            import boto3

            session = boto3.session.Session(profile_name=AWS_PROFILE)
            sts = session.client("sts", region_name=AWS_REGION)
            identity = sts.get_caller_identity()
            print(json.dumps(identity, indent=2, default=str))
            assert identity["Account"] == ACCOUNT_ID, (
                f"Expected account {ACCOUNT_ID}, got {identity['Account']}"
            )
            """
        ),
        _md(
            """
            ## Run the bootstrap script

            `infra/aws/setup/bootstrap_ec2_resources.py` is idempotent. It
            checks for the bucket / role / profile and creates anything
            missing. Output below is a single JSON summary.
            """
        ),
        _code(
            """
            import subprocess

            cmd = [
                sys.executable,
                str(REPO_ROOT / "infra" / "aws" / "setup" / "bootstrap_ec2_resources.py"),
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
            """
        ),
        _md(
            """
            ## Verify the resources

            Confirms the bucket exists, the role exists and is trusted by EC2,
            and the instance profile carries the role.
            """
        ),
        _code(
            """
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
            """
        ),
        _md(
            """
            ## Summary

            Bucket and instance profile are ready. Use the values above as
            inputs to notebooks 2/3/4. The instance profile name is also
            stored in `.env` as `EC2_INSTANCE_PROFILE_NAME` so the launch
            scripts pick it up automatically.
            """
        ),
        _code(
            """
            print("Setup complete. Use these values in subsequent notebooks:")
            print(json.dumps({
                "AWS_PROFILE": AWS_PROFILE,
                "AWS_REGION": AWS_REGION,
                "RESULTS_BUCKET": RESULTS_BUCKET,
                "EC2_INSTANCE_PROFILE_NAME": EC2_INSTANCE_ROLE_NAME,
            }, indent=2))
            """
        ),
    ]
    write_notebook(NOTEBOOKS / "01_setup_aws.ipynb", cells)


# ---------------------------------------------------------------------------
# Notebook 2: 02_prune_llm.ipynb
# ---------------------------------------------------------------------------


def build_notebook_02() -> None:
    cells = [
        _md(
            """
            # 02 - Prune an LLM (general)

            Launches a single EC2 spot GPU box that:

            1. loads any Hugging Face causal LM (default Qwen2-72B);
            2. splits the chosen calibration dataset 80/20 with a seed;
            3. runs WANDA-style activation-stat collection on the train
               (calibration) split;
            4. uploads `wanda_stats.pt` + `manifest.json` + `split.json` to
               `s3://<bucket>/pruning_artifacts/<run_id>/`.

            The artifact URI printed at the end is the input to notebooks 3
            and 4. The same calibration artifact can be re-used for any number
            of downstream evaluations on different test datasets.

            Notebook 1 must have run first to provision the S3 bucket and
            EC2 instance profile.
            """
        ),
        _code(COMMON_BOOTSTRAP_CELL),
        _md(
            """
            ## Configuration

            * `BASE_MODEL_ID`: any HF causal LM. Defaults to Qwen2-72B (~145 GiB
              bf16; needs `p4de.24xlarge` or `p5.48xlarge`). Use
              `Qwen/Qwen2-1.5B-Instruct` for a smoke pass on `g5.xlarge`.
            * `CALIBRATION_DATASET_SPEC`: any task-adapter spec. Examples:
              `coding` (HumanEval+; seeded 80/20 fallback over the test
              split), `math:gsm8k:main` (uses GSM8K's native train + test
              splits), `mcq:allenai/ai2_arc:ARC-Challenge` (uses ARC's
              native train + test splits).
            * `PRUNING_LEVELS`: percent levels recorded in the manifest;
              the actual pruning is re-derived deterministically per level
              by notebooks 3 and 4.
            * `SPLIT_SEED`, `TRAIN_FRAC`: only used for the seeded fallback
              when a dataset has no native train split (e.g. HumanEval+),
              and for reproducible truncation when `MAX_CALIBRATION_SAMPLES`
              caps a native train set.
            * `EXPLICIT_TRAIN_IDS` / `EXPLICIT_TEST_IDS`: optional task-id
              overrides; force the seeded splitter and missing ids fall
              back to the shuffle.
            * `INSTANCE_TYPE_PRIORITY` and `REGION_PRIORITY`: search order
              for the spot capacity probe.
            """
        ),
        _code(
            """
            import json

            AWS_PROFILE = os.environ.get("AWS_PROFILE", "rengz")
            RESULTS_BUCKET = os.environ.get(
                "RESULTS_BUCKET", "pruning-metrics-results-414266451290"
            )

            BASE_MODEL_ID = "Qwen/Qwen2-72B"

            # Spec format: ``<adapter>:<dataset_name>[:<config>][:<split>]``
            # See pruning_metrics.evals.tasks.registry for accepted adapters.
            CALIBRATION_DATASET_SPEC = "coding:evalplus/humanevalplus:test"

            PRUNING_LEVELS = [0, 20, 40, 60, 80]
            SPLIT_SEED = 65320
            TRAIN_FRAC = 0.8
            EXPLICIT_TRAIN_IDS = None  # or ["HumanEval/0", ...]
            EXPLICIT_TEST_IDS = None
            MAX_CALIBRATION_SAMPLES = 0  # 0 = use full train split
            MAX_CALIBRATION_TOKENS = 512

            INSTANCE_TYPE_PRIORITY = ["p4de.24xlarge", "p5.48xlarge", "p4d.24xlarge"]
            REGION_PRIORITY = ["us-east-1", "us-west-2", "us-east-2"]
            HF_TOKEN = os.environ.get("HF_TOKEN", "")

            print(json.dumps({
                "BASE_MODEL_ID": BASE_MODEL_ID,
                "CALIBRATION_DATASET_SPEC": CALIBRATION_DATASET_SPEC,
                "PRUNING_LEVELS": PRUNING_LEVELS,
                "SPLIT_SEED": SPLIT_SEED,
                "TRAIN_FRAC": TRAIN_FRAC,
                "INSTANCE_TYPE_PRIORITY": INSTANCE_TYPE_PRIORITY,
                "REGION_PRIORITY": REGION_PRIORITY,
            }, indent=2))
            """
        ),
        _md(
            """
            ## Find a viable spot capacity candidate

            Queries `describe_spot_price_history` and `describe_instance_type_offerings`
            across the priority list and returns the cheapest-and-still-available
            `(region, AZ, instance_type)` tuple. Re-run this cell if your
            launch fails with `InsufficientInstanceCapacity` to get fresh
            candidates.
            """
        ),
        _code(
            """
            from pruning_metrics.notebook_helpers import find_capacity

            candidates = find_capacity(
                regions=tuple(REGION_PRIORITY),
                instance_types=tuple(INSTANCE_TYPE_PRIORITY),
                aws_profile=AWS_PROFILE,
            )
            assert candidates, "No spot capacity available in the priority list."
            print(f"Top 3 candidates (of {len(candidates)} total):")
            for cand in candidates[:3]:
                print(json.dumps(cand, indent=2, default=str))
            chosen = candidates[0]
            print("\\nChosen:")
            print(json.dumps(chosen, indent=2, default=str))
            """
        ),
        _md(
            """
            ## Launch the calibration runner

            Tars the repo, uploads to S3, resolves the latest Deep Learning
            AMI, and calls `RunInstances` with the `pruning-metrics-ec2`
            instance profile attached. The runner uploads its artifact to
            `s3://<bucket>/pruning_artifacts/<run_id>/`.

            Pass `dry_run=True` to render the user-data without launching.
            """
        ),
        _code(
            """
            from pruning_metrics.notebook_helpers import launch_runner, render_run_id_default

            run_id = render_run_id_default()

            runner_env = {
                "BASE_MODEL_ID": BASE_MODEL_ID,
                "CALIBRATION_DATASET_SPEC": CALIBRATION_DATASET_SPEC,
                "PRUNING_LEVELS": ",".join(str(level) for level in PRUNING_LEVELS),
                "SPLIT_SEED": SPLIT_SEED,
                "TRAIN_FRAC": TRAIN_FRAC,
                "MAX_CALIBRATION_SAMPLES": MAX_CALIBRATION_SAMPLES,
                "MAX_CALIBRATION_TOKENS": MAX_CALIBRATION_TOKENS,
            }
            if EXPLICIT_TRAIN_IDS:
                runner_env["EXPLICIT_TRAIN_IDS"] = ",".join(EXPLICIT_TRAIN_IDS)
            if EXPLICIT_TEST_IDS:
                runner_env["EXPLICIT_TEST_IDS"] = ",".join(EXPLICIT_TEST_IDS)

            launched = launch_runner(
                runner="pruning_calibration",
                runner_env=runner_env,
                region=chosen["region"],
                availability_zone=chosen["availability_zone"],
                instance_type=chosen["instance_type"],
                max_spot_price=float(chosen["max_bid_usd_per_hour"]),
                results_bucket=RESULTS_BUCKET,
                results_prefix="pruning_artifacts",
                run_id=run_id,
                aws_profile=AWS_PROFILE,
                hf_token=HF_TOKEN,
                name_tag="pruning-metrics-calibration",
            )

            PRUNING_ARTIFACT_URI = launched.results_uri  # exported for downstream notebooks
            print(json.dumps(launched.raw_plan, indent=2, default=str))
            print("\\nArtifact will land at:", PRUNING_ARTIFACT_URI)
            """
        ),
        _md(
            """
            ## Wait for the calibration artifact

            Polls S3 until `wanda_stats.pt` (the heaviest artifact) is
            present. Calibration on `p4de.24xlarge` typically takes 6-15
            minutes for Qwen2-72B (mostly model download + WANDA hooks).
            """
        ),
        _code(
            """
            from pruning_metrics.notebook_helpers import wait_for_artifact, list_results

            artifact_prefix = f"pruning_artifacts/{launched.run_id}"
            head = wait_for_artifact(
                bucket=RESULTS_BUCKET,
                key=f"{artifact_prefix}/wanda_stats.pt",
                aws_profile=AWS_PROFILE,
                poll_seconds=30.0,
                timeout_seconds=60 * 60 * 2,
            )
            print("wanda_stats.pt size:", head["ContentLength"], "bytes")
            for entry in list_results(RESULTS_BUCKET, artifact_prefix, aws_profile=AWS_PROFILE):
                print(f"  {entry['size']:>12d}  {entry['key']}")
            """
        ),
        _md(
            """
            ## Inspect manifest + split

            The manifest records every input that produced the artifact so
            downstream notebooks can replay deterministically.
            """
        ),
        _code(
            """
            import boto3, io

            session = boto3.session.Session(profile_name=AWS_PROFILE)
            s3 = session.client("s3")
            bucket, prefix = RESULTS_BUCKET, f"pruning_artifacts/{launched.run_id}"

            manifest = json.loads(
                s3.get_object(Bucket=bucket, Key=f"{prefix}/manifest.json")["Body"].read()
            )
            split = json.loads(
                s3.get_object(Bucket=bucket, Key=f"{prefix}/split.json")["Body"].read()
            )
            print("--- manifest ---")
            print(json.dumps(manifest, indent=2))
            print("--- split (counts only) ---")
            print(json.dumps({k: v for k, v in split.items() if "task_ids" not in k}, indent=2))
            print("--- first 5 train ids ---", split["train_task_ids"][:5])
            print("--- first 5 test  ids ---", split["test_task_ids"][:5])
            """
        ),
        _md(
            """
            ## Hand-off to notebooks 3 & 4

            Copy the artifact URI below into the configuration cell of either
            downstream notebook. The same artifact can be reused across many
            downstream evaluations on different datasets.
            """
        ),
        _code(
            """
            print("PRUNING_ARTIFACT_URI =", repr(PRUNING_ARTIFACT_URI))
            """
        ),
    ]
    write_notebook(NOTEBOOKS / "02_prune_llm.ipynb", cells)


# ---------------------------------------------------------------------------
# Notebook 3: 03_freeform_eval.ipynb
# ---------------------------------------------------------------------------


def build_notebook_03() -> None:
    cells = [
        _md(
            """
            # 03 - Free-form evaluation per pruning level

            Given a pruning calibration artifact (output of notebook 2) and a
            task-adapter spec, this notebook launches a GPU box that:

            1. downloads the artifact (`manifest.json`, `wanda_stats.pt`);
            2. loads the matching base model;
            3. for each requested pruning level, applies WANDA from the stats,
               generates greedy completions on the test split, and runs the
               adapter's verifier (subprocess pass@1 for coding, numeric
               match for math, regex letter match for MCQ);
            4. uploads per-level `eval_records.jsonl` and a rolling
               `summary.json` to `s3://<bucket>/freeform_eval/<run_id>/`.

            **No teacher forcing is used here.** This is the raw "is the
            pruned model still useful?" measurement.
            """
        ),
        _code(COMMON_BOOTSTRAP_CELL),
        _md(
            """
            ## Configuration

            * `PRUNING_ARTIFACT_URI`: full S3 URI from notebook 2.
            * `EVAL_DATASET_SPEC`: pick any registered adapter; defaults to
              the calibration dataset for parity. Examples:
              `coding:evalplus/humanevalplus:test` (HumanEval+ test),
              `math:gsm8k:main` (GSM8K Hub ``train`` + ``test``; there is no
              separate ``validation`` split on ``main``, and the adapters
              never assume one),
              `mcq:allenai/ai2_arc:ARC-Challenge` (ARC Hub ``train`` + ``test``).
            * `EVAL_LEVELS`: subset of `manifest.pruning_levels`.
            * `GENERATION_SEED`: feeds `torch.manual_seed`. Greedy decoding
              is otherwise deterministic; the seed exists so anyone curious
              can flip to sampling without touching the runner.
            * `MAX_TEST_SAMPLES`: cap on the number of test records (0 = all).
            """
        ),
        _code(
            """
            import json

            AWS_PROFILE = os.environ.get("AWS_PROFILE", "rengz")
            RESULTS_BUCKET = os.environ.get(
                "RESULTS_BUCKET", "pruning-metrics-results-414266451290"
            )

            PRUNING_ARTIFACT_URI = (
                # paste the URI printed at the end of notebook 2 here:
                "s3://pruning-metrics-results-414266451290/pruning_artifacts/<run_id>/"
            )
            EVAL_DATASET_SPEC = ""  # empty -> reuse the artifact's calibration spec
            EVAL_LEVELS = [0, 20, 40, 60, 80]
            GENERATION_SEED = 65320
            MAX_NEW_TOKENS = 512
            TIMEOUT_SECONDS = 10.0
            MAX_TEST_SAMPLES = 0

            INSTANCE_TYPE_PRIORITY = ["p4de.24xlarge", "p5.48xlarge", "p4d.24xlarge"]
            REGION_PRIORITY = ["us-east-1", "us-west-2", "us-east-2"]
            HF_TOKEN = os.environ.get("HF_TOKEN", "")

            print(json.dumps({
                "PRUNING_ARTIFACT_URI": PRUNING_ARTIFACT_URI,
                "EVAL_DATASET_SPEC": EVAL_DATASET_SPEC or "(use artifact's)",
                "EVAL_LEVELS": EVAL_LEVELS,
                "GENERATION_SEED": GENERATION_SEED,
            }, indent=2))
            assert PRUNING_ARTIFACT_URI.startswith("s3://"), "Set PRUNING_ARTIFACT_URI."
            assert "<" not in PRUNING_ARTIFACT_URI and ">" not in PRUNING_ARTIFACT_URI, (
                "PRUNING_ARTIFACT_URI must be the real URI from notebook 2, not a "
                "placeholder (remove <run_id> and use the printed timestamp id)."
            )
            """
        ),
        _md(
            """
            ## Find capacity & launch the eval runner

            Same boilerplate as notebook 2 -- find a viable spot candidate
            and shell out to `launch_gpu_instance.py` with `--runner freeform_eval`.
            """
        ),
        _code(
            """
            from pruning_metrics.notebook_helpers import (
                find_capacity,
                launch_runner,
                render_run_id_default,
            )

            candidates = find_capacity(
                regions=tuple(REGION_PRIORITY),
                instance_types=tuple(INSTANCE_TYPE_PRIORITY),
                aws_profile=AWS_PROFILE,
            )
            assert candidates, "No spot capacity available."
            chosen = candidates[0]
            print("Chosen:", chosen["region"], chosen["availability_zone"], chosen["instance_type"])

            run_id = render_run_id_default()
            runner_env = {
                "PRUNING_ARTIFACT_URI": PRUNING_ARTIFACT_URI,
                "EVAL_LEVELS": ",".join(str(level) for level in EVAL_LEVELS),
                "GENERATION_SEED": GENERATION_SEED,
                "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
                "TIMEOUT_SECONDS": TIMEOUT_SECONDS,
                "MAX_TEST_SAMPLES": MAX_TEST_SAMPLES,
            }
            if EVAL_DATASET_SPEC:
                runner_env["EVAL_DATASET_SPEC"] = EVAL_DATASET_SPEC

            launched = launch_runner(
                runner="freeform_eval",
                runner_env=runner_env,
                region=chosen["region"],
                availability_zone=chosen["availability_zone"],
                instance_type=chosen["instance_type"],
                max_spot_price=float(chosen["max_bid_usd_per_hour"]),
                results_bucket=RESULTS_BUCKET,
                results_prefix="freeform_eval",
                run_id=run_id,
                aws_profile=AWS_PROFILE,
                hf_token=HF_TOKEN,
                name_tag="pruning-metrics-freeform-eval",
            )
            print(json.dumps(launched.raw_plan, indent=2, default=str))
            FREEFORM_RESULTS_URI = launched.results_uri
            """
        ),
        _md(
            """
            ## Poll until the run finishes

            The runner uploads `summary.json` after **each** completed pruning
            level. If the job exits before level 0 finishes (bad
            `PRUNING_ARTIFACT_URI`, OOM, etc.), that file never appears and the
            next cell will time out with a hint to inspect `_logs/userdata.log`
            under this run prefix.

            We poll until the instance terminates (or shuts down post-run),
            then wait for `summary.json` before reading metrics.
            """
        ),
        _code(
            """
            from pruning_metrics.notebook_helpers import (
                describe_instance,
                list_results,
                wait_for_instance_terminated,
            )

            print("Streaming progress (Ctrl-C in the kernel to stop polling without affecting EC2):")
            try:
                wait_for_instance_terminated(
                    instance_id=launched.instance_id,
                    region=launched.region,
                    aws_profile=AWS_PROFILE,
                    poll_seconds=60.0,
                    timeout_seconds=60 * 60 * 6,
                )
            except KeyboardInterrupt:
                print("Stopped polling. The EC2 box keeps running until it self-shuts.")

            for entry in list_results(
                RESULTS_BUCKET, f"freeform_eval/{launched.run_id}", aws_profile=AWS_PROFILE
            ):
                print(f"  {entry['size']:>12d}  {entry['key']}")
            """
        ),
        _md(
            """
            ## Pull and display the per-level metrics

            Waits for `summary.json` (the runner creates it after the first
            level completes), then reads the per-level aggregates for pandas +
            matplotlib.
            """
        ),
        _code(
            """
            import boto3
            import pandas as pd

            from pruning_metrics.notebook_helpers import list_results, wait_for_artifact

            session = boto3.session.Session(profile_name=AWS_PROFILE)
            s3 = session.client("s3")
            prefix = f"freeform_eval/{launched.run_id}"
            summary_key = f"{prefix}/summary.json"

            print("Waiting for", summary_key, "...")
            try:
                wait_for_artifact(
                    RESULTS_BUCKET,
                    summary_key,
                    aws_profile=AWS_PROFILE,
                    poll_seconds=15.0,
                    timeout_seconds=60 * 60 * 6,
                )
            except TimeoutError:
                print(
                    "Timed out: no summary.json. Common causes: invalid "
                    "PRUNING_ARTIFACT_URI (404 on manifest), model OOM, or the "
                    "runner crashed before finishing level 0. Objects under prefix:"
                )
                for entry in list_results(RESULTS_BUCKET, prefix, aws_profile=AWS_PROFILE):
                    print(f"  {entry['key']}")
                raise

            summary = json.loads(
                s3.get_object(Bucket=RESULTS_BUCKET, Key=summary_key)["Body"].read()
            )
            print("Calibration dataset :", summary.get("calibration_dataset_spec"))
            print("Eval dataset        :", summary.get("eval_dataset_spec"))
            print("Base model          :", summary.get("base_model_id"))
            print("Generation seed     :", summary.get("generation_seed"))

            level_rows = []
            for entry in summary.get("levels", []):
                level_rows.append({
                    "pruning_level": entry["pruning_level"],
                    "num_test_tasks": entry["num_test_tasks"],
                    "num_passed": entry["num_passed"],
                    "pass_at_1": entry["pass_at_1"],
                    "elapsed_seconds": entry.get("elapsed_seconds"),
                })
            df = pd.DataFrame(level_rows).sort_values("pruning_level").reset_index(drop=True)
            df
            """
        ),
        _code(
            """
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            if df.empty:
                ax.text(0.5, 0.5, "No levels in summary.json", ha="center", va="center")
            else:
                ax.plot(df["pruning_level"], df["pass_at_1"], "o-", linewidth=2)
                ymax = max(0.05, float(df["pass_at_1"].max()) * 1.1)
                ax.set_ylim(0.0, ymax)
            ax.set_xlabel("Pruning level (% sparsity)")
            ax.set_ylabel("pass@1 (or accuracy for math/MCQ)")
            ax.set_title(
                f"{summary.get('base_model_id', '?')}\\n{summary.get('eval_dataset_spec', '?')}"
            )
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig
            """
        ),
        _md(
            """
            ## Optional: inspect a single task's verification record

            Every level writes one JSON record per test task. Inspect any
            level / task to see the prompt, the model's generation, the
            target text, and the verifier's status.
            """
        ),
        _code(
            """
            from botocore.exceptions import ClientError

            inspect_level = EVAL_LEVELS[0]
            level_label = str(int(inspect_level)) if float(inspect_level).is_integer() else str(inspect_level)
            jsonl_key = f"{prefix}/level={level_label}/eval_records.jsonl"
            try:
                body = s3.get_object(Bucket=RESULTS_BUCKET, Key=jsonl_key)["Body"].read().decode("utf-8")
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") == "NoSuchKey":
                    print(f"Missing {jsonl_key!r} (level may not have completed or label differs).")
                    raise
                raise
            records = [json.loads(line) for line in body.strip().splitlines()]
            print(f"Loaded {len(records)} records at level {level_label}.")
            print("First record (truncated):")
            sample = records[0]
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + " ..."
                print(f"  {key}: {value!r}")
            """
        ),
        _md(
            """
            ## Done

            Final artifacts are at the URI below. Re-run notebook 4
            (teacher-forced log-probs) to compare per-token confidences.
            """
        ),
        _code(
            """
            print("FREEFORM_RESULTS_URI =", repr(FREEFORM_RESULTS_URI))
            """
        ),
    ]
    write_notebook(NOTEBOOKS / "03_freeform_eval.ipynb", cells)


# ---------------------------------------------------------------------------
# Notebook 4: 04_teacher_forced.ipynb
# ---------------------------------------------------------------------------


def build_notebook_04() -> None:
    cells = [
        _md(
            """
            # 04 - Teacher-forced next-token predictions

            Given a pruning calibration artifact (notebook 2) plus a
            task-adapter spec, this notebook launches a GPU box that picks
            `NUM_TF_SAMPLES` records from the test split deterministically
            using `TF_SEED`, and for each requested pruning level computes
            the per-token log-probability of the **ground-truth answer**
            under perfect teacher forcing (one forward pass; the model never
            sees its own outputs while scoring).

            Outputs land at
            `s3://<bucket>/teacher_forced/<run_id>/level=NN/sample=KKK.../per_token.json`,
            with a rolling `summary.json` aggregating average log-probability
            and perplexity per (sample, level).
            """
        ),
        _code(COMMON_BOOTSTRAP_CELL),
        _md(
            """
            ## Configuration

            * `PRUNING_ARTIFACT_URI`: from notebook 2.
            * `EVAL_DATASET_SPEC`: empty -> reuse the artifact's calibration
              spec. Otherwise any registered adapter spec.
            * `EVAL_LEVELS`: subset of `manifest.pruning_levels`.
            * `TF_SEED`: deterministically picks records from the test
              split. Default 65320 matches the rest of the project.
            * `NUM_TF_SAMPLES`: number of records scored (>= 1).
            * `EXPLICIT_SAMPLE_TASK_IDS`: optional override; must be a subset
              of the test split.
            """
        ),
        _code(
            """
            import json

            AWS_PROFILE = os.environ.get("AWS_PROFILE", "rengz")
            RESULTS_BUCKET = os.environ.get(
                "RESULTS_BUCKET", "pruning-metrics-results-414266451290"
            )

            PRUNING_ARTIFACT_URI = (
                # paste the URI printed at the end of notebook 2 here:
                "s3://pruning-metrics-results-414266451290/pruning_artifacts/<run_id>/"
            )
            EVAL_DATASET_SPEC = ""
            EVAL_LEVELS = [0, 20, 40, 60, 80]
            TF_SEED = 65320
            NUM_TF_SAMPLES = 1
            EXPLICIT_SAMPLE_TASK_IDS = None  # or ["HumanEval/137", ...]
            TF_TOP_K = 5

            INSTANCE_TYPE_PRIORITY = ["p4de.24xlarge", "p5.48xlarge", "p4d.24xlarge"]
            REGION_PRIORITY = ["us-east-1", "us-west-2", "us-east-2"]
            HF_TOKEN = os.environ.get("HF_TOKEN", "")

            print(json.dumps({
                "PRUNING_ARTIFACT_URI": PRUNING_ARTIFACT_URI,
                "EVAL_LEVELS": EVAL_LEVELS,
                "TF_SEED": TF_SEED,
                "NUM_TF_SAMPLES": NUM_TF_SAMPLES,
                "TF_TOP_K": TF_TOP_K,
            }, indent=2))
            assert PRUNING_ARTIFACT_URI.startswith("s3://"), "Set PRUNING_ARTIFACT_URI."
            assert "<" not in PRUNING_ARTIFACT_URI and ">" not in PRUNING_ARTIFACT_URI, (
                "PRUNING_ARTIFACT_URI must be the real URI from notebook 2, not a "
                "placeholder (remove <run_id> and use the printed timestamp id)."
            )
            """
        ),
        _md(
            """
            ## Find capacity & launch the teacher-forced runner
            """
        ),
        _code(
            """
            from pruning_metrics.notebook_helpers import (
                find_capacity,
                launch_runner,
                render_run_id_default,
            )

            candidates = find_capacity(
                regions=tuple(REGION_PRIORITY),
                instance_types=tuple(INSTANCE_TYPE_PRIORITY),
                aws_profile=AWS_PROFILE,
            )
            assert candidates, "No spot capacity available."
            chosen = candidates[0]
            print("Chosen:", chosen["region"], chosen["availability_zone"], chosen["instance_type"])

            run_id = render_run_id_default()
            runner_env = {
                "PRUNING_ARTIFACT_URI": PRUNING_ARTIFACT_URI,
                "EVAL_LEVELS": ",".join(str(level) for level in EVAL_LEVELS),
                "TF_SEED": TF_SEED,
                "NUM_TF_SAMPLES": NUM_TF_SAMPLES,
                "TF_TOP_K": TF_TOP_K,
            }
            if EVAL_DATASET_SPEC:
                runner_env["EVAL_DATASET_SPEC"] = EVAL_DATASET_SPEC
            if EXPLICIT_SAMPLE_TASK_IDS:
                runner_env["EXPLICIT_SAMPLE_TASK_IDS"] = ",".join(EXPLICIT_SAMPLE_TASK_IDS)

            launched = launch_runner(
                runner="teacher_forced",
                runner_env=runner_env,
                region=chosen["region"],
                availability_zone=chosen["availability_zone"],
                instance_type=chosen["instance_type"],
                max_spot_price=float(chosen["max_bid_usd_per_hour"]),
                results_bucket=RESULTS_BUCKET,
                results_prefix="teacher_forced",
                run_id=run_id,
                aws_profile=AWS_PROFILE,
                hf_token=HF_TOKEN,
                name_tag="pruning-metrics-teacher-forced",
            )
            print(json.dumps(launched.raw_plan, indent=2, default=str))
            TF_RESULTS_URI = launched.results_uri
            """
        ),
        _md(
            """
            ## Wait for the run to terminate

            `summary.json` is written after the first pruning level completes. If
            the runner crashes before that (bad artifact URI, OOM, etc.), the
            next cell will time out and list objects under this run prefix.
            """
        ),
        _code(
            """
            from pruning_metrics.notebook_helpers import (
                list_results,
                wait_for_instance_terminated,
            )

            print("Polling EC2 state every 60s until the box self-shuts...")
            try:
                wait_for_instance_terminated(
                    instance_id=launched.instance_id,
                    region=launched.region,
                    aws_profile=AWS_PROFILE,
                    poll_seconds=60.0,
                    timeout_seconds=60 * 60 * 4,
                )
            except KeyboardInterrupt:
                print("Stopped polling; instance keeps running.")

            for entry in list_results(
                RESULTS_BUCKET, f"teacher_forced/{launched.run_id}", aws_profile=AWS_PROFILE
            ):
                print(f"  {entry['size']:>12d}  {entry['key']}")
            """
        ),
        _md(
            """
            ## Per-level summary table

            Waits for `summary.json` before loading it (avoids S3 `NoSuchKey`
            if you run this cell before the first level finishes uploading).
            """
        ),
        _code(
            """
            import boto3
            import pandas as pd

            from pruning_metrics.notebook_helpers import list_results, wait_for_artifact

            session = boto3.session.Session(profile_name=AWS_PROFILE)
            s3 = session.client("s3")
            prefix = f"teacher_forced/{launched.run_id}"
            summary_key = f"{prefix}/summary.json"

            print("Waiting for", summary_key, "...")
            try:
                wait_for_artifact(
                    RESULTS_BUCKET,
                    summary_key,
                    aws_profile=AWS_PROFILE,
                    poll_seconds=15.0,
                    timeout_seconds=60 * 60 * 4,
                )
            except TimeoutError:
                print(
                    "Timed out: no summary.json. Check PRUNING_ARTIFACT_URI and "
                    "`_logs/userdata.log` under this prefix:"
                )
                for entry in list_results(RESULTS_BUCKET, prefix, aws_profile=AWS_PROFILE):
                    print(f"  {entry['key']}")
                raise

            summary = json.loads(
                s3.get_object(Bucket=RESULTS_BUCKET, Key=summary_key)["Body"].read()
            )
            print("Eval dataset:", summary.get("eval_dataset_spec"))
            print("Base model  :", summary.get("base_model_id"))
            print("TF seed     :", summary.get("tf_seed"))

            rows = []
            for task_id, payload in summary.get("samples", {}).items():
                for entry in payload.get("by_level", []):
                    rows.append({
                        "task_id": task_id,
                        "pruning_level": entry["pruning_level"],
                        "num_answer_tokens": entry["num_answer_tokens"],
                        "average_logprob": entry["average_logprob"],
                        "perplexity": entry["perplexity"],
                    })
            df = pd.DataFrame(rows).sort_values(["task_id", "pruning_level"]).reset_index(drop=True)
            df
            """
        ),
        _code(
            """
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            if df.empty:
                ax.text(0.5, 0.5, "No rows in summary.json", ha="center", va="center")
            else:
                for task_id, group in df.groupby("task_id"):
                    ax.plot(
                        group["pruning_level"],
                        group["average_logprob"],
                        "o-",
                        label=task_id,
                    )
                ax.legend(fontsize="small")
            ax.set_xlabel("Pruning level (% sparsity)")
            ax.set_ylabel("Average log-probability of gold answer")
            ax.set_title("Teacher-forced confidence vs. sparsity")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig
            """
        ),
        _md(
            """
            ## Per-token table for one (sample, level)

            Picks the first sampled task and the first level by default; tweak
            `inspect_task_id` and `inspect_level` to compare other slices.
            Each row is one teacher-forced position with the ground-truth
            target token's log-probability and the model's top-`TF_TOP_K`
            alternatives.
            """
        ),
        _code(
            """
            inspect_task_id = next(iter(summary["samples"]))
            inspect_level = EVAL_LEVELS[0]
            level_label = str(int(inspect_level)) if float(inspect_level).is_integer() else str(inspect_level)
            safe_task = inspect_task_id.replace("/", "_").replace(" ", "_")

            # The runner stores the per-token JSON under a deterministic path; iterate
            # the listing to find the one matching this task id.
            sample_dir_prefix = f"{prefix}/level={level_label}/"
            entries = list_results(RESULTS_BUCKET, sample_dir_prefix, aws_profile=AWS_PROFILE)
            target_keys = [e["key"] for e in entries if safe_task in e["key"] and e["key"].endswith("per_token.json")]
            assert target_keys, f"No per_token.json found for {inspect_task_id} at level {level_label}"
            payload = json.loads(s3.get_object(Bucket=RESULTS_BUCKET, Key=target_keys[0])["Body"].read())

            preview_rows = []
            for step in payload["per_token"][:50]:
                preview_rows.append({
                    "pos": step["position"],
                    "target_token": step["target_token_text"],
                    "target_logp": step["target_logprob"],
                    "rank": step["rank"],
                    "alt1": step["top_alternatives"][0]["token_text"] if step["top_alternatives"] else "",
                    "alt1_logp": step["top_alternatives"][0]["logprob"] if step["top_alternatives"] else None,
                    "alt2": step["top_alternatives"][1]["token_text"] if len(step["top_alternatives"]) > 1 else "",
                })
            pd.DataFrame(preview_rows)
            """
        ),
        _md(
            """
            ## Summary

            Per-token log-probabilities are now in S3 alongside the
            free-form eval results. The combination of the two notebooks gives
            you both the autoregressive performance metric (pass@1, accuracy)
            and the calibrated next-token confidence on the gold answer at
            each pruning level.
            """
        ),
        _code(
            """
            print("TF_RESULTS_URI =", repr(TF_RESULTS_URI))
            """
        ),
    ]
    write_notebook(NOTEBOOKS / "04_teacher_forced.ipynb", cells)


def main() -> None:
    """Build all four notebooks (idempotent; overwrites)."""

    build_notebook_01()
    build_notebook_02()
    build_notebook_03()
    build_notebook_04()


if __name__ == "__main__":  # pragma: no cover
    main()
