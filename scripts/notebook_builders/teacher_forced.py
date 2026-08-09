"""Builder for 04_teacher_forced.ipynb: teacher-forced next-token log-probs."""

# See notebook_builders.prune_llm for why duplicate-code is disabled here:
# notebooks 2/3/4 intentionally share orchestration snippets, and this
# refactor is explicitly scoped to NOT dedupe generated cell content.
# pylint: disable=duplicate-code
from ._shared import COMMON_BOOTSTRAP_CELL, NOTEBOOKS, _code, _md, write_notebook

# ---------------------------------------------------------------------------
# Notebook 4: 04_teacher_forced.ipynb
# ---------------------------------------------------------------------------


def build_notebook_04() -> None:
    cells = [
        _md("""
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
            """),
        _code(COMMON_BOOTSTRAP_CELL),
        _md("""
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
            """),
        _code("""
            import json
            import boto3

            AWS_PROFILE = os.environ.get("AWS_PROFILE", "rengz")
            RESULTS_BUCKET = os.environ.get(
                "RESULTS_BUCKET", "pruning-metrics-results-414266451290"
            )

            def _discover_latest_pruning_artifact_uri(results_bucket, aws_profile):
                session = boto3.session.Session(profile_name=aws_profile)
                s3 = session.client("s3")
                paginator = s3.get_paginator("list_objects_v2")
                run_ids = set()
                for page in paginator.paginate(Bucket=results_bucket, Prefix="pruning_artifacts/"):
                    for entry in page.get("Contents", []) or []:
                        key = entry["Key"]
                        parts = key.split("/")
                        if len(parts) >= 3 and parts[0] == "pruning_artifacts":
                            run_ids.add(parts[1])
                if not run_ids:
                    return ""
                latest = sorted(run_ids)[-1]
                return f"s3://{results_bucket}/pruning_artifacts/{latest}/"

            PRUNING_ARTIFACT_URI = os.environ.get("PRUNING_ARTIFACT_URI", "").strip()
            if "<" in PRUNING_ARTIFACT_URI or ">" in PRUNING_ARTIFACT_URI:
                PRUNING_ARTIFACT_URI = ""
            if not PRUNING_ARTIFACT_URI:
                PRUNING_ARTIFACT_URI = _discover_latest_pruning_artifact_uri(
                    RESULTS_BUCKET, AWS_PROFILE
                )
            if not PRUNING_ARTIFACT_URI:
                PRUNING_ARTIFACT_URI = "s3://pruning-metrics-results-414266451290/pruning_artifacts/<run_id>/"
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
                "Could not resolve PRUNING_ARTIFACT_URI. Run notebook 2 to completion "
                "(or set PRUNING_ARTIFACT_URI in .env) and re-run this cell."
            )
            """),
        _md("""
            ## Find capacity & launch the teacher-forced runner
            """),
        _code("""
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
            """),
        _md("""
            ## Wait until the run is **finished**

            Same as notebook 3: ``wait_for_runner_completion`` waits for
            `ended_at_utc` in `summary.json` (or instance termination as a
            fallback). Incremental summaries exist before the full job completes.

            `summary.json` is written after the first pruning level completes. If
            the runner crashes before that (bad artifact URI, OOM, etc.), the
            next cell will time out and list objects under this run prefix.
            """),
        _code("""
            from pruning_metrics.notebook_helpers import (
                list_results,
                wait_for_runner_completion,
            )

            print(
                "Waiting for final summary (ended_at_utc) or instance shutdown..."
            )
            try:
                reason, _summary = wait_for_runner_completion(
                    bucket=RESULTS_BUCKET,
                    summary_key=f"teacher_forced/{launched.run_id}/summary.json",
                    instance_id=launched.instance_id,
                    region=launched.region,
                    aws_profile=AWS_PROFILE,
                    poll_seconds=60.0,
                    timeout_seconds=60 * 60 * 4,
                    progress_log_interval_seconds=300.0,
                )
                print("Completion signal:", reason)
            except KeyboardInterrupt:
                print("Stopped polling; instance keeps running.")

            for entry in list_results(
                RESULTS_BUCKET, f"teacher_forced/{launched.run_id}", aws_profile=AWS_PROFILE
            ):
                print(f"  {entry['size']:>12d}  {entry['key']}")
            """),
        _md("""
            ## Per-level summary table

            Waits for `summary.json` before loading it (avoids S3 `NoSuchKey`
            if you run this cell before the first level finishes uploading).
            """),
        _code("""
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
            """),
        _code("""
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
            """),
        _md("""
            ## Per-token table for one (sample, level)

            Picks the first sampled task and the first level by default; tweak
            `inspect_task_id` and `inspect_level` to compare other slices.
            Each row is one teacher-forced position with the ground-truth
            target token's log-probability and the model's top-`TF_TOP_K`
            alternatives.
            """),
        _code("""
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
            """),
        _md("""
            ## Summary

            Per-token log-probabilities are now in S3 alongside the
            free-form eval results. The combination of the two notebooks gives
            you both the autoregressive performance metric (pass@1, accuracy)
            and the calibrated next-token confidence on the gold answer at
            each pruning level.
            """),
        _code("""
            print("TF_RESULTS_URI =", repr(TF_RESULTS_URI))
            """),
    ]
    write_notebook(NOTEBOOKS / "04_teacher_forced.ipynb", cells)
