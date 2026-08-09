"""Builder for 03_freeform_eval.ipynb: free-form eval per pruning level."""

# See notebook_builders.prune_llm for why duplicate-code is disabled here:
# notebooks 2/3/4 intentionally share orchestration snippets, and this
# refactor is explicitly scoped to NOT dedupe generated cell content.
# pylint: disable=duplicate-code
from ._shared import (
    COMMON_BOOTSTRAP_CELL,
    NOTEBOOKS,
    _code,
    _md,
    artifact_config_cell,
    write_notebook,
)

# ---------------------------------------------------------------------------
# Notebook 3: 03_freeform_eval.ipynb
# ---------------------------------------------------------------------------


def build_notebook_03() -> None:
    cells = [
        _md("""
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
            """),
        _code(COMMON_BOOTSTRAP_CELL),
        _md("""
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
            """),
        _code(
            artifact_config_cell(
                knobs="""\
EVAL_DATASET_SPEC = ""  # empty -> reuse the artifact's calibration spec
EVAL_LEVELS = [0, 20, 40, 60, 80]
GENERATION_SEED = 65320
MAX_NEW_TOKENS = 512
TIMEOUT_SECONDS = 10.0
MAX_TEST_SAMPLES = 0
""",
                summary_keys="""\
    "EVAL_DATASET_SPEC": EVAL_DATASET_SPEC or "(use artifact's)",
    "EVAL_LEVELS": EVAL_LEVELS,
    "GENERATION_SEED": GENERATION_SEED,
""",
            )
        ),
        _md("""
            ## Find capacity & launch the eval runner

            Same boilerplate as notebook 2 -- find a viable spot candidate
            and shell out to `launch_gpu_instance.py` with `--runner freeform_eval`.
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
            """),
        _md("""
            ## Wait until the run is **finished** (not merely started)

            Seeing `summary.json` in S3 does **not** mean the eval is done: the
            runner rewrites that file after **each** pruning level, with
            `ended_at_utc` left null until the final `finally` block. The
            instance can stay `running` for a long time after the first upload.

            This cell calls ``wait_for_runner_completion``, which returns as soon
            as `summary.json` contains a non-null `ended_at_utc`, or when the
            EC2 instance terminates (fallback if the process dies without a
            final write). The next cell still waits for the object to exist
            before plotting (instant once the final summary landed).
            """),
        _code("""
            from pruning_metrics.notebook_helpers import (
                list_results,
                wait_for_runner_completion,
            )

            print(
                "Waiting for final summary (ended_at_utc) or instance shutdown. "
                "Ctrl-C stops this kernel loop only; the instance keeps running."
            )
            try:
                reason, _summary = wait_for_runner_completion(
                    bucket=RESULTS_BUCKET,
                    summary_key=f"freeform_eval/{launched.run_id}/summary.json",
                    instance_id=launched.instance_id,
                    region=launched.region,
                    aws_profile=AWS_PROFILE,
                    poll_seconds=60.0,
                    timeout_seconds=60 * 60 * 6,
                    progress_log_interval_seconds=300.0,
                )
                print("Completion signal:", reason)
            except KeyboardInterrupt:
                print("Stopped polling. The EC2 box keeps running until it self-shuts.")

            for entry in list_results(
                RESULTS_BUCKET, f"freeform_eval/{launched.run_id}", aws_profile=AWS_PROFILE
            ):
                print(f"  {entry['size']:>12d}  {entry['key']}")
            """),
        _md("""
            ## Pull and display the per-level metrics

            Waits for `summary.json` (the runner creates it after the first
            level completes), then reads the per-level aggregates for pandas +
            matplotlib.
            """),
        _code("""
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
                    "average_perplexity": entry.get("average_perplexity"),
                    "elapsed_seconds": entry.get("elapsed_seconds"),
                })
            df = pd.DataFrame(level_rows).sort_values("pruning_level").reset_index(drop=True)
            df
            """),
        _code("""
            import matplotlib.pyplot as plt

            title = (
                f"{summary.get('base_model_id', '?')}"
                f"\\n{summary.get('eval_dataset_spec', '?')}"
            )

            fig, (ax_acc, ax_ppl) = plt.subplots(1, 2, figsize=(12, 4))

            if df.empty:
                for ax in (ax_acc, ax_ppl):
                    ax.text(0.5, 0.5, "No levels in summary.json", ha="center", va="center")
            else:
                # Left: accuracy / pass@1
                ax_acc.plot(df["pruning_level"], df["pass_at_1"], "o-", linewidth=2)
                ymax = max(0.05, float(df["pass_at_1"].max()) * 1.1)
                ax_acc.set_ylim(0.0, ymax)
                ax_acc.set_xlabel("Pruning level (% sparsity)")
                ax_acc.set_ylabel("pass@1 (or accuracy for math/MCQ)")
                ax_acc.set_title(title)
                ax_acc.grid(True, alpha=0.3)

                # Right: perplexity of ground-truth answer under teacher forcing
                ppl_series = df["average_perplexity"].dropna()
                if not ppl_series.empty:
                    ax_ppl.plot(
                        df.loc[ppl_series.index, "pruning_level"],
                        ppl_series,
                        "s--",
                        color="tab:orange",
                        linewidth=2,
                    )
                else:
                    ax_ppl.text(0.5, 0.5, "No perplexity data", ha="center", va="center")
                ax_ppl.set_xlabel("Pruning level (% sparsity)")
                ax_ppl.set_ylabel("Perplexity of gold answer (teacher-forced)")
                ax_ppl.set_title(title)
                ax_ppl.grid(True, alpha=0.3)

            fig.tight_layout()
            fig
            """),
        _md("""
            ## Optional: inspect a single task's verification record

            Every level writes one JSON record per test task. Inspect any
            level / task to see the prompt, the model's generation, the
            target text, and the verifier's status.
            """),
        _code("""
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
            """),
        _md("""
            ## Done

            Final artifacts are at the URI below. Re-run notebook 4
            (teacher-forced log-probs) to compare per-token confidences.
            """),
        _code("""
            print("FREEFORM_RESULTS_URI =", repr(FREEFORM_RESULTS_URI))
            """),
    ]
    write_notebook(NOTEBOOKS / "03_freeform_eval.ipynb", cells)
