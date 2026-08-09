"""Builder for 02_prune_llm.ipynb: launches the pruning calibration run."""

# Design: notebooks 2/3/4 share several near-identical orchestration
# snippets (capacity lookup, launch_runner call shape, etc.) because they
# are genuinely running the same EC2-launch playbook against different
# runners. Splitting the monolithic build_notebooks.py into one module per
# notebook (per the target layout) makes pylint's cross-file duplicate-code
# checker newly visible to this pre-existing duplication -- it could not
# see across notebook sections while they lived in one file. Deduping the
# cell *content* itself is explicitly out of scope for this refactor (see
# task spec), so the checker is silenced here rather than worked around by
# templating notebook cells.
# pylint: disable=duplicate-code
from ._shared import COMMON_BOOTSTRAP_CELL, NOTEBOOKS, _code, _md, write_notebook

# ---------------------------------------------------------------------------
# Notebook 2: 02_prune_llm.ipynb
# ---------------------------------------------------------------------------


def build_notebook_02() -> None:
    cells = [
        _md("""
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
            """),
        _code(COMMON_BOOTSTRAP_CELL),
        _md("""
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
            """),
        _code("""
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
            """),
        _md("""
            ## Find a viable spot capacity candidate

            Queries `describe_spot_price_history` and `describe_instance_type_offerings`
            across the priority list and returns the cheapest-and-still-available
            `(region, AZ, instance_type)` tuple. Re-run this cell if your
            launch fails with `InsufficientInstanceCapacity` to get fresh
            candidates.
            """),
        _code("""
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
            """),
        _md("""
            ## Launch the calibration runner

            Tars the repo, uploads to S3, resolves the latest Deep Learning
            AMI, and calls `RunInstances` with the `pruning-metrics-ec2`
            instance profile attached. The runner uploads its artifact to
            `s3://<bucket>/pruning_artifacts/<run_id>/`.

            Pass `dry_run=True` to render the user-data without launching.
            """),
        _code("""
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
            """),
        _md("""
            ## Wait for the calibration artifact

            Polls S3 until `wanda_stats.pt` (the heaviest artifact) is
            present. Calibration on `p4de.24xlarge` typically takes 6-15
            minutes for Qwen2-72B (mostly model download + WANDA hooks).
            """),
        _code("""
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
            """),
        _md("""
            ## Inspect manifest + split

            The manifest records every input that produced the artifact so
            downstream notebooks can replay deterministically.
            """),
        _code("""
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
            """),
        _md("""
            ## Hand-off to notebooks 3 & 4

            Persist the artifact URI into `.env` so notebooks 3 and 4 pick it
            up automatically on next bootstrap (`load_dotenv` in cell 1).
            The same artifact can be reused across many downstream evaluations
            on different datasets.
            """),
        _code("""
            def _upsert_env_key(env_path, key, value):
                env_path = Path(env_path)
                lines = []
                if env_path.exists():
                    lines = env_path.read_text(encoding="utf-8").splitlines()
                replaced = False
                out = []
                for line in lines:
                    if line.startswith(f"{key}="):
                        out.append(f"{key}={value}")
                        replaced = True
                    else:
                        out.append(line)
                if not replaced:
                    if out and out[-1].strip():
                        out.append("")
                    out.append(f"{key}={value}")
                env_path.write_text("\\n".join(out).rstrip() + "\\n", encoding="utf-8")

            os.environ["PRUNING_ARTIFACT_URI"] = PRUNING_ARTIFACT_URI
            _upsert_env_key(REPO_ROOT / ".env", "PRUNING_ARTIFACT_URI", PRUNING_ARTIFACT_URI)
            print("Persisted PRUNING_ARTIFACT_URI to", REPO_ROOT / ".env")
            print("PRUNING_ARTIFACT_URI =", repr(PRUNING_ARTIFACT_URI))
            """),
    ]
    write_notebook(NOTEBOOKS / "02_prune_llm.ipynb", cells)
