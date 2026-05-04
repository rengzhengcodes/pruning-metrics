# pruning-metrics
Making a metric space on pruning scenarios.

## HumanEval+ bootstrap
This repository now includes a bootstrap coding-evaluation pipeline that:
- loads tasks from `evalplus/humanevalplus`,
- sends prompts to a pluggable LLM interface (mock, Bedrock placeholder, SageMaker placeholder),
- verifies model outputs with the HumanEval+ test harness.

The HumanEval+ dataset is sourced from Hugging Face:
- https://huggingface.co/datasets/evalplus/humanevalplus

## Project structure
- `src/pruning_metrics/evals/coding/humaneval_plus_dataset.py`: dataset loader and task schema.
- `src/pruning_metrics/evals/coding/llm_client.py`: provider-agnostic LLM client interface.
- `src/pruning_metrics/evals/coding/verifier.py`: subprocess-based correctness verifier.
- `src/pruning_metrics/evals/coding/pipeline.py`: end-to-end prompt/generate/verify orchestration.
- `scripts/run_humaneval_plus.py`: CLI runner.
- `tests/evals/coding/`: dataset, verifier, and pipeline smoke tests.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run a 3-task smoke evaluation
By default, the runner uses the mock client.

```bash
python scripts/run_humaneval_plus.py --provider mock --max-samples 3
```

To provide deterministic mock completions, pass a JSON file:

```json
{
  "HumanEval/0": "def has_close_elements(numbers, threshold):\n    return False\n"
}
```

```bash
python scripts/run_humaneval_plus.py \
  --provider mock \
  --max-samples 3 \
  --mock-completions-file path/to/mock_completions.json
```

## Provider configuration
- `--provider mock|bedrock|sagemaker`: selects inference backend.
- `--bedrock-model-id`: required with `--provider bedrock`.
- `--sagemaker-endpoint-name`: required with `--provider sagemaker`.
- `--pruning-level`: required with `--provider sagemaker`.
- `--seed`: required with `--provider sagemaker`.

Bedrock remains a placeholder adapter. SageMaker client invocation is implemented and
expects an endpoint that returns JSON with `generated_text` and metadata fields such
as `logits_s3_uri`.

## Output artifacts
The runner writes files under `artifacts/humaneval_plus/` by default:
- `records.jsonl`: one JSON record per task with prompt, generated code, and verification status.
- `summary.json`: aggregate metrics (`num_tasks`, `num_passed`, `pass_at_1`, status breakdown).

## Notebook demo
- `notebooks/aws_sagemaker_pruning_and_logprobs.ipynb`: end-to-end SageMaker walkthrough for pruning, deployment, invocation, and token log-probability reconstruction from logits artifacts. One-time AWS bootstrap: `make -f infra/aws/Makefile setup` from the repository root (see `infra/aws/sagemaker/README.md`). Suitable for **smaller-model** demos that fit on the SageMaker endpoint quota of the target account.

## Qwen2-72B EC2 GPU experiment
For the actual Qwen2-72B WANDA prune + HumanEval+ + teacher-forced log-probs run, the SageMaker-endpoint workflow is bypassed: GPU-endpoint quota for the instance types large enough to host a 72 B model is `0` in this account. Instead, a single EC2 spot GPU instance (p5.48xlarge / p4d.24xlarge) does pruning, evaluation, and teacher-forced scoring in one process. See `infra/ec2/README.md` for the full runbook.