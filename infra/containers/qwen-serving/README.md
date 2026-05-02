# qwen-serving container

This container implements SageMaker-compatible inference for Qwen2.5-72B pruned
checkpoints.

## Request payload

```json
{
  "prompt": "Write a function foo(x).",
  "task_id": "HumanEval/0",
  "pruning_level": 40,
  "seed": 7,
  "max_new_tokens": 256,
  "temperature": 0.0,
  "top_p": 1.0
}
```

## Response payload

```json
{
  "generated_text": "def foo(x):\n    return x\n",
  "task_id": "HumanEval/0",
  "pruning_level": 40,
  "seed": 7,
  "token_count": 18,
  "request_id": "uuid",
  "logits_s3_uri": "s3://bucket/logits/..."
}
```

## Environment variables

- `LOGITS_S3_BUCKET` (required): Destination bucket for per-token logits.
- `LOGITS_S3_PREFIX` (optional): Prefix under the bucket, default `logits`.
- `MODEL_MANIFEST_PATH` (optional): Path to pruning-level model manifest.
- `BASE_MODEL_PATH` (optional): Tokenizer source path, defaults to model dir.
