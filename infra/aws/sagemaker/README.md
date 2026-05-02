# SageMaker Qwen pruning workflow

This directory contains scripts to prune Qwen2.5-72B at levels
`0,20,40,60,80`, publish artifacts to S3, deploy a single SageMaker endpoint,
and invoke that endpoint with per-request seed and pruning level.

## 1) Prune and register model levels

```bash
python infra/aws/sagemaker/prune_and_register.py \
  --base-model-id Qwen/Qwen2.5-72B \
  --calibration-source humanevalplus_prompts \
  --artifact-bucket <artifact-bucket> \
  --artifact-prefix qwen-pruning \
  --region us-east-1
```

Custom calibration source format:

```text
hf:<dataset_name>:<split>:<text_field>
```

Example:

```text
hf:codeparrot/github-code:train:code
```

## 2) Build and publish serving image

```bash
docker build -t qwen-serving:latest infra/containers/qwen-serving
# tag + push to ECR (replace placeholders)
docker tag qwen-serving:latest <account>.dkr.ecr.<region>.amazonaws.com/qwen-serving:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/qwen-serving:latest
```

## 3) Deploy or update endpoint

```bash
python infra/aws/sagemaker/deploy_endpoint.py \
  --container-image-uri <ecr-image-uri> \
  --manifest-s3-uri s3://<artifact-bucket>/qwen-pruning/model_manifest.json \
  --region us-east-1 \
  --role-arn <sagemaker-role-arn> \
  --logits-bucket <logits-bucket> \
  --logits-prefix logits
```

## 4) Invoke endpoint

```bash
python infra/aws/sagemaker/invoke_endpoint.py \
  --endpoint-name qwen-pruning-endpoint \
  --prompt "Write a Python function that returns n squared." \
  --task-id manual/debug \
  --pruning-level 40 \
  --seed 7 \
  --region us-east-1
```

The response includes `logits_s3_uri` for downstream analysis.
