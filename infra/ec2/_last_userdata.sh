#!/bin/bash
# EC2 user-data bootstrap for the Qwen2-72B WANDA pruning experiment.
#
# This script runs once on first boot of the spot GPU instance. It is rendered
# from infra/ec2/launch_gpu_instance.py with template variables substituted.
#
# Responsibilities:
# 1. Stream stdout+stderr to a CloudWatch-friendly log (no agent needed; we use
#    the SSM-managed instance core role + tail via SSM Session Manager).
# 2. Pull the repository tarball from S3 (uploaded by the launcher).
# 3. Install/refresh Python dependencies into the Deep Learning AMI's
#    pre-existing PyTorch conda environment.
# 4. Run infra/ec2/run_qwen_pruning_experiment.py with the configured
#    environment variables.
# 5. On success or failure, sync residual logs to S3 and shutdown to release
#    the spot reservation. Spot interruption notifications also trigger
#    shutdown so partial S3 sync still happens.
#
# Template variables (rendered by the launcher):
#   pruning-metrics-results-414266451290         S3 bucket for the tarball + results.
#   qwen2_72b_pruning/20260503T221034Z-d5b497/code/repo.tar.gz       S3 key under that bucket holding repo.tar.gz.
#   qwen2_72b_pruning         Object prefix for results (run id appended in script).
#   20260503T221034Z-d5b497                 Run identifier.
#   Qwen/Qwen2-72B          HF model id to load and prune.
#   0,20,40,60,80         Comma-separated pruning percentages.
#   65320             HumanEval+ split seed.
#   0.8             Train fraction.
#   65320                Teacher-forcing seed.
#                  Optional HF token (empty when not gated).

set -uo pipefail

# Cloud-init runs user-data as root.
LOG_DIR=/var/log/pruning-experiment
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/userdata.log") 2>&1

echo "===== userdata bootstrap started at $(date -u +%FT%TZ) ====="
echo "Instance metadata:"
TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" || true)
INSTANCE_ID=$(curl -sS -H "X-aws-ec2-metadata-token: ${TOKEN}" \
    http://169.254.169.254/latest/meta-data/instance-id || echo unknown)
INSTANCE_REGION=$(curl -sS -H "X-aws-ec2-metadata-token: ${TOKEN}" \
    http://169.254.169.254/latest/meta-data/placement/region || echo us-east-1)
echo "InstanceId=${INSTANCE_ID} Region=${INSTANCE_REGION}"

# Substituted by the launcher (single quotes prevent shell expansion of literals).
RESULTS_BUCKET='pruning-metrics-results-414266451290'
REPO_TARBALL_KEY='qwen2_72b_pruning/20260503T221034Z-d5b497/code/repo.tar.gz'
RESULTS_PREFIX='qwen2_72b_pruning'
RUN_ID='20260503T221034Z-d5b497'
BASE_MODEL_ID='Qwen/Qwen2-72B'
PRUNING_LEVELS='0,20,40,60,80'
SPLIT_SEED='65320'
TRAIN_FRAC='0.8'
TF_SEED='65320'
HF_TOKEN=''

WORK_DIR=/opt/pruning-metrics
RESULTS_DIR=/opt/results
mkdir -p "$WORK_DIR" "$RESULTS_DIR"

# Sync logs at the end no matter what (spot interrupt, crash, success).
cleanup() {
    rc=$?
    echo "===== cleanup (exit=$rc) at $(date -u +%FT%TZ) ====="
    if [ -n "$RESULTS_BUCKET" ]; then
        aws s3 cp "$LOG_DIR/userdata.log" \
            "s3://${RESULTS_BUCKET}/${RESULTS_PREFIX}/${RUN_ID}/_logs/userdata.log" \
            --region "$INSTANCE_REGION" || true
        if [ -d "$RESULTS_DIR" ]; then
            aws s3 sync "$RESULTS_DIR" \
                "s3://${RESULTS_BUCKET}/${RESULTS_PREFIX}/${RUN_ID}/" \
                --region "$INSTANCE_REGION" || true
        fi
    fi
    if [ "${SHUTDOWN_ON_EXIT:-yes}" = "yes" ]; then
        echo "Shutting down to release spot reservation."
        shutdown -h +1 || true
    fi
    exit $rc
}
trap cleanup EXIT

# Spot interruption signal (best-effort): poll the metadata service in the
# background and, if interruption is announced, sync results immediately.
(
    while true; do
        sleep 5
        STATUS=$(curl -sS -H "X-aws-ec2-metadata-token: ${TOKEN}" \
            -o /dev/null -w "%{http_code}" \
            http://169.254.169.254/latest/meta-data/spot/instance-action || echo 000)
        if [ "$STATUS" = "200" ]; then
            echo "===== spot interruption detected; syncing results ====="
            if [ -n "$RESULTS_BUCKET" ] && [ -d "$RESULTS_DIR" ]; then
                aws s3 sync "$RESULTS_DIR" \
                    "s3://${RESULTS_BUCKET}/${RESULTS_PREFIX}/${RUN_ID}/" \
                    --region "$INSTANCE_REGION" || true
            fi
            break
        fi
    done
) &

# AWS Deep Learning OSS Nvidia AMI for Ubuntu has Python 3.10+ + Pytorch under
# /opt/pytorch (DLAMI varies; we just pick whichever python is on PATH).
echo "Using python: $(which python3)"
python3 --version

# Fetch the repository tarball from S3.
cd "$WORK_DIR"
echo "Pulling repo tarball s3://${RESULTS_BUCKET}/${REPO_TARBALL_KEY}"
for attempt in 1 2 3; do
    if aws s3 cp "s3://${RESULTS_BUCKET}/${REPO_TARBALL_KEY}" /tmp/repo.tar.gz \
        --region "$INSTANCE_REGION"; then
        break
    fi
    echo "S3 cp failed (attempt $attempt); sleeping before retry"
    sleep 10
done
tar -xzf /tmp/repo.tar.gz -C "$WORK_DIR" --strip-components=0

# Install Python deps. transformers + datasets are usually in the DLAMI but we
# upgrade defensively. accelerate is required for device_map='auto'.
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade \
    "transformers>=4.42" \
    "accelerate>=0.30" \
    "datasets>=2.18" \
    "boto3>=1.34" \
    "python-dotenv>=1.0" \
    "huggingface_hub>=0.23"

if [ -n "$HF_TOKEN" ]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    export HF_TOKEN="$HF_TOKEN"
    python3 -c "from huggingface_hub import login; login('$HF_TOKEN', add_to_git_credential=False)" || true
fi

# Make the project importable without an editable install (avoids needing C-extension
# builds on the DLAMI).
export PYTHONPATH="$WORK_DIR/src:$PYTHONPATH"

# Run the experiment. Timestamp inside RUN_ID is already unique.
cd "$WORK_DIR"
mkdir -p "$RESULTS_DIR"

echo "===== launching run_qwen_pruning_experiment.py ====="
RESULTS_BUCKET="$RESULTS_BUCKET" \
RESULTS_PREFIX="$RESULTS_PREFIX" \
RUN_ID="$RUN_ID" \
BASE_MODEL_ID="$BASE_MODEL_ID" \
PRUNING_LEVELS="$PRUNING_LEVELS" \
HUMANEVAL_SPLIT_SEED="$SPLIT_SEED" \
HUMANEVAL_TRAIN_FRAC="$TRAIN_FRAC" \
RESULTS_LOCAL_DIR="$RESULTS_DIR" \
python3 -u "$WORK_DIR/infra/ec2/run_qwen_pruning_experiment.py" \
    --base-model-id "$BASE_MODEL_ID" \
    --pruning-levels "$PRUNING_LEVELS" \
    --split-seed "$SPLIT_SEED" \
    --train-frac "$TRAIN_FRAC" \
    --teacher-forcing-seed "$TF_SEED" \
    --output-dir "$RESULTS_DIR" \
    --results-bucket "$RESULTS_BUCKET" \
    --results-prefix "$RESULTS_PREFIX" \
    --run-id "$RUN_ID"

echo "===== run_qwen_pruning_experiment.py finished at $(date -u +%FT%TZ) ====="
