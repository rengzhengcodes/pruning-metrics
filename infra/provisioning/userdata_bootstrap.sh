#!/bin/bash
# EC2 user-data bootstrap for the pruning-metrics GPU runners.
#
# This script runs once on first boot of the spot GPU instance. It is rendered
# by infra/provisioning/launch_gpu_instance.py with the following template variables
# substituted in:
#
# See ``render_userdata`` in launch_gpu_instance.py for the full list of
# substituted placeholders. The placeholder names use a ``__NAME__`` shape
# to keep them visually distinct from real shell variables; do NOT echo the
# placeholder names anywhere else in this script (a naive substitution would
# pick them up and break the rendered output).
#
# Responsibilities:
# 1. Stream stdout+stderr to a log file synced to S3 on exit.
# 2. Pull the repository tarball from S3 (uploaded by the launcher).
# 3. Probe DLAMI conda envs for a python with torch pre-installed; fall back
#    to system pip if none.
# 4. Export runner-specific env vars and run the chosen runner script.
# 5. On success, failure, or spot interruption, sync residual logs + results
#    to S3 and shut down the instance.
#
# NOTE: deliberately not using `set -u`. AWS DLAMI shells leave several
# environment variables (PYTHONPATH, LD_LIBRARY_PATH, ...) unset, and we
# want to be tolerant when appending to them. We do use `set -o pipefail`
# so silent failures inside pipelines surface in the log.

set -o pipefail

LOG_DIR=/var/log/pruning-experiment
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/userdata.log") 2>&1

echo "===== userdata bootstrap started at $(date -u +%FT%TZ) ====="
# Diagnostics only: all S3 traffic uses RESULTS_BUCKET_REGION below, not
# the instance's own region.
echo "Instance metadata:"
TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" || true)
INSTANCE_ID=$(curl -sS -H "X-aws-ec2-metadata-token: ${TOKEN}" \
    http://169.254.169.254/latest/meta-data/instance-id || echo unknown)
INSTANCE_REGION=$(curl -sS -H "X-aws-ec2-metadata-token: ${TOKEN}" \
    http://169.254.169.254/latest/meta-data/placement/region || echo us-east-1)
echo "InstanceId=${INSTANCE_ID} Region=${INSTANCE_REGION}"

# Substituted by the launcher (single quotes prevent shell expansion of literals).
RESULTS_BUCKET='__RESULTS_BUCKET__'
# Region where the S3 bucket lives -- may differ from this instance's region.
# Using the bucket's region avoids PermanentRedirect (301) errors that occur
# when an instance in us-east-2 sends S3 requests to the us-east-1 bucket
# while specifying --region us-east-2.
RESULTS_BUCKET_REGION='__RESULTS_BUCKET_REGION__'
REPO_TARBALL_KEY='__REPO_TARBALL_KEY__'
RESULTS_PREFIX='__RESULTS_PREFIX__'
RUN_ID='__RUN_ID__'
HF_TOKEN='__HF_TOKEN__'
RUNNER_RELPATH='__RUNNER_RELPATH__'

WORK_DIR=/opt/pruning-metrics
RESULTS_DIR=/opt/results
mkdir -p "$WORK_DIR" "$RESULTS_DIR"

# Let the AWS CLI retry throttled/transient failures itself instead of
# wrapping every call in a shell loop.
export AWS_RETRY_MODE=standard
export AWS_MAX_ATTEMPTS=5

# Sync logs at the end no matter what (spot interrupt, crash, success).
cleanup() {
    rc=$?
    echo "===== cleanup (exit=$rc) at $(date -u +%FT%TZ) ====="
    if [ -n "$RESULTS_BUCKET" ]; then
        aws s3 cp "$LOG_DIR/userdata.log" \
            "s3://${RESULTS_BUCKET}/${RESULTS_PREFIX}/${RUN_ID}/_logs/userdata.log" \
            --region "$RESULTS_BUCKET_REGION" || true
        if [ -d "$RESULTS_DIR" ]; then
            aws s3 sync "$RESULTS_DIR" \
                "s3://${RESULTS_BUCKET}/${RESULTS_PREFIX}/${RUN_ID}/" \
                --region "$RESULTS_BUCKET_REGION" || true
        fi
    fi
    if [ "${SHUTDOWN_ON_EXIT:-yes}" = "yes" ]; then
        echo "Shutting down to release spot reservation."
        shutdown -h +1 || true
    fi
    exit $rc
}
trap cleanup EXIT

# Spot-interruption watchdog: best-effort sync when AWS announces a stop.
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
                    --region "$RESULTS_BUCKET_REGION" || true
            fi
            break
        fi
    done
) &

# Probe DLAMI python locations for one that already has torch.
PYTHON_BIN=""
for candidate in \
    /opt/pytorch/bin/python \
    /opt/conda/envs/pytorch/bin/python \
    /opt/conda/bin/python \
    /usr/local/bin/python3 \
    /usr/bin/python3
do
    if [ -x "$candidate" ] && "$candidate" -c "import torch" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        echo "Selected python with torch pre-installed: $PYTHON_BIN"
        break
    fi
done

# Runtime deps installed on top of the AMI's python. Keep the shared pins
# in sync with pyproject.toml [project.dependencies]; the extra entries
# (transformers/accelerate/huggingface_hub) are GPU-only runtime deps the
# local package deliberately does not declare.
PIP_PACKAGES=(
    "transformers>=4.42"
    "accelerate>=0.30"
    "datasets>=2.20"
    "boto3>=1.34"
    "python-dotenv>=1.0"
    "huggingface_hub>=0.23"
)

if [ -z "$PYTHON_BIN" ]; then
    echo "No DLAMI python had torch pre-installed; bootstrapping system python."
    apt-get update -y
    apt-get install -y python3-pip python3-venv
    PYTHON_BIN=/usr/bin/python3
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install --upgrade "torch>=2.3" "${PIP_PACKAGES[@]}"
else
    "$PYTHON_BIN" -m pip install --upgrade "${PIP_PACKAGES[@]}"
fi

echo "Using python: $PYTHON_BIN"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

# Fetch the repository tarball from S3. Retried by hand on top of the CLI's
# own retries because on first boot the instance-profile credentials can lag
# IMDS availability, and the CLI does not retry credential-resolution
# failures.
cd "$WORK_DIR"
echo "Pulling repo tarball s3://${RESULTS_BUCKET}/${REPO_TARBALL_KEY}"
for attempt in 1 2 3; do
    if aws s3 cp "s3://${RESULTS_BUCKET}/${REPO_TARBALL_KEY}" /tmp/repo.tar.gz \
        --region "$RESULTS_BUCKET_REGION"; then
        break
    fi
    echo "S3 cp failed (attempt $attempt); sleeping before retry"
    sleep 10
done
tar -xzf /tmp/repo.tar.gz -C "$WORK_DIR" --strip-components=0

# huggingface_hub reads these env vars directly; no explicit login() call
# needed (which would also put the token into a process argv).
if [ -n "$HF_TOKEN" ]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    export HF_TOKEN="$HF_TOKEN"
fi

# Disable the hf_xet Xet-CDN protocol. Unauthenticated Xet transfers are
# severely throttled (dataset parquet files take hours instead of seconds).
# Plain HTTPS is fast for both authenticated and unauthenticated users.
export HF_HUB_DISABLE_XET=1

# Point all boto3 / aws-cli calls at the bucket's region so cross-region
# instances (e.g. us-east-2 instance, us-east-1 bucket) don't get
# PermanentRedirect (301) errors.
export AWS_DEFAULT_REGION="$RESULTS_BUCKET_REGION"

# ${PYTHONPATH:-} guards against the variable being unset.
export PYTHONPATH="$WORK_DIR/src:$WORK_DIR:${PYTHONPATH:-}"

# Runner-specific env exports (rendered by the launcher).
export RESULTS_BUCKET RESULTS_PREFIX RUN_ID
export RESULTS_LOCAL_DIR="$RESULTS_DIR"
__RUNNER_ENV_EXPORTS__

cd "$WORK_DIR"
mkdir -p "$RESULTS_DIR"

echo "===== launching $RUNNER_RELPATH ====="
"$PYTHON_BIN" -u "$WORK_DIR/$RUNNER_RELPATH"

echo "===== $RUNNER_RELPATH finished at $(date -u +%FT%TZ) ====="
