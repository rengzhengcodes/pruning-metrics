# AWS prerequisites for the SageMaker pruning notebook (see infra/aws/sagemaker/README.md).
# Run from the repository root:
#   make -f infra/aws/Makefile verify
# PYTHONPATH must include the repo root so ``infra.*`` imports work.

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
PYTHON ?= python3
export PYTHONPATH := $(REPO_ROOT)$(if $(PYTHONPATH),:$(PYTHONPATH),)

SETUP := $(PYTHON) infra/aws/sagemaker/setup_prerequisites.py

.PHONY: verify ensure-s3 ensure-ecr setup iam-print

verify:
	cd $(REPO_ROOT) && $(SETUP) verify

ensure-s3:
	cd $(REPO_ROOT) && $(SETUP) ensure-s3-buckets

ensure-ecr:
	cd $(REPO_ROOT) && $(SETUP) ensure-ecr-repo

setup: ensure-s3 ensure-ecr verify

iam-print:
	cd $(REPO_ROOT) && $(SETUP) iam-print
