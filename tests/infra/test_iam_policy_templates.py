"""Sanity checks for IAM policy JSON templates used by setup_prerequisites."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICIES = REPO_ROOT / "infra" / "aws" / "iam" / "policies"


def test_policy_templates_are_valid_json() -> None:
    """Each template file must parse as JSON (placeholders live inside string literals)."""

    for name in (
        "notebook-operator-policy.json",
        "sagemaker-execution-policy.json",
    ):
        path = POLICIES / name
        assert path.is_file(), f"missing {path}"
        json.loads(path.read_text(encoding="utf-8"))
