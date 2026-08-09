"""Shared notebook-authoring helpers for the ``notebook_builders`` package.

Every per-notebook module (``setup_aws``, ``prune_llm``, ``freeform_eval``,
``teacher_forced``) imports its cell constructors, ``write_notebook``, and
the common bootstrap cell from here, so there is exactly one implementation
of "how a notebook gets written to disk" shared across all four builders.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat import v4 as nb

# Design: this module lives one directory deeper than the original
# scripts/build_notebooks.py (scripts/notebook_builders/_shared.py vs.
# scripts/build_notebooks.py), so the walk up to the repo root needs one
# extra `.parent` step -- parents[2] instead of parents[1] -- to keep
# resolving to the same <repo>/notebooks/aws_tutorial directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO_ROOT / "notebooks" / "aws_tutorial"
NOTEBOOKS.mkdir(parents=True, exist_ok=True)


def _md(text: str) -> nbformat.NotebookNode:
    """Build a markdown cell from a dedented heredoc."""

    return nb.new_markdown_cell(dedent(text).strip() + "\n")


def _code(text: str) -> nbformat.NotebookNode:
    """Build a code cell from a dedented heredoc."""

    return nb.new_code_cell(dedent(text).strip() + "\n")


def write_notebook(path: Path, cells: list[nbformat.NotebookNode]) -> None:
    """Materialise the notebook on disk with python3 metadata."""

    notebook = nb.new_notebook(cells=cells)
    # nbformat gives every new cell a random id, which would make each
    # rebuild a spurious diff. Pin ids to cell position so regeneration is
    # byte-stable and `git diff` after a rebuild shows real changes only.
    for index, cell in enumerate(notebook.cells):
        cell.id = f"cell-{index:02d}"
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    nbformat.write(notebook, path)
    print(f"Wrote {path}")


# Common header used across every notebook so users know the prerequisites.
COMMON_BOOTSTRAP_CELL = """
import os
import sys
from pathlib import Path

# Allow the notebook to be run from anywhere by pinning to the repo root.
REPO_ROOT = Path.cwd()
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "pyproject.toml").is_file():
    REPO_ROOT = REPO_ROOT.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Optional: load .env so AWS_PROFILE etc. surface in the kernel.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:
    pass

print("REPO_ROOT =", REPO_ROOT)
print("AWS_PROFILE =", os.environ.get("AWS_PROFILE"))
print("AWS_REGION  =", os.environ.get("AWS_REGION"))
"""


# ---------------------------------------------------------------------------
# Shared artifact-config cell (notebooks 03 and 04)
# ---------------------------------------------------------------------------

#: Shared text of the eval notebooks' artifact-config cell: the artifact
#: discovery helper, URI resolution and fallback. Identical in 03 and 04.
_ARTIFACT_CONFIG_HEAD = """import json
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
"""

#: Shared middle: instance/region priorities and the summary-print opener.
_ARTIFACT_CONFIG_MID = """
INSTANCE_TYPE_PRIORITY = ["p4de.24xlarge", "p5.48xlarge", "p4d.24xlarge"]
REGION_PRIORITY = ["us-east-1", "us-west-2", "us-east-2"]
HF_TOKEN = os.environ.get("HF_TOKEN", "")

print(json.dumps({
    "PRUNING_ARTIFACT_URI": PRUNING_ARTIFACT_URI,
"""

#: Shared tail: summary-print closer and the artifact-URI asserts.
_ARTIFACT_CONFIG_TAIL = """}, indent=2))
assert PRUNING_ARTIFACT_URI.startswith("s3://"), "Set PRUNING_ARTIFACT_URI."
assert "<" not in PRUNING_ARTIFACT_URI and ">" not in PRUNING_ARTIFACT_URI, (
    "Could not resolve PRUNING_ARTIFACT_URI. Run notebook 2 to completion "
    "(or set PRUNING_ARTIFACT_URI in .env) and re-run this cell."
)
"""


def artifact_config_cell(*, knobs: str, summary_keys: str) -> str:
    """Assemble notebook 03/04's artifact-config cell from shared text.

    The 30-line artifact-discovery helper, priority lists, and asserts are
    identical between the two eval notebooks; only each runner's knob
    block and printed summary keys differ.

    Parameters
    ----------
    knobs:
        Runner-specific configuration lines placed between the artifact
        fallback and the instance-priority lists. Newline-terminated.
    summary_keys:
        Runner-specific entries of the printed JSON summary.
        Newline-terminated.

    Returns
    -------
    str
        The complete cell source, ready for :func:`_code`.
    """
    return (
        _ARTIFACT_CONFIG_HEAD
        + knobs
        + _ARTIFACT_CONFIG_MID
        + summary_keys
        + _ARTIFACT_CONFIG_TAIL
    )
