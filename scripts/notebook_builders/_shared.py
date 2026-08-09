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
