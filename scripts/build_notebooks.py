"""Generate the four orchestration notebooks programmatically.

Authoring notebooks via :mod:`nbformat` keeps them under version control as
plain Python and avoids the noise of hand-edited ``.ipynb`` JSON. Run with
``python scripts/build_notebooks.py`` from the repo root; the script
overwrites the four notebook files in ``notebooks/aws_tutorial/``.
Regeneration is deterministic, so a rebuild with no source changes leaves
``git diff`` empty. It also strips any saved cell outputs.

The notebooks themselves stay short: each cell is either a markdown
explanation or a thin orchestration call that delegates to
``pruning_metrics.notebook_helpers``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Design: scripts/ is not a package (no __init__.py), so `notebook_builders`
# is not importable via the normal package-relative mechanism when this file
# is run directly (``python scripts/build_notebooks.py``) from an arbitrary
# cwd. Prepending this file's own directory to sys.path makes
# `notebook_builders` resolve as a top-level import regardless of the
# caller's cwd, matching how the pre-split monolithic script could be run
# from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from notebook_builders import (  # noqa: E402  pylint: disable=wrong-import-position
    build_notebook_01,
    build_notebook_02,
    build_notebook_03,
    build_notebook_04,
)


def main() -> None:
    """Build all four notebooks (idempotent; overwrites)."""

    build_notebook_01()
    build_notebook_02()
    build_notebook_03()
    build_notebook_04()


if __name__ == "__main__":  # pragma: no cover
    main()
