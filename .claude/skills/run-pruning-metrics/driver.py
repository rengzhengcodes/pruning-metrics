#!/usr/bin/env python3
"""Agent driver for the pruning-metrics project.

Runs the locally-runnable surface of the project from any cwd: venv setup,
metrics-library smoke test, pytest, headless notebook execution, figure
listing, and a read-only AWS credential check.

The script itself needs only the stdlib (any python3); every subcommand that
needs project dependencies shells out to the repo venv's interpreter, so run
``setup`` first on a clean machine.

Usage
-----
    python3 .claude/skills/run-pruning-metrics/driver.py setup
    python3 .claude/skills/run-pruning-metrics/driver.py smoke
    python3 .claude/skills/run-pruning-metrics/driver.py test
    python3 .claude/skills/run-pruning-metrics/driver.py notebook notebooks/experiment/05_tsne.ipynb
    python3 .claude/skills/run-pruning-metrics/driver.py figures
    python3 .claude/skills/run-pruning-metrics/driver.py aws-check

This driver never launches EC2 instances. ``aws-check`` performs a single
read-only STS ``GetCallerIdentity`` call and nothing else.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"

# nbformat/nbclient/ipykernel: headless notebook execution.
# matplotlib/pandas/scipy/scikit-learn/umap-learn: imports made by the
# notebooks in notebooks/experiment/ (05_tsne needs sklearn + umap).
ANALYSIS_DEPS = [
    "nbformat",
    "nbclient",
    "ipykernel",
    "matplotlib",
    "pandas",
    "scipy",
    "scikit-learn",
    "umap-learn",
]

SMOKE_SNIPPET = """
import json, math, sys
from pathlib import Path

repo = Path({repo!r})
sys.path.insert(0, str(repo / "src"))
from pruning_metrics.metrics import compute_kld, compute_jsd, compute_emd, compute_chamfer

cache = repo / "notebooks/experiment/results/tf_cache/gsm8k_gsm8k"
base_path = next((cache / "level=0").glob("sample=*/per_token.json"), None)
if base_path is None:
    sys.exit(
        "No cached teacher-forced data under notebooks/experiment/results/tf_cache/. "
        "The smoke test needs the results cache (synced by 04_metric_spaces.ipynb)."
    )
pruned_path = cache / "level=80" / base_path.parent.name / "per_token.json"
base = json.load(base_path.open())["per_token"]
pruned = json.load(pruned_path.open())["per_token"]
print(f"smoke: {{base_path.parent.name}} level 0 vs 80, {{len(base)}} positions")
failures = []
for name, fn in [("kld", compute_kld), ("jsd", compute_jsd),
                 ("emd", compute_emd), ("chamfer", compute_chamfer)]:
    v = fn(base, pruned)
    ok = math.isfinite(v) and v >= 0.0
    print(f"  {{name:8s}} {{v:.6f}}  {{'OK' if ok else 'BAD'}}")
    if not ok:
        failures.append(name)
sys.exit(f"non-finite/negative metrics: {{failures}}" if failures else 0)
"""

NOTEBOOK_SNIPPET = """
import sys, time
import nbformat
from nbclient import NotebookClient

nb_path, out_path = sys.argv[1], sys.argv[2]
t0 = time.time()
nb = nbformat.read(nb_path, as_version=4)
NotebookClient(nb, timeout=900, kernel_name="python3").execute()
nbformat.write(nb, out_path)
print(f"executed {{nb_path}} in {{time.time() - t0:.0f}}s")
print(f"executed copy: {{out_path}}")
""".format()

AWS_CHECK_SNIPPET = """
import os, sys
from dotenv import load_dotenv

load_dotenv({repo!r} + "/.env", override=False)
import boto3

profile = os.environ.get("AWS_PROFILE") or None
region = os.environ.get("AWS_REGION", "us-east-1")
print(f"profile={{profile}} region={{region}}")
try:
    session = boto3.session.Session(profile_name=profile)
    ident = session.client("sts", region_name=region).get_caller_identity()
    print(f"LIVE  account={{ident['Account']}}  arn={{ident['Arn']}}")
except Exception as exc:  # expired SSO, missing profile, no network, ...
    print(f"NOT USABLE: {{type(exc).__name__}}: {{exc}}")
    sys.exit(1)
"""


def _need_venv() -> None:
    if not VENV_PY.exists():
        sys.exit(f"{VENV_PY} missing — run the 'setup' subcommand first.")


def _run(cmd: list[str | Path], **kwargs) -> int:
    shown = [str(c) if len(str(c)) < 120 else "<inline script>" for c in cmd]
    print(f"$ {' '.join(shown)}", flush=True)
    return subprocess.run([str(c) for c in cmd], **kwargs).returncode


def cmd_setup(_args: argparse.Namespace) -> int:
    if not VENV_PY.exists():
        rc = _run([sys.executable, "-m", "venv", REPO_ROOT / ".venv"])
        if rc:
            return rc
    for pip_args in (
        ["--upgrade", "pip"],
        ["-e", str(REPO_ROOT) + "[dev]"],
        ANALYSIS_DEPS,
    ):
        rc = _run([VENV_PY, "-m", "pip", "install", "-q", *pip_args], cwd=REPO_ROOT)
        if rc:
            return rc
    print("setup OK:", VENV_PY)
    return 0


def cmd_smoke(_args: argparse.Namespace) -> int:
    _need_venv()
    return _run([VENV_PY, "-c", SMOKE_SNIPPET.format(repo=str(REPO_ROOT))])


def cmd_test(_args: argparse.Namespace) -> int:
    _need_venv()
    return _run([VENV_PY, "-m", "pytest", "-q"], cwd=REPO_ROOT)


def cmd_notebook(args: argparse.Namespace) -> int:
    _need_venv()
    nb_path = (REPO_ROOT / args.path).resolve() if not Path(args.path).is_absolute() else Path(args.path)
    if not nb_path.exists():
        sys.exit(f"notebook not found: {nb_path}")
    out = args.out or str(Path(tempfile.gettempdir()) / (nb_path.stem + ".executed.ipynb"))
    # cwd must be the notebook's own directory: cells derive RESULTS_DIR from
    # Path.cwd(), so running from repo root would read/write the wrong paths.
    return _run(
        [VENV_PY, "-c", NOTEBOOK_SNIPPET, nb_path, out],
        cwd=nb_path.parent,
    )


def cmd_figures(_args: argparse.Namespace) -> int:
    results = REPO_ROOT / "notebooks" / "experiment" / "results"
    pngs = sorted(results.glob("*_figures/*.png"))
    if not pngs:
        print(f"no figures under {results}/*_figures/")
        return 1
    for p in pngs:
        import datetime

        mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime)
        print(f"{mtime:%Y-%m-%d %H:%M:%S}  {p.stat().st_size // 1024:>5d} KB  {p.relative_to(REPO_ROOT)}")
    return 0


def cmd_aws_check(_args: argparse.Namespace) -> int:
    _need_venv()
    return _run([VENV_PY, "-c", AWS_CHECK_SNIPPET.format(repo=str(REPO_ROOT))])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="create .venv and install all deps").set_defaults(fn=cmd_setup)
    sub.add_parser("smoke", help="compute all 4 metrics on real cached data").set_defaults(fn=cmd_smoke)
    sub.add_parser("test", help="run pytest").set_defaults(fn=cmd_test)
    nb = sub.add_parser("notebook", help="execute a notebook headlessly (cwd = its dir)")
    nb.add_argument("path", help="notebook path, relative to repo root")
    nb.add_argument("--out", help="where to write the executed copy (default: $TMPDIR)")
    nb.set_defaults(fn=cmd_notebook)
    sub.add_parser("figures", help="list figure PNGs with mtimes").set_defaults(fn=cmd_figures)
    sub.add_parser("aws-check", help="read-only STS identity check (never launches)").set_defaults(fn=cmd_aws_check)
    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
