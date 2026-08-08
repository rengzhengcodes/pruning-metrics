"""Run the v2 diagnosticity analysis (notebook 07) on a CPU spot box.

The local workstation cannot hold a ~1 h compute process (container pid
exhaustion from an unrelated workload), so this runner executes
``notebooks/experiment/07_diagnosticity.ipynb`` on EC2 instead:

1. ``pip install`` the notebook's analysis dependencies (no torch needed).
2. Pull every run's ``per_token.json`` / ``*.digest.npz`` / ``summary.json``
   from ``s3://<bucket>/prune_eval_v2/`` into the notebook's local cache dir
   (in-region, concurrent).
3. Execute the notebook headlessly with ``jupyter nbconvert`` (the notebook
   itself is checkpoint-resumable and tolerant of partial data).
4. Upload ``07_executed.ipynb`` plus ``results/v2_*`` artifacts to
   ``s3://<bucket>/<results-prefix>/<run_id>/``.

The repo tarball shipped by the launcher includes
``notebooks/experiment/experiment_config_v2.json`` (untracked but present),
which is the notebook's run inventory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from infra.runners._runner_common import (  # noqa: E402
    LOGGER,
    add_common_runner_args,
    configure_logging,
    env_or,
    write_json,
)

ANALYSIS_DEPS = [
    "nbformat",
    "nbconvert",
    "jupyter",
    "ipykernel",
    "scikit-learn",
    "scipy",
    "pandas",
    "matplotlib",
    # Section 7 embeds with all five reducers; UMAP is the only one not in
    # scikit-learn, and without it that whole section raises ImportError.
    "umap-learn",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_runner_args(parser, default_results_prefix="prune_eval_v2_analysis")
    parser.add_argument(
        "--data-prefix",
        default=env_or("V2_DATA_PREFIX", default="prune_eval_v2"),
        help="S3 prefix holding the prune/eval runs to analyse.",
    )
    parser.add_argument(
        "--permutations",
        default=env_or("V2_PERMUTATIONS", default="4999"),
    )
    parser.add_argument(
        "--benches",
        default=env_or("V2_BENCHES", default=""),
        help=(
            "Comma-separated benchmark substrings to analyse (default: all). "
            "Scopes both the S3 cache sync and the notebook itself, which "
            "matters because one benchmark's matrices can cost 20x another's."
        ),
    )
    return parser.parse_args()


def _pip_install() -> None:
    LOGGER.info("Installing analysis dependencies: %s", ANALYSIS_DEPS)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *ANALYSIS_DEPS],
        check=True,
    )


def _sync_cache(
    bucket: str, data_prefix: str, cache_dir: Path, benches: list[str] | None = None
) -> int:
    """Concurrent pull of analysis inputs from S3 into the notebook cache.

    Parameters
    ----------
    bucket, data_prefix:
        Source location of the prune/eval runs.
    cache_dir:
        Local ``results/v2_cache`` directory to mirror into.
    benches:
        Optional benchmark substrings. When given, only ``per_token.json``
        under a matching ``bench=`` path component is fetched -- the whole set
        is ~163 k objects / 15 GB, and a single-benchmark analysis needs a
        fraction of it. Mask digests and summaries are always fetched: they are
        small and are not benchmark-scoped.

    Returns
    -------
    int
        Number of objects actually downloaded (already-present files are
        skipped, so a resumed run fetches nothing).
    """

    import boto3

    s3 = boto3.session.Session().client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{data_prefix}/"):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if key.endswith((".digest.npz", "summary.json", "manifest.json")):
                keys.append(key)
            elif key.endswith("per_token.json"):
                if not benches or any(f"bench={b}" in key or b in key for b in benches):
                    keys.append(key)
    LOGGER.info("Cache sync: %d objects to fetch", len(keys))

    def _fetch(key: str) -> bool:
        rel = key.split("/", 1)[1]  # strip "<data_prefix>/"
        dest = cache_dir / rel
        if dest.exists() and dest.stat().st_size > 0:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest))
        return True

    fetched = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        for got in pool.map(_fetch, keys, chunksize=16):
            fetched += int(got)
    LOGGER.info("Cache sync done: %d fetched, %d already present.", fetched, len(keys) - fetched)
    return fetched


def _upload_results(bucket: str, results_prefix: str, run_id: str, nb_dir: Path) -> int:
    import boto3

    s3 = boto3.session.Session().client("s3")
    uploads: list[Path] = []
    executed = nb_dir / "07_executed.ipynb"
    if executed.exists():
        uploads.append(executed)
    results_dir = nb_dir / "results"
    for pattern in ("v2_pairwise_*", "v2_mask_*", "v2_jaccard*", "v2_*.csv", "v2_*.json"):
        uploads.extend(results_dir.glob(pattern))
    for path in uploads:
        rel = path.relative_to(nb_dir)
        s3.upload_file(str(path), bucket, f"{results_prefix}/{run_id}/{rel}")
    LOGGER.info("Uploaded %d result files.", len(uploads))
    return len(uploads)


def main() -> int:
    configure_logging()
    args = parse_args()
    nb_dir = REPO_ROOT / "notebooks" / "experiment"
    cache_dir = nb_dir / "results" / "v2_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    benches = [b.strip() for b in str(args.benches).split(",") if b.strip()]

    _pip_install()
    _sync_cache(args.results_bucket, args.data_prefix, cache_dir, benches)

    env = dict(os.environ)
    env["V2_PERMUTATIONS"] = str(args.permutations)
    if benches:
        env["V2_BENCHES"] = ",".join(benches)
    # The cache was just mirrored above, so the notebook's own S3 sync would
    # only re-list ~200 k objects to find nothing new.
    env["V2_SKIP_SYNC"] = "1"
    LOGGER.info(
        "Executing 07_diagnosticity.ipynb (permutations=%s, benches=%s) ...",
        args.permutations, benches or "all",
    )
    proc = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "07_diagnosticity.ipynb",
            "--output", "07_executed.ipynb",
            "--ExecutePreprocessor.timeout=-1",
        ],
        cwd=nb_dir,
        env=env,
        check=False,
    )
    LOGGER.info("nbconvert exit code: %s", proc.returncode)

    n_up = _upload_results(args.results_bucket, args.results_prefix, args.run_id, nb_dir)
    write_json(
        Path(args.output_dir) / "analysis_metadata.json",
        {
            "run_id": args.run_id,
            "nbconvert_exit": proc.returncode,
            "uploaded_files": n_up,
            "permutations": str(args.permutations),
            "data_prefix": args.data_prefix,
        },
    )
    # Non-zero nbconvert exit still uploads whatever was produced, then fails
    # loudly so the operator sees it in the instance exit status.
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
