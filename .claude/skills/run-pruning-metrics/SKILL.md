---
name: run-pruning-metrics
description: Build, run, test, or screenshot pruning-metrics — set up the venv, smoke-test the metrics library, run pytest, headlessly execute the analysis notebooks (t-SNE/metric-space figures), and check AWS creds. Use when asked to run the project, execute a notebook, regenerate figures, or verify a change works. Do NOT use for launching GPU spot sweeps or the aws_tutorial notebooks — those launch paid instances and are human-attended only.
---

# Run pruning-metrics

Research tooling for studying LLM pruning degradation. Two very different
surfaces: a **local analysis pipeline** (python library + Jupyter notebooks
over a 2.2 GB cached results directory — runs headless in minutes) and an
**AWS GPU pipeline** (notebooks that launch paid EC2 spot instances — never
run these unattended). The driver covers the local surface end-to-end.

All paths below are relative to the repo root. All commands were verified on
Linux, system python 3.12, in this container.

## Prerequisites

- python3 ≥ 3.10 with `venv` (no OS packages needed — headless matplotlib Agg,
  no GUI, no xvfb).
- Network access to pypi (setup) — and to S3 only for `04_metric_spaces.ipynb`
  and `aws-check`.
- The gitignored `notebooks/experiment/results/` cache (2.2 GB) must exist for
  the smoke test and notebooks 04/05. Without it there is no local data path:
  rebuilding it requires the paid GPU sweep + an S3 sync.

## Run (agent path) — the driver

```bash
python3 .claude/skills/run-pruning-metrics/driver.py setup       # venv + all deps (~2 min clean)
python3 .claude/skills/run-pruning-metrics/driver.py smoke       # 4 metrics on real cached data, <5 s
python3 .claude/skills/run-pruning-metrics/driver.py test        # pytest: 233 passed, 4 skipped (~70 s)
python3 .claude/skills/run-pruning-metrics/driver.py notebook notebooks/experiment/05_tsne.ipynb   # ~50 s
python3 .claude/skills/run-pruning-metrics/driver.py figures     # list output PNGs with mtimes
python3 .claude/skills/run-pruning-metrics/driver.py aws-check   # read-only STS identity, never launches
```

- `setup` recreates `.venv` if missing and installs `-e ".[dev]"` plus the
  notebook/analysis deps (nbclient, matplotlib, pandas, scipy, scikit-learn,
  umap-learn). Idempotent.
- `smoke` is the direct-invocation check for the layer most changes touch
  (`src/pruning_metrics/metrics/distributions.py`): loads a real cached
  teacher-forced token pair (gsm8k, level 0 vs 80) and asserts all four
  metrics (kld/jsd/emd/chamfer) are finite and non-negative.
- `notebook <path>` executes any notebook headlessly via nbclient **with cwd
  set to the notebook's own directory** (required — see Gotchas), writing the
  executed copy to `$TMPDIR/<name>.executed.ipynb` (override with `--out`).
  `--timeout` sets the **per-cell** limit (default 900 s); `07_diagnosticity`
  needs far more — see the matrix below.
- The pytest skips are torch-dependent tests (WANDA / teacher forcing);
  torch is deliberately not installed on workstations.

**Verify a change visually:** run the notebook, then `figures` — the PNGs under
`notebooks/experiment/results/{tsne,umap,pca,isomap,lle}_figures/`,
`notebooks/experiment/results/cal_signal_figures/` and
`notebooks/experiment/results/v2_embedding_figures/`
regenerate in place with fresh mtimes; open one to confirm it rendered.

**Dimensionality reduction lives in two places.** `05_tsne.ipynb` runs all five
reducers (PCA, t-SNE, UMAP, Isomap, LLE) over the **v1** 13×13 matrices;
`07_diagnosticity.ipynb` §7 runs the same five over the **v2** 232-variant
matrices and additionally scores each embedding (trustworthiness, continuity,
Kruskal stress-1, Shepard ρ) into `results/v2_embedding_quality.csv`. The shared
math is in `src/pruning_metrics/embedding.py` and
`src/pruning_metrics/metrics/embedding_quality.py` — both unit-tested, so prefer
changing those over editing notebook cells.

**Both notebooks also regress real degradation on embedding radius** (distance
from the unpruned baseline) — 05 against measured `pass_at_1_drop`
(`results/v1_embedding_r2.csv`, `results/r2_figures/`), 07 against
log-perplexity increase read from each run's `summary.json`
(`results/v2_embedding_r2.csv`). Every table carries a `raw` control column
using distance in the original matrix, which is the ceiling each reducer is
trying to preserve; t-SNE and UMAP lose most of it because their distance scale
saturates.

Notebook execution matrix (all verified):

| Notebook | Needs AWS? | Time | Outputs |
|---|---|---|---|
| `notebooks/experiment/05_tsne.ipynb` | no — fully cache-local | ~95 s | `results/{tsne,umap,pca,isomap,lle}_figures/`, `results/cal_signal_figures/`, `results/r2_figures/`, `results/v1_embedding_r2.csv` |
| `notebooks/experiment/04_metric_spaces.ipynb` | yes, read-only S3 (9 small summary.json + listings; run `aws-check` first) | ~6.5 min | `results/metric_space_*.csv`, pairwise `.npy` caches |
| `notebooks/experiment/07_diagnosticity.ipynb` | only to sync new runs — set `V2_SKIP_SYNC=1` to run purely off the local cache | **~25 min** on the one cached benchmark; **hours** if it has to build matrices (see below) | `results/v2_embedding_figures/`, `results/v2_embedding_quality.csv`, `results/v2_embeddings/`, `results/v2_jaccard.npy` |
| `notebooks/experiment/01–03`, `notebooks/aws_tutorial/01–04` | yes — **launches paid GPU spot instances** (01/02) | hours | S3 |

Never execute the GPU-launching notebooks headlessly. They call EC2
`RunInstances` on p4de/p5-class instances; a forgotten instance costs real
money. The driver has no subcommand for them on purpose.

### Scoping a `07_diagnosticity` run

Its cell 10 builds a 232×232 distance matrix per `(benchmark, metric)` from
163 k cached `per_token.json` files. Measured cost of the four benchmarks that
have no cached matrix: **~31 core-hours, 93 % of it `math:openai_gsm8k:main`**
alone (112-token answers, versus 3–6 for the MCQ sets). Three env knobs scope a
run; always pass `--timeout` well above the default 900 s:

```bash
# validate the analysis against the one benchmark that is already cached (~25 min)
V2_BENCHES='coding:evalplus_humanevalplus:test' V2_SKIP_SYNC=1 V2_PERMUTATIONS=999 \
  python3 .claude/skills/run-pruning-metrics/driver.py \
    notebook notebooks/experiment/07_diagnosticity.ipynb --timeout 43200
```

- `V2_BENCHES` — comma-separated substrings; restricts which benchmarks are built.
- `V2_SKIP_SYNC=1` — skip the S3 mirror entirely (no AWS needed).
- `V2_PERMUTATIONS` — default 4999; lower it for a smoke pass.

Cell 10 checkpoints every 25 tasks to `results/v2_ckpt_<bench>.npz` and resumes,
so a long build survives a killed kernel. Watch progress with
`len(np.load(ckpt, allow_pickle=True)["done_tasks"])` — nbclient buffers cell
output, so the cell prints nothing until it finishes.

## Direct invocation (no notebook)

```python
# .venv/bin/python, from repo root
import sys, json; sys.path.insert(0, "src")
from pruning_metrics.metrics import compute_kld, compute_jsd, compute_emd, compute_chamfer
steps = json.load(open("notebooks/experiment/results/tf_cache/gsm8k_gsm8k/level=0/sample=000_task=gsm8k_test_00000/per_token.json"))["per_token"]
```

Each metric takes two `list[TokenStepDict]` (the `per_token` field) and
returns a float. `pruning_metrics.notebook_helpers.launch_runner` accepts
`dry_run=True` to render an EC2 launch plan without launching.

## Run (human path)

`jupyter lab notebooks/experiment/` and run cells in order (kernel = the
`.venv` interpreter). The AWS bootstrap flow is `notebooks/aws_tutorial/01–04`
with `.env` configured from `template.env`. Useless headless; use the driver.

## Gotchas

- **Notebooks are cwd-sensitive.** Every notebook derives `REPO_ROOT` and
  `RESULTS_DIR` from `Path.cwd()`. Execute them with cwd = the notebook's own
  directory (the driver does this) or they silently read/write wrong paths.
- **The README's notebook paths are stale.** It documents `notebooks/*.ipynb`;
  the real files are `notebooks/aws_tutorial/*.ipynb` (AWS walkthrough) and
  `notebooks/experiment/*.ipynb` (the actual experiment + analysis).
- **`scripts/build_notebooks.py` is stale too**: it regenerates the tutorial
  notebooks into `notebooks/` root, not `notebooks/aws_tutorial/`. Don't run
  it expecting to rebuild the tutorial in place.
- **`results/` is gitignored** — figures and CSVs regenerate in place with no
  git noise, but also no version history. `04_metric_spaces` rewrites
  `metric_space_{combined,distances,r2}.csv` on every run.
- `.venv` is currently complete and working (sklearn 1.9, umap-learn 0.5.12,
  numpy 2.4.6, scipy 1.18, matplotlib 3.11). `.venv312` lacks `umap-learn` and
  cannot run 05's UMAP section — prefer `.venv`, which is what the driver uses.
  `setup` rebuilds it if missing.
- `pruning_metrics.metrics` **raises ImportError at import time** if scipy is
  missing (EMD dependency) — not at call time.
- **Mask digests must not be loaded unpacked in bulk.** `load_digest` returns
  bool arrays — 204 MB per variant, ~47 GB for all 232. Use
  `load_digest_packed` / `jaccard_matrix_packed`, which keep the on-disk bit
  packing (25 MB each) and tile the pair loop.
- `r_squared.png` at the repo root **is tracked in git** (verified with
  `git ls-files`), though it is not reproducible from the committed notebooks.
  Leave it alone.

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| Figures/results "disappear" when checked right after running a notebook | You `cd`'d into `notebooks/experiment` earlier and your relative path doubled up. Use absolute paths; the data was fine. |
| `IPKernelApp WARNING … Kernel is running over TCP without encryption` on every nbclient run | Harmless local-kernel noise; ignore. |
| `04_metric_spaces` cell 9 prints `SKIP s3://…` for every run, then a pandas KeyError | AWS creds expired (freeform summaries have no local cache). Run `aws-check`; refresh SSO (`aws sso login --profile rengz`), retry. |
| `driver.py smoke` exits with "No cached teacher-forced data" | The `results/` cache is absent on this machine. Only `test` and `setup` are runnable; the data path needs the S3 sync inside `04_metric_spaces.ipynb` (live creds) or a full GPU sweep. |
