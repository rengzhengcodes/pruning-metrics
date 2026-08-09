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
python3 .claude/skills/run-pruning-metrics/driver.py test        # pytest: 359 passed, 4 skipped (~120 s)
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
`notebooks/experiment/results/cal_signal_figures/`,
`notebooks/experiment/results/metric_family_figures/`,
`notebooks/experiment/results/sweep_figures/`,
`notebooks/experiment/results/grid_figures/` and
`notebooks/experiment/results/v2_embedding_figures/`
regenerate in place with fresh mtimes; open one to confirm it rendered.

**`REDUCERS` holds fourteen reducers, not five.** `05_tsne.ipynb` and
`07_diagnosticity.ipynb` §7 still use only the original five (PCA, t-SNE, UMAP,
Isomap, LLE) — 05 over the **v1** 13×13 matrices, 07 over the **v2** 232-variant
ones with quality scores into `results/v2_embedding_quality.csv`. The other nine
(`mds`, `nmds`, `spectral`, `kpca_rbf`, `lle_modified`, `lle_hessian`, `ltsa`,
`ica`, `random`) are exercised by **`09_reducer_sweep.ipynb`**, which runs the
full distance × reducer cross product. The shared math is in
`src/pruning_metrics/embedding.py` and
`src/pruning_metrics/metrics/embedding_quality.py` — both unit-tested, so prefer
changing those over editing notebook cells.

Two reducer rows are not candidates and must not be read as such:
`random` (Gaussian random projection) is the **control** — the empirical noise
floor a real reducer has to beat — and `ica` differs from `pca` only by
FastICA's whitening, because a rotation cannot change any pairwise distance and
therefore cannot change any score in this repository. Eight of the fourteen have
no `metric="precomputed"` mode and are fed `classical_mds_coords(D)`, which
silently discards the negative eigenvalues; always score an embedding against
the **original** `D`, never against those coordinates, or a coords-based reducer
gets graded on its own preprocessed input.

**There are sixteen distributional distances, not four.**
They live one-module-per-measure in `src/pruning_metrics/prob_measures/`
(`kld`, `jsd`, `emd`, `chamfer`, `rkld`, `jeffreys`, `tv`, `hellinger`,
`bhattacharyya`, `renyi05`, `chisq`, `renyi2`, `triangular`, `l2`, `cosine`,
`wasserstein2`); `metrics/distributions.py` is a backwards-compatible facade
re-exporting them. Notebooks 04/05/07 still use the original four; **08
compares all sixteen**. When computing more than one, call `compute_all` rather than
looping over the individual functions — it shares the per-position union-support
alignment, so all sixteen cost about as much as one, and it returns bit-identical
floats (there is a provenance cell in 08 that asserts this against the cached
matrices). `METRIC_INFO` carries each measure's family, symmetry, boundedness and
formula for tables and axis labels.

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
| `notebooks/experiment/08_distribution_metrics.ipynb` | no — fully cache-local | **~4.5 min** cold (builds 36 matrices), **~50 s** once cached | `results/metric_family_figures/`, `results/metric_{scale_audit,mds_spectrum,agreement,family_r2}.csv`, 36 new `pairwise_dist_*.npy` |
| `notebooks/experiment/09_reducer_sweep.ipynb` | no — fully cache-local | **~5.5 min** (952 embeddings; `V1_ONLY=1` cuts it to ~40 s but skips the v2 grid sheets) | `results/sweep_figures/`, `results/grid_figures/`, `results/reducer_sweep_{v1,v2}.csv` |
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
- **`scripts/build_notebooks.py` regenerates the tutorial in place.** It
  writes `notebooks/aws_tutorial/*.ipynb` deterministically (pinned cell
  ids), so a rebuild with no source change leaves `git diff` empty. It also
  strips any saved cell outputs — don't run it to preserve executed output.
- **`results/` is gitignored** — figures and CSVs regenerate in place with no
  git noise, but also no version history. `04_metric_spaces` rewrites
  `metric_space_{combined,distances,r2}.csv` on every run.
- **Never delete or overwrite `results/pairwise_dist_{bench}_{kld,jsd,emd,chamfer}.npy`.**
  Those four per benchmark are the provenance of the figures in 04/05/07 and
  predate the batch builder. Notebook 08 loads them and writes only the twelve
  new metrics per benchmark; its cell 11 asserts a fresh batch build reproduces
  them bit-for-bit, which is the only thing tying old figures to new numbers.
  Deleting one silently rebuilds it — same values, but the check stops meaning
  anything.
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
