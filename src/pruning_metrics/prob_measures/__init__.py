"""Distributional distance functions for teacher-forced token predictions.

Each public metric function accepts two ``list[TokenStepDict]`` arguments —
one from the base (level=0) model and one from a pruned model — and
returns a single non-negative float aggregated over all token positions.
The input dicts match the serialised form of
:class:`pruning_metrics.evals.coding.teacher_forcing.TeacherForcedTokenStep`
so callers can pass ``json.load(per_token_path)["per_token"]`` directly.

Sixteen measures are provided, in four families (see :data:`METRIC_INFO`
for the machine-readable version of this table):

* *Aligned f-divergences and vector geometry* — computed on the two
  probability vectors at each position after they have been aligned onto
  the union of the two top-k support sets: ``kld``, ``rkld``,
  ``jeffreys``, ``jsd``, ``tv``, ``hellinger``, ``bhattacharyya``,
  ``renyi05``, ``chisq``, ``renyi2``, ``triangular``, ``l2``, ``cosine``.
* *Optimal transport* — computed on each model's own logprob values as
  atom positions on the real line: ``emd`` (Wasserstein-1),
  ``wasserstein2``.
* *Point cloud* — ``chamfer``, which matches positions by nearest
  neighbour rather than by index.

Layout:
    One module per measure (``kld.py``, ``jsd.py``, ...), each exposing
    ``NAME``, ``INFO``, ``compute`` and — for the union-support family —
    the per-position ``kernel``; see
    :mod:`pruning_metrics.prob_measures.base` for the contract and for
    the shared representation notes (missing-token fill, epsilon guards).
    :mod:`pruning_metrics.prob_measures.registry` assembles the modules
    into :data:`METRIC_FUNCS` / :data:`METRIC_INFO` and provides
    :func:`compute_all`.

Computing several measures separately re-parses the same JSON and
rebuilds the same union-support vectors once per measure.
:func:`compute_all` does one pass and returns every measure, which is
what the notebooks use; the single-metric functions remain the readable
reference implementations and are what the unit tests check
``compute_all`` against.

``scipy`` is required for EMD; an ``ImportError`` with an install hint is
raised at import time if it is missing.
"""

from __future__ import annotations

from pruning_metrics.prob_measures.base import (
    MetricInfo,
    TokenStepDict,
    TopAlternativeDict,
)
from pruning_metrics.prob_measures.bhattacharyya import compute_bhattacharyya
from pruning_metrics.prob_measures.chamfer import compute_chamfer
from pruning_metrics.prob_measures.chisq import compute_chisq
from pruning_metrics.prob_measures.cosine import compute_cosine
from pruning_metrics.prob_measures.emd import compute_emd
from pruning_metrics.prob_measures.hellinger import compute_hellinger
from pruning_metrics.prob_measures.jeffreys import compute_jeffreys
from pruning_metrics.prob_measures.jsd import compute_jsd
from pruning_metrics.prob_measures.kld import compute_kld
from pruning_metrics.prob_measures.l2 import compute_l2
from pruning_metrics.prob_measures.registry import (
    METRIC_FUNCS,
    METRIC_INFO,
    METRIC_NAMES,
    compute_all,
)
from pruning_metrics.prob_measures.renyi05 import compute_renyi05
from pruning_metrics.prob_measures.renyi2 import compute_renyi2
from pruning_metrics.prob_measures.rkld import compute_rkld
from pruning_metrics.prob_measures.triangular import compute_triangular
from pruning_metrics.prob_measures.tv import compute_tv
from pruning_metrics.prob_measures.wasserstein2 import compute_wasserstein2

__all__ = [
    "METRIC_FUNCS",
    "METRIC_INFO",
    "METRIC_NAMES",
    "MetricInfo",
    "TokenStepDict",
    "TopAlternativeDict",
    "compute_all",
    "compute_bhattacharyya",
    "compute_chamfer",
    "compute_chisq",
    "compute_cosine",
    "compute_emd",
    "compute_hellinger",
    "compute_jeffreys",
    "compute_jsd",
    "compute_kld",
    "compute_l2",
    "compute_renyi05",
    "compute_renyi2",
    "compute_rkld",
    "compute_triangular",
    "compute_tv",
    "compute_wasserstein2",
]
