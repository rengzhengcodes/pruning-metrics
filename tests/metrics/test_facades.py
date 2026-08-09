"""Parity tests for the backwards-compatible facade modules.

``pruning_metrics.embedding`` and ``pruning_metrics.metrics.distributions``
re-export their packages' surfaces via wildcard imports with a derived
``__all__``. These tests pin the contract: every canonical name is present
on the facade, is the *same object* (not a copy), and the aggregate
``pruning_metrics.metrics`` namespace covers all of its source modules.

The source modules are loaded via :func:`importlib.import_module` because
``pruning_metrics.metrics`` re-exports a *function* named
``embedding_quality`` that shadows the submodule of the same name on the
package namespace.
"""

from __future__ import annotations

import importlib

from pruning_metrics import dim_reduction, embedding, metrics, prob_measures

_distributions = importlib.import_module("pruning_metrics.metrics.distributions")
_masks = importlib.import_module("pruning_metrics.metrics.masks")
_cluster_stats = importlib.import_module("pruning_metrics.metrics.cluster_stats")
_embedding_quality = importlib.import_module(
    "pruning_metrics.metrics.embedding_quality"
)


def test_distributions_facade_matches_prob_measures() -> None:
    """The distributions facade mirrors ``prob_measures`` exactly."""
    assert set(_distributions.__all__) == set(prob_measures.__all__)
    for name in prob_measures.__all__:
        assert getattr(_distributions, name) is getattr(prob_measures, name)


def test_embedding_facade_matches_dim_reduction() -> None:
    """The embedding facade mirrors ``dim_reduction`` exactly."""
    assert set(embedding.__all__) == set(dim_reduction.__all__)
    for name in dim_reduction.__all__:
        assert getattr(embedding, name) is getattr(dim_reduction, name)


def test_metrics_package_covers_all_sources() -> None:
    """``pruning_metrics.metrics`` re-exports every source module's surface."""
    expected = (
        set(prob_measures.__all__)
        | set(_masks.__all__)
        | set(_cluster_stats.__all__)
        | set(_embedding_quality.__all__)
    )
    assert expected <= set(metrics.__all__)
    for name in expected:
        assert getattr(metrics, name) is not None
