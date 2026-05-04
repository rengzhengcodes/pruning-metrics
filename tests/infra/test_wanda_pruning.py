"""Tests for the canonical WANDA per-output-row pruning routine.

The pruning helpers in ``infra/aws/sagemaker/prune_and_register.py`` and
``infra/ec2/run_qwen_pruning_experiment.py`` should both:

1. Score weights as ``|W| * rms(input_channel)``.
2. For each output row, zero the bottom ``prune_ratio`` fraction of entries
   (NOT a global per-layer threshold — that path runs into
   ``torch.quantile``'s 1.2 B-element limit on Qwen2-72B).

These tests exercise a tiny Linear so they run in milliseconds without
``transformers`` or a GPU. If ``torch`` is unavailable the whole module is
skipped (we do not run on the operator workstation in this project).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, file_path: Path):
    """Import a script-style module by file path without ``sys.path`` mutation."""

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_model() -> nn.Module:
    """A two-Linear stub mimicking the named-modules access pattern."""

    class _Stub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.Linear(8, 4, bias=False)
            with torch.no_grad():
                # Make scores per row monotone so the test is deterministic.
                self.layer.weight.copy_(
                    torch.tensor(
                        [
                            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0],
                            [9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0],
                        ]
                    )
                )

    return _Stub()


def _build_stats(model: nn.Module) -> dict[str, torch.Tensor]:
    """All-ones channel RMS so the score equals ``|W|`` for clarity."""

    return {"layer": torch.ones(8, dtype=torch.float32)}


def test_runner_prune_zeroes_bottom_half_per_row(tiny_model) -> None:
    """50 %% pruning must zero exactly 4 entries per row (out of 8)."""

    runner = _load_module(
        "_pm_runner_for_test",
        REPO_ROOT / "infra" / "ec2" / "run_qwen_pruning_experiment.py",
    )
    stats = _build_stats(tiny_model)
    runner.apply_wanda_pruning(tiny_model, stats, prune_ratio=0.5)

    weight = tiny_model.layer.weight.data
    zeros_per_row = (weight == 0).sum(dim=1)
    assert torch.equal(zeros_per_row, torch.tensor([4, 4, 4, 4]))

    # Row 0: smallest |w| are 1, 2, 3, 4 -> pruned. 5..8 must remain.
    assert (weight[0] == torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0])).all()
    # Row 1: smallest |w| are 4, 3, 2, 1 (last four columns).
    assert (weight[1] == torch.tensor([8.0, 7.0, 6.0, 5.0, 0.0, 0.0, 0.0, 0.0])).all()
    # Rows 2/3: prune the four 1s, keep the four 9s.
    assert (weight[2] == torch.tensor([0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0])).all()
    assert (weight[3] == torch.tensor([9.0, 9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 0.0])).all()


def test_runner_prune_ratio_zero_is_identity(tiny_model) -> None:
    """``prune_ratio=0`` must leave weights unchanged (used for the level-0 baseline)."""

    runner = _load_module(
        "_pm_runner_for_test",
        REPO_ROOT / "infra" / "ec2" / "run_qwen_pruning_experiment.py",
    )
    snapshot = tiny_model.layer.weight.data.detach().clone()
    runner.apply_wanda_pruning(tiny_model, _build_stats(tiny_model), prune_ratio=0.0)
    assert torch.equal(tiny_model.layer.weight.data, snapshot)


def test_runner_handles_huge_score_tensor() -> None:
    """Smoke: a Linear too large for ``torch.quantile`` must still prune.

    PyTorch's ``torch.quantile`` raises ``"input tensor is too large"`` for
    1-D tensors with more than 2**24 elements. Our implementation must work
    on at least that range. This test materialises a 6000x6000 weight
    (~36 M elements) which would tip ``quantile`` over.
    """

    runner = _load_module(
        "_pm_runner_for_test",
        REPO_ROOT / "infra" / "ec2" / "run_qwen_pruning_experiment.py",
    )

    class _Big(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(6000, 6000, bias=False)

    model = _Big()
    stats = {"linear": torch.ones(6000, dtype=torch.float32)}
    runner.apply_wanda_pruning(model, stats, prune_ratio=0.2)
    fraction_zero = (model.linear.weight.data == 0).float().mean().item()
    # Should be ~0.2; allow small slack from rounding to int per row.
    assert 0.18 < fraction_zero < 0.22


def test_sagemaker_path_matches_runner(tiny_model) -> None:
    """The SageMaker variant must produce the same mask as the EC2 runner."""

    runner = _load_module(
        "_pm_runner_for_test",
        REPO_ROOT / "infra" / "ec2" / "run_qwen_pruning_experiment.py",
    )
    sagemaker = _load_module(
        "_pm_sm_for_test",
        REPO_ROOT / "infra" / "aws" / "sagemaker" / "prune_and_register.py",
    )

    a = tiny_model
    b = type(tiny_model)()  # fresh stub with the same initial weights
    with torch.no_grad():
        b.layer.weight.copy_(a.layer.weight)

    runner.apply_wanda_pruning(a, _build_stats(a), prune_ratio=0.25)
    sagemaker.apply_wanda_pruning(b, _build_stats(b), prune_ratio=0.25)
    assert torch.equal(a.layer.weight.data, b.layer.weight.data)
