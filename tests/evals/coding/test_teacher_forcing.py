"""Tests for teacher-forced log-prob extraction.

Uses a fake tokenizer + model rather than a real HuggingFace checkpoint so the
test runs in seconds with only ``torch`` (no transformers, no GPU).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from pruning_metrics.evals.coding.teacher_forcing import (
    compute_teacher_forced_logprobs,
    write_teacher_forced_record,
)


class _DummyTokenizer:
    """Whitespace-character tokenizer with a tiny vocabulary."""

    def __init__(self) -> None:
        # ASCII printable characters; ids = ord(ch). Vocabulary size 128.
        self.vocab_size = 128

    def __call__(self, text: str, return_tensors: str | None = None):
        ids = torch.tensor([[ord(ch) for ch in text]], dtype=torch.long)
        if return_tensors == "pt":
            return _TokOut(ids)
        return _TokOut(ids)

    def decode(self, ids):
        return "".join(chr(int(idx)) for idx in ids)


class _TokOut:
    """Tiny duck-typed container that mirrors HF tokenizer outputs."""

    def __init__(self, input_ids: torch.Tensor) -> None:
        self.input_ids = input_ids


class _DummyModel:
    """Causal LM whose logits assign max prob to the *next* character.

    This guarantees every teacher-forced ground-truth target is the model's
    argmax, which lets the test assert rank == 1 and sane log-probs.
    """

    def __init__(self) -> None:
        self.vocab_size = 128
        self._device = torch.device("cpu")

    def parameters(self):
        return iter([torch.zeros(1, requires_grad=False)])

    def __call__(self, *, input_ids: torch.Tensor, use_cache: bool, return_dict: bool):
        del use_cache, return_dict
        # logits[b, t] should peak at input_ids[b, t+1] when teacher-forced.
        # We only need shape (1, T, V); set the gold next-token logit high and
        # everything else uniform low.
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size), -10.0, dtype=torch.float32
        )
        for b in range(batch_size):
            for t in range(seq_len - 1):
                gold = int(input_ids[b, t + 1].item())
                logits[b, t, gold] = 10.0
        # The final position has no gold target; we still need a row.
        return _ModelOut(logits)


class _ModelOut:
    """Duck type matching HF causal LM output."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


def test_teacher_forced_record_perfect_prediction(tmp_path) -> None:
    """When the dummy model perfectly predicts each next char, rank is always 1."""

    tokenizer = _DummyTokenizer()
    model = _DummyModel()

    record = compute_teacher_forced_logprobs(
        model=model,
        tokenizer=tokenizer,
        prompt="abc",
        answer="def",
        model_id="dummy",
        task_id="HumanEval/test",
        seed=65320,
        top_k=3,
    )

    assert record.num_prompt_tokens == 3
    assert record.num_answer_tokens == 3
    assert all(step.rank == 1 for step in record.per_token)
    # log_softmax of [10, -10, -10, ...] is dominated by the gold logit.
    # Expected logprob ~ 0 (close to log(1)).
    assert record.average_logprob > -0.01
    assert math.isfinite(record.perplexity)
    assert record.perplexity < 1.5

    output_path = tmp_path / "tf.json"
    write_teacher_forced_record(record, output_path)
    assert output_path.is_file()
    contents = output_path.read_text(encoding="utf-8")
    assert "per_token" in contents
    assert "average_logprob" in contents


def test_teacher_forced_rejects_empty_answer() -> None:
    """An empty answer cannot be teacher-forced; we must fail loudly."""

    with pytest.raises(ValueError, match="non-empty"):
        compute_teacher_forced_logprobs(
            model=_DummyModel(),
            tokenizer=_DummyTokenizer(),
            prompt="abc",
            answer="",
            model_id="dummy",
            task_id="HumanEval/test",
            seed=0,
        )
