"""Teacher-forced next-token log-probability extraction for causal LMs.

This module computes, for a single ``(prompt, answer)`` pair, the per-token
probability that a causal language model assigns to the **correct** next token
when the model is conditioned on the **ground-truth** prefix at every step.

That is the textbook definition of "teacher forcing": the model never sees its
own output during the scoring pass — at position ``t`` we always feed
``answer[:t]`` even if the model would have predicted something different.

For a HuggingFace causal LM with full causal attention, teacher forcing is
exactly equivalent to a single forward pass over the concatenated
``prompt + answer`` token sequence: ``logits[:, i, :]`` is conditioned on
tokens ``[0..i]`` (with the standard causal mask), so the prediction for
position ``i + 1`` only depends on the ground-truth tokens up to ``i``.

The output is a ``TeacherForcedRecord`` dataclass that serializes to JSON for
storage in S3 alongside the autoregressive evaluation records.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class TopAlternative:
    """One competing next-token candidate at a teacher-forced position.

    Parameters
    ----------
    token_id:
        Vocabulary id of the candidate token.
    token_text:
        Decoded text of the candidate token (with special characters intact).
    logprob:
        Natural-log probability the model assigned to this candidate at the
        teacher-forced position.
    """

    token_id: int
    token_text: str
    logprob: float


@dataclass(frozen=True)
class TeacherForcedTokenStep:
    """Per-position record from a teacher-forced scoring pass.

    Parameters
    ----------
    position:
        0-indexed offset into the answer tokens.
    target_token_id:
        Vocabulary id of the ground-truth (teacher-forced) next token.
    target_token_text:
        Decoded text of the ground-truth token.
    target_logprob:
        Natural-log probability the model assigned to ``target_token_id``.
    target_prob:
        Linear probability mass assigned to the ground-truth token.
    rank:
        1-indexed rank of the ground-truth token among the full vocabulary
        ordering (lower is better; ``1`` means the model's argmax matched).
    top_alternatives:
        Top-``k`` candidates by descending logprob (``k`` controlled by
        ``compute_teacher_forced_logprobs``).
    """

    position: int
    target_token_id: int
    target_token_text: str
    target_logprob: float
    target_prob: float
    rank: int
    top_alternatives: list[TopAlternative]


@dataclass(frozen=True)
class TeacherForcedRecord:
    """Full teacher-forced scoring record for one prompt/answer pair.

    Parameters
    ----------
    model_id:
        Identifier of the model evaluated (e.g. ``Qwen/Qwen2-72B`` plus a
        pruning-level suffix when applicable).
    task_id:
        HumanEval+ task id whose canonical solution acts as the teacher.
    seed:
        Seed used by callers to (deterministically) pick this pair.
    prompt:
        Prompt text fed to the model.
    answer:
        Ground-truth answer text fed token-by-token (in batched form) for
        teacher forcing.
    num_prompt_tokens:
        Tokenized prompt length (number of leading positions skipped when
        scoring answer tokens).
    num_answer_tokens:
        Number of answer tokens scored.
    average_logprob:
        Mean of ``target_logprob`` across answer tokens.
    perplexity:
        ``exp(-average_logprob)`` — the geometric-mean inverse probability of
        the answer tokens; lower means the model finds the gold answer more
        likely.
    per_token:
        Detailed step-by-step record.
    """

    model_id: str
    task_id: str
    seed: int
    prompt: str
    answer: str
    num_prompt_tokens: int
    num_answer_tokens: int
    average_logprob: float
    perplexity: float
    per_token: list[TeacherForcedTokenStep]


def compute_teacher_forced_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    answer: str,
    model_id: str,
    task_id: str,
    seed: int,
    top_k: int = 5,
) -> TeacherForcedRecord:
    """Score ``answer`` against ``model`` under perfect teacher forcing.

    The function tokenizes ``prompt`` and ``prompt + answer`` separately so we
    know exactly which token positions correspond to answer tokens (the
    boundary handling avoids tokenizer "merge across boundary" surprises by
    re-tokenizing the prompt alone, then taking the suffix).

    Parameters
    ----------
    model:
        A causal LM exposing ``model(input_ids=..., return_dict=True).logits``
        and a ``device`` / ``hf_device_map``-managed parameter placement.
    tokenizer:
        Matching tokenizer providing ``__call__`` returning ``input_ids`` and
        ``decode`` for individual token rendering.
    prompt:
        Conditioning text. May be empty.
    answer:
        Ground-truth continuation text. Must be non-empty.
    model_id:
        Identifier carried into the output record (caller is free to embed a
        pruning-level suffix).
    task_id:
        HumanEval+ task id carried into the output record.
    seed:
        Seed used by the caller to pick this pair (recorded for traceability).
    top_k:
        Number of top alternative tokens to record per step.

    Returns
    -------
    TeacherForcedRecord
        Per-step + summary statistics.

    Preconditions
    -------------
    ``answer`` tokenizes to at least one token under ``tokenizer``.

    Postconditions
    --------------
    - The returned record is internally consistent: ``len(per_token) ==
      num_answer_tokens`` and ``average_logprob`` equals the mean of
      ``target_logprob`` across steps.
    - The model is invoked exactly once with no gradient tape attached.
    """

    if not answer:
        raise ValueError("answer must be a non-empty string for teacher forcing.")

    import torch  # local import keeps unit tests of this module light

    # Tokenize prompt-alone and prompt+answer; the suffix difference gives the
    # exact token positions corresponding to answer tokens, even when the
    # tokenizer would have merged a boundary character had we tokenized
    # ``answer`` in isolation.
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    full_ids = tokenizer(prompt + answer, return_tensors="pt").input_ids[0]

    num_prompt_tokens = int(prompt_ids.shape[0])
    if int(full_ids.shape[0]) <= num_prompt_tokens:
        raise ValueError(
            "Tokenizing prompt + answer did not extend the prompt sequence; "
            "cannot score answer tokens."
        )
    answer_ids = full_ids[num_prompt_tokens:]
    num_answer_tokens = int(answer_ids.shape[0])

    # Determine target device. ``device_map='auto'`` models expose hf_device_map;
    # otherwise fall back to the parameter device or CPU.
    target_device = _resolve_input_device(model)

    input_ids = full_ids.unsqueeze(0).to(target_device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
    logits = outputs.logits[0]

    # logits[i] predicts token i+1 given tokens [0..i]. Shift so each row lines
    # up with the token whose probability we want.
    shift_logits = logits[:-1, :]
    shift_targets = full_ids[1:].to(shift_logits.device)
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)

    # Per-position log-probabilities of the (teacher-forced) ground-truth token.
    target_logprobs = log_probs.gather(
        dim=-1, index=shift_targets.unsqueeze(-1)
    ).squeeze(-1)

    # Slice out the answer-token region. Position ``num_prompt_tokens - 1`` is
    # the first answer token's predictor; we want the log-probs starting there.
    answer_logprobs = target_logprobs[num_prompt_tokens - 1 :]
    answer_log_distributions = log_probs[num_prompt_tokens - 1 :]

    per_token: list[TeacherForcedTokenStep] = []
    for offset in range(num_answer_tokens):
        position_logprobs = answer_log_distributions[offset]
        target_logprob = float(answer_logprobs[offset].item())
        target_token_id = int(answer_ids[offset].item())
        target_token_text = tokenizer.decode([target_token_id])

        # Top-k alternatives at this step (sorted descending by logprob).
        top_values, top_indices = torch.topk(
            position_logprobs, k=min(top_k, position_logprobs.shape[-1])
        )
        alternatives = [
            TopAlternative(
                token_id=int(idx.item()),
                token_text=tokenizer.decode([int(idx.item())]),
                logprob=float(val.item()),
            )
            for val, idx in zip(top_values, top_indices)
        ]

        # Rank: count tokens with strictly higher logprob; +1 for 1-indexed.
        rank = int((position_logprobs > target_logprob).sum().item()) + 1

        per_token.append(
            TeacherForcedTokenStep(
                position=offset,
                target_token_id=target_token_id,
                target_token_text=target_token_text,
                target_logprob=target_logprob,
                target_prob=float(np.exp(target_logprob)),
                rank=rank,
                top_alternatives=alternatives,
            )
        )

    average_logprob = (
        float(np.mean([step.target_logprob for step in per_token]))
        if per_token
        else 0.0
    )
    perplexity = float(np.exp(-average_logprob)) if per_token else float("inf")

    return TeacherForcedRecord(
        model_id=model_id,
        task_id=task_id,
        seed=seed,
        prompt=prompt,
        answer=answer,
        num_prompt_tokens=num_prompt_tokens,
        num_answer_tokens=num_answer_tokens,
        average_logprob=average_logprob,
        perplexity=perplexity,
        per_token=per_token,
    )


def write_teacher_forced_record(
    record: TeacherForcedRecord, output_path: str | Path
) -> None:
    """Persist a teacher-forced record as pretty-printed JSON.

    Parameters
    ----------
    record:
        The record returned by :func:`compute_teacher_forced_logprobs`.
    output_path:
        Local filesystem path (parent directories are created if needed).
    """

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(asdict(record), handle, indent=2)


def teacher_forced_records_to_summary(
    records: Sequence[TeacherForcedRecord],
) -> dict[str, Any]:
    """Reduce a set of records into a compact comparison summary.

    Used by the experiment runner to produce a single ``summary.json`` that
    aligns each pruning level's average log-prob and perplexity for the seeded
    sample so downstream analysis does not need to re-parse every per-token
    record.
    """

    return {
        "num_records": len(records),
        "records": [
            {
                "model_id": r.model_id,
                "task_id": r.task_id,
                "num_answer_tokens": r.num_answer_tokens,
                "average_logprob": r.average_logprob,
                "perplexity": r.perplexity,
            }
            for r in records
        ],
    }


def _resolve_input_device(model: Any) -> Any:
    """Find a sensible device on which to place input ids for ``model``.

    Sharded ``device_map='auto'`` models expose ``hf_device_map`` as a dict
    keyed by submodule name; the embedding layer is reliably present and gives
    us the device to send token ids to. Falls back to the first parameter's
    device for non-sharded models, then to CPU.
    """

    import torch

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for key in (
            "model.embed_tokens",
            "transformer.wte",
            "model.tok_embeddings",
        ):
            if key in device_map:
                value = device_map[key]
                if isinstance(value, str):
                    return torch.device(value)
                return value
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
