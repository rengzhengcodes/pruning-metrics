# Adding a new task type

The four notebooks treat any `(prompt, target_text)` dataset as
first-class: free-form generation runs the model on the inference prompt
and scores the result with a task-specific verifier; teacher forcing
scores the model's confidence on `target_text` directly. To add a new
task you only need to write **one adapter class** that implements the
`TaskAdapter` protocol.

## The contract

The protocol lives in
[`src/pruning_metrics/evals/tasks/base.py`](../src/pruning_metrics/evals/tasks/base.py).
A minimal adapter looks like this:

```python
from typing import Sequence
from pruning_metrics.evals.tasks.base import (
    TaskAdapter,
    TaskRecord,
    VerificationOutcome,
    deterministic_split,
)


class MyAdapter(TaskAdapter):
    """One-line description of the task."""

    name = "my_task"           # short id used by build_adapter_from_spec()

    def __init__(
        self,
        dataset_name: str = "<hf-dataset-name>",
        split: str = "test",
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.dataset_spec = f"my_task:{dataset_name}:{split}"
        self._records: list[TaskRecord] | None = None

    def load_records(self) -> list[TaskRecord]:
        if self._records is not None:
            return list(self._records)
        # ... read the HF dataset, build TaskRecord objects ...
        records = [
            TaskRecord(
                task_id="...",
                prompt="...",          # the natural prefix for teacher forcing
                target_text="...",     # the gold continuation
                metadata={...},        # free-form
            )
            for ...
        ]
        self._records = records
        return list(records)

    def train_test_split(
        self,
        seed: int = 65320,
        train_frac: float = 0.8,
        explicit_train_ids: Sequence[str] | None = None,
        explicit_test_ids: Sequence[str] | None = None,
    ) -> tuple[list[TaskRecord], list[TaskRecord]]:
        return deterministic_split(
            self.load_records(),
            seed=seed,
            train_frac=train_frac,
            explicit_train_ids=explicit_train_ids,
            explicit_test_ids=explicit_test_ids,
        )

    def build_inference_prompt(self, record: TaskRecord) -> str:
        # If `record.prompt` already includes any required instruction text
        # for free-form generation, just return it. Otherwise wrap it here.
        return record.prompt

    def verify(
        self,
        record: TaskRecord,
        generated_text: str,
        timeout_seconds: float = 10.0,
    ) -> VerificationOutcome:
        # Implement task-specific scoring. Must return a VerificationOutcome
        # with status in {"pass", "fail", "timeout", "runtime_error",
        # "parse_error"}.
        ...
```

Then register it in
[`src/pruning_metrics/evals/tasks/registry.py`](../src/pruning_metrics/evals/tasks/registry.py):

```python
from pruning_metrics.evals.tasks.my_adapter import MyAdapter

TASK_REGISTRY = {
    "coding": CodingTaskAdapter,
    "math": MathTaskAdapter,
    "mcq": MCQTaskAdapter,
    "my_task": MyAdapter,    # <-- add
}
```

The notebooks now accept `CALIBRATION_DATASET_SPEC = "my_task"` (or
`"my_task:<dataset>:<split>"` for fully-qualified) without any other
changes.

## Two flavours of "prompt"

`record.prompt` is the **teacher-forcing prompt** -- the runner feeds
`prompt + target_text` through the model in a single forward pass and
scores the suffix tokens. So `prompt + target_text` must read naturally
when concatenated.

`build_inference_prompt(record)` returns the **free-form inference
prompt** -- the runner feeds this to `model.generate` and the model's
decoded output is what the verifier sees. Most tasks include any required
instruction text in `record.prompt`; the coding adapter is the exception
because the canonical solution is a function body that needs to follow
the partial `def` (no instruction wrapper) for teacher forcing, but the
free-form prompt needs an instruction "return only Python code defining
`<entry_point>`" so the verifier can find a complete callable.

## Status values

`VerificationOutcome.status` belongs to a small fixed set:

* `"pass"` -- the generated answer is correct.
* `"fail"` -- the generated answer parsed but did not match the gold.
* `"parse_error"` -- the generated text could not be parsed (no number,
  no choice letter, no function definition, etc.).
* `"runtime_error"` -- the verifier blew up (e.g. subprocess crashed
  for coding tasks).
* `"timeout"` -- a timed-out subprocess (only meaningful for tasks that
  execute generated code).

The free-form runner aggregates `pass / num_test_tasks` as `pass@1`.
Other statuses are all "non-pass" but useful when auditing low scores.

## Test your adapter

Add a new file under `tests/evals/tasks/` mirroring the pattern of
[`test_math_adapter.py`](../tests/evals/tasks/test_math_adapter.py) or
[`test_mcq_adapter.py`](../tests/evals/tasks/test_mcq_adapter.py):

* `monkeypatch` the HF `load_dataset` call to return a tiny synthetic
  list of dicts;
* assert that `load_records` builds the right `TaskRecord` shape;
* assert that `train_test_split(seed=65320)` is byte-stable across two
  calls;
* hit `verify` with hand-crafted "good" and "bad" generations.

## Worked example: the coding adapter

[`src/pruning_metrics/evals/tasks/coding.py`](../src/pruning_metrics/evals/tasks/coding.py)
wraps the existing HumanEval+ helpers. Highlights:

* `target_text = task.canonical_solution` so teacher forcing scores the
  reference Python code.
* `metadata = {"entry_point": ..., "test": ...}` so `verify` can
  reconstruct a HumanEval+ task and forward to the existing subprocess
  test harness.
* `build_inference_prompt` injects the instruction wrapper from
  `pipeline.build_coding_prompt` so the model emits a complete callable
  rather than just a function body.

## Worked example: the GSM8K adapter

[`src/pruning_metrics/evals/tasks/math.py`](../src/pruning_metrics/evals/tasks/math.py)
shows numeric verification for math word problems. Highlights:

* `prompt` is wrapped with a brief instruction asking for `#### <number>`.
* `target_text` is the full chain-of-thought + GSM8K divider so teacher
  forcing measures the model's confidence on the reasoning trace, not
  just the final number.
* `verify` extracts the last number using a small priority list:
  `#### N` first (GSM convention), then LaTeX `\boxed{}`, then the last
  number-shaped substring. Comparison tolerates 1e-6 absolute error.

## Worked example: the ARC-Challenge adapter

[`src/pruning_metrics/evals/tasks/mcq.py`](../src/pruning_metrics/evals/tasks/mcq.py)
demonstrates regex-letter scoring. Highlights:

* `prompt` enumerates the choices `A) ... B) ...` and asks for a single
  letter answer.
* `target_text` is the **body** of the correct choice, not its letter,
  so teacher forcing measures content confidence.
* `verify` regex-extracts the first `A-E` token and compares against the
  gold `answerKey`.
