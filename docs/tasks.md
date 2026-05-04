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
    native_or_seeded_split,
)


class MyAdapter(TaskAdapter):
    """One-line description of the task."""

    name = "my_task"           # short id used by build_adapter_from_spec()

    def __init__(
        self,
        dataset_name: str = "<hf-dataset-name>",
        train_split: str | None = "train",  # None -> seeded fallback
        test_split: str = "test",
    ) -> None:
        self.dataset_name = dataset_name
        self.train_split = train_split
        self.test_split = test_split
        split_label = (
            f"{train_split}+{test_split}" if train_split else test_split
        )
        self.dataset_spec = f"my_task:{dataset_name}:{split_label}"
        self._train_records: list[TaskRecord] | None = None
        self._test_records: list[TaskRecord] | None = None

    def _load_split(self, split: str) -> list[TaskRecord]:
        # Read the HF dataset, build TaskRecord objects for ``split``.
        ...

    def load_records(self) -> list[TaskRecord]:
        if self._test_records is None:
            self._test_records = self._load_split(self.test_split)
        if self.train_split is not None and self._train_records is None:
            self._train_records = self._load_split(self.train_split)
        if self._train_records is not None:
            return list(self._train_records) + list(self._test_records)
        return list(self._test_records)

    def train_test_split(
        self,
        seed: int = 65320,
        train_frac: float = 0.8,
        explicit_train_ids: Sequence[str] | None = None,
        explicit_test_ids: Sequence[str] | None = None,
    ) -> tuple[list[TaskRecord], list[TaskRecord]]:
        self.load_records()  # populate both partitions lazily
        return native_or_seeded_split(
            self._train_records,
            self._test_records,
            seed=seed,
            train_frac=train_frac,
            explicit_train_ids=explicit_train_ids,
            explicit_test_ids=explicit_test_ids,
        )

    def build_inference_prompt(self, record: TaskRecord) -> str:
        # If ``record.prompt`` already includes any required instruction
        # text for free-form generation, just return it. Otherwise wrap.
        return record.prompt

    def verify(
        self,
        record: TaskRecord,
        generated_text: str,
        timeout_seconds: float = 10.0,
    ) -> VerificationOutcome:
        # Task-specific scoring. Must return a VerificationOutcome with
        # status in {"pass", "fail", "timeout", "runtime_error",
        # "parse_error"}.
        ...
```

`native_or_seeded_split` is the shared helper in `tasks/base.py` that
routes between native splits and the seeded 80/20 fallback:

* If both `train_split` and `test_split` are configured (and no explicit
  task-id overrides are passed), it returns the natively-loaded
  partitions verbatim, ignoring `seed` and `train_frac`.
* If `train_split` is `None`, or if the caller forces task-id overrides,
  it falls back to `deterministic_split` -- a stable seeded shuffle
  keyed by `seed` over the union of records.

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

The notebooks now accept `CALIBRATION_DATASET_SPEC = "my_task"` (or any
of the longer spec shapes) without any other changes.

## Spec grammar (`build_adapter_from_spec`)

Each adapter's spec parser is one-line in
[`registry.py`](../src/pruning_metrics/evals/tasks/registry.py); the
project ships three:

| Adapter | Spec grammar | Default behaviour |
|---------|--------------|-------------------|
| `coding` | `coding[:<dataset>[:<test_split>]]` | HumanEval+, single test split, seeded 80/20 fallback for calibration. |
| `math` | `math[:<dataset>[:<config>[:<train_split>[:<test_split>]]]]` | Defaults to `math:gsm8k:main:train:test` (native splits). Pass `""` for `train_split` to force the seeded fallback. |
| `mcq` | `mcq[:<dataset>[:<config>[:<train_split>[:<test_split>]]]]` | Defaults to `mcq:allenai/ai2_arc:ARC-Challenge:train:test` (native splits). |

Examples:

```python
build_adapter_from_spec("coding")                                # HumanEval+
build_adapter_from_spec("math:gsm8k:main")                       # GSM8K native splits
build_adapter_from_spec("math:gsm8k:main::test")                 # GSM8K test only -> seeded fallback
build_adapter_from_spec("mcq:allenai/ai2_arc:ARC-Challenge")     # ARC native splits
```

## Train, test, and validation splits

GSM8K (`gsm8k`, config `main`) and ARC-Challenge expose Hub splits named
`train` and `test` by default; neither ships a separate `validation` split
on those configs, and the built-in adapters **never** assume one exists.
Calibration rows come from `train_split` (default `"train"`) and
evaluation rows from `test_split` (default `"test"`). If you point
`MathTaskAdapter` or `MCQTaskAdapter` at another dataset that only has
`train` + `test`, keep the defaults; if a dataset adds `validation` and
you want to calibrate on it, pass that split name explicitly when
constructing the adapter (or extend the spec parser). Datasets with only a
single public split should set `train_split=None` so the seeded fallback
partitions that split (same pattern as HumanEval+).

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
* `build_inference_prompt` injects an instruction wrapper that asks the
  model to emit a complete callable rather than just a function body, so
  the verifier can find the entry-point function.

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
