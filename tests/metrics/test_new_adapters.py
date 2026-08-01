"""Tests for the v2 domain adapters: MBPP (coding) and MathQA (MCQ).

Two layers of coverage:

* Fast unit tests that monkeypatch ``load_dataset`` with a handful of
  synthetic rows so the parsing / entry-point / verification logic runs
  without any network access (the MBPP verify path shells out to a subprocess
  but only for a trivial function, so it finishes in milliseconds).
* ``@pytest.mark.network`` smoke tests that build each adapter from its
  registry spec string, download the real Hugging Face dataset, and assert one
  train and one test record carry the fields the runners depend on.
"""

from __future__ import annotations

import pytest

from pruning_metrics.evals.tasks import coding as coding_module
from pruning_metrics.evals.tasks import mcq as mcq_module
from pruning_metrics.evals.tasks.coding import (
    MbppTaskAdapter,
    _build_mbpp_check,
    _mbpp_entry_point,
)
from pruning_metrics.evals.tasks.mcq import (
    MathQaTaskAdapter,
    _parse_mathqa_options,
)
from pruning_metrics.evals.tasks.registry import (
    TASK_REGISTRY,
    build_adapter,
    build_adapter_from_spec,
)


# --------------------------------------------------------------------------- #
# Synthetic dataset rows
# --------------------------------------------------------------------------- #
def _fake_mbpp_rows() -> list[dict[str, object]]:
    return [
        {
            "task_id": 11,
            "prompt": "Write a function to add two numbers.",
            "code": "\ndef add(a, b):\n    return a + b\n",
            "source_file": "x",
            "test_imports": [],
            "test_list": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
            "test": "assert add(1, 2) == 3\n",
        },
        {
            "task_id": 12,
            "prompt": "Write a function to return the maximum of a list.",
            # A helper def precedes the real entry point.
            "code": (
                "\ndef _key(x):\n    return x\n\n"
                "def max_of(values):\n    return max(values, key=_key)\n"
            ),
            "source_file": "x",
            "test_imports": ["import math"],
            "test_list": ["assert max_of([1, 5, 3]) == 5"],
            "test": "assert max_of([1, 5, 3]) == 5\n",
        },
    ]


def _fake_mathqa_rows() -> list[dict[str, object]]:
    return [
        {
            "Problem": "what is 2 + 2 ?",
            "options": "a ) 3 , b ) 4 , c ) 5 , d ) 6 , e ) none of these",
            "correct": "b",
            "Rationale": "2 + 2 = 4",
        },
        {
            # Bracket/quote-wrapped options, commas inside a body.
            "Problem": "present worth ?",
            "options": "['a ) rs . 400', 'b ) rs . 300', 'c ) rs . 500']",
            "correct": "c",
            "Rationale": "...",
        },
    ]


# --------------------------------------------------------------------------- #
# MBPP unit tests
# --------------------------------------------------------------------------- #
def test_mbpp_entry_point_prefers_called_function() -> None:
    code = "def _key(x):\n    return x\n\ndef max_of(v):\n    return max(v)\n"
    assert _mbpp_entry_point(code, ["assert max_of([1]) == 1"]) == "max_of"


def test_mbpp_entry_point_falls_back_to_last_def() -> None:
    code = "def helper():\n    return 1\n\ndef solution():\n    return 2\n"
    assert _mbpp_entry_point(code, ["assert True"]) == "solution"


def test_build_mbpp_check_defines_check_function() -> None:
    source = _build_mbpp_check(["import math"], ["assert add(1, 2) == 3"])
    assert "def check(candidate):" in source
    assert "import math" in source
    assert "    assert add(1, 2) == 3" in source


def test_build_mbpp_check_handles_empty_test_list() -> None:
    source = _build_mbpp_check([], [])
    # Must still be a syntactically valid, callable check.
    ns: dict[str, object] = {}
    exec(compile(source, "<check>", "exec"), ns)  # noqa: S102 -- trusted test input
    assert callable(ns["check"])


def test_mbpp_adapter_loads_and_builds_records(monkeypatch) -> None:
    monkeypatch.setattr(
        coding_module, "load_dataset", lambda _name, split: _fake_mbpp_rows()
    )
    adapter = MbppTaskAdapter()
    assert adapter.name == "coding"
    assert adapter.train_split is None
    records = adapter.load_records()
    assert [r.task_id for r in records] == ["mbpp/test/11", "mbpp/test/12"]
    first = records[0]
    # target_text is the canonical solution; prompt carries the description.
    assert "return a + b" in first.target_text
    assert "add two numbers" in first.prompt
    assert first.metadata["entry_point"] == "add"
    # Helper-def row resolves to the tested function, not the helper.
    assert records[1].metadata["entry_point"] == "max_of"


def test_mbpp_adapter_verify_pass_and_fail(monkeypatch) -> None:
    monkeypatch.setattr(
        coding_module, "load_dataset", lambda _name, split: _fake_mbpp_rows()
    )
    adapter = MbppTaskAdapter()
    record = adapter.load_records()[0]
    good = "def add(a, b):\n    return a + b\n"
    bad = "def add(a, b):\n    return a - b\n"
    assert adapter.verify(record, good, timeout_seconds=10.0).status == "pass"
    assert adapter.verify(record, bad, timeout_seconds=10.0).status in (
        "fail",
        "runtime_error",
    )


def test_mbpp_adapter_seeded_split_covers_all_records(monkeypatch) -> None:
    monkeypatch.setattr(
        coding_module, "load_dataset", lambda _name, split: _fake_mbpp_rows()
    )
    adapter = MbppTaskAdapter()
    train, test = adapter.train_test_split(seed=65320, train_frac=0.5)
    assert len(train) + len(test) == len(_fake_mbpp_rows())
    assert adapter.dataset_spec == "mbpp:evalplus/mbppplus:test"


# --------------------------------------------------------------------------- #
# MathQA unit tests
# --------------------------------------------------------------------------- #
def test_parse_mathqa_plain_options() -> None:
    pairs = _parse_mathqa_options("a ) rs . 400 , b ) rs . 300 , e ) none")
    assert pairs == [("A", "rs . 400"), ("B", "rs . 300"), ("E", "none")]


def test_parse_mathqa_letter_paren_inside_body() -> None:
    """A bare ``b )`` inside an option body must not start a new option."""

    pairs = _parse_mathqa_options("a ) ( x + b ) , b ) 5 , c ) 6 , d ) 7 , e ) 8")
    assert pairs == [
        ("A", "( x + b )"),
        ("B", "5"),
        ("C", "6"),
        ("D", "7"),
        ("E", "8"),
    ]


def test_parse_mathqa_bracket_wrapped_options() -> None:
    pairs = _parse_mathqa_options("['a ) 8', 'b ) 16', 'c ) 24']")
    assert pairs == [("A", "8"), ("B", "16"), ("C", "24")]


def test_parse_mathqa_list_input() -> None:
    pairs = _parse_mathqa_options(["a ) 1", "b ) 2"])
    assert pairs == [("A", "1"), ("B", "2")]


def test_parse_mathqa_doubled_markers() -> None:
    """Rows like ``"a ) a ) 56 , b ) b ) 35"`` collapse to the real bodies."""

    pairs = _parse_mathqa_options("a ) a ) 56 , b ) b ) 35 , c ) c ) 39")
    assert pairs == [("A", "56"), ("B", "35"), ("C", "39")]


def test_mathqa_adapter_loads_and_builds_records(monkeypatch) -> None:
    monkeypatch.setattr(
        mcq_module,
        "load_dataset",
        lambda _name, split, revision: _fake_mathqa_rows(),
    )
    # train_split=None isolates a single split so ids are deterministic.
    adapter = MathQaTaskAdapter(train_split=None)
    assert adapter.name == "mcq"
    records = adapter.load_records()
    assert [r.task_id for r in records] == ["mathqa/test/00000", "mathqa/test/00001"]
    first = records[0]
    assert first.target_text == "4"  # body of the correct choice "b"
    assert first.metadata["answer_key"] == "B"
    assert first.metadata["choice_labels"] == ["A", "B", "C", "D", "E"]
    # Commas inside a body survive parsing.
    assert records[1].target_text == "rs . 500"


def test_mathqa_adapter_verify_uses_letter_matching(monkeypatch) -> None:
    monkeypatch.setattr(
        mcq_module,
        "load_dataset",
        lambda _name, split, revision: _fake_mathqa_rows(),
    )
    adapter = MathQaTaskAdapter(train_split=None)
    record = adapter.load_records()[0]  # correct answer is "B"
    assert adapter.verify(record, "The answer is B").status == "pass"
    assert adapter.verify(record, "A").status == "fail"
    assert adapter.verify(record, "no idea").status == "parse_error"


def test_mathqa_adapter_dataset_spec(monkeypatch) -> None:
    monkeypatch.setattr(
        mcq_module,
        "load_dataset",
        lambda _name, split, revision: _fake_mathqa_rows(),
    )
    adapter = MathQaTaskAdapter()
    assert adapter.dataset_spec == "mathqa:allenai/math_qa:train+test"
    assert adapter.revision == "refs/convert/parquet"


# --------------------------------------------------------------------------- #
# Registry wiring
# --------------------------------------------------------------------------- #
def test_core_registry_unchanged() -> None:
    """The v2 adapters must not pollute the three-way core registry."""

    assert set(TASK_REGISTRY) == {"coding", "math", "mcq"}


def test_build_adapter_from_spec_mbpp() -> None:
    adapter = build_adapter_from_spec("mbpp")
    assert isinstance(adapter, MbppTaskAdapter)
    assert adapter.dataset_spec == "mbpp:evalplus/mbppplus:test"


def test_build_adapter_from_spec_mathqa() -> None:
    adapter = build_adapter_from_spec("mathqa")
    assert isinstance(adapter, MathQaTaskAdapter)
    assert adapter.dataset_spec == "mathqa:allenai/math_qa:train+test"


def test_build_adapter_by_name_v2() -> None:
    assert isinstance(build_adapter("mbpp"), MbppTaskAdapter)
    assert isinstance(build_adapter("mathqa"), MathQaTaskAdapter)


def test_build_adapter_from_spec_mbpp_with_dataset_override() -> None:
    adapter = build_adapter_from_spec("mbpp:google-research-datasets/mbpp:test")
    assert isinstance(adapter, MbppTaskAdapter)
    assert adapter.dataset_name == "google-research-datasets/mbpp"
    assert adapter.test_split == "test"


# --------------------------------------------------------------------------- #
# Network smoke tests (download the real datasets)
# --------------------------------------------------------------------------- #
@pytest.mark.network
def test_mbpp_spec_network_smoke() -> None:
    adapter = build_adapter_from_spec("mbpp")
    train, test = adapter.train_test_split()
    assert len(train) >= 1 and len(test) >= 1
    for record in (train[0], test[0]):
        assert record.prompt.strip()
        assert record.target_text.strip()
        assert record.metadata["entry_point"]
        assert "def check(candidate):" in record.metadata["test"]


@pytest.mark.network
def test_mathqa_spec_network_smoke() -> None:
    adapter = build_adapter_from_spec("mathqa")
    train, test = adapter.train_test_split()
    # MathQA train must be a usable calibration source (>= 384 records).
    assert len(train) >= 384
    assert len(test) >= 1
    for record in (train[0], test[0]):
        assert record.prompt.strip()
        assert record.target_text.strip()
        assert record.metadata["answer_key"] in record.metadata["choice_labels"]
