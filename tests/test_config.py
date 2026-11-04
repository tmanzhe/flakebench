from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from flakebench.config import DEFAULT_RATES, FAILURES, failure_rates, load_suite


def write_suite(tmp_path, suite):
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    return path


def test_suite_defaults_and_task_overrides(tmp_path, task):
    task.pop("max_steps")
    task.pop("timeout")
    second = deepcopy(task)
    second.update(id="second", max_steps=2, timeout=0.5)
    suite = load_suite(write_suite(tmp_path, {"tasks": [task, second], "max_steps": 7, "timeout": 3}))
    assert suite["tasks"][0]["max_steps"] == 7
    assert suite["tasks"][0]["timeout"] == 3
    assert suite["tasks"][1]["max_steps"] == 2
    assert suite["tasks"][1]["timeout"] == 0.5


@pytest.mark.parametrize("suite", [None, [], {}, {"tasks": []}, {"tasks": "bad"}, {"tasks": [None]}])
def test_bad_suite_shapes(tmp_path, suite):
    with pytest.raises(ValueError):
        load_suite(write_suite(tmp_path, suite))


@pytest.mark.parametrize("field,value", [
    ("id", ""), ("id", 7), ("prompt", None), ("max_steps", 0),
    ("max_steps", True), ("max_steps", 1.5), ("timeout", 0), ("timeout", -1),
    ("timeout", float("nan")), ("timeout", float("inf")), ("timeout", True),
    ("tools", {}), ("mock", None),
])
def test_invalid_task_fields(tmp_path, task, field, value):
    task[field] = value
    with pytest.raises(ValueError):
        load_suite(write_suite(tmp_path, {"tasks": [task]}))


def test_duplicate_ids_rejected(tmp_path, task):
    with pytest.raises(ValueError, match="unique"):
        load_suite(write_suite(tmp_path, {"tasks": [task, task]}))


def test_duplicate_tool_names_rejected(tmp_path, task, add_tool):
    task["tools"] = [add_tool, add_tool]
    with pytest.raises(ValueError, match="unique"):
        load_suite(write_suite(tmp_path, {"tasks": [task]}))


@pytest.mark.parametrize("name", ["bad name", "__flakebench_fake", "", "x" * 65, 5])
def test_invalid_tool_names(tmp_path, task, add_tool, name):
    add_tool["name"] = name
    task["tools"] = [add_tool]
    with pytest.raises(ValueError):
        load_suite(write_suite(tmp_path, {"tasks": [task]}))


def test_nonobject_tool_schema_rejected(tmp_path, task, add_tool):
    add_tool["parameters"] = {"type": "array"}
    task["tools"] = [add_tool]
    with pytest.raises(ValueError, match="object JSON schema"):
        load_suite(write_suite(tmp_path, {"tasks": [task]}))


@pytest.mark.parametrize("override", [{"unknown": 0.1}, {"timeout": -0.1}, {"timeout": 1.1},
                                       {"timeout": float("nan")}, {"timeout": True},
                                       {"timeout": "0.1"}, {"timeout": 1, "tool_error": 1}, []])
def test_invalid_mock_rates(override):
    with pytest.raises(ValueError):
        failure_rates({"mock": {"failure_rates": override}}, {})


def test_rate_precedence_and_defaults():
    rates = failure_rates({"mock": {"failure_rates": {"timeout": 0.1}}},
                          {"mock": {"failure_rates": {"timeout": 0.2}}})
    assert rates["timeout"] == 0.2
    assert rates["tool_error"] == DEFAULT_RATES["tool_error"]
    assert failure_rates({}, {}) == DEFAULT_RATES


@pytest.mark.parametrize("hint", [{"answer": 4}, {"expected": 4}, {"tool_calls": "bad"},
                                   {"tool_calls": [{"name": "missing"}]},
                                   {"tool_calls": [{"name": "add", "arguments": []}]}])
def test_invalid_mock_hints(tmp_path, task, add_tool, hint):
    task["mock"], task["tools"] = hint, [add_tool]
    with pytest.raises(ValueError):
        load_suite(write_suite(tmp_path, {"tasks": [task]}))


def test_shipped_suites_are_separate_and_cover_features():
    root = Path(__file__).resolve().parents[1]
    dev = load_suite(root / "examples/dev.yaml")
    heldout = load_suite(root / "examples/heldout.yaml")
    assert 6 <= len(heldout["tasks"]) <= 8
    assert {t["id"] for t in dev["tasks"]}.isdisjoint(t["id"] for t in heldout["tasks"])
    assert {t["prompt"] for t in dev["tasks"]}.isdisjoint(t["prompt"] for t in heldout["tasks"])
    assert {t["checker"]["type"] for t in heldout["tasks"]} == {"exact", "regex", "json_schema", "python"}
    assert any(t.get("tools") for t in heldout["tasks"])
