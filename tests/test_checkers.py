import jsonschema
import pytest

from flakebench.checkers import check_answer, resolve, validate_checker
from flakebench.tools import BadToolJSON, implementation, parse_arguments


@pytest.mark.parametrize("answer,expected,strip,passed", [
    ("ok", "ok", False, True), ("OK", "ok", False, False),
    (" ok\n", "ok", False, False), (" ok\n", "ok", True, True),
    ("", "", False, True), ("café", "café", False, True),
])
def test_exact(answer, expected, strip, passed):
    assert check_answer(answer, {"type": "exact", "expected": expected, "strip": strip}) is passed


@pytest.mark.parametrize("pattern,answer,passed", [
    (r"\d+", "answer: 42", True), ("^ok$", "not ok", False),
    ("(?i)^ok$", "OK", True), ("^x+$", "xxx", True), ("^x+$", "xy", False),
])
def test_regex_search(pattern, answer, passed):
    assert check_answer(answer, {"type": "regex", "pattern": pattern}) is passed


@pytest.mark.parametrize("answer,passed", [
    ('{"n": 3}', True), ('{"n": "3"}', False), ('{}', False),
    ('{"n": 3, "x": 0}', False), ('not JSON', False), ('[3]', False),
    ('```json\n{"n":3}\n```', False),
])
def test_json_schema(answer, passed):
    schema = {"type": "object", "properties": {"n": {"type": "integer"}},
              "required": ["n"], "additionalProperties": False}
    assert check_answer(answer, {"type": "json_schema", "schema": schema}) is passed


def test_python_checker():
    checker = {"type": "python", "function": "tests.plugins:palindrome"}
    assert check_answer("level", checker)
    assert not check_answer("hello", checker)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_schema_rejects_non_json_numeric_constants(constant):
    assert not check_answer(constant, {"type": "json_schema", "schema": {"type": "number"}})


@pytest.mark.parametrize("checker", [
    {"type": "unknown"}, {"type": "exact", "expected": 3},
    {"type": "exact", "expected": "ok", "strip": "yes"},
    {"type": "regex", "pattern": "["},
    {"type": "json_schema", "schema": {"type": "made-up"}},
    {"type": "python", "function": "tests.plugins:absent"}, None,
])
def test_invalid_checkers_rejected(checker):
    with pytest.raises(Exception):
        validate_checker(checker)


@pytest.mark.parametrize("ref", ["missing_colon", ":function", "json:", "json:__name__"])
def test_invalid_python_reference(ref):
    with pytest.raises(ValueError):
        resolve(ref)


@pytest.mark.parametrize("raw", ['{', '[]', 'null', '"string"', '{"a":"x","b":2}',
                                  '{"a":1}', '{"a":1,"b":2,"extra":0}', None,
                                  '{"a":NaN,"b":2}', '{"a":Infinity,"b":2}'])
def test_bad_arguments(raw, add_tool):
    with pytest.raises(BadToolJSON):
        parse_arguments(raw, add_tool["parameters"])


def test_valid_arguments_and_tool_execution(add_tool):
    args = parse_arguments('{"a":2,"b":4}', add_tool["parameters"])
    assert implementation(add_tool["implementation"])(**args) == 6


def test_python_tool_import():
    assert implementation("flakebench.example_plugins:reverse")("abcd") == "dcba"


def test_unknown_builtin():
    with pytest.raises(ValueError, match="Unknown builtin"):
        implementation("builtin:missing")
