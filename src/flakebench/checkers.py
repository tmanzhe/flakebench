"""Deterministic answer checks. Python checkers receive the final answer string."""

import importlib
import json
import re

import jsonschema


def _reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def parse_json(text):
    return json.loads(text, parse_constant=_reject_constant)


def resolve(reference):
    module, separator, name = reference.partition(":")
    if not separator or not module or not name:
        raise ValueError("Python references must be module:function")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise ValueError(f"{reference} is not callable")
    return function


def validate_checker(checker):
    if not isinstance(checker, dict):
        raise ValueError("checker must be a mapping")
    kind = checker.get("type")
    if kind == "exact":
        if not isinstance(checker.get("expected"), str):
            raise ValueError("exact checker expected must be a string")
        if "strip" in checker and not isinstance(checker["strip"], bool):
            raise ValueError("exact checker strip must be boolean")
    elif kind == "regex":
        re.compile(checker["pattern"])
    elif kind == "json_schema":
        jsonschema.Draft202012Validator.check_schema(checker["schema"])
    elif kind == "python":
        resolve(checker["function"])
    else:
        raise ValueError(f"Unknown checker type: {kind}")


def check_answer(answer, checker):
    kind = checker["type"]
    if kind == "exact":
        expected = checker["expected"]
        if checker.get("strip", False):
            return answer.strip() == expected.strip()
        return answer == expected
    if kind == "regex":
        return re.search(checker["pattern"], answer) is not None
    if kind == "json_schema":
        try:
            value = parse_json(answer)
            jsonschema.Draft202012Validator(checker["schema"]).validate(value)
            return True
        except (ValueError, jsonschema.ValidationError):
            return False
    if kind == "python":
        return bool(resolve(checker["function"])(answer))
    raise ValueError(f"Unknown checker type: {kind}")
