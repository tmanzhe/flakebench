"""Tool definitions and deliberately small builtins."""

import time

import jsonschema

from .checkers import parse_json, resolve


class BadToolJSON(ValueError):
    pass


def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


def echo(text):
    return text


def lookup(key, data):
    return data[key]


def fail(message="intentional tool failure"):
    raise RuntimeError(message)


def sleep(seconds):
    time.sleep(seconds)
    return "done"


def _mock_probe(fail=False):
    if fail:
        raise RuntimeError("mock injected tool error")
    return "ok"


BUILTINS = {"add": add, "multiply": multiply, "echo": echo, "lookup": lookup,
            "fail": fail, "sleep": sleep}
MOCK_TOOL = {
    "name": "__flakebench_mock_probe",
    "description": "Internal mock failure probe",
    "parameters": {"type": "object", "properties": {"fail": {"type": "boolean"}},
                   "additionalProperties": False},
    "implementation": "builtin:_mock_probe",
}


def implementation(reference):
    if reference == "builtin:_mock_probe":
        return _mock_probe
    if reference.startswith("builtin:"):
        try:
            return BUILTINS[reference.split(":", 1)[1]]
        except KeyError as exc:
            raise ValueError(f"Unknown builtin: {reference}") from exc
    return resolve(reference)


def tool_schema(tool):
    return {"type": "function", "function": {
        "name": tool["name"], "description": tool.get("description", ""),
        "parameters": tool["parameters"],
    }}


def parse_arguments(raw, schema):
    try:
        if not isinstance(raw, str):
            raise ValueError("tool arguments must be a JSON string")
        args = parse_json(raw)
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be an object")
        jsonschema.Draft202012Validator(schema).validate(args)
        return args
    except (ValueError, TypeError, jsonschema.ValidationError) as exc:
        raise BadToolJSON(str(exc)) from exc
