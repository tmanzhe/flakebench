"""Load and validate suites before starting any runs."""

import math
from pathlib import Path
import re

import jsonschema
import yaml

from .checkers import validate_checker
from .tools import implementation

FAILURES = ("wrong_answer", "bad_tool_json", "tool_error", "timeout", "step_limit")
DEFAULT_RATES = dict.fromkeys(FAILURES, 0.05)


def failure_rates(suite, task):
    rates = DEFAULT_RATES.copy()
    for config in (suite.get("mock", {}), task.get("mock", {})):
        override = config.get("failure_rates", {})
        if not isinstance(override, dict) or set(override) - set(FAILURES):
            raise ValueError("mock.failure_rates must map known failure names to rates")
        rates.update(override)
    if any(isinstance(rate, bool) or not isinstance(rate, (int, float))
           or not math.isfinite(rate) or not 0 <= rate <= 1 for rate in rates.values()):
        raise ValueError("mock failure rates must be finite numbers between 0 and 1")
    if sum(rates.values()) > 1 + 1e-12:
        raise ValueError("mock failure rates must sum to at most 1")
    return rates


def load_suite(path):
    with Path(path).open(encoding="utf-8") as handle:
        suite = yaml.safe_load(handle)
    if not isinstance(suite, dict) or not isinstance(suite.get("tasks"), list) or not suite["tasks"]:
        raise ValueError("suite must contain a nonempty tasks list")
    seen = set()
    if not isinstance(suite.get("mock", {}), dict):
        raise ValueError("suite mock must be a mapping")
    for task in suite["tasks"]:
        if not isinstance(task, dict):
            raise ValueError("each task must be a mapping")
        name = task.get("id")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("task ids must be unique nonempty strings")
        seen.add(name)
        if not isinstance(task.get("prompt"), str):
            raise ValueError(f"{name}: prompt must be a string")
        if not isinstance(task.get("mock", {}), dict):
            raise ValueError(f"{name}: mock must be a mapping")
        steps = task.get("max_steps", suite.get("max_steps", 8))
        timeout = task.get("timeout", suite.get("timeout", 30))
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        task["max_steps"], task["timeout"] = steps, timeout
        validate_checker(task.get("checker"))
        tools = task.get("tools", [])
        if not isinstance(tools, list):
            raise ValueError("tools must be a list")
        tool_names = set()
        for tool in tools:
            if not isinstance(tool, dict):
                raise ValueError("each tool must be a mapping")
            tool_name = tool.get("name", "")
            if (not isinstance(tool_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tool_name)
                    or tool_name.startswith("__flakebench_") or tool_name in tool_names):
                raise ValueError("tool names must be unique identifiers, without the __flakebench_ prefix")
            tool_names.add(tool_name)
            schema = tool["parameters"]
            jsonschema.Draft202012Validator.check_schema(schema)
            if not isinstance(schema, dict) or schema.get("type") != "object":
                raise ValueError("tool parameters must be an object JSON schema")
            implementation(tool["implementation"])
        failure_rates(suite, task)
        hint = task.get("mock", {})
        for field in ("answer", "expected"):
            if field in hint and not isinstance(hint[field], str):
                raise ValueError(f"mock.{field} must be a string")
        calls = hint.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ValueError("mock.tool_calls must be a list")
        for call in calls:
            if not isinstance(call, dict) or call.get("name") not in tool_names:
                raise ValueError("mock tool calls must name a task tool")
            if not isinstance(call.get("arguments", {}), (dict, str)):
                raise ValueError("mock tool arguments must be an object or JSON string")
    return suite
