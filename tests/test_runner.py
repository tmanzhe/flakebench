import json
import multiprocessing
import time

import pytest

from flakebench.config import FAILURES
from flakebench.models import HTTPModel, MockModel
from flakebench.runner import derive_seed, run_agent, run_one


def tool_call(name="add", arguments='{"a":2,"b":3}', call_id="c1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class ScriptedModel:
    is_mock = False

    def __init__(self, messages):
        self.messages = iter(messages)

    def complete(self, *args):
        return next(self.messages)


def test_successful_mock_run(task, zero_suite):
    result = run_agent(task, MockModel(zero_suite, task, 1), 1)
    assert result["passed"] and result["failure"] is None
    assert result["answer"] == "ok" and result["steps"] == 1


def test_mock_exact_without_hint(task, zero_suite):
    task.pop("mock")
    assert run_agent(task, MockModel(zero_suite, task, 3), 3)["passed"]


def test_mock_uses_tools_and_final_answer(task, zero_suite, add_tool):
    task["tools"] = [add_tool]
    task["checker"]["expected"] = "5"
    task["mock"] = {"answer": "5", "tool_calls": [{"name": "add", "arguments": {"a": 2, "b": 3}}]}
    result = run_agent(task, MockModel(zero_suite, task, 5), 5)
    assert result["passed"] and result["steps"] == 2
    assert json.loads(result["messages"][2]["content"]) == 5
    assert result["messages"][2]["tool_call_id"] == "mock_call_1"


@pytest.mark.parametrize("failure", FAILURES)
def test_mock_injects_each_taxonomy(task, zero_suite, failure):
    zero_suite["mock"]["failure_rates"] = {name: float(name == failure) for name in FAILURES}
    task["timeout"] = 0.4 if failure == "timeout" else 2
    result = run_one(zero_suite, task, 1, 42, "mock")
    assert not result["passed"] and result["failure"] == failure
    assert result["is_mock"] and result["seed"] == 42


def test_seed_determinism_and_independent_task_streams(task, zero_suite):
    zero_suite["mock"]["failure_rates"] = dict.fromkeys(FAILURES, 0.15)
    def sequence(seed):
        return [MockModel(zero_suite, task, derive_seed(seed, task["id"], i)).failure for i in range(100)]
    assert sequence(7) == sequence(7)
    assert sequence(7) != sequence(8)
    assert set(sequence(7)) == {None, *FAILURES}
    assert derive_seed(7, "one", 1) != derive_seed(7, "two", 1)
    assert derive_seed(7, "one", 1) != derive_seed(7, "one", 2)


def test_process_results_repeat_by_seed(task, zero_suite):
    first = run_one(zero_suite, task, 2, 8, "mock")
    second = run_one(zero_suite, task, 2, 8, "mock")
    first.pop("duration_seconds")
    second.pop("duration_seconds")
    assert first == second


@pytest.mark.parametrize("raw", ['{', '[]', '{"a":"bad","b":2}', '{"a":1}'])
def test_agent_bad_tool_json(task, add_tool, raw):
    task["tools"] = [add_tool]
    result = run_agent(task, ScriptedModel([{"tool_calls": [tool_call(arguments=raw)]}]), 0)
    assert result["failure"] == "bad_tool_json"


def test_unknown_tool_is_tool_error(task):
    result = run_agent(task, ScriptedModel([{"tool_calls": [tool_call(name="missing")]}]), 0)
    assert result["failure"] == "tool_error"
    assert "Unknown tool" in result["detail"]


@pytest.mark.parametrize("reference", ["builtin:fail", "tests.plugins:not_json"])
def test_raised_and_unserializable_tools(task, reference):
    task["tools"] = [{"name": "broken", "implementation": reference, "parameters": {"type": "object"}}]
    result = run_agent(task, ScriptedModel([{"tool_calls": [tool_call("broken", "{}")]}]), 0)
    assert result["failure"] == "tool_error"


def test_multiple_tools_in_one_assistant_message(task, add_tool):
    task["tools"] = [add_tool]
    model = ScriptedModel([
        {"content": None, "tool_calls": [tool_call(call_id="one"), tool_call(call_id="two")]},
        {"content": "ok"},
    ])
    result = run_agent(task, model, 0)
    assert result["passed"] and result["steps"] == 2
    assert [message["tool_call_id"] for message in result["messages"] if message["role"] == "tool"] == ["one", "two"]


def test_step_limit_counts_model_calls(task, add_tool):
    task["tools"], task["max_steps"] = [add_tool], 2
    result = run_agent(task, ScriptedModel([{"tool_calls": [tool_call()]}] * 2), 0)
    assert result["failure"] == "step_limit" and result["steps"] == 2


def test_final_answer_on_last_allowed_step(task, add_tool):
    task["tools"], task["max_steps"] = [add_tool], 2
    result = run_agent(task, ScriptedModel([{"tool_calls": [tool_call()]}, {"content": "ok"}]), 0)
    assert result["passed"]


def test_wrong_answer(task):
    assert run_agent(task, ScriptedModel([{"content": "wrong"}]), 0)["failure"] == "wrong_answer"


def test_checker_exception_is_wrong_answer_with_detail(task):
    task["checker"] = {"type": "python", "function": "tests.plugins:broken_checker"}
    result = run_agent(task, ScriptedModel([{"content": "ok"}]), 0)
    assert result["failure"] == "wrong_answer" and "checker broke" in result["detail"]


@pytest.mark.parametrize("message", [{"content": None}, {"tool_calls": "bad"},
                                       {"tool_calls": [{}]}, {"tool_calls": [None]}])
def test_protocol_errors_are_model_errors(task, message):
    assert run_agent(task, ScriptedModel([message]), 0)["failure"] == "model_error"


@pytest.mark.parametrize("blocked", ["tool", "checker"])
def test_process_deadline_stops_user_code(task, zero_suite, blocked):
    task["timeout"] = 0.6
    if blocked == "checker":
        task["checker"] = {"type": "python", "function": "tests.plugins:blocked_checker"}
    else:
        task["tools"] = [{"name": "blocked", "implementation": "tests.plugins:blocked_tool",
                          "parameters": {"type": "object"}}]
        task["mock"]["tool_calls"] = [{"name": "blocked", "arguments": {}}]
    before = {process.pid for process in multiprocessing.active_children()}
    started = time.monotonic()
    result = run_one(zero_suite, task, 1, 1, "mock")
    assert result["failure"] == "timeout"
    assert time.monotonic() - started < 2
    assert {process.pid for process in multiprocessing.active_children()} == before


def test_http_agent_tool_loop_and_request_contract(task, add_tool, http_server):
    task["tools"] = [add_tool]
    task["system"] = "Be concise"
    url, requests = http_server([
        {"message": {"role": "assistant", "content": None, "tool_calls": [tool_call()]}},
        {"message": {"content": "ok"}},
    ])
    result = run_agent(task, HTTPModel("test-model", url, "test-key"), 17)
    assert result["passed"]
    first, second = requests
    assert first["path"] == "/v1/chat/completions"
    assert first["headers"]["Authorization"] == "Bearer test-key"
    assert first["payload"]["model"] == "test-model"
    assert first["payload"]["seed"] == 17 and second["payload"]["seed"] == 18
    assert first["payload"]["temperature"] == 0
    assert first["payload"]["tools"][0]["function"]["parameters"] == add_tool["parameters"]
    assert "implementation" not in json.dumps(first["payload"])
    assert first["payload"]["messages"][0] == {"role": "system", "content": "Be concise"}
    assert second["payload"]["messages"][-1] == {"role": "tool", "content": "5", "tool_call_id": "c1"}


def test_http_without_tools_or_key(task, http_server):
    url, requests = http_server([{"message": {"content": "ok"}}])
    assert run_agent(task, HTTPModel("test", url + "/chat/completions"), 0)["passed"]
    assert "Authorization" not in requests[0]["headers"]
    assert "tools" not in requests[0]["payload"]


@pytest.mark.parametrize("response", [{"status": 401}, {"status": 500}, {"raw": "not json"},
                                       {"body": {}}, {"body": {"choices": []}},
                                       {"body": {"choices": [{"message": "bad"}]}}])
def test_http_errors_are_model_errors(task, http_server, response):
    url, _ = http_server([response])
    assert run_agent(task, HTTPModel("test", url), 0)["failure"] == "model_error"


def test_http_request_timeout(task, http_server):
    task["timeout"] = 0.05
    url, _ = http_server([{"delay": 0.2}])
    assert run_agent(task, HTTPModel("test", url), 0)["failure"] == "timeout"


@pytest.mark.parametrize("failure", ["bad_tool_json", "tool_error"])
def test_http_tool_failures_use_agent_taxonomy(task, add_tool, http_server, failure):
    task["tools"] = [add_tool]
    call = tool_call(arguments="{" if failure == "bad_tool_json" else "{}")
    if failure == "tool_error":
        call["function"]["name"] = "missing"
    url, _ = http_server([{"message": {"tool_calls": [call]}}])
    result = run_agent(task, HTTPModel("test", url), 0)
    assert result["failure"] == failure


def test_http_in_subprocess_with_tool_loop(task, zero_suite, add_tool, http_server):
    task["tools"] = [add_tool]
    url, requests = http_server([
        {"message": {"tool_calls": [tool_call()]}}, {"message": {"content": "ok"}}])
    result = run_one(zero_suite, task, 1, 12, "real-test", url)
    assert result["passed"] and not result["is_mock"]
    assert len(requests) == 2
