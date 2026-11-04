"""Agent loop and process-level run deadlines."""

import hashlib
import json
import multiprocessing
import time

from .checkers import check_answer
from .models import HTTPModel, MockModel, ModelError
from .tools import BadToolJSON, MOCK_TOOL, implementation, parse_arguments, tool_schema


def derive_seed(seed, task_id, run):
    payload = json.dumps([seed, task_id, run], ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31)


def run_agent(task, model, seed):
    """Execute one run. The caller isolates this in a process to enforce deadlines."""
    started = time.monotonic()
    deadline = started + task["timeout"]
    messages = []
    if task.get("system"):
        messages.append({"role": "system", "content": task["system"]})
    messages.append({"role": "user", "content": task["prompt"]})
    definitions = list(task.get("tools", []))
    if model.is_mock:
        definitions.append(MOCK_TOOL)
    available = {tool["name"]: tool for tool in definitions}
    schemas = [tool_schema(tool) for tool in definitions]
    steps = 0
    answer = None

    def result(failure=None, detail=None):
        return {"passed": failure is None, "failure": failure, "detail": detail,
                "answer": answer, "steps": steps, "messages": messages}

    try:
        for steps in range(1, task["max_steps"] + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return result("timeout", "Run deadline exceeded")
            response = model.complete(messages, schemas, (seed + steps - 1) % (2**31), remaining)
            if time.monotonic() >= deadline:
                return result("timeout", "Run deadline exceeded")
            if not isinstance(response, dict):
                raise ModelError("Assistant message must be an object")
            calls = response.get("tool_calls")
            if calls is not None and not isinstance(calls, list):
                raise ModelError("tool_calls must be a list")
            messages.append({**response, "role": "assistant"})
            if calls:
                for call in calls:
                    if (not isinstance(call, dict) or not isinstance(call.get("function"), dict)
                            or not isinstance(call.get("id"), str) or not call["id"]):
                        raise ModelError("Malformed tool call envelope")
                    function = call["function"]
                    name = function.get("name")
                    if not isinstance(name, str) or name not in available:
                        return result("tool_error", f"Unknown tool: {name}")
                    tool = available[name]
                    args = parse_arguments(function.get("arguments"), tool["parameters"])
                    try:
                        output = implementation(tool["implementation"])(**args)
                        content = json.dumps(output, ensure_ascii=False, allow_nan=False)
                    except Exception as exc:
                        return result("tool_error", f"{type(exc).__name__}: {str(exc)[:1000]}")
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": content})
                    if time.monotonic() >= deadline:
                        return result("timeout", "Run deadline exceeded")
            else:
                answer = response.get("content")
                if not isinstance(answer, str):
                    raise ModelError("Final answer content must be a string")
                try:
                    passed = check_answer(answer, task["checker"])
                except Exception as exc:
                    return result("wrong_answer", f"Checker raised {type(exc).__name__}: {str(exc)[:1000]}")
                if time.monotonic() >= deadline:
                    return result("timeout", "Run deadline exceeded")
                if model.is_mock and model.failure == "wrong_answer":
                    return result("wrong_answer", "Mock injected wrong answer")
                return result(None if passed else "wrong_answer")
        return result("step_limit", "No final answer within max_steps model calls")
    except BadToolJSON as exc:
        return result("bad_tool_json", str(exc)[:1000])
    except TimeoutError:
        return result("timeout", "Model request timed out")
    except Exception as exc:
        return result("model_error", f"{type(exc).__name__}: {str(exc)[:1000]}")


def _worker(connection, suite, task, seed, model_name, base_url, api_key):
    try:
        model = MockModel(suite, task, seed) if model_name == "mock" else HTTPModel(model_name, base_url, api_key)
        connection.send(run_agent(task, model, seed))
    finally:
        connection.close()


def run_one(suite, task, run, seed, model_name, base_url=None, api_key=None):
    """Timeout includes child startup, HTTP, tools, and answer checking."""
    run_seed = derive_seed(seed, task["id"], run)
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(send, suite, task, run_seed, model_name, base_url, api_key))
    started = time.monotonic()
    outcome = {"passed": False, "failure": "timeout", "detail": "Run deadline exceeded",
               "answer": None, "steps": None, "messages": []}
    try:
        process.start()
        send.close()
        remaining = max(0, task["timeout"] - (time.monotonic() - started))
        if receive.poll(remaining):
            try:
                outcome = receive.recv()
            except EOFError:
                outcome.update(failure="model_error", detail="Run worker exited without a result")
    finally:
        if process.pid is not None:
            process.join(0.05)
            if process.is_alive():
                process.terminate()
                process.join(0.2)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        receive.close()
        send.close()
    return {"task_id": task["id"], "run": run, "seed": seed, "run_seed": run_seed,
            "model": model_name, "is_mock": model_name == "mock",
            "max_steps": task["max_steps"], "timeout": task["timeout"],
            "duration_seconds": round(time.monotonic() - started, 6), **outcome}
