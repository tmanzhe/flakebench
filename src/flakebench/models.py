"""OpenAI-compatible HTTP and seeded offline mock clients."""

import json
import random
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import failure_rates
from .tools import MOCK_TOOL


class ModelError(RuntimeError):
    pass


class HTTPModel:
    is_mock = False

    def __init__(self, model, base_url, api_key=None):
        self.model = model
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.api_key = api_key

    def complete(self, messages, tools, seed, timeout):
        payload = {"model": self.model, "messages": messages, "seed": seed, "temperature": 0}
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:
                data = json.load(response)
            message = data["choices"][0]["message"]
            if not isinstance(message, dict):
                raise ValueError("message must be an object")
            return message
        except HTTPError as exc:
            raise ModelError(f"HTTP {exc.code} from chat completions endpoint") from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise TimeoutError("model request timed out") from exc
            raise ModelError(f"Cannot reach chat completions endpoint: {exc.reason}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelError("Invalid chat completions response") from exc


class MockModel:
    is_mock = True

    def __init__(self, suite, task, seed):
        self.task = task
        self.hint = task.get("mock", {})
        draw = random.Random(seed).random()
        self.failure = None
        cumulative = 0.0
        for name, rate in failure_rates(suite, task).items():
            cumulative += rate
            if draw < cumulative:
                self.failure = name
                break
        self.round = 0

    def complete(self, messages, tools, seed, timeout):
        self.round += 1
        if self.failure == "timeout":
            time.sleep(timeout + 1)
            raise TimeoutError("mock injected timeout")
        if self.failure in ("bad_tool_json", "tool_error", "step_limit"):
            args = "{" if self.failure == "bad_tool_json" else json.dumps({"fail": self.failure == "tool_error"})
            return self._call(MOCK_TOOL["name"], args)
        calls = self.hint.get("tool_calls", [])
        if self.round <= len(calls):
            call = calls[self.round - 1]
            args = call.get("arguments", {})
            return self._call(call["name"], args if isinstance(args, str) else json.dumps(args))
        # The malformed JSON sentinel also fails permissive JSON object schemas.
        answer = "__MOCK_WRONG_ANSWER__" if self.failure == "wrong_answer" else self.hint.get(
            "answer", self.hint.get("expected", self.task["checker"].get("expected", "")))
        return {"role": "assistant", "content": answer}

    def _call(self, name, arguments):
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"mock_call_{self.round}", "type": "function",
            "function": {"name": name, "arguments": arguments},
        }]}
