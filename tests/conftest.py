from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest

from flakebench.config import FAILURES


@pytest.fixture
def task():
    return {"id": "test", "prompt": "Say ok", "max_steps": 4, "timeout": 2,
            "checker": {"type": "exact", "expected": "ok"}, "mock": {"answer": "ok"}}


@pytest.fixture
def zero_suite(task):
    return {"tasks": [task], "mock": {"failure_rates": dict.fromkeys(FAILURES, 0)}}


@pytest.fixture
def add_tool():
    return {"name": "add", "implementation": "builtin:add", "parameters": {
        "type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"], "additionalProperties": False}}


@pytest.fixture
def http_server():
    servers = []

    def start(responses):
        scripted = deepcopy(responses)
        records = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                records.append({"path": self.path, "headers": dict(self.headers), "payload": payload})
                entry = scripted.pop(0) if scripted else {"status": 500}
                time.sleep(entry.get("delay", 0))
                self.send_response(entry.get("status", 200))
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                body = entry.get("raw")
                if body is None:
                    body = json.dumps(entry.get("body", {"choices": [{"message": entry.get("message", {"content": "ok"})}]}))
                try:
                    self.wfile.write(body.encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}/v1", records

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join()
