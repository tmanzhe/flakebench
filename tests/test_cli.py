import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from flakebench.report import aggregate


EXECUTABLE = str(Path(sys.executable).with_name("flakebench"))


def invoke(*args, cwd=None, env=None):
    clean = os.environ.copy()
    clean.pop("BASE_URL", None)
    clean.pop("API_KEY", None)
    clean.update(env or {})
    return subprocess.run([EXECUTABLE, *map(str, args)], cwd=cwd, env=clean,
                          capture_output=True, text=True, timeout=60)


def save_suite(tmp_path, suite):
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    return path


def test_cli_mock_end_to_end_and_jsonl(tmp_path, zero_suite):
    suite_path = save_suite(tmp_path, zero_suite)
    out = tmp_path / "output"
    command = invoke("run", suite_path, "--model", "mock", "--runs", 3, "--seed", 12, "--out", out,
                     env={"BASE_URL": "http://127.0.0.1:1", "API_KEY": "do-not-record-this"})
    assert command.returncode == 0, command.stderr
    assert "MOCK test: 3/3 passed" in command.stdout
    lines = (out / "results.jsonl").read_text().splitlines()
    assert len(lines) == 3
    rows = [json.loads(line) for line in lines]
    assert [row["run"] for row in rows] == [1, 2, 3]
    assert len({row["run_seed"] for row in rows}) == 3
    for row in rows:
        assert row["seed"] == 12 and row["is_mock"] is True
        assert row["suite"] == str(suite_path) and row["task_id"] == "test"
        assert row["passed"] and row["failure"] is None and row["answer"] == "ok"
        assert row["duration_seconds"] > 0 and row["max_steps"] == 4
    report = (out / "report.md").read_text()
    assert "# MOCK flakebench report" in report and "3/3 (100.0%)" in report
    assert str(suite_path) in report
    assert "do-not-record-this" not in report + "\n".join(lines) + command.stdout


def test_cli_default_twenty_runs_and_generated_seed(tmp_path, zero_suite):
    zero_suite["model"] = "mock"
    path = save_suite(tmp_path, zero_suite)
    result = invoke("run", path, "--out", tmp_path / "out")
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (tmp_path / "out/results.jsonl").read_text().splitlines()]
    assert len(rows) == 20
    assert isinstance(rows[0]["seed"], int)
    assert all(row["seed"] == rows[0]["seed"] for row in rows)
    assert "Runs per task: 20." in (tmp_path / "out/report.md").read_text()


def test_cli_default_output_directory(tmp_path, zero_suite):
    path = save_suite(tmp_path, zero_suite)
    result = invoke("run", path, "--model", "mock", "--runs", 1, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(list((tmp_path / "results").glob("*/report.md"))) == 1


@pytest.mark.parametrize("runs", [0, -1, "abc"])
def test_cli_invalid_runs(tmp_path, zero_suite, runs):
    path = save_suite(tmp_path, zero_suite)
    result = invoke("run", path, "--model", "mock", "--runs", runs)
    assert result.returncode == 2


def test_cli_missing_base_url(tmp_path, zero_suite):
    result = invoke("run", save_suite(tmp_path, zero_suite), "--model", "real")
    assert result.returncode == 2 and "BASE_URL" in result.stderr


@pytest.mark.parametrize("content", ["tasks: [", "tasks: []", "tasks: [{id: a, prompt: hi, checker: {type: regex, pattern: '['}}]"])
def test_cli_bad_yaml_and_configuration_no_traceback(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    result = invoke("run", path, "--model", "mock")
    assert result.returncode == 2 and "Traceback" not in result.stderr
    assert "flakebench:" in result.stderr


def test_cli_refuses_to_overwrite_results(tmp_path, zero_suite):
    path = save_suite(tmp_path, zero_suite)
    out = tmp_path / "out"
    out.mkdir()
    (out / "results.jsonl").write_text("keep me")
    result = invoke("run", path, "--model", "mock", "--out", out)
    assert result.returncode == 2 and "already exist" in result.stderr
    assert (out / "results.jsonl").read_text() == "keep me"


def test_cli_real_endpoint_offline(tmp_path, zero_suite, http_server):
    url, requests = http_server([{"message": {"content": "ok"}}])
    path = save_suite(tmp_path, zero_suite)
    result = invoke("run", path, "--model", "local", "--runs", 1, "--seed", 3,
                    "--out", tmp_path / "out", env={"BASE_URL": url})
    assert result.returncode == 0, result.stderr
    row = json.loads((tmp_path / "out/results.jsonl").read_text())
    assert row["passed"] and not row["is_mock"]
    assert len(requests) == 1 and requests[0]["payload"]["model"] == "local"
    assert "MOCK" not in (tmp_path / "out/report.md").read_text()


def test_cli_imports_plugins_from_working_directory(tmp_path, task, zero_suite):
    (tmp_path / "my_plugin.py").write_text("def accepts(answer):\n    return answer == 'ok'\n")
    task["checker"] = {"type": "python", "function": "my_plugin:accepts"}
    path = save_suite(tmp_path, zero_suite)
    result = invoke("run", path, "--model", "mock", "--runs", 1, "--out", tmp_path / "out", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "out/results.jsonl").read_text())["passed"]


def test_cli_seed_replays_failure_outcomes_and_report(tmp_path, task, zero_suite):
    zero_suite["mock"]["failure_rates"].update(wrong_answer=0.5, bad_tool_json=0.25, step_limit=0.25)
    path = save_suite(tmp_path, zero_suite)
    result_sets = []
    for name in ("first", "second"):
        result = invoke("run", path, "--model", "mock", "--runs", 5, "--seed", 9, "--out", tmp_path / name)
        assert result.returncode == 0, result.stderr
        rows = [json.loads(line) for line in (tmp_path / name / "results.jsonl").read_text().splitlines()]
        for row in rows:
            row.pop("duration_seconds")
        result_sets.append(rows)
    assert result_sets[0] == result_sets[1]
    assert (tmp_path / "first/report.md").read_text() == (tmp_path / "second/report.md").read_text()
    stats = aggregate(result_sets[0])
    assert sum(stats["taxonomy"].values()) == 5
