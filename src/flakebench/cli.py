"""Command-line entry point."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
import sys
import uuid

import jsonschema
import yaml

from .config import load_suite
from .report import render_report
from .runner import run_one


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(prog="flakebench", description="Measure repeated agent task outcomes.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run every task in a YAML suite repeatedly")
    run.add_argument("suite", type=Path)
    run.add_argument("--model", help="Model name, or mock for offline simulation")
    run.add_argument("--runs", type=positive_int, default=20)
    run.add_argument("--seed", type=int)
    run.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    # Console scripts otherwise put the venv bin directory first, hiding local plugins.
    sys.path.insert(0, str(Path.cwd()))
    try:
        suite = load_suite(args.suite)
        model = args.model or suite.get("model", "gpt-4o-mini")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a nonempty string")
        base_url = os.environ.get("BASE_URL")
        if model != "mock" and not base_url:
            raise ValueError("Set BASE_URL for a real model, or use --model mock")
        if base_url and model != "mock" and not base_url.startswith(("http://", "https://")):
            raise ValueError("BASE_URL must start with http:// or https://")
        seed = args.seed if args.seed is not None else secrets.randbits(63)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = args.out or Path("results") / f"{stamp}-{uuid.uuid4().hex[:8]}"
        output.mkdir(parents=True, exist_ok=True)
        jsonl_path, report_path = output / "results.jsonl", output / "report.md"
        if jsonl_path.exists() or report_path.exists():
            raise ValueError(f"Results already exist in {output}; choose a new --out directory")
        rows = []
        suite_name = suite.get("name", args.suite.stem)
        with jsonl_path.open("x", encoding="utf-8") as handle:
            for task in suite["tasks"]:
                for index in range(1, args.runs + 1):
                    row = run_one(suite, task, index, seed, model, base_url, os.environ.get("API_KEY"))
                    row.update(suite=str(args.suite), suite_name=suite_name)
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                label = "MOCK " if model == "mock" else ""
                passed = sum(row["passed"] for row in rows if row["task_id"] == task["id"])
                print(f"{label}{task['id']}: {passed}/{args.runs} passed", flush=True)
        report_path.write_text(render_report(rows, str(args.suite), suite_name, model, seed, args.runs), encoding="utf-8")
        print(f"{'MOCK ' if model == 'mock' else ''}seed: {seed}")
        print(f"Results: {jsonl_path}\nReport: {report_path}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError, AttributeError,
            re.error, yaml.YAMLError, jsonschema.SchemaError) as exc:
        print(f"flakebench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
