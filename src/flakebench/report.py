"""Aggregation and Markdown reporting, with Wilson score confidence intervals."""

from collections import Counter
import math

from .config import FAILURES


def wilson(passes, total, z=1.959963984540054):
    if total < 0 or passes < 0 or passes > total:
        raise ValueError("Require 0 <= passes <= total")
    if total == 0:
        return (0.0, 1.0)
    proportion = passes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def aggregate(results):
    tasks = {}
    for row in results:
        stats = tasks.setdefault(row["task_id"], {"passes": 0, "total": 0})
        stats["total"] += 1
        stats["passes"] += bool(row["passed"])
    flaky = sum(0 < item["passes"] < item["total"] for item in tasks.values())
    return {"tasks": tasks, "passes": sum(item["passes"] for item in tasks.values()),
            "total": len(results), "flaky_tasks": flaky,
            "flakiness": flaky / len(tasks) if tasks else 0.0,
            "taxonomy": dict(Counter(row["failure"] for row in results if not row["passed"]))}


def _escape(value):
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_report(results, suite_path, suite_name, model, seed, runs):
    stats = aggregate(results)
    tag = "MOCK " if model == "mock" else ""
    lines = [f"# {tag}flakebench report", "", f"Suite: {_escape(suite_name)} (`{_escape(suite_path)}`)",
             f"Model: `{_escape(model)}`", ""]
    if tag:
        lines.extend(["MOCK simulation only. These numbers do not measure a real model.", ""])
    lines.extend([f"## {tag}run settings", "", f"Seed: {seed}. Runs per task: {runs}.", "",
                  f"## {tag}overall numbers", ""])
    passes, total = stats["passes"], stats["total"]
    lo, hi = wilson(passes, total)
    rate = passes / total if total else 0.0
    lines.extend([f"Pass rate: {passes}/{total} ({rate:.1%}), Wilson 95% CI [{lo:.1%}, {hi:.1%}].",
                  f"Flakiness: {stats['flakiness']:.1%} ({stats['flaky_tasks']}/{len(stats['tasks'])} tasks pass sometimes and fail sometimes).",
                  "", f"## {tag}per-task numbers", "",
                  "| Task | Passes | Runs | Pass rate | Wilson 95% CI | Flaky |",
                  "| --- | ---: | ---: | ---: | --- | --- |"])
    for name, item in stats["tasks"].items():
        passed, count = item["passes"], item["total"]
        lo, hi = wilson(passed, count)
        flaky = "yes" if 0 < passed < count else "no"
        lines.append(f"| {_escape(name)} | {passed} | {count} | {passed/count:.1%} | [{lo:.1%}, {hi:.1%}] | {flaky} |")
    lines.extend(["", f"## {tag}failure taxonomy counts", "", "| Failure | Count |", "| --- | ---: |"])
    for failure in (*FAILURES, "model_error"):
        lines.append(f"| {failure} | {stats['taxonomy'].get(failure, 0)} |")
    lines.extend(["", "Overall pass rate pools all runs. Flakiness weights each task equally.",
                  "Wilson intervals describe binary run outcomes; they do not establish task independence.", ""])
    return "\n".join(lines)
