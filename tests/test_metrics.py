import pytest

from flakebench.report import aggregate, render_report, wilson


@pytest.mark.parametrize("passed,total,expected", [
    (0, 0, (0, 1)), (0, 10, (0, 0.277532799863)),
    (10, 10, (0.722467200137, 1)), (5, 10, (0.236593090513, 0.763406909487)),
    (1, 1, (0.206549314377, 1)), (50, 100, (0.403831530366, 0.596168469634)),
])
def test_wilson_known_values(passed, total, expected):
    assert wilson(passed, total) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("passed,total", [(-1, 2), (3, 2), (0, -1)])
def test_wilson_invalid_counts(passed, total):
    with pytest.raises(ValueError):
        wilson(passed, total)


def rows_for(outcomes):
    return [{"task_id": name, "passed": passed, "failure": None if passed else "wrong_answer"}
            for name, values in outcomes.items() for passed in values]


@pytest.mark.parametrize("outcomes,flakiness,passes,total", [
    ({}, 0, 0, 0), ({"a": [True, True]}, 0, 2, 2),
    ({"a": [False, False]}, 0, 0, 2), ({"a": [True, False]}, 1, 1, 2),
    ({"a": [True, False], "b": [False, False]}, 0.5, 1, 4),
    ({"a": [True], "b": [False]}, 0, 1, 2),
])
def test_flakiness(outcomes, flakiness, passes, total):
    stats = aggregate(rows_for(outcomes))
    assert stats["flakiness"] == flakiness
    assert (stats["passes"], stats["total"]) == (passes, total)
    assert stats["taxonomy"].get("wrong_answer", 0) == total - passes


def test_report_counts_and_intervals():
    rows = rows_for({"a": [True, False], "b": [True, True]})
    report = render_report(rows, "examples/heldout.yaml", "held-out", "live", 42, 2)
    assert report.startswith("# flakebench report")
    assert "examples/heldout.yaml" in report
    assert "Seed: 42. Runs per task: 2." in report
    assert "3/4 (75.0%)" in report
    assert "Flakiness: 50.0% (1/2 tasks" in report
    assert "| a | 1 | 2 | 50.0% | [9.5%, 90.5%] | yes |" in report
    assert "| wrong_answer | 1 |" in report


def test_mock_report_every_number_section_is_labeled():
    report = render_report(rows_for({"a": [True, False]}), "heldout.yaml", "held-out", "mock", 42, 2)
    assert report.startswith("# MOCK flakebench report")
    assert "MOCK simulation only" in report
    assert all(line.startswith("## MOCK ") for line in report.splitlines() if line.startswith("## "))
    for failure in ("wrong_answer", "bad_tool_json", "tool_error", "timeout", "step_limit", "model_error"):
        assert f"| {failure} |" in report


def test_markdown_task_names_are_escaped():
    report = render_report(rows_for({"a|b\nc": [True]}), "x", "name", "mock", 0, 1)
    assert "| a\\|b c |" in report
