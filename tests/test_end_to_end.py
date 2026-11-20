"""End-to-end: baseline target over the real golden dataset, checked against the committed baseline."""

import json

import pytest
from conftest import BASELINE_PATH, DATASET_PATH

from eval_harness.cli import main
from eval_harness.dataset import load_dataset
from eval_harness.gate import compare, load_report
from eval_harness.runner import run_eval
from eval_harness.scorers import default_scorers
from eval_harness.targets import RuleBasedTarget, Target, build_target


@pytest.fixture(scope="module")
def report():
    return run_eval(load_dataset(DATASET_PATH), RuleBasedTarget(), default_scorers(judge=False), run_id="test")


def test_baseline_is_realistic_not_perfect(report):
    summary = report["summary"]
    assert 0.5 < summary["accuracy"] < 0.95
    assert summary["errors"] == 0
    assert summary["case_pass_rate"] < summary["accuracy"]  # reply checks catch extra failures


def test_report_matches_committed_baseline(report):
    baseline = load_report(BASELINE_PATH)
    assert baseline["dataset"]["sha256"] == report["dataset"]["sha256"], "dataset changed: refresh the baseline"
    for key in ("accuracy", "macro_f1", "case_pass_rate", "scorer_pass_rates"):
        assert report["summary"][key] == pytest.approx(baseline["summary"][key])
    assert compare(baseline, report).passed


def test_confusion_matrix_totals(report):
    cm = report["summary"]["confusion_matrix"]
    assert sum(map(sum, cm["matrix"])) == report["summary"]["cases"]


def test_target_exceptions_are_recorded_not_raised():
    class Broken(Target):
        name = "broken"

        def predict(self, message):
            raise TimeoutError("upstream timed out")

    ds = load_dataset(DATASET_PATH)
    result = run_eval(ds, Broken(), default_scorers(judge=False))
    assert result["summary"]["errors"] == len(ds)
    assert result["summary"]["case_pass_rate"] == 0.0
    assert "TimeoutError" in result["cases"][0]["error"]


def test_build_target_unknown_name():
    with pytest.raises(ValueError):
        build_target("gpt")


def test_claude_target_without_key_exits_cleanly(tmp_path):
    assert main(["run", "--target", "claude", "--dataset", str(DATASET_PATH), "--out", str(tmp_path)]) == 2


def test_run_cli_writes_reports_and_gate_passes(tmp_path, capsys):
    code = main(
        [
            "run",
            "--run-name",
            "e2e",
            "--dataset",
            str(DATASET_PATH),
            "--out",
            str(tmp_path),
            "--gate",
            str(BASELINE_PATH),
        ]
    )
    assert code == 0
    assert "GATE PASSED" in capsys.readouterr().out
    data = json.loads((tmp_path / "e2e.json").read_text(encoding="utf-8"))
    assert data["summary"]["scorers_skipped"] == ["llm_judge"]
    markdown = (tmp_path / "e2e.md").read_text(encoding="utf-8")
    assert "## Confusion matrix" in markdown and "## Regression gate" in markdown


def test_run_cli_gate_fails_on_regression(tmp_path):
    baseline = load_report(BASELINE_PATH)
    baseline["summary"]["accuracy"] = 0.99  # pretend the baseline used to be much better
    inflated = tmp_path / "baseline.json"
    inflated.write_text(json.dumps(baseline), encoding="utf-8")
    assert (
        main(["run", "--no-judge", "--dataset", str(DATASET_PATH), "--out", str(tmp_path), "--gate", str(inflated)])
        == 1
    )
