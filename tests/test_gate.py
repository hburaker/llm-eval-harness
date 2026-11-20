import copy
import json

import pytest

from eval_harness.cli import main
from eval_harness.gate import compare, make_baseline


def payload(accuracy=0.8, macro_f1=0.75, case_pass=0.7, keywords=0.9, sha="abc"):
    return {
        "run_id": "r",
        "dataset": {"name": "d", "version": "1.0.0", "sha256": sha},
        "summary": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "case_pass_rate": case_pass,
            "scorer_pass_rates": {"keywords": keywords},
        },
    }


def test_identical_runs_pass():
    gate = compare(payload(), payload())
    assert gate.passed
    assert {d.name for d in gate.deltas} == {"accuracy", "macro_f1", "case_pass_rate", "pass_rate.keywords"}


def test_drop_beyond_threshold_fails():
    gate = compare(payload(), payload(accuracy=0.75), threshold=0.02)
    assert not gate.passed
    assert [d.name for d in gate.regressions] == ["accuracy"]
    assert gate.regressions[0].delta == pytest.approx(-0.05)


def test_drop_within_threshold_and_improvements_pass():
    assert compare(payload(), payload(accuracy=0.79, keywords=1.0), threshold=0.02).passed


def test_drop_exactly_at_threshold_passes():
    assert compare(payload(accuracy=0.80), payload(accuracy=0.78), threshold=0.02).passed


def test_scorer_pass_rate_regression_is_gated():
    gate = compare(payload(), payload(keywords=0.5))
    assert [d.name for d in gate.regressions] == ["pass_rate.keywords"]


def test_dataset_change_and_missing_metric_warn():
    current = payload(sha="different")
    del current["summary"]["scorer_pass_rates"]["keywords"]
    gate = compare(payload(), current)
    assert gate.passed
    assert any("dataset changed" in w for w in gate.warnings)
    assert any("pass_rate.keywords" in w for w in gate.warnings)


def test_make_baseline_drops_per_case_details():
    report = {**payload(), "cases": [{"id": "x"}], "schema_version": 1}
    baseline = make_baseline(report)
    assert "cases" not in baseline and baseline["summary"] == report["summary"]


def test_compare_cli_exit_codes(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(payload()), encoding="utf-8")
    b.write_text(json.dumps(copy.deepcopy(payload())), encoding="utf-8")
    assert main(["compare", str(a), str(b)]) == 0
    assert "GATE PASSED" in capsys.readouterr().out

    b.write_text(json.dumps(payload(macro_f1=0.5)), encoding="utf-8")
    assert main(["compare", str(a), str(b)]) == 1
    assert "REGRESSED" in capsys.readouterr().out


def test_compare_cli_rejects_non_report_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert main(["compare", str(bad), str(bad)]) == 2
