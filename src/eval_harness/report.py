"""Console, Markdown and JSON rendering of eval results."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval_harness.gate import GateResult


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    """Plain-text table with left-aligned first column and right-aligned numbers."""
    cells = [list(map(str, headers))] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]

    def fmt(row: Sequence[str]) -> str:
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths, strict=True)))

    return [fmt(cells[0]), "  ".join("-" * w for w in widths)] + [fmt(r) for r in cells[1:]]


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], numeric: bool = True) -> list[str]:
    """Markdown table; first column left-aligned, the rest right-aligned when ``numeric``."""
    rest = "---:" if numeric else ":---"
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([":---"] + [rest] * (len(headers) - 1)) + "|"]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows]
    return lines


def _failures(report: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    rows = []
    for case in report["cases"]:
        if case["passed"]:
            continue
        failed = [s for s in case["scores"] if s["passed"] is False]
        reasons = "; ".join(f"{s['scorer']}: {s['detail']}" for s in failed) or (case["error"] or "")
        rows.append((case["id"], case["expected_intent"], case["predicted_intent"], reasons))
    return rows


def render_console(report: dict[str, Any]) -> str:
    s = report["summary"]
    target = report["target"]
    lines = [
        f"Run {report['run_id']}  |  target={target['name']} model={target.get('model', '-')}  |  "
        f"dataset={report['dataset']['name']}@{report['dataset']['version']} ({s['cases']} cases)",
        "",
        f"Accuracy {_pct(s['accuracy'])}   Macro-F1 {s['macro_f1']:.3f}   "
        f"Case pass rate {_pct(s['case_pass_rate'])}   Errors {s['errors']}",
        "",
    ]
    lines += _table(
        ["intent", "precision", "recall", "f1", "support"],
        [
            (k, f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1']:.2f}", v["support"])
            for k, v in s["per_intent"].items()
        ],
    )
    lines.append("")
    scorer_rows = [(k, _pct(v)) for k, v in s["scorer_pass_rates"].items()]
    scorer_rows += [(k, "skipped") for k in s["scorers_skipped"]]
    lines += _table(["scorer", "pass rate"], scorer_rows)
    lat = s["latency_ms"]
    lines += [
        "",
        f"Latency ms  mean {lat['mean']:.2f}  p50 {lat['p50']:.2f}  p95 {lat['p95']:.2f}  max {lat['max']:.2f}",
    ]
    failures = _failures(report)
    if failures:
        lines += ["", f"Failing cases ({len(failures)}):"]
        lines += [f"  {cid:<8} {exp} -> {pred}  [{why}]" for cid, exp, pred, why in failures]
    return "\n".join(lines)


def render_gate(gate: GateResult) -> str:
    lines = [f"Regression gate (max allowed drop: {gate.threshold * 100:.1f} pts)"]
    lines += _table(
        ["metric", "baseline", "current", "delta", "status"],
        [
            (
                d.name,
                _pct(d.baseline),
                _pct(d.current),
                f"{d.delta * 100:+.1f} pts",
                "REGRESSED" if d.regressed else "ok",
            )
            for d in gate.deltas
        ],
    )
    lines += [f"warning: {w}" for w in gate.warnings]
    lines.append("GATE PASSED" if gate.passed else f"GATE FAILED: {len(gate.regressions)} metric(s) regressed")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any], gate: GateResult | None = None) -> str:
    s = report["summary"]
    target, ds = report["target"], report["dataset"]
    lines = [
        f"# Eval report: `{report['run_id']}`",
        "",
        f"- **Target:** {target['name']} ({target.get('kind', '-')}, model `{target.get('model', '-')}`)",
        f"- **Dataset:** `{ds['name']}` v{ds['version']} - {ds['cases']} cases - sha256 `{ds['sha256'][:12]}`",
        f"- **Created:** {report['created_at']} - harness v{report['harness_version']}",
        "",
        "## Headline metrics",
        "",
    ]
    lines += _md_table(
        ["Metric", "Value"],
        [
            ("Intent accuracy", _pct(s["accuracy"])),
            ("Macro-F1", f"{s['macro_f1']:.3f}"),
            ("Case pass rate (all scorers)", _pct(s["case_pass_rate"])),
            ("Target errors", s["errors"]),
            ("Latency p50 / p95 (ms)", f"{s['latency_ms']['p50']:.2f} / {s['latency_ms']['p95']:.2f}"),
        ],
    )
    lines += ["", "## Scorer pass rates", ""]
    lines += _md_table(
        ["Scorer", "Pass rate"],
        [(k, _pct(v)) for k, v in s["scorer_pass_rates"].items()] + [(k, "skipped") for k in s["scorers_skipped"]],
    )
    lines += ["", "## Per-intent metrics", ""]
    lines += _md_table(
        ["Intent", "Precision", "Recall", "F1", "Support"],
        [
            (k, f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1']:.2f}", v["support"])
            for k, v in s["per_intent"].items()
        ],
    )
    cm = s["confusion_matrix"]
    lines += ["", "## Confusion matrix", "", "Rows: expected intent. Columns: predicted intent.", ""]
    lines += _md_table(
        ["expected \\ predicted"] + cm["labels"],
        [[lbl, *row] for lbl, row in zip(cm["labels"], cm["matrix"], strict=True)],
    )

    failures = _failures(report)
    lines += ["", f"## Failing cases ({len(failures)})", ""]
    if failures:
        lines += _md_table(["Case", "Expected", "Predicted", "Failed checks"], failures, numeric=False)
    else:
        lines.append("None.")

    if gate is not None:
        lines += ["", "## Regression gate", "", "```text", render_gate(gate), "```"]
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any], out_dir: str | Path, gate: GateResult | None = None) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{report['run_id']}.json"
    md_path = out / f"{report['run_id']}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report, gate), encoding="utf-8")
    return json_path, md_path
