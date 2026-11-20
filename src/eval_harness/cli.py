"""Command-line interface.

    python -m eval_harness run --target baseline [--gate baselines/baseline.json]
    python -m eval_harness compare reports/a.json reports/b.json
    python -m eval_harness validate datasets/support_intents.jsonl

Exit codes: 0 success, 1 regression detected, 2 configuration or input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval_harness.dataset import DatasetError, load_dataset
from eval_harness.gate import DEFAULT_THRESHOLD, compare, load_report, make_baseline
from eval_harness.llm import LLMUnavailableError, api_key_present
from eval_harness.report import render_console, render_gate, write_reports
from eval_harness.runner import run_eval
from eval_harness.scorers import default_scorers
from eval_harness.targets import TARGET_NAMES, build_target

EXIT_OK, EXIT_REGRESSION, EXIT_ERROR = 0, 1, 2
DEFAULT_DATASET = "datasets/support_intents.jsonl"


def _cmd_run(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    target = build_target(args.target, args.model)
    judge = not args.no_judge
    if judge and not api_key_present():
        print("note: ANTHROPIC_API_KEY not set - the llm_judge scorer will be skipped", file=sys.stderr)
    scorers = default_scorers(judge=judge)

    def progress(done: int, total: int) -> None:
        print(f"\r  evaluating {done}/{total}", end="", file=sys.stderr, flush=True)

    interactive = sys.stderr.isatty()
    report = run_eval(dataset, target, scorers, run_id=args.run_name, progress=progress if interactive else None)
    if interactive:
        print(file=sys.stderr)

    gate = None
    if args.gate:
        gate = compare(load_report(args.gate), report, threshold=args.threshold)

    json_path, md_path = write_reports(report, args.out, gate)
    print(render_console(report))
    if gate:
        print("\n" + render_gate(gate))
    print(f"\nReports written: {json_path}  {md_path}")

    if args.update_baseline:
        path = Path(args.update_baseline)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(make_baseline(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Baseline updated: {path}")

    return EXIT_REGRESSION if gate and not gate.passed else EXIT_OK


def _cmd_compare(args: argparse.Namespace) -> int:
    baseline, current = load_report(args.baseline), load_report(args.current)
    print(f"Comparing {baseline.get('run_id', args.baseline)} (baseline) -> {current.get('run_id', args.current)}\n")
    gate = compare(baseline, current, threshold=args.threshold)
    print(render_gate(gate))
    return EXIT_OK if gate.passed else EXIT_REGRESSION


def _cmd_validate(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    print(f"{dataset.name} v{dataset.version}: {len(dataset)} valid cases (sha256 {dataset.sha256[:12]})")
    for intent, count in dataset.intent_distribution().items():
        print(f"  {intent:<15} {count}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval_harness", description="Evaluate a support-intent LLM system.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a target over the golden dataset")
    run.add_argument("--target", choices=TARGET_NAMES, default="baseline")
    run.add_argument("--model", help="model id for LLM targets (default: claude-sonnet-5-5)")
    run.add_argument("--dataset", default=DEFAULT_DATASET)
    run.add_argument("--out", default="reports", help="directory for the JSON and Markdown reports")
    run.add_argument("--run-name", help="report file name (default: <target>-<UTC timestamp>)")
    run.add_argument("--gate", metavar="BASELINE_JSON", help="fail (exit 1) if metrics regress vs this baseline")
    run.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="max allowed absolute drop (0.02 = 2 pts)"
    )
    run.add_argument("--no-judge", action="store_true", help="disable the LLM-as-judge scorer")
    run.add_argument("--update-baseline", metavar="PATH", help="write this run's summary as the new baseline")
    run.set_defaults(func=_cmd_run)

    cmp_ = sub.add_parser("compare", help="compare two report/baseline JSON files")
    cmp_.add_argument("baseline")
    cmp_.add_argument("current")
    cmp_.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    cmp_.set_defaults(func=_cmd_compare)

    val = sub.add_parser("validate", help="validate a dataset file")
    val.add_argument("dataset", nargs="?", default=DEFAULT_DATASET)
    val.set_defaults(func=_cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (DatasetError, LLMUnavailableError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
