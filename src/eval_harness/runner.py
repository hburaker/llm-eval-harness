"""Runs a target over a dataset, applies scorers and builds the report payload."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from eval_harness import INTENTS, __version__, metrics
from eval_harness.dataset import Case, Dataset
from eval_harness.scorers import Scorer
from eval_harness.targets.base import Prediction, Target

REPORT_SCHEMA_VERSION = 1


def predict_timed(target: Target, case: Case) -> Prediction:
    start = time.perf_counter()
    try:
        prediction = target.predict(case.message)
    except Exception as exc:  # a failing call is a data point, not a crash
        prediction = Prediction(intent="other", reply="", error=f"{type(exc).__name__}: {exc}")
    elapsed_ms = (time.perf_counter() - start) * 1000
    return replace(prediction, latency_ms=elapsed_ms)


def evaluate_case(case: Case, prediction: Prediction, scorers: Sequence[Scorer]) -> dict[str, Any]:
    scores = [scorer.score(case, prediction).to_dict() for scorer in scorers]
    applicable = [s for s in scores if s["passed"] is not None]
    return {
        "id": case.id,
        "tags": list(case.tags),
        "expected_intent": case.expected_intent,
        "predicted_intent": prediction.intent,
        "reply": prediction.reply,
        "latency_ms": round(prediction.latency_ms, 3),
        "error": prediction.error,
        "passed": prediction.error is None and all(s["passed"] for s in applicable),
        "scores": scores,
    }


def summarize(case_results: Sequence[dict[str, Any]], labels: Sequence[str] = INTENTS) -> dict[str, Any]:
    y_true = [c["expected_intent"] for c in case_results]
    y_pred = [c["predicted_intent"] for c in case_results]
    matrix = metrics.confusion_matrix(y_true, y_pred, labels)
    rates = metrics.pass_rates([c["scores"] for c in case_results])
    n = len(case_results)
    return {
        "cases": n,
        "accuracy": metrics.accuracy(y_true, y_pred),
        "macro_f1": metrics.macro_f1(matrix, labels),
        "case_pass_rate": sum(c["passed"] for c in case_results) / n if n else 0.0,
        "errors": sum(c["error"] is not None for c in case_results),
        "per_intent": metrics.per_class_metrics(matrix, labels),
        "confusion_matrix": {"labels": list(labels), "matrix": matrix},
        "scorer_pass_rates": {k: v for k, v in rates.items() if v is not None},
        "scorers_skipped": sorted(k for k, v in rates.items() if v is None),
        "latency_ms": metrics.latency_summary([c["latency_ms"] for c in case_results]),
    }


def run_eval(
    dataset: Dataset,
    target: Target,
    scorers: Sequence[Scorer],
    run_id: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    created = datetime.now(timezone.utc)
    case_results = []
    for i, case in enumerate(dataset.cases, start=1):
        case_results.append(evaluate_case(case, predict_timed(target, case), scorers))
        if progress:
            progress(i, len(dataset))

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id or f"{target.name}-{created.strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": created.isoformat(timespec="seconds"),
        "harness_version": __version__,
        "target": target.describe(),
        "dataset": dataset.metadata(),
        "scorers": [s.name for s in scorers],
        "summary": summarize(case_results),
        "cases": case_results,
    }
