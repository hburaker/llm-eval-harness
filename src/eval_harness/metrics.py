"""Pure-Python classification and aggregate metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    """Rows are true labels, columns are predicted labels, both in ``labels`` order."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred, strict=True):
        if t not in index or p not in index:
            raise ValueError(f"label not in labels: {t if t not in index else p}")
        matrix[index[t]][index[p]] += 1
    return matrix


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def per_class_metrics(matrix: list[list[int]], labels: Sequence[str]) -> dict[str, dict[str, float]]:
    """Precision, recall, F1 and support for each label (0.0 when undefined)."""
    results: dict[str, dict[str, float]] = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        predicted = sum(row[i] for row in matrix)
        support = sum(matrix[i])
        precision = _safe_div(tp, predicted)
        recall = _safe_div(tp, support)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        results[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return results


def macro_f1(matrix: list[list[int]], labels: Sequence[str]) -> float:
    """Unweighted mean F1 over labels that occur in either the truth or the predictions."""
    per_class = per_class_metrics(matrix, labels)
    present = [label for i, label in enumerate(labels) if sum(matrix[i]) > 0 or sum(row[i] for row in matrix) > 0]
    return _safe_div(sum(per_class[label]["f1"] for label in present), len(present))


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return _safe_div(sum(t == p for t, p in zip(y_true, y_pred, strict=True)), len(y_true))


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile (same definition as numpy's default)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": sum(latencies_ms) / len(latencies_ms),
        "p50": percentile(latencies_ms, 50),
        "p95": percentile(latencies_ms, 95),
        "max": max(latencies_ms),
    }


def pass_rates(case_scores: Sequence[Sequence[dict[str, Any]]]) -> dict[str, float | None]:
    """Pass rate per scorer across cases, ignoring skipped results (``passed is None``).

    Returns ``None`` for a scorer that was skipped on every case.
    """
    totals: dict[str, list[int]] = {}
    for scores in case_scores:
        for s in scores:
            bucket = totals.setdefault(s["scorer"], [0, 0])
            if s["passed"] is not None:
                bucket[0] += int(s["passed"])
                bucket[1] += 1
    return {name: (_safe_div(p, n) if n else None) for name, (p, n) in totals.items()}
