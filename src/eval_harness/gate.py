"""Regression gate: compare a run against a saved baseline and fail on drops.

Quality metrics are gated on an absolute drop larger than ``threshold``.
Latency is reported but not gated - it is too noisy across machines to fail
a build on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HEADLINE_METRICS = ("accuracy", "macro_f1", "case_pass_rate")
DEFAULT_THRESHOLD = 0.02


@dataclass(frozen=True)
class MetricDelta:
    name: str
    baseline: float
    current: float
    regressed: bool

    @property
    def delta(self) -> float:
        return self.current - self.baseline


@dataclass
class GateResult:
    threshold: float
    deltas: list[MetricDelta] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.regressed]

    @property
    def passed(self) -> bool:
        return not self.regressions


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a report or baseline JSON file; both carry a ``summary`` block."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "summary" not in data:
        raise ValueError(f"{path}: not a report or baseline file (no 'summary' key)")
    return data


def gated_metrics(summary: dict[str, Any]) -> dict[str, float]:
    values = {name: float(summary[name]) for name in HEADLINE_METRICS if name in summary}
    for scorer, rate in summary.get("scorer_pass_rates", {}).items():
        values[f"pass_rate.{scorer}"] = float(rate)
    return values


def compare(baseline: dict[str, Any], current: dict[str, Any], threshold: float = DEFAULT_THRESHOLD) -> GateResult:
    """Compare two report/baseline payloads. Only metrics present in both are gated."""
    result = GateResult(threshold=threshold)

    base_ds, cur_ds = baseline.get("dataset", {}), current.get("dataset", {})
    if base_ds.get("sha256") and cur_ds.get("sha256") and base_ds["sha256"] != cur_ds["sha256"]:
        result.warnings.append(
            f"dataset changed since the baseline (version {base_ds.get('version')} -> {cur_ds.get('version')}); "
            "metrics may not be directly comparable"
        )

    base_metrics = gated_metrics(baseline["summary"])
    cur_metrics = gated_metrics(current["summary"])
    for name, base_value in base_metrics.items():
        if name not in cur_metrics:
            result.warnings.append(f"metric '{name}' is in the baseline but missing from the current run")
            continue
        cur_value = cur_metrics[name]
        regressed = (base_value - cur_value) > threshold + 1e-12
        result.deltas.append(MetricDelta(name, base_value, cur_value, regressed))
    return result


def make_baseline(report: dict[str, Any]) -> dict[str, Any]:
    """The subset of a report worth committing as a baseline (no per-case replies)."""
    keys = ("schema_version", "run_id", "created_at", "harness_version", "target", "dataset", "scorers", "summary")
    return {k: report[k] for k in keys if k in report}
