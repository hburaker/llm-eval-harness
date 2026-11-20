"""Systems under test and a small registry to build them by name."""

from __future__ import annotations

from eval_harness.targets.base import Prediction, Target
from eval_harness.targets.rule_based import RuleBasedTarget

TARGET_NAMES = ("baseline", "claude")


def build_target(name: str, model: str | None = None) -> Target:
    if name == "baseline":
        return RuleBasedTarget()
    if name == "claude":
        from eval_harness.targets.claude import ClaudeTarget  # imported lazily: optional dependency

        return ClaudeTarget(model=model) if model else ClaudeTarget()
    raise ValueError(f"unknown target '{name}' (available: {', '.join(TARGET_NAMES)})")


__all__ = ["TARGET_NAMES", "Prediction", "RuleBasedTarget", "Target", "build_target"]
