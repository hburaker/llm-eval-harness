from __future__ import annotations

from pathlib import Path

import pytest

from eval_harness.dataset import Case

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "datasets" / "support_intents.jsonl"
BASELINE_PATH = ROOT / "baselines" / "baseline.json"


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must be offline and deterministic regardless of the developer's shell."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def make_case(**overrides) -> Case:
    fields = {"id": "t-1", "message": "Where is my order?", "expected_intent": "order_status"}
    fields.update(overrides)
    return Case(**fields)
