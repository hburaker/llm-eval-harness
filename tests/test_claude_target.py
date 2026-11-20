"""The Claude target is tested offline against a fake client that mimics the SDK response shape."""

import json

import pytest

from eval_harness.llm import FALLBACK_BETA
from eval_harness.prompts import SYSTEM_PROMPT
from eval_harness.targets.claude import ClaudeTarget


class FakeClient:
    def __init__(self, payload: dict, stop_reason: str = "end_turn") -> None:
        self.calls: list[dict] = []
        block = type("Block", (), {"type": "text", "text": json.dumps(payload)})()
        response = type("Response", (), {"stop_reason": stop_reason, "content": (block,)})()
        outer = self

        class _Messages:
            @staticmethod
            def create(**kwargs):
                outer.calls.append(kwargs)
                return response

        self.beta = type("Beta", (), {"messages": _Messages})()


def test_parses_structured_output_and_sends_expected_request():
    client = FakeClient({"intent": "order_status", "reply": "You can track it under Orders > Track order."})
    prediction = ClaudeTarget(client=client).predict("Where is my order?")

    assert prediction.intent == "order_status"
    assert prediction.reply.startswith("You can track")
    request = client.calls[0]
    assert request["model"] == "claude-sonnet-5-5"
    assert request["system"] == SYSTEM_PROMPT
    assert FALLBACK_BETA in request["betas"]
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "Where is my order?" in request["messages"][0]["content"]


def test_unknown_intent_is_coerced_to_other():
    client = FakeClient({"intent": "refund", "reply": "Sure, let me help with that refund right away."})
    assert ClaudeTarget(client=client).predict("refund pls").intent == "other"


def test_refusal_raises_so_the_runner_records_an_error():
    client = FakeClient({"intent": "other", "reply": ""}, stop_reason="refusal")
    with pytest.raises(RuntimeError, match="refused"):
        ClaudeTarget(client=client).predict("...")
