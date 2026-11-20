"""Claude-backed target: one structured-output call returns both intent and reply."""

from __future__ import annotations

from typing import Any

from eval_harness import INTENTS
from eval_harness.llm import DEFAULT_MODEL, make_client, structured_call
from eval_harness.prompts import SYSTEM_PROMPT, TARGET_OUTPUT_SCHEMA
from eval_harness.targets.base import Prediction, Target


class ClaudeTarget(Target):
    name = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, client: Any | None = None) -> None:
        self.model = model
        self.client = client or make_client()

    def predict(self, message: str) -> Prediction:
        data = structured_call(
            self.client,
            model=self.model,
            system=SYSTEM_PROMPT,
            user=f"<customer_message>\n{message}\n</customer_message>",
            schema=TARGET_OUTPUT_SCHEMA,
        )
        intent = data.get("intent", "other")
        if intent not in INTENTS:
            intent = "other"
        return Prediction(intent=intent, reply=str(data.get("reply", "")), meta={"model": self.model})

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "kind": "LLM (Anthropic Messages API)", "model": self.model}
