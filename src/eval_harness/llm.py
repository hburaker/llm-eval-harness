"""Thin wrapper around the Anthropic SDK used by the Claude target and the LLM judge.

The ``anthropic`` package is an optional dependency; everything else in the
harness runs without it.
"""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_MODEL = "claude-sonnet-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailableError(RuntimeError):
    """Raised when the API key or the ``anthropic`` package is missing."""


def api_key_present() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def make_client() -> Any:
    if not api_key_present():
        raise LLMUnavailableError("ANTHROPIC_API_KEY is not set")
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise LLMUnavailableError(
            'the "anthropic" package is not installed; run: pip install -e ".[anthropic]"'
        ) from exc
    return anthropic.Anthropic()


def structured_call(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    max_tokens: int = 2048,
    effort: str = "low",
) -> dict[str, Any]:
    """Send one request constrained to ``schema`` and return the parsed JSON object.

    Server-side refusal fallbacks are enabled, so a request declined by a
    safety classifier is retried on a fallback model inside the same call.
    """
    response = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("model refused the request")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("response truncated at max_tokens")
    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        raise RuntimeError("response contained no text block")
    return json.loads(text)
