"""The system-under-test abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Prediction:
    """What a target produced for one message.

    ``latency_ms`` is filled in by the runner, not by the target, so every
    implementation is timed the same way.
    """

    intent: str
    reply: str
    latency_ms: float = 0.0
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Target(ABC):
    """A system under test: takes a customer message, returns an intent and a reply."""

    name: str = "target"

    @abstractmethod
    def predict(self, message: str) -> Prediction:
        """Classify ``message`` and draft a reply. May raise; the runner records the error."""

    def describe(self) -> dict[str, Any]:
        """Metadata recorded in every report (model id, version, ...)."""
        return {"name": self.name}
