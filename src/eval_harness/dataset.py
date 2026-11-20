"""Golden dataset loading and validation.

A dataset is a JSONL file with one test case per line. Its version lives in a
``manifest.json`` next to it; the content hash is computed on load so every
report records exactly which data it was produced from.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_harness import INTENTS

REQUIRED_FIELDS = ("id", "message", "expected_intent")
OPTIONAL_LIST_FIELDS = ("must_include", "must_not_include", "tags")
ALLOWED_FIELDS = frozenset(REQUIRED_FIELDS + OPTIONAL_LIST_FIELDS + ("notes",))


class DatasetError(ValueError):
    """Raised when a dataset file is missing, malformed or fails validation."""


@dataclass(frozen=True)
class Case:
    id: str
    message: str
    expected_intent: str
    must_include: tuple[str, ...] = ()
    must_not_include: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    notes: str = ""

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags


@dataclass(frozen=True)
class Dataset:
    name: str
    version: str
    sha256: str
    cases: tuple[Case, ...]

    def __len__(self) -> int:
        return len(self.cases)

    def intent_distribution(self) -> dict[str, int]:
        counts = Counter(case.expected_intent for case in self.cases)
        return {intent: counts.get(intent, 0) for intent in INTENTS}

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "sha256": self.sha256, "cases": len(self)}


def parse_case(raw: Any, line_no: int) -> Case:
    """Validate one decoded JSON object and turn it into a :class:`Case`."""
    where = f"line {line_no}"
    if not isinstance(raw, dict):
        raise DatasetError(f"{where}: expected a JSON object")

    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        raise DatasetError(f"{where}: missing required field(s): {', '.join(missing)}")
    unknown = sorted(set(raw) - ALLOWED_FIELDS)
    if unknown:
        raise DatasetError(f"{where}: unknown field(s): {', '.join(unknown)}")

    for field in REQUIRED_FIELDS:
        if not isinstance(raw[field], str):
            raise DatasetError(f"{where}: '{field}' must be a string")
    if not raw["id"].strip():
        raise DatasetError(f"{where}: 'id' must not be blank")
    if raw["expected_intent"] not in INTENTS:
        raise DatasetError(f"{where}: unknown intent '{raw['expected_intent']}' (allowed: {', '.join(INTENTS)})")

    lists: dict[str, tuple[str, ...]] = {}
    for field in OPTIONAL_LIST_FIELDS:
        value = raw.get(field, [])
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            raise DatasetError(f"{where}: '{field}' must be a list of non-empty strings")
        lists[field] = tuple(value)

    return Case(
        id=raw["id"],
        message=raw["message"],
        expected_intent=raw["expected_intent"],
        notes=str(raw.get("notes", "")),
        **lists,
    )


def _read_version(path: Path) -> str:
    manifest = path.parent / "manifest.json"
    if not manifest.exists():
        return "unversioned"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{manifest}: invalid JSON ({exc})") from exc
    return str(data.get(path.stem, {}).get("version", "unversioned"))


def load_dataset(path: str | Path) -> Dataset:
    """Load, validate and fingerprint a JSONL dataset."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"dataset not found: {path}")

    # Normalise line endings so the fingerprint is identical on Windows and Linux checkouts.
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    cases: list[Case] = []
    seen: set[str] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"line {line_no}: invalid JSON ({exc.msg})") from exc
        case = parse_case(raw, line_no)
        if case.id in seen:
            raise DatasetError(f"line {line_no}: duplicate case id '{case.id}'")
        seen.add(case.id)
        cases.append(case)

    if not cases:
        raise DatasetError(f"dataset is empty: {path}")

    return Dataset(
        name=path.stem,
        version=_read_version(path),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        cases=tuple(cases),
    )
