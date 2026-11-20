import json

import pytest
from conftest import DATASET_PATH

from eval_harness import INTENTS
from eval_harness.dataset import DatasetError, load_dataset


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET_PATH)


def test_golden_dataset_loads_with_version_and_fingerprint(dataset):
    assert 30 <= len(dataset) <= 40
    assert dataset.version == "1.0.0"
    assert len(dataset.sha256) == 64


def test_every_intent_is_covered(dataset):
    distribution = dataset.intent_distribution()
    assert set(distribution) == set(INTENTS)
    assert all(count >= 3 for count in distribution.values())


def test_adversarial_and_edge_cases_are_present(dataset):
    tags = {tag for case in dataset.cases for tag in case.tags}
    assert {"injection", "multilingual", "empty", "ambiguous"} <= tags


def write_jsonl(tmp_path, rows):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows), encoding="utf-8")
    return path


VALID = {"id": "a", "message": "hello", "expected_intent": "other"}


def test_minimal_valid_file_and_blank_lines(tmp_path):
    ds = load_dataset(write_jsonl(tmp_path, [VALID, "", {**VALID, "id": "b", "tags": ["x"]}]))
    assert [c.id for c in ds.cases] == ["a", "b"]
    assert ds.version == "unversioned"


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("{not json", "invalid JSON"),
        ({"id": "a", "message": "hi"}, "missing required field"),
        ({**VALID, "expected_intent": "refund"}, "unknown intent"),
        ({**VALID, "must_include": "track"}, "must be a list"),
        ({**VALID, "colour": "red"}, "unknown field"),
        ([1, 2], "expected a JSON object"),
    ],
)
def test_invalid_rows_are_rejected_with_line_numbers(tmp_path, row, message):
    with pytest.raises(DatasetError, match=message) as exc:
        load_dataset(write_jsonl(tmp_path, [VALID, row]))
    assert "line 2" in str(exc.value)


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(DatasetError, match="duplicate case id"):
        load_dataset(write_jsonl(tmp_path, [VALID, VALID]))


def test_empty_and_missing_files_rejected(tmp_path):
    with pytest.raises(DatasetError, match="empty"):
        load_dataset(write_jsonl(tmp_path, ["", "  "]))
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope.jsonl")


def test_fingerprint_ignores_line_endings(tmp_path):
    lf = tmp_path / "lf" / "d.jsonl"
    crlf = tmp_path / "crlf" / "d.jsonl"
    for p in (lf, crlf):
        p.parent.mkdir()
    body = json.dumps(VALID) + "\n" + json.dumps({**VALID, "id": "b"}) + "\n"
    lf.write_bytes(body.encode())
    crlf.write_bytes(body.replace("\n", "\r\n").encode())
    assert load_dataset(lf).sha256 == load_dataset(crlf).sha256
