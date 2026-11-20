import pytest

from eval_harness import metrics

LABELS = ["a", "b", "c"]
Y_TRUE = ["a", "a", "a", "b", "b", "c"]
Y_PRED = ["a", "a", "b", "b", "c", "c"]


def test_confusion_matrix_rows_are_truth_columns_are_predictions():
    assert metrics.confusion_matrix(Y_TRUE, Y_PRED, LABELS) == [
        [2, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ]


def test_confusion_matrix_rejects_unknown_labels_and_length_mismatch():
    with pytest.raises(ValueError):
        metrics.confusion_matrix(["a"], ["z"], LABELS)
    with pytest.raises(ValueError):
        metrics.confusion_matrix(["a", "b"], ["a"], LABELS)


def test_per_class_precision_recall_f1_by_hand():
    cm = metrics.confusion_matrix(Y_TRUE, Y_PRED, LABELS)
    per = metrics.per_class_metrics(cm, LABELS)
    # a: tp=2, predicted=2, support=3
    assert per["a"]["precision"] == pytest.approx(1.0)
    assert per["a"]["recall"] == pytest.approx(2 / 3)
    assert per["a"]["f1"] == pytest.approx(0.8)
    # b: tp=1, predicted=2, support=2
    assert per["b"]["precision"] == pytest.approx(0.5)
    assert per["b"]["recall"] == pytest.approx(0.5)
    assert per["b"]["f1"] == pytest.approx(0.5)
    # c: tp=1, predicted=2, support=1
    assert per["c"]["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert per["c"]["support"] == 1


def test_zero_division_yields_zero_not_error():
    cm = metrics.confusion_matrix(["a", "a"], ["a", "a"], LABELS)
    per = metrics.per_class_metrics(cm, LABELS)
    assert per["b"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


def test_macro_f1_ignores_labels_absent_from_truth_and_predictions():
    cm = metrics.confusion_matrix(["a", "b"], ["a", "b"], LABELS)
    assert metrics.macro_f1(cm, LABELS) == pytest.approx(1.0)  # "c" never appears


def test_macro_f1_matches_mean_of_present_labels():
    cm = metrics.confusion_matrix(Y_TRUE, Y_PRED, LABELS)
    expected = (0.8 + 0.5 + 2 / 3) / 3
    assert metrics.macro_f1(cm, LABELS) == pytest.approx(expected)


def test_accuracy():
    assert metrics.accuracy(Y_TRUE, Y_PRED) == pytest.approx(4 / 6)
    assert metrics.accuracy([], []) == 0.0


@pytest.mark.parametrize(
    ("pct", "expected"),
    [(0, 1.0), (50, 2.5), (95, 3.85), (100, 4.0)],
)
def test_percentile_linear_interpolation(pct, expected):
    assert metrics.percentile([4.0, 1.0, 3.0, 2.0], pct) == pytest.approx(expected)


def test_pass_rates_exclude_skipped_results():
    rates = metrics.pass_rates(
        [
            [{"scorer": "x", "passed": True}, {"scorer": "judge", "passed": None}],
            [{"scorer": "x", "passed": False}, {"scorer": "judge", "passed": None}],
        ]
    )
    assert rates == {"x": 0.5, "judge": None}
