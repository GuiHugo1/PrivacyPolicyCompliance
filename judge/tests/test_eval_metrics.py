import json

from judge.eval_metrics import (
    COMPLIANCE_CLASSES,
    json_validity_rate,
    macro_f1,
    parse_verdict,
    per_class_prf1,
)
from judge.schema_utils import DEFAULT_SCHEMA_PATH, load_schema

SCHEMA = load_schema(DEFAULT_SCHEMA_PATH)


def _verdict(status: str) -> str:
    return json.dumps(
        {
            "article": "5(1)(a)",
            "requirement_present": True,
            "compliance_status": status,
            "evidence_span": "span",
            "rationale": "rationale",
            "confidence": 0.5,
        }
    )


def test_parse_verdict_valid():
    parsed = parse_verdict(_verdict("compliant"), SCHEMA)
    assert parsed.json_valid is True
    assert parsed.verdict["compliance_status"] == "compliant"


def test_parse_verdict_invalid_json():
    parsed = parse_verdict("not json", SCHEMA)
    assert parsed.json_valid is False
    assert parsed.verdict is None


def test_parse_verdict_schema_violation():
    parsed = parse_verdict(_verdict("maybe"), SCHEMA)
    assert parsed.json_valid is False
    assert parsed.verdict is None


def test_json_validity_rate():
    parsed = [
        parse_verdict(_verdict("compliant"), SCHEMA),
        parse_verdict(_verdict("partial"), SCHEMA),
        parse_verdict("garbage", SCHEMA),
        parse_verdict("also garbage", SCHEMA),
    ]
    assert json_validity_rate(parsed) == 0.5


def test_json_validity_rate_empty():
    assert json_validity_rate([]) == 0.0


def test_per_class_prf1_perfect_predictions():
    y_true = ["compliant", "partial", "non_compliant", "not_applicable"]
    y_pred = ["compliant", "partial", "non_compliant", "not_applicable"]
    metrics = per_class_prf1(y_true, y_pred)
    for label in COMPLIANCE_CLASSES:
        m = metrics[label]
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.support == 1
    assert macro_f1(metrics) == 1.0


def test_per_class_prf1_confusion_and_invalid_prediction():
    # 4 examples: one correct, one confused compliant<->partial each way,
    # one gold "compliant" example the model failed to produce valid JSON for.
    y_true = ["compliant", "compliant", "partial", "compliant"]
    y_pred = ["compliant", "partial", "compliant", None]
    metrics = per_class_prf1(y_true, y_pred)

    compliant = metrics["compliant"]
    assert compliant.support == 3
    assert compliant.tp == 1  # example 0
    assert compliant.fp == 1  # example 2 (gold partial, predicted compliant)
    assert compliant.fn == 2  # examples 1 and 3
    assert compliant.precision == 0.5
    assert compliant.recall == 1 / 3

    partial = metrics["partial"]
    assert partial.support == 1
    assert partial.tp == 0
    assert partial.fp == 1  # example 1 (gold compliant, predicted partial)
    assert partial.fn == 1  # example 2

    non_compliant = metrics["non_compliant"]
    assert non_compliant.support == 0
    assert non_compliant.precision == 0.0
    assert non_compliant.recall == 0.0
    assert non_compliant.f1 == 0.0

    # The None (invalid-JSON) prediction never scores a TP/FP for any class:
    # only the 3 examples with an actual predicted label contribute one
    # tp-or-fp count each (example 0 -> TP compliant, example 1 -> FP
    # partial, example 2 -> FP compliant); example 3's None contributes none.
    assert sum(m.tp + m.fp for m in metrics.values()) == 3


def test_per_class_prf1_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        per_class_prf1(["compliant"], ["compliant", "partial"])


def test_macro_f1_empty():
    assert macro_f1({}) == 0.0
