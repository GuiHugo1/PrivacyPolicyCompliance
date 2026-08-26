import json

import pytest

from judge.schema_utils import DEFAULT_SCHEMA_PATH, load_schema

# eval_qlora.py imports torch/peft/transformers at module scope for its
# model-loading + generation code; skip these tests where that heavy
# `judge` dependency group isn't installed, but still exercise the pure
# aggregation logic (`build_test_report`) when it is.
torch = pytest.importorskip("torch")
eval_qlora = pytest.importorskip("judge.eval_qlora")

SCHEMA = load_schema(DEFAULT_SCHEMA_PATH)


def _gold(status: str) -> str:
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


def _gen(status: str) -> str:
    return _gold(status)


def test_build_test_report_perfect_predictions():
    gold = [_gold("compliant"), _gold("partial")]
    generations = [_gen("compliant"), _gen("partial")]
    report = eval_qlora.build_test_report(generations, gold, SCHEMA)
    assert report["n_examples"] == 2
    assert report["json_validity_rate"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["per_class"]["compliant"]["precision"] == 1.0


def test_build_test_report_counts_invalid_generation():
    gold = [_gold("compliant"), _gold("partial")]
    generations = ["not json", _gen("partial")]
    report = eval_qlora.build_test_report(generations, gold, SCHEMA)
    assert report["json_validity_rate"] == 0.5
    assert report["per_class"]["compliant"]["recall"] == 0.0


def test_build_test_report_raises_on_bad_gold():
    with pytest.raises(ValueError):
        eval_qlora.build_test_report(["irrelevant"], ["not a verdict"], SCHEMA)
