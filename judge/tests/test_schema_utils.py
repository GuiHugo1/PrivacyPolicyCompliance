import json

import pytest

from judge.schema_utils import (
    DEFAULT_SCHEMA_PATH,
    extract_json_object,
    is_valid_verdict,
    load_schema,
    validate_against_schema,
)

VALID_VERDICT = {
    "article": "13(1)(c)",
    "requirement_present": True,
    "compliance_status": "compliant",
    "evidence_span": "we use your data to provide the service",
    "rationale": "The clause states the purpose of processing.",
    "confidence": 0.9,
}


@pytest.fixture
def schema():
    return load_schema(DEFAULT_SCHEMA_PATH)


def test_default_schema_loads_and_matches_valid_verdict(schema):
    assert validate_against_schema(VALID_VERDICT, schema) == []


def test_extract_json_object_strict():
    text = json.dumps(VALID_VERDICT)
    assert extract_json_object(text) == VALID_VERDICT


def test_extract_json_object_tolerates_surrounding_text():
    text = f"Sure, here is the verdict:\n{json.dumps(VALID_VERDICT)}\nHope that helps!"
    assert extract_json_object(text) == VALID_VERDICT


def test_extract_json_object_returns_none_for_garbage():
    assert extract_json_object("not json at all") is None
    assert extract_json_object("") is None


def test_validate_against_schema_missing_required_key(schema):
    bad = dict(VALID_VERDICT)
    del bad["confidence"]
    errors = validate_against_schema(bad, schema)
    assert any("confidence" in e for e in errors)


def test_validate_against_schema_bad_enum(schema):
    bad = dict(VALID_VERDICT, compliance_status="sort_of")
    errors = validate_against_schema(bad, schema)
    assert any("compliance_status" in e for e in errors)


def test_validate_against_schema_wrong_type(schema):
    bad = dict(VALID_VERDICT, requirement_present="yes")
    errors = validate_against_schema(bad, schema)
    assert any("requirement_present" in e for e in errors)


def test_validate_against_schema_bool_is_not_a_number(schema):
    # bool is a subclass of int in Python; confidence must reject True/False.
    bad = dict(VALID_VERDICT, confidence=True)
    errors = validate_against_schema(bad, schema)
    assert any("confidence" in e for e in errors)


def test_validate_against_schema_out_of_range_confidence(schema):
    bad = dict(VALID_VERDICT, confidence=1.5)
    errors = validate_against_schema(bad, schema)
    assert any("confidence" in e for e in errors)


def test_validate_against_schema_rejects_extra_keys(schema):
    bad = dict(VALID_VERDICT, extra_field="nope")
    errors = validate_against_schema(bad, schema)
    assert any("extra_field" in e for e in errors)


def test_validate_against_schema_non_dict_input(schema):
    errors = validate_against_schema(["not", "a", "dict"], schema)
    assert len(errors) == 1


def test_is_valid_verdict(schema):
    assert is_valid_verdict(json.dumps(VALID_VERDICT), schema) is True
    assert is_valid_verdict("garbage", schema) is False
    invalid = json.dumps(dict(VALID_VERDICT, compliance_status="bad"))
    assert is_valid_verdict(invalid, schema) is False
