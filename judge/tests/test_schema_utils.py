import copy
import json
from pathlib import Path

import pytest

from judge.build_sft_dataset import JUDGE_SYSTEM_PROMPT
from judge.mapping import ALLOWED_COMPLIANCE_STATUS
from judge.schema_utils import (
    DEFAULT_SCHEMA_PATH,
    extract_json_object,
    is_valid_verdict,
    load_schema,
    validate_against_schema,
)

OUTPUT_SCHEMA_PATH = Path(__file__).parent.parent / "output_schema.json"

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


# ---------------------------------------------------------------------------
# Cross-schema consistency: the `compliance_status` enum is duplicated in
# three independent places (judge_schema.json's own enum, judge/mapping.py's
# ALLOWED_COMPLIANCE_STATUS, and the schema description hardcoded into
# JUDGE_SYSTEM_PROMPT for the model prompt) -- these must never drift apart,
# or the judge would be prompted/trained against a different vocabulary than
# what schema_utils actually validates its output against.
# ---------------------------------------------------------------------------


def test_judge_schema_compliance_status_enum_matches_mapping_module(schema):
    schema_enum = set(schema["properties"]["compliance_status"]["enum"])
    assert schema_enum == ALLOWED_COMPLIANCE_STATUS


def test_judge_system_prompt_enum_matches_judge_schema(schema):
    # JUDGE_SYSTEM_PROMPT hardcodes the same enum as a `"a"|"b"|"c"` string
    # (see build_sft_dataset.py) rather than importing the schema, since it's
    # baked into every training example's prompt text verbatim -- so this
    # guards against the two drifting apart independently.
    schema_enum = schema["properties"]["compliance_status"]["enum"]
    quoted = "|".join(f'"{status}"' for status in schema_enum)
    assert quoted in JUDGE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Nested/$ref schema support: judge/output_schema.json (the full pipeline
# output judge/pipeline.py assembles, not just one verdict) nests objects
# and arrays several levels deep and uses $ref/$defs -- unlike the flat
# judge_schema.json above, which never exercises that code path.
# ---------------------------------------------------------------------------


@pytest.fixture
def output_schema():
    return load_schema(OUTPUT_SCHEMA_PATH)


VALID_CLAUSE_VERDICT = {
    "clause_id": "clause-0",
    "article": "13",
    "article_number": "13",
    "source_type": "gdpr_article",
    "chunk_id": "gdpr-article-13",
    "retrieval_rank": 1,
    "retrieval_score": 0.87,
    "requirement_present": True,
    "compliance_status": "compliant",
    "evidence_span": "we collect your email to create an account",
    "rationale": "Discloses the data collected and the purpose.",
    "confidence": 0.9,
    "retry_used": False,
    "error": None,
}

VALID_ARTICLE_SUMMARY = {
    "article": "13",
    "clauses_addressing_it": [
        {
            "clause_id": "clause-0",
            "compliance_status": "compliant",
            "requirement_present": True,
            "evidence_span": "we collect your email to create an account",
            "confidence": 0.9,
        }
    ],
    "best_compliance_status": "compliant",
    "evidence": "we collect your email to create an account",
    "rationale": "Discloses the data collected and the purpose.",
}

VALID_PIPELINE_OUTPUT = {
    "meta": {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "pipeline_version": "0.1.0",
        "judge_model": {
            "config": "judge/config/qlora_judge.yaml",
            "adapter": "judge/checkpoints/qwen2.5-7b-qlora-judge",
        },
        "retrieval": {"k": 3, "hybrid": True, "rerank": True, "fetch_k": None, "rerank_top_n": 20},
    },
    "policy": {"source": "judge/examples/sample_policy.pdf", "n_clauses": 1},
    "clauses": [{"id": "clause-0", "index": 0, "text": "We collect your email address."}],
    "clause_verdicts": [VALID_CLAUSE_VERDICT],
    "articles": [VALID_ARTICLE_SUMMARY],
}


def test_valid_pipeline_output_matches_output_schema(output_schema):
    assert validate_against_schema(VALID_PIPELINE_OUTPUT, output_schema) == []


def test_output_schema_catches_missing_nested_required_key(output_schema):
    bad = copy.deepcopy(VALID_PIPELINE_OUTPUT)
    del bad["meta"]["judge_model"]["adapter"]
    errors = validate_against_schema(bad, output_schema)
    assert any("meta.judge_model.adapter" in e for e in errors)


def test_output_schema_catches_bad_array_item(output_schema):
    bad = copy.deepcopy(VALID_PIPELINE_OUTPUT)
    del bad["clauses"][0]["text"]
    errors = validate_against_schema(bad, output_schema)
    assert any("clauses[0].text" in e for e in errors)


def test_output_schema_resolves_ref_enum_inside_array_item(output_schema):
    bad = copy.deepcopy(VALID_PIPELINE_OUTPUT)
    bad["clause_verdicts"][0]["compliance_status"] = "invalid_status"
    errors = validate_against_schema(bad, output_schema)
    assert any("clause_verdicts[0].compliance_status" in e for e in errors)


def test_output_schema_needs_review_is_valid_only_via_ref(output_schema):
    ok = copy.deepcopy(VALID_PIPELINE_OUTPUT)
    ok["clause_verdicts"][0]["compliance_status"] = "needs_review"
    ok["clause_verdicts"][0]["error"] = "Judge did not produce schema-valid JSON."
    assert validate_against_schema(ok, output_schema) == []


def test_output_schema_nullable_union_type_accepts_null_and_int(output_schema):
    ok_null = copy.deepcopy(VALID_PIPELINE_OUTPUT)
    ok_null["meta"]["retrieval"]["fetch_k"] = None
    assert validate_against_schema(ok_null, output_schema) == []

    ok_int = copy.deepcopy(VALID_PIPELINE_OUTPUT)
    ok_int["meta"]["retrieval"]["fetch_k"] = 50
    assert validate_against_schema(ok_int, output_schema) == []


def test_output_schema_nullable_union_type_rejects_wrong_type(output_schema):
    bad = copy.deepcopy(VALID_PIPELINE_OUTPUT)
    bad["meta"]["retrieval"]["fetch_k"] = "fifty"
    errors = validate_against_schema(bad, output_schema)
    assert any("meta.retrieval.fetch_k" in e for e in errors)


def test_output_schema_rejects_additional_properties_in_nested_object(output_schema):
    bad = copy.deepcopy(VALID_PIPELINE_OUTPUT)
    bad["meta"]["judge_model"]["extra"] = "nope"
    errors = validate_against_schema(bad, output_schema)
    assert any("meta.judge_model.extra" in e for e in errors)
