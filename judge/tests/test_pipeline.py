"""Tests for the dependency-free parts of judge/pipeline.py: clause
segmentation, chunk-label/article-key derivation, the retry/repair control
flow, and per-article aggregation. ``JudgeModel`` itself (real model
loading) needs torch/transformers/peft and is exercised the same way
judge/eval_qlora.py's model-loading code is -- by an actual (even tiny)
run, not pytest -- see judge/README.md's testing note.
"""

from __future__ import annotations

import json

import pytest

from judge.pipeline import (
    _article_keys_for_chunk,
    _chunk_label,
    aggregate_articles,
    generate_verdict_with_repair,
    known_articles,
    segment_clauses,
)
from judge.schema_utils import DEFAULT_SCHEMA_PATH, load_schema

VALID_VERDICT = {
    "article": "13",
    "requirement_present": True,
    "compliance_status": "compliant",
    "evidence_span": "we collect your email to create an account",
    "rationale": "Discloses the data collected and the purpose.",
    "confidence": 0.9,
}


@pytest.fixture
def schema():
    return load_schema(DEFAULT_SCHEMA_PATH)


# ---------------------------------------------------------------------------
# segment_clauses
# ---------------------------------------------------------------------------


def test_segment_clauses_splits_on_blank_lines():
    text = "First paragraph about data collection.\n\nSecond paragraph about retention."
    clauses = segment_clauses(text)
    assert [c.text for c in clauses] == [
        "First paragraph about data collection.",
        "Second paragraph about retention.",
    ]
    assert [c.id for c in clauses] == ["clause-0", "clause-1"]
    assert [c.index for c in clauses] == [0, 1]


def test_segment_clauses_skips_blank_paragraphs():
    text = "One.\n\n\n\n   \n\nTwo."
    clauses = segment_clauses(text)
    assert [c.text for c in clauses] == ["One.", "Two."]


def test_segment_clauses_collapses_hard_wrapped_whitespace():
    text = "This clause is\nhard-wrapped   across\nseveral   lines."
    clauses = segment_clauses(text)
    assert clauses[0].text == "This clause is hard-wrapped across several lines."


def test_segment_clauses_splits_long_paragraph_into_sentences():
    sentence = "We collect your data for analytics purposes only."
    long_paragraph = " ".join([sentence] * 10)
    clauses = segment_clauses(long_paragraph, max_clause_chars=120)
    assert len(clauses) > 1
    for clause in clauses:
        assert len(clause.text) <= 120 + len(sentence)  # merge can slightly overshoot one sentence
    assert "".join(c.text for c in clauses).replace(" ", "") == long_paragraph.replace(" ", "")


def test_segment_clauses_empty_text():
    assert segment_clauses("") == []
    assert segment_clauses("   \n\n  ") == []


# ---------------------------------------------------------------------------
# chunk label / article-key derivation
# ---------------------------------------------------------------------------


def test_chunk_label_article():
    assert _chunk_label({"source_type": "gdpr_article", "article_number": "13"}) == "13"


def test_chunk_label_concept():
    meta = {"source_type": "gdpr_concept", "concept_articles": "6,21"}
    assert _chunk_label(meta) == "concept:6,21"


def test_chunk_label_recital():
    assert _chunk_label({"source_type": "gdpr_recital", "recital_number": "47"}) == "Recital 47"


def test_chunk_label_edpb_guideline():
    meta = {"source_type": "edpb_guideline", "section_heading": "3. Transparency obligations"}
    assert _chunk_label(meta) == "3. Transparency obligations"


def test_chunk_label_unknown_falls_back_to_chunk_id():
    assert _chunk_label({"chunk_id": "mystery-1"}) == "mystery-1"


def test_article_keys_for_chunk_article():
    assert _article_keys_for_chunk({"source_type": "gdpr_article", "article_number": "13"}) == [
        "13"
    ]


def test_article_keys_for_chunk_concept_fans_out():
    meta = {"source_type": "gdpr_concept", "concept_articles": "6,21"}
    assert _article_keys_for_chunk(meta) == ["6", "21"]


def test_article_keys_for_chunk_recital_and_edpb_are_not_articles():
    assert _article_keys_for_chunk({"source_type": "gdpr_recital", "recital_number": "47"}) == []
    assert _article_keys_for_chunk({"source_type": "edpb_guideline"}) == []


# ---------------------------------------------------------------------------
# generate_verdict_with_repair
# ---------------------------------------------------------------------------


def test_generate_verdict_with_repair_valid_first_try(schema):
    calls = []

    def generate_fn(messages):
        calls.append(messages)
        return json.dumps(VALID_VERDICT)

    result = generate_verdict_with_repair(generate_fn, "clause text", "13", "article text", schema)

    assert len(calls) == 1
    assert result["compliance_status"] == "compliant"
    assert result["retry_used"] is False
    assert result["error"] is None


def test_generate_verdict_with_repair_recovers_on_retry(schema):
    responses = iter(["not valid json at all", json.dumps(VALID_VERDICT)])
    calls = []

    def generate_fn(messages):
        calls.append(messages)
        return next(responses)

    result = generate_verdict_with_repair(generate_fn, "clause text", "13", "article text", schema)

    assert len(calls) == 2
    # The repair turn includes the model's own bad output and an error message.
    repair_messages = calls[1]
    assert repair_messages[-2] == {"role": "assistant", "content": "not valid json at all"}
    assert "not valid JSON" in repair_messages[-1]["content"]
    assert result["compliance_status"] == "compliant"
    assert result["retry_used"] is True
    assert result["error"] is None


def test_generate_verdict_with_repair_falls_back_to_needs_review(schema):
    def generate_fn(messages):
        return "still not json"

    result = generate_verdict_with_repair(generate_fn, "clause text", "13", "article text", schema)

    assert result["compliance_status"] == "needs_review"
    assert result["article"] == "13"
    assert result["requirement_present"] is False
    assert result["confidence"] == 0.0
    assert result["retry_used"] is True
    assert result["error"] is not None


def test_generate_verdict_with_repair_never_raises_on_schema_violation(schema):
    def generate_fn(messages):
        bad = dict(VALID_VERDICT, compliance_status="sort_of")
        return json.dumps(bad)

    result = generate_verdict_with_repair(generate_fn, "clause text", "13", "article text", schema)
    assert result["compliance_status"] == "needs_review"


# ---------------------------------------------------------------------------
# aggregate_articles
# ---------------------------------------------------------------------------


def _clause_verdict(article_number, compliance_status, clause_id="clause-0", **overrides):
    base = {
        "clause_id": clause_id,
        "article": article_number,
        "article_number": article_number,
        "source_type": "gdpr_article",
        "chunk_id": f"article-{article_number}",
        "retrieval_rank": 1,
        "retrieval_score": 0.9,
        "requirement_present": True,
        "compliance_status": compliance_status,
        "evidence_span": f"evidence for {article_number}",
        "rationale": f"rationale for {article_number}",
        "confidence": 0.8,
        "retry_used": False,
        "error": None,
    }
    base.update(overrides)
    return base


def test_aggregate_articles_never_retrieved_is_not_addressed():
    articles = aggregate_articles([], known={"13", "15"})
    by_number = {a["article"]: a for a in articles}
    assert by_number["13"]["best_compliance_status"] == "not_addressed"
    assert by_number["13"]["clauses_addressing_it"] == []
    assert by_number["15"]["best_compliance_status"] == "not_addressed"


def test_aggregate_articles_not_applicable_is_distinct_from_not_addressed():
    verdicts = [_clause_verdict("13", "not_applicable", requirement_present=False)]
    articles = aggregate_articles(verdicts, known={"13"})
    assert articles[0]["article"] == "13"
    assert articles[0]["best_compliance_status"] == "not_applicable"
    assert len(articles[0]["clauses_addressing_it"]) == 1


def test_aggregate_articles_best_status_prefers_compliant_over_non_compliant():
    verdicts = [
        _clause_verdict("13", "non_compliant", clause_id="clause-0"),
        _clause_verdict("13", "compliant", clause_id="clause-5"),
        _clause_verdict("13", "partial", clause_id="clause-9"),
    ]
    articles = aggregate_articles(verdicts, known={"13"})
    assert articles[0]["best_compliance_status"] == "compliant"
    assert articles[0]["evidence"] == "evidence for 13"
    assert len(articles[0]["clauses_addressing_it"]) == 3


def test_aggregate_articles_includes_articles_only_seen_in_verdicts():
    # An article retrieved by a clause but absent from `known` (e.g. a stale
    # index) is still reported rather than silently dropped.
    verdicts = [_clause_verdict("99", "compliant")]
    articles = aggregate_articles(verdicts, known=set())
    assert [a["article"] for a in articles] == ["99"]


def test_aggregate_articles_sorted_numerically():
    verdicts = [_clause_verdict("2", "compliant"), _clause_verdict("10", "compliant")]
    articles = aggregate_articles(verdicts, known={"2", "10"})
    assert [a["article"] for a in articles] == ["2", "10"]


def test_aggregate_articles_ignores_entries_without_article_number():
    verdicts = [_clause_verdict("13", "compliant"), _clause_verdict(None, "compliant")]
    articles = aggregate_articles(verdicts, known={"13"})
    assert [a["article"] for a in articles] == ["13"]
    assert len(articles[0]["clauses_addressing_it"]) == 1


# ---------------------------------------------------------------------------
# known_articles
# ---------------------------------------------------------------------------


class _FakeCollection:
    def __init__(self, metadatas):
        self._metadatas = metadatas

    def get(self, where=None, include=None):
        if where == {"source_type": "gdpr_article"}:
            return {"metadatas": self._metadatas}
        return {"metadatas": []}


def test_known_articles_returns_distinct_article_numbers():
    collection = _FakeCollection(
        [
            {"source_type": "gdpr_article", "article_number": "13"},
            {"source_type": "gdpr_article", "article_number": "13", "paragraph_number": "1"},
            {"source_type": "gdpr_article", "article_number": "15"},
        ]
    )
    assert known_articles(collection) == {"13", "15"}


def test_known_articles_empty_collection():
    assert known_articles(_FakeCollection([])) == set()
