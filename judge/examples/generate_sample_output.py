"""Produces `judge/examples/sample_output.json` from `sample_policy.pdf`
without a trained adapter, a built RAG index, or any downloaded model
weights -- retrieval and the judge model are stand-ins that return one fixed,
schema-valid verdict, so this only exercises the *real* PDF ingestion,
clause segmentation, retry/repair result shape, and per-article aggregation
code in `judge/pipeline.py`.

This is NOT what production judging looks like (see judge/README.md's "Try
it out" section for that) -- it exists purely as a fast, concrete way to see
the exact shape of `judge/output_schema.json` filled in, and to regenerate
`sample_output.json` if that schema or `sample_policy.*` ever changes.

Usage::

    uv sync --group rag --group judge
    python -m judge.examples.generate_sample_output
"""

from __future__ import annotations

import json
from pathlib import Path

from judge.pipeline import aggregate_articles as _aggregate_articles
from judge.pipeline import build_output, judge_clauses, known_articles, segment_clauses
from judge.pipeline import load_policy_text as _load_policy_text
from rag.retriever import RetrievedChunk

EXAMPLES_DIR = Path(__file__).parent


class _FakeCollection:
    """Stands in for the real Chroma collection: reports article 13 as the
    only article this toy index "knows about" (see known_articles)."""

    def get(self, where=None, include=None):
        if where == {"source_type": "gdpr_article"}:
            return {"metadatas": [{"source_type": "gdpr_article", "article_number": "13"}]}
        return {"metadatas": []}


class _FakeRetrieverContext:
    """Stands in for RetrieverContext: always retrieves the same one
    (fake) Article 13 chunk, regardless of the clause text."""

    def __init__(self):
        self.collection = _FakeCollection()
        self.hybrid = True
        self.rerank = True
        self.fetch_k = None
        self.rerank_top_n = 20
        self._chunk = RetrievedChunk(
            text=(
                "Article 13 -- Information to be provided where personal data are "
                "collected from the data subject: the controller shall, at the time "
                "when personal data are obtained, provide the data subject with the "
                "identity of the controller, the purposes of the processing, and the "
                "recipients of the personal data."
            ),
            metadata={"source_type": "gdpr_article", "article_number": "13"},
            score=0.91,
            id="gdpr-article-13",
        )

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return [self._chunk][:k]


class _FakeJudgeModel:
    """Stands in for JudgeModel: always returns the same fixed, schema-valid
    verdict instead of running a real base model + LoRA adapter."""

    def judge(self, clause_text: str, article_cite: str, article_text: str) -> dict:
        return {
            "article": article_cite,
            "requirement_present": True,
            "compliance_status": "compliant",
            "evidence_span": clause_text[:80],
            "rationale": "Fake judge model: always returns a fixed compliant verdict.",
            "confidence": 0.9,
            "retry_used": False,
            "error": None,
        }


def main() -> None:
    policy_source = "judge/examples/sample_policy.pdf"
    policy_text = _load_policy_text(EXAMPLES_DIR / "sample_policy.pdf")

    retriever_ctx = _FakeRetrieverContext()
    judge_model = _FakeJudgeModel()

    clauses = segment_clauses(policy_text)
    # _FakeRetrieverContext/_FakeJudgeModel are deliberate duck-typed stand-ins
    # (see their docstrings), not real RetrieverContext/JudgeModel instances.
    clause_verdicts = judge_clauses(clauses, retriever_ctx, judge_model, k=1)  # type: ignore[arg-type]
    articles = _aggregate_articles(clause_verdicts, known_articles(retriever_ctx.collection))

    result = build_output(
        policy_source=policy_source,
        clauses=clauses,
        clause_verdicts=clause_verdicts,
        articles=articles,
        judge_config_path="judge/config/qlora_judge.yaml",
        adapter_path="judge/checkpoints/qwen2.5-7b-qlora-judge",
        retrieval_meta={
            "k": 1,
            "hybrid": True,
            "rerank": True,
            "fetch_k": None,
            "rerank_top_n": 20,
        },
    )

    out_path = EXAMPLES_DIR / "sample_output.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(clauses)} clauses, {len(clause_verdicts)} clause verdicts -> {out_path}")


if __name__ == "__main__":
    main()
