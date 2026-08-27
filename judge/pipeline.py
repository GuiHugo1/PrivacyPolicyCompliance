"""End-to-end GDPR compliance-judging pipeline.

Takes a full privacy policy -- a ``.pdf`` or a plain-text/``.txt`` file, see
``load_policy_text`` -- and produces one JSON object matching
``judge/output_schema.json``: the text segmented into clauses, one judge
verdict per (clause, retrieved-reference) pair, and a policy-level
aggregation of those verdicts per GDPR article.

Pipeline stages:

0. ``load_policy_text`` -- reads the input file, extracting text from PDF
   pages via ``pypdf`` (already a ``rag`` dependency-group requirement) when
   ``--policy`` ends in ``.pdf``, or reading it as plain text otherwise --
   most privacy policies are published as PDFs, so this is the common case,
   not a fallback.
1. ``segment_clauses`` -- paragraph + sentence-heuristic clause segmentation.
   See its docstring for why this is deliberately simple and where it's
   expected to be improved later.
2. For each clause, ``RetrieverContext.retrieve`` calls ``rag.retriever.retrieve``
   for the top-``k`` (default 3) most relevant chunks. Per
   ``eval/RAG_ANALYSIS.md``'s conclusion (also documented in
   ``judge/README.md``'s "Alignment with the RAG pipeline"), this project's
   recommended retrieval configuration is ``hybrid=True, rerank=True``, so
   that's this pipeline's default -- overridable via ``--hybrid``/``--rerank``.
3. Each retrieved chunk is judged individually: ``JudgeModel.judge`` (via the
   dependency-free ``generate_verdict_with_repair``) prompts the fine-tuned
   LoRA judge with ``(clause, chunk text)`` in exactly the
   ``judge/build_sft_dataset.py`` training format (system prompt + "Clause:
   ...\\n\\nRetrieved GDPR Article ...:\\n...") and parses its JSON verdict
   against ``judge/judge_schema.json``. One verdict per (clause, chunk) pair
   -- not one verdict per clause bundling all k references -- because that's
   what the judge model was actually fine-tuned to do (see
   ``build_sft_dataset.build_example``: one training example per
   (clause, single article) pair). A ``gdpr_concept`` chunk (see
   ``rag/parsers/gdpr.py``) names two articles in one chunk, so its one
   judge call is recorded under both articles.
4. Invalid/unparseable judge output gets exactly one repair retry (the
   model's own bad output plus a message describing the schema error,
   asking it to correct itself); if that also fails, the pair falls back to
   a ``needs_review`` verdict rather than raising -- see
   ``generate_verdict_with_repair``.
5. ``aggregate_articles`` groups every clause_verdicts entry by GDPR article
   number into the policy-level ``articles`` list, picking a
   ``best_compliance_status`` per article (see its docstring for the
   ranking rule).
6. Articles present in the RAG index but never retrieved by any clause are
   included with ``best_compliance_status: "not_addressed"`` -- distinct
   from ``not_applicable``, which means a clause *was* checked against the
   article and the judge decided it doesn't apply. The "known" article
   universe is read from the live Chroma collection (``known_articles``),
   not a hardcoded list: ``data/raw/gdpr.json`` isn't committed to this repo
   (see ``.gitignore``'s ``data/`` rules and ``rag/README.md``), so whatever
   was actually indexed -- the full regulation, a subset, or a test fixture
   -- is the only real source of truth for "which articles exist here".

Known limitation: retrieval draws from the whole corpus (GDPR articles,
recitals, and EDPB guideline sections all share one Chroma collection -- see
``rag/store.py``), matching the RAG architecture as built. But the judge was
only fine-tuned on GDPR-article-shaped grounding text (``judge/gdpr_source.py``
never loads recitals or EDPB PDFs). When retrieval surfaces a recital or
EDPB chunk, this pipeline still submits it to the judge for a best-effort
verdict (never crashes; retry/repair still applies) and records it in
``clause_verdicts`` with ``article_number: null`` (excluded from the
per-article ``articles`` aggregation, which only covers real GDPR articles).
Improving judge coverage of non-article references is future work, not
solved here.

Model loading mirrors ``judge/eval_qlora.py``'s ``load_model_for_eval``
exactly (same config schema, same 4-bit/bf16/device-backend resolution via
``judge/train_qlora.py``), so any ``judge/config/qlora_judge*.yaml`` +
matching LoRA adapter checkpoint trained by ``judge/train_qlora.py`` works
here unchanged.

Usage::

    uv sync --group rag --group judge

    python -m judge.pipeline \\
        --policy path/to/policy.pdf \\
        --adapter judge/checkpoints/qwen2.5-7b-qlora-judge \\
        --out result.json

    # a plain-text policy works the same way:
    python -m judge.pipeline \\
        --policy path/to/policy.txt \\
        --adapter judge/checkpoints/qwen2.5-7b-qlora-judge \\
        --out result.json

    # against the 0.5B smoke-test adapter instead of the 7B judge:
    python -m judge.pipeline \\
        --policy path/to/policy.pdf \\
        --config judge/config/qlora_judge_0.5b_gpu.yaml \\
        --adapter judge/checkpoints/qwen2.5-0.5b-gpu-qlora-judge \\
        --out result.json

See ``judge/README.md``'s "Try it out" walkthrough for a complete,
runnable example against ``judge/examples/sample_policy.pdf``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from judge.build_sft_dataset import JUDGE_SYSTEM_PROMPT
from judge.eval_metrics import parse_verdict
from judge.schema_utils import (
    DEFAULT_SCHEMA_PATH,
    extract_json_object,
    load_schema,
    validate_against_schema,
)
from rag.embeddings import Embedder, get_embedder
from rag.lexical import BM25Index
from rag.rerank import get_reranker
from rag.retriever import DEFAULT_RERANK_TOP_N, RetrievedChunk, retrieve
from rag.store import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR, get_or_create_collection

PIPELINE_VERSION = "0.1.0"

# Matches judge/build_sft_dataset.py's _MAX_CLAUSE_CHARS, so inference-time
# clause length matches what the judge was trained on.
DEFAULT_MAX_CLAUSE_CHARS = 2000

# ---------------------------------------------------------------------------
# 1. Clause segmentation
# ---------------------------------------------------------------------------

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_INTRA_PARAGRAPH_WHITESPACE_RE = re.compile(r"\s+")
# Split after sentence-ending punctuation followed by whitespace and a
# capital letter/opening quote/paren -- a plain heuristic, not a real
# sentence boundary detector.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(])")


@dataclass
class Clause:
    id: str
    index: int
    text: str


def _merge_sentences(sentences: list[str], max_clause_chars: int) -> list[str]:
    """Greedily re-merges sentence fragments up to ``max_clause_chars`` so a
    clause stays a multi-sentence unit instead of one bare sentence each."""
    merged: list[str] = []
    buf = ""
    for sentence in sentences:
        candidate = f"{buf} {sentence}".strip() if buf else sentence
        if buf and len(candidate) > max_clause_chars:
            merged.append(buf)
            buf = sentence
        else:
            buf = candidate
    if buf:
        merged.append(buf)
    return merged


def segment_clauses(
    policy_text: str, max_clause_chars: int = DEFAULT_MAX_CLAUSE_CHARS
) -> list[Clause]:
    """Segments a full policy text into clauses: split on blank lines
    (paragraphs), then, within any paragraph longer than
    ``max_clause_chars``, split into sentences and greedily re-merge them
    back up to that length.

    AREA FOR IMPROVEMENT: this is a deliberately simple heuristic, not a
    real sentence/clause segmenter. It doesn't handle abbreviations (e.g.,
    "U.S.", "Inc.", "Art."), nested/lettered lists, footnotes, or
    legal cross-references, and a hard-wrapped source paragraph is only
    recovered via whitespace collapsing, not re-flowed intelligently. A
    production version should swap this for a proper sentence/clause
    segmenter (e.g. spaCy, or a rule set tuned on this project's actual
    policy corpus) -- flagged here rather than solved, per the project's
    current priorities.
    """
    clauses: list[Clause] = []
    for paragraph in _PARAGRAPH_SPLIT_RE.split(policy_text):
        paragraph = _INTRA_PARAGRAPH_WHITESPACE_RE.sub(" ", paragraph).strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_clause_chars:
            segments = [paragraph]
        else:
            segments = _merge_sentences(_SENTENCE_SPLIT_RE.split(paragraph), max_clause_chars)
        for segment in segments:
            segment = segment.strip()
            if segment:
                clauses.append(
                    Clause(id=f"clause-{len(clauses)}", index=len(clauses), text=segment)
                )
    return clauses


# ---------------------------------------------------------------------------
# 2/3. Retrieval
# ---------------------------------------------------------------------------


@dataclass
class RetrieverContext:
    """Bundles the shared retrieval objects (collection/embedder/BM25 index/
    reranker) built once per pipeline run -- rebuilding the BM25 index or
    reloading the cross-encoder per clause would be wasteful, per
    rag/README.md's "A caller doing many queries..." note."""

    collection: Any
    embedder: Any
    hybrid: bool
    rerank: bool
    bm25_index: Any | None = None
    reranker: Any | None = None
    fetch_k: int | None = None
    rerank_top_n: int = DEFAULT_RERANK_TOP_N

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return retrieve(
            query,
            k=k,
            collection=self.collection,
            embedder=self.embedder,
            hybrid=self.hybrid,
            bm25_index=self.bm25_index,
            fetch_k=self.fetch_k,
            rerank=self.rerank,
            reranker=self.reranker,
            rerank_top_n=self.rerank_top_n,
        )


def build_retriever_context(
    *,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    hybrid: bool = True,
    rerank: bool = True,
    fetch_k: int | None = None,
    rerank_top_n: int = DEFAULT_RERANK_TOP_N,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
) -> RetrieverContext:
    collection = get_or_create_collection(persist_dir, collection_name)
    embedder = Embedder(embedding_model) if embedding_model else get_embedder()

    bm25_index = BM25Index.from_collection(collection) if hybrid else None
    reranker = None
    if rerank:
        reranker = get_reranker(reranker_model) if reranker_model else get_reranker()

    return RetrieverContext(
        collection=collection,
        embedder=embedder,
        hybrid=hybrid,
        rerank=rerank,
        bm25_index=bm25_index,
        reranker=reranker,
        fetch_k=fetch_k,
        rerank_top_n=rerank_top_n,
    )


def known_articles(collection: Any) -> set[str]:
    """Every base GDPR article number actually present in the persisted RAG
    index, read live from the collection rather than a hardcoded list --
    see the module docstring's point 6 for why: this repo doesn't commit
    ``data/raw/gdpr.json``, so there is no static article list to hardcode
    against, and this way the "never addressed" check always matches
    whatever corpus was actually indexed."""
    got = collection.get(where={"source_type": "gdpr_article"}, include=["metadatas"])
    return {
        meta["article_number"]
        for meta in got.get("metadatas") or []
        if meta and meta.get("article_number")
    }


def _chunk_label(metadata: dict[str, Any]) -> str:
    """Human-readable label for what was retrieved, used both in the judge
    prompt ("Retrieved GDPR Article {label}:") and in each clause_verdicts
    entry's ``article`` field."""
    source_type = metadata.get("source_type")
    if source_type == "gdpr_article":
        return metadata.get("article_number", "unknown")
    if source_type == "gdpr_concept":
        return f"concept:{metadata.get('concept_articles', 'unknown')}"
    if source_type == "gdpr_recital":
        return f"Recital {metadata.get('recital_number', '?')}"
    if source_type == "edpb_guideline":
        return (
            metadata.get("section_heading") or metadata.get("guideline_title") or "EDPB guideline"
        )
    return metadata.get("chunk_id") or "unknown"


def _article_keys_for_chunk(metadata: dict[str, Any]) -> list[str]:
    """Base GDPR article number(s) a retrieved chunk should be aggregated
    under in ``articles[]``. Empty for recital/EDPB chunks -- they aren't
    themselves a GDPR article (see the module docstring's "Known
    limitation")."""
    source_type = metadata.get("source_type")
    if source_type == "gdpr_article":
        number = metadata.get("article_number")
        return [number] if number else []
    if source_type == "gdpr_concept":
        return [a.strip() for a in metadata.get("concept_articles", "").split(",") if a.strip()]
    return []


# ---------------------------------------------------------------------------
# 4. Judge model + retry/repair
# ---------------------------------------------------------------------------

GenerateFn = Callable[[list[dict[str, str]]], str]


def _describe_schema_errors(raw_text: str, schema: dict[str, Any]) -> str:
    obj = extract_json_object(raw_text)
    if obj is None:
        return "output is not valid JSON"
    return "; ".join(validate_against_schema(obj, schema)) or "unknown schema error"


def generate_verdict_with_repair(
    generate_fn: GenerateFn,
    clause_text: str,
    article_cite: str,
    article_text: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Runs one (clause, article) pair through ``generate_fn``, matching
    ``judge/build_sft_dataset.py``'s exact training prompt shape (system
    prompt + "Clause:\\n...\\n\\nRetrieved GDPR Article ...:\\n...").

    If the model's output isn't schema-valid JSON, retries once with its own
    bad output plus a message describing the schema error and asking it to
    correct itself. If the retry also fails, falls back to a
    ``needs_review`` verdict rather than raising -- this function never
    crashes on model output.

    ``generate_fn`` is injected (chat-message list -> raw decoded
    completion) so this retry/repair control flow is unit-testable without
    loading the actual model -- see ``JudgeModel.judge``, which is the only
    caller that passes a real model-backed ``generate_fn``.

    Returns a dict with ``judge_schema.json``'s fields (``article``,
    ``requirement_present``, ``compliance_status``, ``evidence_span``,
    ``rationale``, ``confidence``) plus pipeline-only ``retry_used``/``error``.
    """
    user_content = (
        f"Clause:\n{clause_text}\n\nRetrieved GDPR Article {article_cite}:\n{article_text}"
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    raw = generate_fn(messages)
    parsed = parse_verdict(raw, schema)
    if parsed.verdict is not None:
        return {**parsed.verdict, "retry_used": False, "error": None}

    repair_messages = messages + [
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                "That response was not valid JSON matching the required schema. "
                f"Errors: {_describe_schema_errors(raw, schema)}. Respond again with "
                "ONLY a single corrected JSON object matching the schema -- no extra "
                "text, no markdown fences."
            ),
        },
    ]
    raw_retry = generate_fn(repair_messages)
    parsed_retry = parse_verdict(raw_retry, schema)
    if parsed_retry.verdict is not None:
        return {**parsed_retry.verdict, "retry_used": True, "error": None}

    return {
        "article": article_cite,
        "requirement_present": False,
        "compliance_status": "needs_review",
        "evidence_span": "",
        "rationale": "Judge did not produce schema-valid JSON after one repair attempt.",
        "confidence": 0.0,
        "retry_used": True,
        "error": _describe_schema_errors(raw_retry, schema),
    }


class JudgeModel:
    """Loads the base Qwen model + LoRA adapter and generates verdicts.

    Mirrors ``judge/eval_qlora.py``'s ``load_model_for_eval``/
    ``generate_verdicts`` exactly (same config schema, same 4-bit/bf16/
    device-backend resolution via ``judge/train_qlora.py``), so it accepts
    the same ``judge/config/qlora_judge*.yaml`` + adapter checkpoint pairs.

    All ML imports are deferred to ``__init__`` so the rest of this module
    (segmentation, retrieval wiring, aggregation, retry/repair) stays
    importable and unit-testable without ``torch``/``transformers``/``peft``
    installed.
    """

    def __init__(self, cfg: dict[str, Any], adapter_path: str | Path, schema: dict[str, Any]):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from judge.train_qlora import build_bnb_config, resolve_device_backend

        self.schema = schema
        self.gen_kwargs: dict[str, Any] = cfg.get("generation", {})
        self.max_seq_length: int = cfg["model"].get("max_seq_length", 2048)
        self._torch = torch

        model_cfg = cfg["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["base_model"], trust_remote_code=model_cfg.get("trust_remote_code", False)
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        backend, device = resolve_device_backend(cfg.get("device", {}).get("backend", "auto"))
        print(f"[judge.pipeline] using device backend: {backend} ({device})")

        bnb_config = build_bnb_config(cfg["quantization"], backend)
        device_map = model_cfg.get("device_map")
        if device_map is None:
            device_map = "auto" if bnb_config is not None else None
        base_model = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model"],
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        if bnb_config is None and device_map is None and backend != "cpu":
            base_model = base_model.to(device)

        self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
        self.model.eval()

    def _generate(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_length
        ).to(self.model.device)
        with self._torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_kwargs.get("max_new_tokens", 256),
                do_sample=self.gen_kwargs.get("do_sample", False),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def judge(self, clause_text: str, article_cite: str, article_text: str) -> dict[str, Any]:
        return generate_verdict_with_repair(
            self._generate, clause_text, article_cite, article_text, self.schema
        )


# ---------------------------------------------------------------------------
# 5/6. Aggregation
# ---------------------------------------------------------------------------

# Ranking for articles[].best_compliance_status: earlier = more favorable.
# `not_applicable`/`needs_review` are data-quality signals more than
# compliance findings, so they rank ahead of a confirmed `non_compliant`.
# This is a convenience default, not a scoring judgment -- a scoring engine
# that wants a different rule can recompute one from the full
# clauses_addressing_it list this endpoint also returns.
_STATUS_RANK = {
    "compliant": 0,
    "partial": 1,
    "not_applicable": 2,
    "needs_review": 3,
    "non_compliant": 4,
}

_NOT_ADDRESSED_RATIONALE = (
    "No policy clause retrieved this article among any clause's top-k retrieval results."
)


def _article_sort_key(article_number: str) -> tuple[int, str]:
    match = re.match(r"\d+", article_number)
    return (int(match.group()), article_number) if match else (10**9, article_number)


def aggregate_articles(
    clause_verdicts: list[dict[str, Any]], known: set[str]
) -> list[dict[str, Any]]:
    """Groups ``clause_verdicts`` by ``article_number`` into the policy-level
    per-article structure, and adds a ``not_addressed`` entry for every
    article in ``known`` that no clause_verdicts entry names at all.

    ``best_compliance_status`` is the most favorable status (per
    ``_STATUS_RANK``) among the article's ``clauses_addressing_it`` --
    e.g. if any clause fully satisfies the article somewhere in the policy,
    the article is reported ``compliant`` even if other retrieved clauses
    for it were irrelevant or worse.
    """
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for verdict in clause_verdicts:
        number = verdict.get("article_number")
        if number:
            by_article[number].append(verdict)

    all_numbers = known | set(by_article)
    articles: list[dict[str, Any]] = []
    for number in sorted(all_numbers, key=_article_sort_key):
        entries = by_article.get(number, [])
        if not entries:
            articles.append(
                {
                    "article": number,
                    "clauses_addressing_it": [],
                    "best_compliance_status": "not_addressed",
                    "evidence": "",
                    "rationale": _NOT_ADDRESSED_RATIONALE,
                }
            )
            continue

        best = min(
            entries, key=lambda e: _STATUS_RANK.get(e["compliance_status"], len(_STATUS_RANK))
        )
        articles.append(
            {
                "article": number,
                "clauses_addressing_it": [
                    {
                        "clause_id": e["clause_id"],
                        "compliance_status": e["compliance_status"],
                        "requirement_present": e["requirement_present"],
                        "evidence_span": e["evidence_span"],
                        "confidence": e["confidence"],
                    }
                    for e in entries
                ],
                "best_compliance_status": best["compliance_status"],
                "evidence": best["evidence_span"],
                "rationale": best["rationale"],
            }
        )
    return articles


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def judge_clauses(
    clauses: list[Clause],
    retriever_ctx: RetrieverContext,
    judge_model: JudgeModel,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Runs stages 2-4 for every clause: retrieve top-k, judge each
    retrieved chunk, and flatten into the ``clause_verdicts`` list (fanning
    a ``gdpr_concept`` chunk's single verdict out to each article it
    links)."""
    clause_verdicts: list[dict[str, Any]] = []
    for clause in clauses:
        retrieved = retriever_ctx.retrieve(clause.text, k=k)
        for rank, chunk in enumerate(retrieved, start=1):
            label = _chunk_label(chunk.metadata)
            verdict = judge_model.judge(clause.text, label, chunk.text)
            matched_articles = _article_keys_for_chunk(chunk.metadata)
            article_numbers: list[str | None] = (
                list(matched_articles) if matched_articles else [None]
            )
            for article_number in article_numbers:
                clause_verdicts.append(
                    {
                        "clause_id": clause.id,
                        "article": verdict.get("article") or label,
                        "article_number": article_number,
                        "source_type": chunk.metadata.get("source_type"),
                        "chunk_id": chunk.id,
                        "retrieval_rank": rank,
                        "retrieval_score": chunk.score,
                        "requirement_present": verdict.get("requirement_present"),
                        "compliance_status": verdict.get("compliance_status"),
                        "evidence_span": verdict.get("evidence_span"),
                        "rationale": verdict.get("rationale"),
                        "confidence": verdict.get("confidence"),
                        "retry_used": verdict.get("retry_used", False),
                        "error": verdict.get("error"),
                    }
                )
    return clause_verdicts


def build_output(
    *,
    policy_source: str,
    clauses: list[Clause],
    clause_verdicts: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    judge_config_path: str | Path,
    adapter_path: str | Path,
    retrieval_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "judge_model": {"config": str(judge_config_path), "adapter": str(adapter_path)},
            "retrieval": retrieval_meta,
        },
        "policy": {"source": policy_source, "n_clauses": len(clauses)},
        "clauses": [asdict(c) for c in clauses],
        "clause_verdicts": clause_verdicts,
        "articles": articles,
    }


def run_pipeline(
    policy_text: str,
    *,
    policy_source: str,
    retriever_ctx: RetrieverContext,
    judge_model: JudgeModel,
    judge_config_path: str | Path,
    adapter_path: str | Path,
    k: int = 3,
    max_clause_chars: int = DEFAULT_MAX_CLAUSE_CHARS,
) -> dict[str, Any]:
    clauses = segment_clauses(policy_text, max_clause_chars=max_clause_chars)
    clause_verdicts = judge_clauses(clauses, retriever_ctx, judge_model, k=k)
    articles = aggregate_articles(clause_verdicts, known_articles(retriever_ctx.collection))

    retrieval_meta = {
        "k": k,
        "hybrid": retriever_ctx.hybrid,
        "rerank": retriever_ctx.rerank,
        "fetch_k": retriever_ctx.fetch_k,
        "rerank_top_n": retriever_ctx.rerank_top_n if retriever_ctx.rerank else None,
    }
    return build_output(
        policy_source=policy_source,
        clauses=clauses,
        clause_verdicts=clause_verdicts,
        articles=articles,
        judge_config_path=judge_config_path,
        adapter_path=adapter_path,
        retrieval_meta=retrieval_meta,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_config(path: Path | str) -> dict[str, Any]:
    """Tiny standalone YAML loader -- deliberately not imported from
    judge/train_qlora.py, which pulls in torch/transformers/peft/datasets
    at module import time (see that module's top-level `import torch`).
    Importing it here would force every consumer of this module's
    dependency-free pieces (segmentation, aggregation, retry/repair) to
    have the full judge dependency group installed just to parse YAML."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_policy_text(path: Path | str) -> str:
    """Reads the input privacy-policy document as plain text for
    ``segment_clauses``, extracting it from PDF pages via ``pypdf`` (already
    a ``rag`` dependency-group requirement -- see ``rag/parsers/edpb.py``,
    which extracts EDPB guideline PDFs the same way) whenever ``path`` ends
    in ``.pdf``, so a policy can be judged straight from the PDF a company
    actually publishes instead of requiring a manually pre-extracted
    ``.txt`` copy. Any other suffix (``.txt``, no suffix, etc.) is read as
    plain UTF-8 text, unchanged from this function's previous behavior.

    Uses pypdf's ``extraction_mode="layout"``, which reconstructs blank-line
    paragraph gaps from the PDF's actual text layout, rather than the
    default mode, which concatenates every visual line with a single ``\\n``
    and loses paragraph boundaries entirely -- since ``segment_clauses``'s
    primary split is on blank lines, that default mode would collapse a
    whole multi-paragraph PDF into one paragraph, falling back entirely on
    the cruder sentence-merge heuristic. Layout extraction is still a
    heuristic (a PDF has no explicit paragraph markup, just glyph
    positions), not a guarantee: an unusually laid-out PDF (multi-column,
    tables, no vertical gap between paragraphs) can still lose or merge
    paragraph breaks -- the same class of limitation ``segment_clauses``'s
    own docstring already flags for hard-wrapped text.
    """
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        return path.read_text(encoding="utf-8")

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text(extraction_mode="layout") or "" for page in reader.pages)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--policy",
        required=True,
        type=Path,
        help="Path to the privacy policy to judge -- a .pdf (text extracted via pypdf) or "
        "a plain-text/.txt file.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON to.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("judge/config/qlora_judge.yaml"),
        help="judge/config/qlora_judge*.yaml -- picks the base model and generation settings.",
    )
    parser.add_argument(
        "--adapter", required=True, type=Path, help="Path to the trained LoRA adapter directory."
    )
    parser.add_argument("--k", type=int, default=3, help="Top-k retrieved references per clause.")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument(
        "--hybrid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse dense cosine search with a BM25 lexical pass (default: on).",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rerank candidates with a cross-encoder (default: on).",
    )
    parser.add_argument("--fetch-k", type=int, default=None)
    parser.add_argument("--rerank-top-n", type=int, default=DEFAULT_RERANK_TOP_N)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--max-clause-chars", type=int, default=DEFAULT_MAX_CLAUSE_CHARS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    policy_text = load_policy_text(args.policy)

    cfg = load_config(args.config)
    schema = load_schema(cfg["data"].get("schema_path", DEFAULT_SCHEMA_PATH))

    retriever_ctx = build_retriever_context(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        hybrid=args.hybrid,
        rerank=args.rerank,
        fetch_k=args.fetch_k,
        rerank_top_n=args.rerank_top_n,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
    )
    judge_model = JudgeModel(cfg, args.adapter, schema)

    result = run_pipeline(
        policy_text,
        policy_source=str(args.policy),
        retriever_ctx=retriever_ctx,
        judge_model=judge_model,
        judge_config_path=args.config,
        adapter_path=args.adapter,
        k=args.k,
        max_clause_chars=args.max_clause_chars,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[judge.pipeline] {len(result['clauses'])} clauses, "
        f"{len(result['clause_verdicts'])} clause verdicts, "
        f"{len(result['articles'])} articles -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
