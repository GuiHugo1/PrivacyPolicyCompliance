"""Generates GDPR compliance-judge SFT examples from OPP-115 annotations and
the OPP-115 -> GDPR mapping config (``judge/config/opp115_gdpr_mapping.yaml``).

One SFT example is generated per (policy_id, segment_id, category, primary
GDPR article) -- i.e. per annotated practice x the GDPR article(s)
``judge.mapping`` marks ``primary`` for that OPP-115 category. Multiple
annotators independently labeling the same practice are collapsed via
``judge.opp115.majority_attributes`` before the mapping rule is resolved,
so the dataset isn't inflated with near-duplicate examples of the same
underlying clause.

Grounding text for the "retrieved GDPR article" half of each example comes,
by default, straight from the mapped article's canonical text in the same
``data/raw/gdpr.json`` source ``rag.build_index`` indexes (see
``judge.gdpr_source``) -- i.e. *gold* grounding, not retrieval output, so the
judge is trained against correct context rather than the RAG pipeline's own
retrieval noise. Pass ``--use-retriever`` to instead run each clause through
the actual ``rag.retriever.retrieve`` (against a pre-built index) and use its
top-hit chunk text, matching production inference-time behavior; per
eval/RAG_ANALYSIS.md's conclusion this project runs the retriever with
``hybrid=True, rerank=True`` (best joint recall across single- and
compound-article clauses), so ``--use-retriever`` does the same. That mode
needs `uv sync --group rag` and a built index and will bake in whatever
retrieval mistakes RAG_ANALYSIS.md documents (compound/hard-clause misses),
so it's opt-in, not the default, and is best used to add a *minority* of
imperfect-retrieval examples (teaching the judge to say `not_applicable`/
low-confidence when the retrieved article doesn't actually match the
clause) rather than as the primary grounding source.

This is a WEAK-SUPERVISION bootstrap dataset -- every generated example
carries ``meta.weak_label: true`` and the mapping config's own header
documents why (OPP-115 records disclosure, not GDPR compliance). Review
generated examples, especially any ``coverage: thin``/``weak_alignment``
category (see ``judge/coverage_report.py``), before trusting them as
training data.

Usage::

    python -m judge.build_sft_dataset \\
        --opp115-dir /data/opp115/annotations \\
        --gdpr data/raw/gdpr.json \\
        --out-dir data/processed/judge_sft
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from judge import gdpr_source, opp115
from judge.mapping import DEFAULT_CONFIG_PATH, GdprArticleRef, Opp115GdprMapping, ResolvedRule

JUDGE_SYSTEM_PROMPT = (
    "You are a GDPR compliance judge. Given a clause from a privacy policy and "
    "the text of a retrieved GDPR article, decide whether the clause satisfies "
    "that article's requirement. Respond with ONLY a JSON object matching this "
    'schema: {"article": str, "requirement_present": bool, "compliance_status": '
    '"compliant"|"partial"|"non_compliant"|"not_applicable", "evidence_span": str, '
    '"rationale": str, "confidence": float (0-1)}.'
)

_MAX_CLAUSE_CHARS = 2000


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _pick_evidence_span(
    attributes: dict[str, dict[str, Any]], rule: ResolvedRule, clause: str
) -> str:
    if rule.matched_attribute is not None:
        span = attributes.get(rule.matched_attribute, {}).get("selectedText")
        if span:
            return span
    best = ""
    for attr in attributes.values():
        span = attr.get("selectedText") or ""
        if len(span) > len(best):
            best = span
    return best or _truncate(clause, 300)


def build_example(
    *,
    example_id: str,
    policy_id: str,
    segment_id: str,
    category: str,
    clause: str,
    article_ref: GdprArticleRef,
    article_text: str,
    rule: ResolvedRule,
    attributes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Builds one chat-format SFT record. ``messages`` is what a standard SFT
    trainer (OpenAI fine-tuning format, TRL's SFTTrainer with a chat
    template, etc.) consumes directly; ``target``/``meta`` are extra,
    trainer-ignored keys kept for QA/traceability (matching the eval set's
    own `notes`-for-humans convention in eval/README.md)."""
    evidence = _pick_evidence_span(attributes, rule, clause)
    verdict = {
        "article": article_ref.article,
        "requirement_present": rule.requirement_present,
        "compliance_status": rule.compliance_status,
        "evidence_span": evidence,
        "rationale": rule.note or "No rule-specific rationale recorded.",
        "confidence": round(rule.confidence, 2),
    }
    user_content = (
        f"Clause:\n{clause}\n\n" f"Retrieved GDPR Article {article_ref.article}:\n{article_text}"
    )
    return {
        "id": example_id,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(verdict, ensure_ascii=False)},
        ],
        "target": verdict,
        "meta": {
            "source": "opp115",
            "policy_id": policy_id,
            "segment_id": segment_id,
            "opp115_category": category,
            "opp115_attributes": {k: v.get("value") for k, v in attributes.items()},
            "mapping_role": article_ref.role,
            "mapping_note": article_ref.note,
            "matched_attribute": rule.matched_attribute,
            "matched_value": rule.matched_value,
            "weak_label": True,
        },
    }


def split_for_policy(policy_id: str, val_frac: float, test_frac: float) -> str:
    """Deterministic train/val/test assignment keyed on policy_id, so every
    example from the same policy lands in the same split (avoids leaking a
    near-duplicate clause from the same policy across splits)."""
    bucket = int(hashlib.md5(policy_id.encode("utf-8")).hexdigest(), 16) % 1000
    val_cut = round(val_frac * 1000)
    test_cut = val_cut + round(test_frac * 1000)
    if bucket < val_cut:
        return "val"
    if bucket < test_cut:
        return "test"
    return "train"


@dataclass
class GenerationStats:
    total_groups: int = 0
    excluded: int = 0
    missing_article_text: int = 0
    generated: int = 0
    by_category: dict[str, int] = field(default_factory=dict)


def generate_examples(
    annotations: list[opp115.Annotation],
    mapping: Opp115GdprMapping,
    article_texts: dict[str, str],
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[dict[str, list[dict[str, Any]]], GenerationStats]:
    stats = GenerationStats()
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    segments = opp115.group_by_segment(annotations)
    segment_category_groups = opp115.group_by_segment_category(annotations)

    for (policy_id, segment_id, category), group in sorted(segment_category_groups.items()):
        stats.total_groups += 1
        if category not in mapping.categories:
            stats.excluded += 1
            continue

        attributes = opp115.majority_attributes(group)
        rule = mapping.resolve(category, attributes)
        if rule.exclude:
            stats.excluded += 1
            continue

        clause = opp115.reconstruct_segment_text(segments[(policy_id, segment_id)])
        if not clause:
            stats.excluded += 1
            continue
        clause = _truncate(clause, _MAX_CLAUSE_CHARS)

        split = split_for_policy(policy_id, val_frac, test_frac)
        for article_ref in mapping.target_articles(category, rule):
            base_number = gdpr_source.base_article_number(article_ref.article)
            article_text = article_texts.get(base_number)
            if article_text is None:
                stats.missing_article_text += 1
                continue

            example_id = f"{policy_id}-{segment_id}-{category}-{article_ref.article}".replace(
                " ", "_"
            )
            example = build_example(
                example_id=example_id,
                policy_id=policy_id,
                segment_id=segment_id,
                category=category,
                clause=clause,
                article_ref=article_ref,
                article_text=article_text,
                rule=rule,
                attributes=attributes,
            )
            splits[split].append(example)
            stats.generated += 1
            stats.by_category[category] = stats.by_category.get(category, 0) + 1

    return splits, stats


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_article_texts_via_retriever(
    clauses: list[str], embedding_model: str | None
) -> dict[str, str]:  # pragma: no cover - exercised manually, needs a built index
    from rag.retriever import retrieve

    texts: dict[str, str] = {}
    for clause in clauses:
        hits = retrieve(clause, k=1, hybrid=True, rerank=True)
        if hits:
            meta = hits[0].metadata
            number = meta.get("article_number") or meta.get("concept_articles", "").split(",")[0]
            if number:
                texts[number] = hits[0].text
    return texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opp115-dir", required=True, type=Path, help="OPP-115 annotations/ directory"
    )
    parser.add_argument(
        "--gdpr", required=True, type=Path, help="GDPR source file (data/raw/gdpr.json)"
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/judge_sft"))
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument(
        "--use-retriever",
        action="store_true",
        help="Ground examples in rag.retriever.retrieve() output (hybrid+rerank) instead of gold "
        "article text. Requires `uv sync --group rag` and a pre-built index; see module docstring.",
    )
    args = parser.parse_args(argv)

    mapping = Opp115GdprMapping.load(args.mapping)
    annotations = opp115.load_annotations_dir(args.opp115_dir)
    if not annotations:
        print(f"No annotation CSVs found in {args.opp115_dir}", file=sys.stderr)
        return 1

    article_texts = gdpr_source.load_article_texts(args.gdpr)

    if args.use_retriever:
        segments = opp115.group_by_segment(annotations)
        clauses = [opp115.reconstruct_segment_text(rows) for rows in segments.values()]
        article_texts.update(_load_article_texts_via_retriever(clauses, embedding_model=None))

    splits, stats = generate_examples(
        annotations, mapping, article_texts, val_frac=args.val_frac, test_frac=args.test_frac
    )

    for split_name, records in splits.items():
        write_jsonl(records, args.out_dir / f"{split_name}.jsonl")

    print(f"OPP-115 (policy_id, segment_id, category) groups: {stats.total_groups}")
    print(f"  excluded (mapped to `exclude: true` or no reconstructable clause): {stats.excluded}")
    print(f"  skipped (no article text for mapped article): {stats.missing_article_text}")
    print(f"  generated examples: {stats.generated}")
    for category, count in sorted(stats.by_category.items()):
        print(f"    {category}: {count}")
    for split_name, records in splits.items():
        print(f"  {split_name}: {len(records)} -> {args.out_dir / f'{split_name}.jsonl'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
