"""Diagnose the 0.0/0.5-recall GDPR retrieval eval cases.

The eval harness (`eval_retrieval.py`) only ever looks at the top-k (k<=10)
window and only scores article-level hits. This script bypasses both: it
queries the live index directly at k=50 (no similarity threshold, since
`rag.retriever.retrieve` never applies one — a Chroma `n_results=k` query
just returns fewer than k hits if the collection is small or a `where`
filter is set) and reports the full ranked list, so we can tell apart:

  - the gold chunk sitting just outside the eval's k window (true ranking
    problem)
  - the gold chunk missing from the index, or buried inside an oversized/
    malformed chunk that doesn't semantically match the query (corpus gap)
  - a plausible non-gold article/recital crowding out the gold one because
    the two genuinely overlap in subject matter (label ambiguity)

It also separately checks whether GDPR recitals are indexed and retrievable
at all (see `--check-recitals`), since several of the failing topics
(cookie consent, legitimate-interest marketing, consent bundled into a
broader agreement) are primarily grounded in recital text, not article
text.

Usage:
    python -m rag.build_index --gdpr data/raw/gdpr.json --reset
    python eval/scripts/diagnose_failures.py --gdpr-source data/raw/gdpr.json
    python eval/scripts/diagnose_failures.py --gdpr-source data/raw/gdpr.json \\
        --case-ids eval-003,eval-042 --output eval/benchmarks/diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rag.parsers.gdpr import parse_gdpr_file
from rag.retriever import RetrievedChunk, retrieve
from rag.store import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR

DEFAULT_EVAL_SET = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "gdpr_retrieval_eval_set.jsonl"
)
DEFAULT_K = 50
TOP_N_FOR_TEXT_COMPARE = 3

# The 7 failing/partial topics called out from the last eval run, resolved to
# eval-set ids. Override with --case-ids if the eval set or results change.
DEFAULT_CASE_IDS = [
    "eval-003",  # consent_clarity                          (0.0)
    "eval-011",  # confidentiality_integrity_principle       (0.5, partial)
    "eval-012",  # accountability_principle                  (0.0)
    "eval-025",  # controller_accountability_measures        (0.0)
    "eval-042",  # legitimate_interest_marketing_objection   (0.0)
    "eval-043",  # cookies_tracking_consent                  (0.0)
    "eval-044",  # retention_period_disclosure                (0.5, partial)
]

# Recitals worth checking explicitly: cookie consent / tracking, legitimate
# interest for marketing, and consent bundled into a broader agreement often
# live primarily in recital text rather than article text.
RECITALS_OF_INTEREST = ["32", "42", "47"]


def load_eval_set(path: str | Path) -> dict[str, dict]:
    items: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            item = json.loads(line)
            items[item["id"]] = item
    return items


def _label(meta: dict[str, Any]) -> str:
    """Human-readable id for a chunk: 'Art 6(1)' / 'Art 6' / 'Recital 47'."""
    if meta.get("source_type") == "gdpr_recital":
        return f"Recital {meta.get('recital_number', '?')}"
    article = meta.get("article_number", "?")
    para = meta.get("paragraph_number")
    return f"Art {article}({para})" if para else f"Art {article}"


def _gold_lookup(gdpr_source: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    """Build article_number -> full text and recital_number -> text lookups
    straight from the source file used to build the index (not from
    whatever retrieval happens to surface), so we can show gold text even
    when retrieval finds nothing for it at all."""
    chunks = parse_gdpr_file(gdpr_source)
    article_text: dict[str, list[str]] = {}
    recital_text: dict[str, str] = {}
    for chunk in chunks:
        meta = chunk.metadata
        if meta.get("source_type") == "gdpr_article":
            article_text.setdefault(str(meta["article_number"]), []).append(chunk.text)
        elif meta.get("source_type") == "gdpr_recital":
            recital_text[str(meta["recital_number"])] = chunk.text
    joined_articles = {num: "\n\n".join(parts) for num, parts in article_text.items()}
    return joined_articles, recital_text


def diagnose_case(
    item: dict,
    k: int,
    gold_article_text: dict[str, str],
    **retrieve_kwargs: Any,
) -> dict[str, Any]:
    clause = item["clause"]
    gold_articles = [str(a) for a in item["gold_articles"]]

    ranked: list[RetrievedChunk] = retrieve(clause, k=k, **retrieve_kwargs)

    ranked_view = [
        {
            "rank": rank,
            "id": chunk.id,
            "label": _label(chunk.metadata),
            "source_type": chunk.metadata.get("source_type"),
            "score": round(chunk.score, 4),
        }
        for rank, chunk in enumerate(ranked, start=1)
    ]

    gold_hits: dict[str, dict[str, Any]] = {}
    for gold in gold_articles:
        rank_found = None
        for rank, chunk in enumerate(ranked, start=1):
            if (
                chunk.metadata.get("source_type") == "gdpr_article"
                and str(chunk.metadata.get("article_number")) == gold
            ):
                rank_found = rank
                break
        gold_hits[gold] = {
            "found_in_top_k": rank_found is not None,
            "rank": rank_found,
            "gold_text_indexed": gold in gold_article_text,
        }

    source_type_counts: dict[str, int] = {}
    for chunk in ranked:
        st = chunk.metadata.get("source_type", "unknown")
        source_type_counts[st] = source_type_counts.get(st, 0) + 1

    return {
        "id": item["id"],
        "topic": item.get("topic", ""),
        "clause": clause,
        "gold_articles": gold_articles,
        "k": k,
        "n_returned": len(ranked),
        "source_type_counts_in_top_k": source_type_counts,
        "ranked": ranked_view,
        "gold_hits": gold_hits,
        "top_n_chunks": [
            {
                "rank": rank,
                "label": _label(chunk.metadata),
                "score": round(chunk.score, 4),
                "text": chunk.text,
            }
            for rank, chunk in enumerate(ranked[:TOP_N_FOR_TEXT_COMPARE], start=1)
        ],
    }


def check_recitals_indexed(
    recital_numbers: list[str],
    recital_source_text: dict[str, str],
    **retrieve_kwargs: Any,
) -> dict[str, Any]:
    """Confirm whether recitals are indexed at all, and whether specific
    recitals of interest are present in the source and independently
    reachable by a query naming them directly."""
    report: dict[str, Any] = {"recitals_in_source_file": len(recital_source_text)}

    per_recital: dict[str, Any] = {}
    for number in recital_numbers:
        in_source = number in recital_source_text
        # A query that names the recital directly should surface it near the
        # top if it's indexed at all — this isolates "is it in the index"
        # from "does a realistic policy-clause query retrieve it".
        probe = retrieve(f"Recital {number}", k=10, **retrieve_kwargs)
        found_rank = None
        for rank, chunk in enumerate(probe, start=1):
            if (
                chunk.metadata.get("source_type") == "gdpr_recital"
                and str(chunk.metadata.get("recital_number")) == number
            ):
                found_rank = rank
                break
        per_recital[number] = {
            "in_source_file": in_source,
            "reachable_by_direct_probe": found_rank is not None,
            "probe_rank": found_rank,
        }
    report["recitals_of_interest"] = per_recital
    return report


def print_case_report(case: dict[str, Any], gold_article_text: dict[str, str]) -> None:
    print("\n" + "=" * 100)
    print(f"[{case['id']}] topic={case['topic']!r}  gold_articles={case['gold_articles']}")
    print(f"clause: {case['clause']}")
    print(f"top-{case['k']} returned {case['n_returned']} chunks, by source_type: "
          f"{case['source_type_counts_in_top_k']}")

    print("\n-- gold article status in top-{}: --".format(case["k"]))
    for gold, info in case["gold_hits"].items():
        if info["found_in_top_k"]:
            status = f"FOUND at rank {info['rank']}"
        elif not info["gold_text_indexed"]:
            status = "NOT FOUND — and not present in the source file at all (corpus gap)"
        else:
            status = f"NOT FOUND in top-{case['k']} (but present in source file)"
        print(f"  Art {gold}: {status}")

    print(f"\n-- full ranked list (top-{case['k']}): --")
    for row in case["ranked"]:
        print(f"  [{row['rank']:>2}] score={row['score']:.4f}  {row['label']:<14} id={row['id']}")

    print(f"\n-- top-{TOP_N_FOR_TEXT_COMPARE} retrieved chunks vs. gold text, side by side: --")
    for gold in case["gold_articles"]:
        gold_text = gold_article_text.get(gold, "(not present in source file)")
        print(f"\n  GOLD Art {gold}:")
        print(_indent(gold_text, 4))
    for chunk in case["top_n_chunks"]:
        print(f"\n  RETRIEVED [{chunk['rank']}] {chunk['label']} (score={chunk['score']:.4f}):")
        print(_indent(chunk["text"], 4))


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def print_recital_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("RECITAL INDEXING CHECK")
    print(f"Recitals present in source file: {report['recitals_in_source_file']}")
    for number, info in report["recitals_of_interest"].items():
        print(f"\n  Recital {number}:")
        print(f"    in source file:          {info['in_source_file']}")
        print(f"    reachable by direct probe: {info['reachable_by_direct_probe']}"
              + (f" (rank {info['probe_rank']})" if info["probe_rank"] else ""))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument(
        "--gdpr-source",
        type=Path,
        required=True,
        help="Path to the same GDPR JSON/XML source used to build the index "
        "(needed to look up gold article/recital text even when retrieval finds nothing).",
    )
    parser.add_argument(
        "--case-ids",
        default=",".join(DEFAULT_CASE_IDS),
        help="Comma-separated eval-set ids to diagnose (default: the 7 known failing/partial cases).",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--check-recitals",
        action="store_true",
        default=True,
        help="Also check whether recitals are indexed at all (default: on).",
    )
    parser.add_argument("--no-check-recitals", dest="check_recitals", action="store_false")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--output", type=Path, default=None, help="Write full JSON dump here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    eval_set = load_eval_set(args.eval_set)
    case_ids = [c.strip() for c in args.case_ids.split(",") if c.strip()]
    missing = [c for c in case_ids if c not in eval_set]
    if missing:
        print(f"Unknown eval-set id(s): {missing}", file=sys.stderr)
        return 1

    gold_article_text, recital_source_text = _gold_lookup(args.gdpr_source)

    retrieve_kwargs = {"persist_dir": args.persist_dir, "collection_name": args.collection}

    cases = []
    for case_id in case_ids:
        item = eval_set[case_id]
        case = diagnose_case(item, args.k, gold_article_text, **retrieve_kwargs)
        cases.append(case)
        print_case_report(case, gold_article_text)

    recital_report = None
    if args.check_recitals:
        recital_report = check_recitals_indexed(
            RECITALS_OF_INTEREST, recital_source_text, **retrieve_kwargs
        )
        print_recital_report(recital_report)

    if args.output:
        dump = {"cases": cases, "recital_check": recital_report}
        args.output.write_text(json.dumps(dump, indent=2), encoding="utf-8")
        print(f"\nFull JSON dump written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
