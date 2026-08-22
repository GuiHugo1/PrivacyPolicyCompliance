"""Measure retrieval recall against the hand-labeled GDPR eval set.

For each policy clause in the eval set, retrieves the top-k GDPR chunks from
the built index and checks whether the hand-labeled gold article(s) were
retrieved. This is a check on the `rag` retrieval step in isolation — it says
nothing about downstream judge/scoring quality — and is meant to be run
before trusting retrieved context to ground compliance judgments.

Usage:
    python -m rag.build_index --gdpr data/raw/gdpr.json --reset
    python eval/scripts/eval_retrieval.py
    python eval/scripts/eval_retrieval.py --k 1,3,5,10 --output eval/benchmarks/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.scripts.retrieval_metrics import (
    ItemResult,
    aggregate,
    aggregate_by_topic,
    ranked_article_hits,
    score_item,
)
from rag.retriever import retrieve
from rag.store import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR

DEFAULT_EVAL_SET = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "gdpr_retrieval_eval_set.jsonl"
)
DEFAULT_K_VALUES = [3, 5, 10]
REQUIRED_FIELDS = ("id", "clause", "gold_articles")


def load_eval_set(path: str | Path) -> list[dict]:
    """Load and structurally validate the JSONL eval set.

    Each non-blank, non-comment line must be a JSON object with at least
    ``id``, ``clause``, and a non-empty ``gold_articles`` list.
    """
    path = Path(path)
    items: list[dict] = []
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            item = json.loads(line)
            missing = [key for key in REQUIRED_FIELDS if not item.get(key)]
            if missing:
                raise ValueError(f"{path}:{line_num}: missing required field(s) {missing}")
            if not isinstance(item["gold_articles"], list) or not item["gold_articles"]:
                raise ValueError(f"{path}:{line_num}: 'gold_articles' must be a non-empty list")

            if item["id"] in seen_ids:
                raise ValueError(f"{path}:{line_num}: duplicate id '{item['id']}'")
            seen_ids.add(item["id"])

            items.append(item)

    return items


def run_eval(
    eval_set: list[dict],
    k_values: list[int],
    **retrieve_kwargs: Any,
) -> list[ItemResult]:
    """Retrieve for every clause and score against its gold articles."""
    max_k = max(k_values)
    results: list[ItemResult] = []

    for item in eval_set:
        chunks = retrieve(item["clause"], k=max_k, **retrieve_kwargs)
        retrieved_articles: list[str] = []
        for chunk in chunks:
            article = chunk.metadata.get("article_number")
            if article:
                retrieved_articles.append(str(article))
        ranked = ranked_article_hits(retrieved_articles)
        results.append(
            score_item(
                item_id=item["id"],
                topic=item.get("topic", ""),
                clause=item["clause"],
                gold_articles=[str(a) for a in item["gold_articles"]],
                ranked_articles=ranked,
                k_values=k_values,
            )
        )

    return results


def print_report(results: list[ItemResult], k_values: list[int]) -> None:
    agg = aggregate(results, k_values)
    max_k = max(k_values)

    print(f"\nGDPR retrieval eval — {agg['n_items']} clauses")
    print("-" * 60)
    print(f"{'metric':<14}" + "".join(f"k={k:<10}" for k in k_values))
    print(f"{'recall':<14}" + "".join(f"{agg[f'recall@{k}']:<12.3f}" for k in k_values))
    print(f"{'hit_rate':<14}" + "".join(f"{agg[f'hit_rate@{k}']:<12.3f}" for k in k_values))
    print(f"\nMRR: {agg['mrr']:.3f}")

    print(f"\nBy topic (recall@{max_k}):")
    for topic, stats in aggregate_by_topic(results, max_k).items():
        print(f"  {topic:<40} {stats[f'recall@{max_k}']:.3f}  (n={stats['n_items']})")

    misses = [r for r in results if not r.hit_at_k[max_k]]
    if misses:
        print(f"\nComplete misses at k={max_k} (no gold article retrieved) — {len(misses)}:")
        for r in misses:
            preview = r.clause if len(r.clause) <= 90 else r.clause[:87] + "..."
            print(f"  [{r.id}] gold={r.gold_articles} got={r.ranked_articles[:max_k]}")
            print(f"      {preview}")
    else:
        print(f"\nNo complete misses at k={max_k}.")


def write_json_report(path: str | Path, results: list[ItemResult], k_values: list[int]) -> None:
    report = {
        "aggregate": aggregate(results, k_values),
        "by_topic": aggregate_by_topic(results, max(k_values)),
        "items": [r.to_dict() for r in results],
    }
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument(
        "--k",
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="Comma-separated list of k values to evaluate, e.g. 3,5,10",
    )
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--output", type=Path, default=None, help="Write full JSON report here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    k_values = sorted({int(k) for k in args.k.split(",")})

    eval_set = load_eval_set(args.eval_set)
    if not eval_set:
        print("Eval set is empty.", file=sys.stderr)
        return 1

    results = run_eval(
        eval_set,
        k_values,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )

    print_report(results, k_values)

    if args.output:
        write_json_report(args.output, results, k_values)
        print(f"\nFull report written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
