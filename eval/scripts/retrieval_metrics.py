"""Pure scoring functions for the GDPR retrieval eval set.

No dependency on the retriever, embedder, or vector store — these operate on
already-ranked lists of GDPR article numbers, so they're cheap to unit test
and reusable if the eval set grows or a different retriever is swapped in.
"""

from __future__ import annotations

from dataclasses import dataclass


def ranked_article_hits(retrieved_article_numbers: list[str]) -> list[str]:
    """Dedupe a rank-ordered list of article numbers, keeping first-seen order.

    A single GDPR article can surface as multiple chunks (e.g. a long article
    split per-paragraph), which would otherwise let one article occupy
    several rank positions and distort recall/MRR.
    """
    seen: set[str] = set()
    ranked: list[str] = []
    for article in retrieved_article_numbers:
        if article not in seen:
            seen.add(article)
            ranked.append(article)
    return ranked


def recall_at_k(gold: list[str], ranked: list[str], k: int) -> float:
    """Fraction of gold articles present in the top-k ranked articles."""
    if not gold:
        return 0.0
    gold_set = set(gold)
    hits = len(gold_set & set(ranked[:k]))
    return hits / len(gold_set)


def hit_at_k(gold: list[str], ranked: list[str], k: int) -> bool:
    """Whether at least one gold article appears in the top-k."""
    return bool(set(gold) & set(ranked[:k]))


def reciprocal_rank(gold: list[str], ranked: list[str]) -> float:
    """1 / rank of the first gold article found, or 0.0 if none is found."""
    gold_set = set(gold)
    for i, article in enumerate(ranked, start=1):
        if article in gold_set:
            return 1.0 / i
    return 0.0


@dataclass
class ItemResult:
    id: str
    topic: str
    clause: str
    gold_articles: list[str]
    ranked_articles: list[str]
    recall_at_k: dict[int, float]
    hit_at_k: dict[int, bool]
    reciprocal_rank: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "clause": self.clause,
            "gold_articles": self.gold_articles,
            "ranked_articles": self.ranked_articles,
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "hit_at_k": {str(k): v for k, v in self.hit_at_k.items()},
            "reciprocal_rank": self.reciprocal_rank,
        }


def score_item(
    item_id: str,
    topic: str,
    clause: str,
    gold_articles: list[str],
    ranked_articles: list[str],
    k_values: list[int],
) -> ItemResult:
    return ItemResult(
        id=item_id,
        topic=topic,
        clause=clause,
        gold_articles=gold_articles,
        ranked_articles=ranked_articles,
        recall_at_k={k: recall_at_k(gold_articles, ranked_articles, k) for k in k_values},
        hit_at_k={k: hit_at_k(gold_articles, ranked_articles, k) for k in k_values},
        reciprocal_rank=reciprocal_rank(gold_articles, ranked_articles),
    )


def aggregate(results: list[ItemResult], k_values: list[int]) -> dict:
    """Macro-average metrics across all items."""
    n = len(results)
    agg: dict = {"n_items": n, "mrr": (sum(r.reciprocal_rank for r in results) / n) if n else 0.0}
    for k in k_values:
        agg[f"recall@{k}"] = (sum(r.recall_at_k[k] for r in results) / n) if n else 0.0
        agg[f"hit_rate@{k}"] = (sum(1 for r in results if r.hit_at_k[k]) / n) if n else 0.0
    return agg


def aggregate_by_topic(results: list[ItemResult], k: int) -> dict[str, dict]:
    """Per-topic breakdown at a single k, to spot systematically weak topics."""
    topics: dict[str, list[ItemResult]] = {}
    for r in results:
        topics.setdefault(r.topic, []).append(r)

    breakdown: dict[str, dict] = {}
    for topic, items in sorted(topics.items()):
        n = len(items)
        breakdown[topic] = {
            "n_items": n,
            f"recall@{k}": sum(i.recall_at_k[k] for i in items) / n,
            f"hit_rate@{k}": sum(1 for i in items if i.hit_at_k[k]) / n,
        }
    return breakdown
