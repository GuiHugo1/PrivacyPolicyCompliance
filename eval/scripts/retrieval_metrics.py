"""Pure scoring functions for the GDPR retrieval eval set.

No dependency on the retriever, embedder, or vector store — these operate on
already-ranked lists of GDPR article numbers, so they're cheap to unit test
and reusable if the eval set grows or a different retriever is swapped in.

Gold articles are multi-label: each eval item can have one or more
"primary" gold articles (the textbook grounding — required for full credit)
and zero or more "secondary" gold articles (a legitimately overlapping
alternative article, e.g. a general-accountability article and its
security-specific instance). Every metric that depends on "the gold set" is
therefore reported twice:

- ``*_strict`` uses only the primary gold articles — a strict pass requires
  every primary article to be found.
- ``*_lenient`` uses the full primary+secondary set — a lenient pass only
  requires that *some* acceptable article was found, primary or secondary.

This module intentionally reports retrieval-only metrics. A future
judge-verdict eval (was the compliance judgment itself correct, not just
"did we retrieve the right article") should live in its own scoring module
and report under its own key in any combined report — see
``eval_retrieval.write_json_report`` — rather than being blended into these
numbers. "Found the right article" and "judged the clause correctly" answer
different questions and must never be averaged into one combined score.
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


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


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
    difficulty: str  # "easy" | "hard"
    split: str  # "train" | "held_out"
    gold_primary: list[str]
    gold_secondary: list[str]
    ranked_articles: list[str]
    recall_at_k_strict: dict[int, float]
    recall_at_k_lenient: dict[int, float]
    hit_at_k_strict: dict[int, bool]
    reciprocal_rank_strict: float
    reciprocal_rank_lenient: float

    @property
    def gold_all(self) -> list[str]:
        """Primary + secondary gold articles, deduped, primary first."""
        return _dedupe_preserve_order(self.gold_primary + self.gold_secondary)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "clause": self.clause,
            "difficulty": self.difficulty,
            "split": self.split,
            "gold_primary": self.gold_primary,
            "gold_secondary": self.gold_secondary,
            "ranked_articles": self.ranked_articles,
            "recall_at_k_strict": {str(k): v for k, v in self.recall_at_k_strict.items()},
            "recall_at_k_lenient": {str(k): v for k, v in self.recall_at_k_lenient.items()},
            "hit_at_k_strict": {str(k): v for k, v in self.hit_at_k_strict.items()},
            "reciprocal_rank_strict": self.reciprocal_rank_strict,
            "reciprocal_rank_lenient": self.reciprocal_rank_lenient,
        }


def score_item(
    item_id: str,
    topic: str,
    clause: str,
    gold_primary: list[str],
    gold_secondary: list[str],
    ranked_articles: list[str],
    k_values: list[int],
    difficulty: str = "easy",
    split: str = "train",
) -> ItemResult:
    gold_all = _dedupe_preserve_order(gold_primary + gold_secondary)
    return ItemResult(
        id=item_id,
        topic=topic,
        clause=clause,
        difficulty=difficulty,
        split=split,
        gold_primary=gold_primary,
        gold_secondary=gold_secondary,
        ranked_articles=ranked_articles,
        recall_at_k_strict={k: recall_at_k(gold_primary, ranked_articles, k) for k in k_values},
        # "Lenient recall" is a hit-test over the full primary+secondary gold
        # set (did we find *anything* acceptable), not a fractional recall
        # over that larger set -- the latter would require finding every
        # secondary article too, which is stricter than plain recall for any
        # item with more than one secondary, defeating the point of "lenient".
        recall_at_k_lenient={
            k: (1.0 if hit_at_k(gold_all, ranked_articles, k) else 0.0) for k in k_values
        },
        hit_at_k_strict={k: hit_at_k(gold_primary, ranked_articles, k) for k in k_values},
        reciprocal_rank_strict=reciprocal_rank(gold_primary, ranked_articles),
        reciprocal_rank_lenient=reciprocal_rank(gold_all, ranked_articles),
    )


def aggregate(results: list[ItemResult], k_values: list[int]) -> dict:
    """Macro-average metrics across all items."""
    n = len(results)
    agg: dict = {
        "n_items": n,
        "mrr_strict": (sum(r.reciprocal_rank_strict for r in results) / n) if n else 0.0,
        "mrr_lenient": (sum(r.reciprocal_rank_lenient for r in results) / n) if n else 0.0,
    }
    for k in k_values:
        agg[f"recall_strict@{k}"] = (
            (sum(r.recall_at_k_strict[k] for r in results) / n) if n else 0.0
        )
        agg[f"recall_lenient@{k}"] = (
            (sum(r.recall_at_k_lenient[k] for r in results) / n) if n else 0.0
        )
        agg[f"hit_rate_strict@{k}"] = (
            (sum(1 for r in results if r.hit_at_k_strict[k]) / n) if n else 0.0
        )
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
            f"recall_strict@{k}": sum(i.recall_at_k_strict[k] for i in items) / n,
            f"recall_lenient@{k}": sum(i.recall_at_k_lenient[k] for i in items) / n,
            f"hit_rate_strict@{k}": sum(1 for i in items if i.hit_at_k_strict[k]) / n,
        }
    return breakdown


def aggregate_by_difficulty(results: list[ItemResult], k_values: list[int]) -> dict[str, dict]:
    """Recall/hit-rate/MRR broken out by difficulty tag (easy vs hard).

    Reused on top of ``aggregate`` rather than reimplemented, so a single
    blended number can't hide a retriever that only works on
    statute-mirroring language and falls over on realistic paraphrase.
    """
    groups: dict[str, list[ItemResult]] = {}
    for r in results:
        groups.setdefault(r.difficulty, []).append(r)

    return {difficulty: aggregate(items, k_values) for difficulty, items in sorted(groups.items())}
