from eval.scripts.retrieval_metrics import (
    aggregate,
    aggregate_by_difficulty,
    aggregate_by_topic,
    hit_at_k,
    ranked_article_hits,
    recall_at_k,
    reciprocal_rank,
    score_item,
)


def test_ranked_article_hits_dedupes_keeping_first_seen_order():
    assert ranked_article_hits(["6", "33", "6", "7", "33"]) == ["6", "33", "7"]


def test_recall_at_k_partial_and_full():
    gold = ["6", "7"]
    ranked = ["33", "6", "12", "7"]
    assert recall_at_k(gold, ranked, k=2) == 0.5
    assert recall_at_k(gold, ranked, k=4) == 1.0
    assert recall_at_k(gold, ranked, k=1) == 0.0


def test_recall_at_k_empty_gold_is_zero():
    assert recall_at_k([], ["6"], k=5) == 0.0


def test_hit_at_k():
    gold = ["17"]
    assert hit_at_k(gold, ["17", "6"], k=1) is True
    assert hit_at_k(gold, ["6", "17"], k=1) is False
    assert hit_at_k(gold, ["6", "17"], k=2) is True
    assert hit_at_k(gold, ["6", "33"], k=5) is False


def test_reciprocal_rank():
    assert reciprocal_rank(["7"], ["6", "7", "33"]) == 0.5
    assert reciprocal_rank(["6"], ["6", "7"]) == 1.0
    assert reciprocal_rank(["99"], ["6", "7"]) == 0.0


def test_score_item_single_primary_no_secondary():
    result = score_item(
        "a", "topic1", "clause a", ["6"], [], ranked_articles=["6", "7"], k_values=[1, 5]
    )
    assert result.recall_at_k_strict[1] == 1.0
    assert result.recall_at_k_lenient[1] == 1.0
    assert result.hit_at_k_strict[1] is True
    assert result.reciprocal_rank_strict == 1.0
    assert result.reciprocal_rank_lenient == 1.0
    assert result.gold_all == ["6"]


def test_score_item_strict_requires_all_primary_lenient_requires_any_gold():
    # Two co-primary articles (AND-required): strict needs both, lenient
    # only needs one gold article (primary or secondary) found.
    result = score_item(
        "eval-042",
        "marketing",
        "clause text",
        gold_primary=["6", "21"],
        gold_secondary=[],
        ranked_articles=["21", "12", "6"],
        k_values=[1, 2, 3],
    )
    assert result.recall_at_k_strict[1] == 0.5  # only "21" in top-1
    assert result.recall_at_k_strict[2] == 0.5  # "6" still not within top-2
    assert result.recall_at_k_strict[3] == 1.0  # both found by top-3
    assert result.hit_at_k_strict[1] is True  # at least one primary found
    assert result.recall_at_k_lenient[1] == 1.0  # lenient is a hit-test, not a fraction
    assert result.reciprocal_rank_strict == 1.0  # first primary article ("21") is rank 1


def test_score_item_secondary_only_satisfies_lenient_not_strict():
    # primary=[5] missed entirely; secondary=[24] found at rank 1.
    result = score_item(
        "eval-012",
        "accountability_principle",
        "clause text",
        gold_primary=["5"],
        gold_secondary=["24"],
        ranked_articles=["24", "12"],
        k_values=[1, 5],
    )
    assert result.recall_at_k_strict[5] == 0.0  # "5" never found
    assert result.hit_at_k_strict[5] is False
    assert result.recall_at_k_lenient[1] == 1.0  # "24" (secondary) found at rank 1
    assert result.reciprocal_rank_strict == 0.0
    assert result.reciprocal_rank_lenient == 1.0
    assert result.gold_all == ["5", "24"]


def test_aggregate_macro_averages():
    r1 = score_item("a", "topic1", "clause a", ["6"], [], ["6", "7"], k_values=[1, 5])
    r2 = score_item("b", "topic1", "clause b", ["17"], [], ["6", "7"], k_values=[1, 5])

    agg = aggregate([r1, r2], k_values=[1, 5])

    assert agg["n_items"] == 2
    assert agg["recall_strict@1"] == 0.5  # r1 hits at k=1 (recall 1.0), r2 misses (recall 0.0)
    assert agg["recall_strict@5"] == 0.5  # r1 hits within top-5, r2 never hits
    assert agg["hit_rate_strict@1"] == 0.5
    assert agg["hit_rate_strict@5"] == 0.5
    assert agg["mrr_strict"] == 0.5  # r1 rank 1 -> 1.0, r2 no hit -> 0.0, mean = 0.5
    assert agg["recall_lenient@1"] == 0.5  # no secondaries here, so lenient == strict
    assert agg["mrr_lenient"] == 0.5


def test_aggregate_empty_results():
    agg = aggregate([], k_values=[5])
    assert agg == {
        "n_items": 0,
        "mrr_strict": 0.0,
        "mrr_lenient": 0.0,
        "recall_strict@5": 0.0,
        "recall_lenient@5": 0.0,
        "hit_rate_strict@5": 0.0,
    }


def test_aggregate_by_topic_groups_and_averages():
    r1 = score_item("a", "consent", "clause a", ["7"], [], ["7"], k_values=[3])
    r2 = score_item("b", "consent", "clause b", ["7"], [], ["33"], k_values=[3])
    r3 = score_item("c", "erasure", "clause c", ["17"], [], ["17"], k_values=[3])

    breakdown = aggregate_by_topic([r1, r2, r3], k=3)

    assert breakdown["consent"]["n_items"] == 2
    assert breakdown["consent"]["recall_strict@3"] == 0.5
    assert breakdown["erasure"]["n_items"] == 1
    assert breakdown["erasure"]["recall_strict@3"] == 1.0


def test_aggregate_by_difficulty_groups_and_averages():
    easy_hit = score_item(
        "a", "t", "c", ["7"], [], ["7"], k_values=[3], difficulty="easy", split="train"
    )
    hard_miss = score_item(
        "b", "t", "c", ["7"], [], ["33"], k_values=[3], difficulty="hard", split="train"
    )
    hard_hit = score_item(
        "c", "t", "c", ["7"], [], ["7"], k_values=[3], difficulty="hard", split="train"
    )

    breakdown = aggregate_by_difficulty([easy_hit, hard_miss, hard_hit], k_values=[3])

    assert breakdown["easy"]["n_items"] == 1
    assert breakdown["easy"]["recall_strict@3"] == 1.0
    assert breakdown["hard"]["n_items"] == 2
    assert breakdown["hard"]["recall_strict@3"] == 0.5


def test_score_item_to_dict_round_trips_key_fields():
    result = score_item(
        "eval-011",
        "confidentiality_integrity_principle",
        "clause",
        gold_primary=["5"],
        gold_secondary=["32"],
        ranked_articles=["32"],
        k_values=[3],
        difficulty="easy",
        split="held_out",
    )
    d = result.to_dict()
    assert d["gold_primary"] == ["5"]
    assert d["gold_secondary"] == ["32"]
    assert d["difficulty"] == "easy"
    assert d["split"] == "held_out"
    assert d["recall_at_k_strict"] == {"3": 0.0}
    assert d["recall_at_k_lenient"] == {"3": 1.0}
