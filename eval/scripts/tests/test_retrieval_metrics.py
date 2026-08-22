from eval.scripts.retrieval_metrics import (
    aggregate,
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


def test_aggregate_macro_averages():
    r1 = score_item("a", "topic1", "clause a", ["6"], ["6", "7"], k_values=[1, 5])
    r2 = score_item("b", "topic1", "clause b", ["17"], ["6", "7"], k_values=[1, 5])

    agg = aggregate([r1, r2], k_values=[1, 5])

    assert agg["n_items"] == 2
    assert agg["recall@1"] == 0.5  # r1 hits at k=1 (recall 1.0), r2 misses (recall 0.0)
    assert agg["recall@5"] == 0.5  # r1 hits within top-5, r2 never hits
    assert agg["hit_rate@1"] == 0.5
    assert agg["hit_rate@5"] == 0.5
    assert agg["mrr"] == 0.5  # r1 rank 1 -> 1.0, r2 no hit -> 0.0, mean = 0.5


def test_aggregate_empty_results():
    agg = aggregate([], k_values=[5])
    assert agg == {"n_items": 0, "mrr": 0.0, "recall@5": 0.0, "hit_rate@5": 0.0}


def test_aggregate_by_topic_groups_and_averages():
    r1 = score_item("a", "consent", "clause a", ["7"], ["7"], k_values=[3])
    r2 = score_item("b", "consent", "clause b", ["7"], ["33"], k_values=[3])
    r3 = score_item("c", "erasure", "clause c", ["17"], ["17"], k_values=[3])

    breakdown = aggregate_by_topic([r1, r2, r3], k=3)

    assert breakdown["consent"]["n_items"] == 2
    assert breakdown["consent"]["recall@3"] == 0.5
    assert breakdown["erasure"]["n_items"] == 1
    assert breakdown["erasure"]["recall@3"] == 1.0


def test_score_item_multi_label_gold():
    result = score_item(
        "eval-042", "marketing", "clause text", ["6", "21"], ["21", "12", "6"], k_values=[1, 2, 3]
    )
    assert result.recall_at_k[1] == 0.5  # only "21" in top-1
    assert result.recall_at_k[2] == 0.5  # "6" still not within top-2
    assert result.recall_at_k[3] == 1.0  # both found by top-3
    assert result.hit_at_k[1] is True
    assert result.reciprocal_rank == 1.0  # first gold article ("21") is rank 1
