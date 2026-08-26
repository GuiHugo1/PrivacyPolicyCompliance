import pytest

from rag.fusion import reciprocal_rank_fusion


def test_rrf_scores_higher_for_items_ranked_first_in_more_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "c"]])
    assert fused["a"] > fused["c"]
    assert fused["b"] > fused["c"]


def test_rrf_matches_formula_directly():
    fused = reciprocal_rank_fusion([["a", "b"]], rrf_k=60)
    assert fused["a"] == pytest.approx(1.0 / 61)
    assert fused["b"] == pytest.approx(1.0 / 62)


def test_rrf_combines_scores_across_lists():
    fused = reciprocal_rank_fusion([["a"], ["a"]], rrf_k=60)
    assert fused["a"] == pytest.approx(2 / 61)


def test_rrf_weights_scale_each_lists_contribution():
    fused_equal = reciprocal_rank_fusion([["a"], ["b"]], rrf_k=60)
    fused_weighted = reciprocal_rank_fusion([["a"], ["b"]], weights=[2.0, 1.0], rrf_k=60)
    assert fused_weighted["a"] == pytest.approx(2 * fused_equal["a"])
    assert fused_weighted["b"] == pytest.approx(fused_equal["b"])


def test_rrf_mismatched_weights_length_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_rrf_id_only_in_one_list_still_scored():
    fused = reciprocal_rank_fusion([["a", "b"], ["c"]])
    assert set(fused) == {"a", "b", "c"}


def test_rrf_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([[], []]) == {}
