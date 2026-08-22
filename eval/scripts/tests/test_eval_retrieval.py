import json

import pytest

from eval.scripts.eval_retrieval import DEFAULT_EVAL_SET, load_eval_set, run_eval
from rag.parsers.gdpr import parse_gdpr_file
from rag.store import add_chunks

VALID_GDPR_ARTICLE_NUMBERS = {str(n) for n in range(1, 100)}


def _write_jsonl(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(line) if isinstance(line, dict) else line for line in lines)
    )
    return path


def _build_test_index(in_memory_collection, fake_embedder, fake_gdpr_path):
    chunks = parse_gdpr_file(fake_gdpr_path)
    embeddings = fake_embedder.embed_documents([c.text for c in chunks])
    add_chunks(in_memory_collection, chunks, embeddings)
    return chunks


class TestLoadEvalSet:
    def test_loads_valid_jsonl(self, tmp_path):
        path = _write_jsonl(
            tmp_path,
            "eval.jsonl",
            [
                {"id": "e1", "topic": "t", "clause": "some clause", "gold_articles": ["6"]},
                {"id": "e2", "topic": "t", "clause": "another clause", "gold_articles": ["7", "6"]},
            ],
        )
        items = load_eval_set(path)
        assert len(items) == 2
        assert items[0]["id"] == "e1"

    def test_skips_blank_lines_and_comments(self, tmp_path):
        path = tmp_path / "eval.jsonl"
        path.write_text(
            "# a comment\n\n"
            + json.dumps({"id": "e1", "clause": "c", "gold_articles": ["6"]})
            + "\n\n"
        )
        items = load_eval_set(path)
        assert len(items) == 1

    def test_missing_required_field_raises(self, tmp_path):
        path = _write_jsonl(tmp_path, "eval.jsonl", [{"id": "e1", "gold_articles": ["6"]}])
        with pytest.raises(ValueError, match="missing required field"):
            load_eval_set(path)

    def test_empty_gold_articles_raises(self, tmp_path):
        # An empty list is falsy, so it's rejected by the same required-field
        # check as an absent key — still a hard failure either way.
        path = _write_jsonl(
            tmp_path, "eval.jsonl", [{"id": "e1", "clause": "c", "gold_articles": []}]
        )
        with pytest.raises(ValueError, match="missing required field"):
            load_eval_set(path)

    def test_non_list_gold_articles_raises(self, tmp_path):
        path = _write_jsonl(
            tmp_path, "eval.jsonl", [{"id": "e1", "clause": "c", "gold_articles": "6"}]
        )
        with pytest.raises(ValueError, match="non-empty list"):
            load_eval_set(path)

    def test_duplicate_id_raises(self, tmp_path):
        path = _write_jsonl(
            tmp_path,
            "eval.jsonl",
            [
                {"id": "e1", "clause": "c1", "gold_articles": ["6"]},
                {"id": "e1", "clause": "c2", "gold_articles": ["7"]},
            ],
        )
        with pytest.raises(ValueError, match="duplicate id"):
            load_eval_set(path)


class TestRunEval:
    def test_run_eval_end_to_end_against_fake_index(
        self, in_memory_collection, fake_embedder, fake_gdpr_path, mini_eval_set_path
    ):
        _build_test_index(in_memory_collection, fake_embedder, fake_gdpr_path)
        eval_set = load_eval_set(mini_eval_set_path)

        results = run_eval(
            eval_set,
            k_values=[3, 5],
            collection=in_memory_collection,
            embedder=fake_embedder,
        )

        assert len(results) == len(eval_set)
        for r in results:
            assert set(r.recall_at_k) == {3, 5}
            assert set(r.hit_at_k) == {3, 5}
            for k in (3, 5):
                assert 0.0 <= r.recall_at_k[k] <= 1.0
            assert 0.0 <= r.reciprocal_rank <= 1.0
            # gold articles for the mini set are single-article, drawn straight
            # from the fixture, so ranked_articles should only ever contain
            # article numbers that actually exist in the fake corpus.
            assert set(r.ranked_articles) <= {"1", "4", "6", "7", "33"}

    def test_run_eval_on_empty_collection_scores_as_misses(
        self, in_memory_collection, fake_embedder, mini_eval_set_path
    ):
        eval_set = load_eval_set(mini_eval_set_path)

        results = run_eval(
            eval_set,
            k_values=[5],
            collection=in_memory_collection,
            embedder=fake_embedder,
        )

        assert all(r.ranked_articles == [] for r in results)
        assert all(r.hit_at_k[5] is False for r in results)
        assert all(r.recall_at_k[5] == 0.0 for r in results)


@pytest.fixture(scope="module")
def gold_eval_set_items():
    return load_eval_set(DEFAULT_EVAL_SET)


class TestGoldEvalSetIsWellFormed:
    """Regression checks on the hand-labeled eval set itself, not the code."""

    @pytest.fixture
    def items(self, gold_eval_set_items):
        return gold_eval_set_items

    def test_has_a_reasonable_sample_size(self, items):
        assert len(items) >= 30

    def test_ids_are_unique(self, items):
        ids = [item["id"] for item in items]
        assert len(ids) == len(set(ids))

    def test_every_item_has_a_topic(self, items):
        for item in items:
            assert item.get("topic"), f"{item['id']} is missing a topic"

    def test_gold_articles_look_like_valid_gdpr_article_numbers(self, items):
        for item in items:
            for article in item["gold_articles"]:
                assert isinstance(article, str), f"{item['id']}: article numbers must be strings"
                assert (
                    article in VALID_GDPR_ARTICLE_NUMBERS
                ), f"{item['id']}: '{article}' is not a plausible GDPR article number (1-99)"

    def test_topics_cover_a_spread_of_gdpr_chapters(self, items):
        # Sanity check that the sample isn't narrowly clustered on one topic —
        # e.g. all consent clauses — which would make recall look better or
        # worse than it will on a real, varied privacy policy.
        all_gold_articles = {a for item in items for a in item["gold_articles"]}
        assert len(all_gold_articles) >= 20
