import json

import pytest

from eval.scripts.eval_retrieval import DEFAULT_EVAL_SET, load_eval_set, run_eval
from rag.parsers.gdpr import parse_gdpr_file
from rag.store import add_chunks

VALID_GDPR_ARTICLE_NUMBERS = {str(n) for n in range(1, 100)}


def _gold(article: str, role: str = "primary") -> dict:
    return {"article": article, "role": role}


def _write_jsonl(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(line) if isinstance(line, dict) else line for line in lines)
    )
    return path


def _base_item(**overrides) -> dict:
    item = {
        "id": "e1",
        "topic": "t",
        "clause": "some clause",
        "gold_articles": [_gold("6")],
        "difficulty": "easy",
        "split": "train",
    }
    item.update(overrides)
    return item


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
                _base_item(id="e1", clause="some clause"),
                _base_item(
                    id="e2", clause="another clause", gold_articles=[_gold("7"), _gold("6")]
                ),
            ],
        )
        items = load_eval_set(path)
        assert len(items) == 2
        assert items[0]["id"] == "e1"

    def test_skips_blank_lines_and_comments(self, tmp_path):
        path = tmp_path / "eval.jsonl"
        path.write_text("# a comment\n\n" + json.dumps(_base_item()) + "\n\n")
        items = load_eval_set(path)
        assert len(items) == 1

    def test_missing_required_field_raises(self, tmp_path):
        item = _base_item()
        del item["clause"]
        path = _write_jsonl(tmp_path, "eval.jsonl", [item])
        with pytest.raises(ValueError, match="missing required field"):
            load_eval_set(path)

    def test_missing_difficulty_raises(self, tmp_path):
        item = _base_item()
        del item["difficulty"]
        path = _write_jsonl(tmp_path, "eval.jsonl", [item])
        with pytest.raises(ValueError, match="missing required field"):
            load_eval_set(path)

    def test_missing_split_raises(self, tmp_path):
        item = _base_item()
        del item["split"]
        path = _write_jsonl(tmp_path, "eval.jsonl", [item])
        with pytest.raises(ValueError, match="missing required field"):
            load_eval_set(path)

    def test_invalid_difficulty_raises(self, tmp_path):
        path = _write_jsonl(tmp_path, "eval.jsonl", [_base_item(difficulty="medium")])
        with pytest.raises(ValueError, match="'difficulty' must be one of"):
            load_eval_set(path)

    def test_invalid_split_raises(self, tmp_path):
        path = _write_jsonl(tmp_path, "eval.jsonl", [_base_item(split="validation")])
        with pytest.raises(ValueError, match="'split' must be one of"):
            load_eval_set(path)

    def test_empty_gold_articles_raises(self, tmp_path):
        # An empty list is falsy, so it's rejected by the same required-field
        # check as an absent key — still a hard failure either way.
        path = _write_jsonl(tmp_path, "eval.jsonl", [_base_item(gold_articles=[])])
        with pytest.raises(ValueError, match="missing required field"):
            load_eval_set(path)

    def test_non_list_gold_articles_raises(self, tmp_path):
        path = _write_jsonl(tmp_path, "eval.jsonl", [_base_item(gold_articles="6")])
        with pytest.raises(ValueError, match="non-empty list"):
            load_eval_set(path)

    def test_gold_article_entry_missing_role_raises(self, tmp_path):
        path = _write_jsonl(tmp_path, "eval.jsonl", [_base_item(gold_articles=[{"article": "6"}])])
        with pytest.raises(ValueError, match="'article' and 'role' keys"):
            load_eval_set(path)

    def test_gold_article_invalid_role_raises(self, tmp_path):
        path = _write_jsonl(
            tmp_path, "eval.jsonl", [_base_item(gold_articles=[_gold("6", "tertiary")])]
        )
        with pytest.raises(ValueError, match="role must be one of"):
            load_eval_set(path)

    def test_gold_articles_with_no_primary_raises(self, tmp_path):
        path = _write_jsonl(
            tmp_path, "eval.jsonl", [_base_item(gold_articles=[_gold("6", "secondary")])]
        )
        with pytest.raises(ValueError, match="at least one primary article"):
            load_eval_set(path)

    def test_gold_articles_with_secondary_role_loads(self, tmp_path):
        item = _base_item(gold_articles=[_gold("5"), _gold("24", "secondary")])
        path = _write_jsonl(tmp_path, "eval.jsonl", [item])
        items = load_eval_set(path)
        assert items[0]["gold_articles"] == [_gold("5"), _gold("24", "secondary")]

    def test_duplicate_id_raises(self, tmp_path):
        path = _write_jsonl(
            tmp_path,
            "eval.jsonl",
            [_base_item(id="e1", clause="c1"), _base_item(id="e1", clause="c2")],
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
            assert set(r.recall_at_k_strict) == {3, 5}
            assert set(r.recall_at_k_lenient) == {3, 5}
            assert set(r.hit_at_k_strict) == {3, 5}
            for k in (3, 5):
                assert 0.0 <= r.recall_at_k_strict[k] <= 1.0
                assert 0.0 <= r.recall_at_k_lenient[k] <= 1.0
            assert 0.0 <= r.reciprocal_rank_strict <= 1.0
            assert 0.0 <= r.reciprocal_rank_lenient <= 1.0
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
        assert all(r.hit_at_k_strict[5] is False for r in results)
        assert all(r.recall_at_k_strict[5] == 0.0 for r in results)

    def test_run_eval_preserves_difficulty_and_split_from_eval_set(
        self, in_memory_collection, fake_embedder, mini_eval_set_path
    ):
        eval_set = load_eval_set(mini_eval_set_path)
        results = run_eval(
            eval_set, k_values=[5], collection=in_memory_collection, embedder=fake_embedder
        )
        by_id = {r.id: r for r in results}
        # mini-004 is the one item tagged split=held_out in the fixture.
        assert by_id["mini-004"].split == "held_out"
        assert by_id["mini-001"].split == "train"
        assert by_id["mini-001"].difficulty == "easy"

    def test_run_eval_splits_gold_primary_and_secondary(
        self, in_memory_collection, fake_embedder, fake_gdpr_path
    ):
        _build_test_index(in_memory_collection, fake_embedder, fake_gdpr_path)
        eval_set = [
            _base_item(
                id="e1",
                clause="consent",
                gold_articles=[_gold("7"), _gold("6", "secondary")],
            )
        ]
        results = run_eval(
            eval_set, k_values=[5], collection=in_memory_collection, embedder=fake_embedder
        )
        assert results[0].gold_primary == ["7"]
        assert results[0].gold_secondary == ["6"]


@pytest.fixture(scope="module")
def gold_eval_set_items():
    return load_eval_set(DEFAULT_EVAL_SET)


class TestGoldEvalSetIsWellFormed:
    """Regression checks on the hand-labeled eval set itself, not the code."""

    @pytest.fixture
    def items(self, gold_eval_set_items):
        return gold_eval_set_items

    def test_has_a_reasonable_sample_size(self, items):
        assert len(items) >= 100

    def test_ids_are_unique(self, items):
        ids = [item["id"] for item in items]
        assert len(ids) == len(set(ids))

    def test_every_item_has_a_topic(self, items):
        for item in items:
            assert item.get("topic"), f"{item['id']} is missing a topic"

    def test_gold_articles_look_like_valid_gdpr_article_numbers(self, items):
        for item in items:
            for entry in item["gold_articles"]:
                article = entry["article"]
                assert isinstance(article, str), f"{item['id']}: article numbers must be strings"
                assert (
                    article in VALID_GDPR_ARTICLE_NUMBERS
                ), f"{item['id']}: '{article}' is not a plausible GDPR article number (1-99)"

    def test_gold_articles_have_at_least_one_primary(self, items):
        for item in items:
            roles = {e["role"] for e in item["gold_articles"]}
            assert "primary" in roles, f"{item['id']}: gold_articles has no primary entry"

    def test_topics_cover_a_spread_of_gdpr_chapters(self, items):
        # Sanity check that the sample isn't narrowly clustered on one topic —
        # e.g. all consent clauses — which would make recall look better or
        # worse than it will on a real, varied privacy policy.
        all_gold_articles = {e["article"] for item in items for e in item["gold_articles"]}
        assert len(all_gold_articles) >= 20

    def test_every_topic_has_at_least_3_examples(self, items):
        topics: dict[str, int] = {}
        for item in items:
            topics[item["topic"]] = topics.get(item["topic"], 0) + 1
        under = {topic: n for topic, n in topics.items() if n < 3}
        assert not under, f"topics with fewer than 3 examples: {under}"

    def test_every_topic_keeps_at_least_one_train_item(self, items):
        topics: dict[str, list[dict]] = {}
        for item in items:
            topics.setdefault(item["topic"], []).append(item)
        zero_train = {
            topic: len(v) for topic, v in topics.items() if all(i["split"] == "held_out" for i in v)
        }
        assert not zero_train, f"topics with no train items left: {zero_train}"

    def test_held_out_fraction_is_roughly_20_percent(self, items):
        held_out = sum(1 for item in items if item["split"] == "held_out")
        fraction = held_out / len(items)
        assert 0.10 <= fraction <= 0.30, f"held-out fraction {fraction:.1%} is far from ~20%"

    def test_difficulty_is_a_real_mix_not_all_one_value(self, items):
        difficulties = {item["difficulty"] for item in items}
        assert difficulties == {"easy", "hard"}
