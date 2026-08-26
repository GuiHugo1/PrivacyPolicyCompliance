import pytest

from judge import gdpr_source
from judge.tests.conftest import MINI_GDPR_PATH


class TestBaseArticleNumber:
    @pytest.mark.parametrize(
        "cite,expected",
        [
            ("5(1)(e)", "5"),
            ("13(1)(a)", "13"),
            ("13(2)(a)", "13"),
            ("32", "32"),
            ("6(1)", "6"),
        ],
    )
    def test_extracts_leading_number(self, cite, expected):
        assert gdpr_source.base_article_number(cite) == expected

    def test_raises_on_unparseable_cite(self):
        with pytest.raises(ValueError):
            gdpr_source.base_article_number("Article Five")


class TestLoadArticleTexts:
    def test_loads_every_article_by_number(self):
        texts = gdpr_source.load_article_texts(MINI_GDPR_PATH)
        assert "13" in texts
        assert "32" in texts
        assert texts["13"].startswith("Article 13 — Information to be provided")

    def test_article_text_includes_all_paragraphs(self):
        texts = gdpr_source.load_article_texts(MINI_GDPR_PATH)
        assert "(1)" in texts["13"]
        assert "(2)" in texts["13"]
        assert "(3)" in texts["13"]
