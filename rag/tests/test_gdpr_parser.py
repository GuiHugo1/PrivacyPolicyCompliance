import json

from rag.parsers.gdpr import parse_gdpr_data, parse_gdpr_file


def test_parses_all_five_articles_and_recitals(fake_gdpr_path):
    chunks = parse_gdpr_file(fake_gdpr_path)
    article_numbers = {
        c.metadata["article_number"] for c in chunks if "article_number" in c.metadata
    }
    assert article_numbers == {"1", "4", "6", "7", "33"}

    recital_numbers = {
        c.metadata["recital_number"] for c in chunks if "recital_number" in c.metadata
    }
    assert recital_numbers == {"1", "39"}


def test_short_article_stays_as_a_single_chunk(fake_gdpr_path):
    data = json.loads(fake_gdpr_path.read_text())
    chunks = parse_gdpr_data(data, max_tokens=500)

    article_1_chunks = [c for c in chunks if c.metadata.get("article_number") == "1"]
    assert len(article_1_chunks) == 1
    assert "paragraph_number" not in article_1_chunks[0].metadata
    # Both paragraphs of article 1 must be present in the single chunk.
    assert "protection of natural persons" in article_1_chunks[0].text
    assert "free movement of personal data" in article_1_chunks[0].text


def test_long_article_splits_by_paragraph_and_keeps_article_number(fake_gdpr_path):
    data = json.loads(fake_gdpr_path.read_text())
    # Force a split by using a small max_tokens threshold.
    chunks = parse_gdpr_data(data, max_tokens=50)

    article_33_chunks = [c for c in chunks if c.metadata.get("article_number") == "33"]
    assert len(article_33_chunks) == 5  # one chunk per paragraph

    paragraph_numbers = {c.metadata["paragraph_number"] for c in article_33_chunks}
    assert paragraph_numbers == {"1", "2", "3", "4", "5"}

    for chunk in article_33_chunks:
        assert chunk.metadata["article_number"] == "33"
        assert (
            chunk.metadata["article_title"]
            == "Notification of a personal data breach to the supervisory authority"
        )


def test_chunks_never_mix_two_articles(fake_gdpr_path):
    chunks = parse_gdpr_file(fake_gdpr_path, max_tokens=500)
    article_chunks = [c for c in chunks if c.metadata.get("source_type") == "gdpr_article"]
    for chunk in article_chunks:
        # Each chunk's text must reference exactly its own article number.
        assert chunk.text.startswith(f"Article {chunk.metadata['article_number']}")


def test_recital_chunks_have_expected_metadata(fake_gdpr_path):
    chunks = parse_gdpr_file(fake_gdpr_path)
    recital_1 = next(c for c in chunks if c.metadata.get("recital_number") == "1")
    assert recital_1.metadata["source_type"] == "gdpr_recital"
    assert "fundamental right" in recital_1.text
