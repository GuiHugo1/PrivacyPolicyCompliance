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


def test_paragraph_with_lettered_subpoints_splits_one_chunk_per_point():
    # A compound paragraph well under the 500-token article threshold, but
    # bundling three distinct legal bases -- exactly the case the
    # aggressive sub-point splitting targets.
    data = {
        "articles": [
            {
                "number": "6",
                "title": "Lawfulness of processing",
                "chapter": "II",
                "paragraphs": [
                    {
                        "number": "1",
                        "text": (
                            "Processing shall be lawful only if and to the extent that at "
                            "least one of the following applies: "
                            "(a) the data subject has given consent to the processing of "
                            "his or her personal data for one or more specific purposes; "
                            "(b) processing is necessary for the performance of a contract "
                            "to which the data subject is party; "
                            "(f) processing is necessary for the purposes of the legitimate "
                            "interests pursued by the controller or by a third party."
                        ),
                    }
                ],
            }
        ],
        "recitals": [],
    }

    chunks = parse_gdpr_data(data, max_tokens=500)

    subpoint_chunks = [c for c in chunks if c.metadata.get("subpoint")]
    assert {c.metadata["subpoint"] for c in subpoint_chunks} == {"a", "b", "f"}
    for c in subpoint_chunks:
        assert c.metadata["source_type"] == "gdpr_article"
        assert c.metadata["article_number"] == "6"
        assert c.metadata["paragraph_number"] == "1"
        assert c.text.startswith("Article 6(1)(")

    legit_interest = next(c for c in subpoint_chunks if c.metadata["subpoint"] == "f")
    assert "legitimate" in legit_interest.text.lower()
    assert "consent" not in legit_interest.text.lower()

    chapeau_chunks = [c for c in chunks if c.metadata.get("chunk_id", "").endswith("-chapeau")]
    assert len(chapeau_chunks) == 1
    assert "at least one of the following applies" in chapeau_chunks[0].text


def test_paragraph_with_stray_cross_reference_is_not_treated_as_a_list():
    # "(c) and (e)" is a cross-reference to points defined elsewhere, not an
    # enumerated list starting at (a) -- must not trigger sub-point splitting.
    data = {
        "articles": [
            {
                "number": "6",
                "title": "Lawfulness of processing",
                "chapter": "II",
                "paragraphs": [
                    {
                        "number": "2",
                        "text": (
                            "Member States may maintain more specific provisions with "
                            "regard to processing for compliance with points (c) and (e) "
                            "of paragraph 1."
                        ),
                    }
                ],
            }
        ],
        "recitals": [],
    }

    chunks = parse_gdpr_data(data, max_tokens=500)

    assert not any(c.metadata.get("subpoint") for c in chunks)
    assert len(chunks) == 1
    assert "paragraph_number" not in chunks[0].metadata  # stayed a single whole-article chunk


def test_concept_chunk_links_related_articles_when_both_present():
    data = {
        "articles": [
            {
                "number": "6",
                "title": "Lawfulness of processing",
                "chapter": "II",
                "paragraphs": [
                    {
                        "number": "1",
                        "text": "Processing shall be lawful only if based on legitimate interests.",
                    }
                ],
            },
            {
                "number": "21",
                "title": "Right to object",
                "chapter": "III",
                "paragraphs": [
                    {
                        "number": "1",
                        "text": "The data subject shall have the right to object to processing "
                        "based on legitimate interests, including profiling.",
                    }
                ],
            },
        ],
        "recitals": [],
    }

    chunks = parse_gdpr_data(data, max_tokens=500)

    concept_chunks = [c for c in chunks if c.metadata.get("source_type") == "gdpr_concept"]
    assert len(concept_chunks) == 1
    concept = concept_chunks[0]
    assert concept.metadata["concept_articles"] == "6,21"
    assert concept.metadata["concept_name"] == "legitimate_interest_and_right_to_object"
    assert "Article 6" in concept.text
    assert "Article 21" in concept.text
    assert "right to object" in concept.text.lower()


def test_concept_chunk_skipped_when_a_linked_article_is_missing():
    data = {
        "articles": [
            {
                "number": "6",
                "title": "Lawfulness of processing",
                "chapter": "II",
                "paragraphs": [{"number": "1", "text": "Processing shall be lawful only if..."}],
            }
        ],
        "recitals": [],
    }

    chunks = parse_gdpr_data(data, max_tokens=500)

    assert not [c for c in chunks if c.metadata.get("source_type") == "gdpr_concept"]


def test_concept_chunks_do_not_set_article_number(fake_gdpr_path):
    # Regression guard for the eval-set article-extraction assumption: a
    # concept chunk carries multiple articles under "concept_articles", not
    # a single "article_number" -- it must never masquerade as a
    # single-article chunk.
    chunks = parse_gdpr_file(fake_gdpr_path)
    for c in chunks:
        if c.metadata.get("source_type") == "gdpr_concept":
            assert "article_number" not in c.metadata
