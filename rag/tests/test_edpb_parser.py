from rag.parsers.edpb import _guess_adoption_date, _split_into_sections

SAMPLE_TEXT = """Guidelines 05/2020 on consent under Regulation 2016/679
Adopted on 4 May 2020

1. Introduction
These guidelines provide guidance on consent as a legal basis for processing.
Consent remains one of six legal bases.

2. Elements of valid consent
Article 4(11) defines consent as any freely given, specific, informed indication.
2.1 Freely given
The element of "free" implies real choice and control for data subjects.
"""


def test_splits_into_expected_sections():
    sections = _split_into_sections(SAMPLE_TEXT)
    headings = [h for h, body in sections if body]
    assert "1. Introduction" in headings
    assert "2. Elements of valid consent" in headings
    assert "2.1 Freely given" in headings


def test_section_bodies_contain_correct_text():
    sections = dict(_split_into_sections(SAMPLE_TEXT))
    assert "six legal bases" in sections["1. Introduction"]
    assert "real choice and control" in sections["2.1 Freely given"]


def test_guesses_adoption_date():
    assert _guess_adoption_date(SAMPLE_TEXT) == "4 May 2020"
