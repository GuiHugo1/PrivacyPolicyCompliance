"""Parser for structured GDPR article/recital source documents (JSON or XML).

Expected JSON shape::

    {
      "articles": [
        {
          "number": "5",
          "title": "Principles relating to processing of personal data",
          "chapter": "II",
          "paragraphs": [
            {"number": "1", "text": "..."},
            {"number": "2", "text": "..."}
          ]
        }
      ],
      "recitals": [
        {"number": "1", "text": "..."}
      ]
    }

The equivalent XML shape uses ``<article number="5" title="..." chapter="II">``
elements each containing ``<paragraph number="1">...</paragraph>`` children,
and a top-level ``<recitals><recital number="1">...</recital></recitals>``.

Chunking rules:

- An article is kept as a single chunk (all paragraphs joined, in order)
  unless its combined text exceeds ``max_tokens`` (~500 tokens by default)
  *or* one of its paragraphs looks like an enumerated sub-point list (e.g.
  Art 6(1)'s six lettered legal bases, "(a) ... (b) ... (f) ..."), in which
  case it splits one-chunk-per-paragraph. This is deliberately more
  aggressive than a pure token-count threshold: a compound paragraph like
  Art 6(1) is well under 500 tokens as a whole, but bundling six distinct
  legal bases into one chunk means a query about e.g. "legitimate interest"
  can only ever retrieve it alongside five unrelated bases. Either way a
  chunk never spans two different articles, and the article number is
  always retained in metadata.
- Within that per-paragraph split, a paragraph that itself looks like an
  enumerated sub-point list is split further, one chunk per lettered point
  (plus a "chapeau" chunk for any lead-in text before the first point), so
  each legal basis / condition / right is independently retrievable.
- Cross-referenced "concept" chunks (``source_type: "gdpr_concept"``) link
  pairs of articles that are frequently invoked together in real privacy-
  policy language but sit in different parts of the regulation (e.g. Art 6's
  legitimate-interest basis and Art 21's right to object to it) -- see
  ``CONCEPT_LINKS``. A compound clause that names both concepts no longer
  has to be served by two separately-ranked chunks; a single concept chunk
  in the results already carries both articles. Only generated when every
  article the link references is actually present in the source.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from rag.chunk import Chunk, estimate_tokens

DEFAULT_MAX_TOKENS = 500

_SUBPOINT_RE = re.compile(r"\(([a-z])\)")

# Pairs of GDPR articles that are frequently invoked together in real
# privacy-policy language but live in different parts of the regulation, so
# a compound clause naming both concepts can be served by a single
# cross-referenced chunk instead of needing two separately-ranked articles
# to both surface in the same top-k list. Extend this list as new recurring
# compound patterns turn up in eval failures -- see the eval set's
# "compound" (multi-primary-gold) items for candidates.
CONCEPT_LINKS: list[dict[str, Any]] = [
    {
        "articles": ["6", "21"],
        "concept": "legitimate_interest_and_right_to_object",
        "note": (
            "Processing on the legitimate-interest legal basis (Article 6(1)(f)) is "
            "counterbalanced by the data subject's right to object to that processing "
            "at any time, including for direct marketing (Article 21)."
        ),
    },
    {
        "articles": ["24", "32"],
        "concept": "accountability_and_security_measures",
        "note": (
            "Article 24's general controller accountability duty to implement "
            "'appropriate technical and organisational measures' is given a "
            "security-specific instance in Article 32, which uses near-identical "
            "language for confidentiality, integrity, and availability of processing."
        ),
    },
    {
        "articles": ["13", "14"],
        "concept": "transparency_notice_at_collection",
        "note": (
            "Article 13 sets the information a controller must give a data subject "
            "when personal data is collected directly from them; Article 14 sets the "
            "parallel requirement when the data is instead obtained from another source."
        ),
    },
    {
        "articles": ["15", "17"],
        "concept": "access_and_erasure_rights",
        "note": (
            "The right of access (Article 15) is often exercised as a precursor to the "
            "right to erasure (Article 17): a data subject confirms what is held before "
            "requesting that it be deleted."
        ),
    },
    {
        "articles": ["33", "34"],
        "concept": "breach_notification_authority_and_subject",
        "note": (
            "Article 33 requires notifying the supervisory authority of a personal data "
            "breach; Article 34 requires notifying the affected data subjects directly "
            "when the breach is likely to result in a high risk to them."
        ),
    },
    {
        "articles": ["6", "9"],
        "concept": "lawful_basis_and_special_category_data",
        "note": (
            "Ordinary personal data needs only an Article 6 legal basis; special "
            "category data (health, biometric, etc.) additionally needs a separate "
            "Article 9 condition lifting the general ban on processing it."
        ),
    },
]


def _looks_like_subpoint_list(text: str) -> bool:
    """True if ``text`` contains a genuine enumerated list of lettered
    sub-points -- (a), (b), (c), ... in order, each appearing once -- as
    opposed to a stray cross-reference like "points (c) and (e) of
    paragraph 1", which does not start at (a) and would otherwise
    false-positive."""
    letters = _SUBPOINT_RE.findall(text)
    if len(letters) < 2 or letters[0] != "a":
        return False
    return len(letters) == len(set(letters)) and letters == sorted(letters)


def _split_by_subpoints(text: str) -> list[tuple[str, str]]:
    """Split text into a leading ("", chapeau) entry (if any lead-in text
    precedes the first "(a)") followed by one (letter, point_text) entry per
    lettered sub-point. Caller must have already confirmed
    ``_looks_like_subpoint_list(text)``."""
    matches = list(_SUBPOINT_RE.finditer(text))
    parts: list[tuple[str, str]] = []

    chapeau = text[: matches[0].start()].strip()
    if chapeau:
        parts.append(("", chapeau))

    for i, match in enumerate(matches):
        letter = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        point_text = text[start:end].strip(" ;.\n")
        if point_text:
            parts.append((letter, point_text))

    return parts


def _article_chunks(article: dict[str, Any], max_tokens: int) -> tuple[list[Chunk], str]:
    """Returns (chunks, full_joined_article_text) -- the latter is reused to
    build concept chunks without re-joining paragraph text."""
    number = str(article["number"])
    title = article.get("title", "")
    chapter = article.get("chapter", "")
    paragraphs = article.get("paragraphs", [])

    base_meta = {
        "source_type": "gdpr_article",
        "article_number": number,
        "article_title": title,
        "chapter": chapter,
    }

    full_text = "\n".join(
        f"({p.get('number', i + 1)}) {p['text']}" for i, p in enumerate(paragraphs)
    ).strip()
    if not full_text:
        return [], ""

    has_subpoints = any(_looks_like_subpoint_list(p["text"]) for p in paragraphs)

    if estimate_tokens(full_text) <= max_tokens and not has_subpoints:
        meta = {**base_meta, "chunk_id": f"article-{number}", "paragraph_range": "all"}
        return [Chunk(text=f"Article {number} — {title}\n\n{full_text}", metadata=meta)], full_text

    chunks: list[Chunk] = []
    for p in paragraphs:
        p_number = str(p.get("number", ""))
        text = p["text"].strip()
        if not text:
            continue

        if _looks_like_subpoint_list(text):
            for letter, part_text in _split_by_subpoints(text):
                label = f"({letter})" if letter else ""
                meta = {
                    **base_meta,
                    "chunk_id": f"article-{number}-para-{p_number}-{letter or 'chapeau'}",
                    "paragraph_number": p_number,
                }
                if letter:
                    meta["subpoint"] = letter
                chunks.append(
                    Chunk(
                        text=f"Article {number}({p_number}){label} — {title}\n\n{part_text}",
                        metadata=meta,
                    )
                )
        else:
            meta = {
                **base_meta,
                "chunk_id": f"article-{number}-para-{p_number}",
                "paragraph_number": p_number,
            }
            chunks.append(
                Chunk(text=f"Article {number}({p_number}) — {title}\n\n{text}", metadata=meta)
            )
    return chunks, full_text


def _recital_chunk(recital: dict[str, Any]) -> Chunk:
    number = str(recital["number"])
    text = recital["text"].strip()
    meta = {
        "source_type": "gdpr_recital",
        "recital_number": number,
        "chunk_id": f"recital-{number}",
    }
    return Chunk(text=f"Recital {number}\n\n{text}", metadata=meta)


def _concept_chunks(article_full_text: dict[str, str]) -> list[Chunk]:
    """Build one composite chunk per CONCEPT_LINKS entry whose articles are
    all present in ``article_full_text``; entries referencing an absent
    article are silently skipped rather than emitting a partial/broken
    chunk."""
    chunks: list[Chunk] = []
    for link in CONCEPT_LINKS:
        articles = link["articles"]
        if not all(a in article_full_text for a in articles):
            continue
        sections = "\n\n".join(f"--- Article {a} ---\n{article_full_text[a]}" for a in articles)
        text = f"Concept: {link['concept']}\n\n{link['note']}\n\n{sections}"
        meta = {
            "source_type": "gdpr_concept",
            "concept_name": link["concept"],
            "concept_articles": ",".join(articles),
            "chunk_id": f"concept-{'-'.join(articles)}",
        }
        chunks.append(Chunk(text=text, metadata=meta))
    return chunks


def parse_gdpr_data(data: dict[str, Any], max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Chunk]:
    """Parse an already-loaded GDPR JSON structure into chunks."""
    chunks: list[Chunk] = []
    article_full_text: dict[str, str] = {}

    for article in data.get("articles", []):
        article_chunks, full_text = _article_chunks(article, max_tokens)
        chunks.extend(article_chunks)
        if full_text:
            number = str(article["number"])
            title = article.get("title", "")
            article_full_text[number] = f"Article {number} — {title}\n\n{full_text}"

    for recital in data.get("recitals", []):
        chunks.append(_recital_chunk(recital))

    chunks.extend(_concept_chunks(article_full_text))

    return chunks


def _xml_to_data(root: ET.Element) -> dict[str, Any]:
    articles = []
    for art_el in root.findall(".//article"):
        paragraphs = [
            {"number": p_el.get("number", str(i + 1)), "text": (p_el.text or "").strip()}
            for i, p_el in enumerate(art_el.findall("paragraph"))
        ]
        articles.append(
            {
                "number": art_el.get("number"),
                "title": art_el.get("title", ""),
                "chapter": art_el.get("chapter", ""),
                "paragraphs": paragraphs,
            }
        )

    recitals = [
        {"number": r_el.get("number"), "text": (r_el.text or "").strip()}
        for r_el in root.findall(".//recital")
    ]
    return {"articles": articles, "recitals": recitals}


def parse_gdpr_file(path: str | Path, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Chunk]:
    """Parse a GDPR source file (``.json`` or ``.xml``) into chunks."""
    path = Path(path)
    if path.suffix.lower() == ".xml":
        tree = ET.parse(path)
        data = _xml_to_data(tree.getroot())
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    return parse_gdpr_data(data, max_tokens=max_tokens)
