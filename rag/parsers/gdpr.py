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

Chunking rule: an article is kept as a single chunk (all paragraphs joined,
in order) unless its combined text exceeds ``max_tokens`` (~500 tokens by
default), in which case it is split one-chunk-per-paragraph. Either way the
article number is always retained in metadata, and a chunk never spans two
different articles.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from rag.chunk import Chunk, estimate_tokens

DEFAULT_MAX_TOKENS = 500


def _article_chunks(article: dict[str, Any], max_tokens: int) -> list[Chunk]:
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
        return []

    if estimate_tokens(full_text) <= max_tokens:
        meta = {**base_meta, "chunk_id": f"article-{number}", "paragraph_range": "all"}
        return [Chunk(text=f"Article {number} — {title}\n\n{full_text}", metadata=meta)]

    chunks: list[Chunk] = []
    for p in paragraphs:
        p_number = str(p.get("number", ""))
        text = p["text"].strip()
        if not text:
            continue
        meta = {
            **base_meta,
            "chunk_id": f"article-{number}-para-{p_number}",
            "paragraph_number": p_number,
        }
        chunks.append(
            Chunk(text=f"Article {number}({p_number}) — {title}\n\n{text}", metadata=meta)
        )
    return chunks


def _recital_chunk(recital: dict[str, Any]) -> Chunk:
    number = str(recital["number"])
    text = recital["text"].strip()
    meta = {
        "source_type": "gdpr_recital",
        "recital_number": number,
        "chunk_id": f"recital-{number}",
    }
    return Chunk(text=f"Recital {number}\n\n{text}", metadata=meta)


def parse_gdpr_data(data: dict[str, Any], max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Chunk]:
    """Parse an already-loaded GDPR JSON structure into chunks."""
    chunks: list[Chunk] = []
    for article in data.get("articles", []):
        chunks.extend(_article_chunks(article, max_tokens))
    for recital in data.get("recitals", []):
        chunks.append(_recital_chunk(recital))
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
