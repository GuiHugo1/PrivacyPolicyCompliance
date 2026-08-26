"""Loads full GDPR article text from the same source file ``rag.build_index``
consumes (``data/raw/gdpr.json``/``.xml`` -- see rag/README.md "GDPR source
format"), so the judge SFT dataset is grounded in the identical article text
the RAG pipeline retrieves at inference time, rather than a hand-copied
excerpt that could drift out of sync with it.

Deliberately independent of ``rag.parsers.gdpr``'s chunking (which splits
long articles into paragraph/sub-point chunks for retrieval granularity):
the judge's grounding context is the whole article text, not a single
retrieved fragment, so this module only re-reads the same source file and
joins each article's paragraphs, it does not chunk them.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

_LEADING_NUMBER_RE = re.compile(r"^(\d{1,3})")


def base_article_number(article_cite: str) -> str:
    """Extract the base article number from a pinpoint cite, e.g.
    ``"5(1)(e)"`` -> ``"5"``, ``"13(1)(a)"`` -> ``"13"``, ``"32"`` -> ``"32"``."""
    match = _LEADING_NUMBER_RE.match(article_cite.strip())
    if not match:
        raise ValueError(f"Cannot parse a base article number from {article_cite!r}")
    return match.group(1)


def _join_paragraphs(paragraphs: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"({p.get('number', i + 1)}) {p['text']}" for i, p in enumerate(paragraphs) if p.get("text")
    ).strip()


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
                "paragraphs": paragraphs,
            }
        )
    return {"articles": articles}


def load_article_texts(path: str | Path) -> dict[str, str]:
    """Returns ``{article_number: "Article N — Title\\n\\n(1) ...\\n(2) ..."}``
    for every article in the source file, keyed by base article number."""
    path = Path(path)
    if path.suffix.lower() == ".xml":
        data = _xml_to_data(ET.parse(path).getroot())
    else:
        data = json.loads(path.read_text(encoding="utf-8"))

    texts: dict[str, str] = {}
    for article in data.get("articles", []):
        number = str(article["number"])
        title = article.get("title", "")
        body = _join_paragraphs(article.get("paragraphs", []))
        if body:
            texts[number] = f"Article {number} — {title}\n\n{body}"
    return texts
