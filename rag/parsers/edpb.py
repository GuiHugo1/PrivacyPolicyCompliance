"""Parser for EDPB guideline PDFs.

Splits a guideline PDF into section-level chunks using a heading heuristic
(numbered headings like ``1.`` / ``1.2`` / ``3.1.4``, or short all-caps
lines). Each chunk carries the guideline title, adoption date, and the
section heading it falls under as metadata.

Title and adoption date can be supplied explicitly (recommended, since they
are authoritative and PDFs are messy to parse reliably), or left unset to
fall back to best-effort extraction from the first page of text.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rag.chunk import Chunk, estimate_tokens

_HEADING_RE = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(?P<title>[A-Z][^\n]{2,100})$")
_ALLCAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 ,'\-/()]{4,80}$")
_ADOPTION_DATE_RE = re.compile(r"[Aa]dopted(?: on)?\s+(\d{1,2}\s+\w+\s+\d{4})")

DEFAULT_MAX_TOKENS = 500


def _extract_pdf_text(path: Path) -> list[str]:
    """Return a list of page texts using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if _HEADING_RE.match(line):
        return True
    if _ALLCAPS_HEADING_RE.match(line) and len(line.split()) <= 12:
        return True
    return False


def _split_into_sections(full_text: str) -> list[tuple[str, str]]:
    """Split text into (heading, body) pairs using the heading heuristic.

    Text before the first detected heading is kept under heading "Preamble".
    """
    sections: list[tuple[str, list[str]]] = [("Preamble", [])]
    for raw_line in full_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_heading(line):
            sections.append((line, []))
        else:
            sections[-1][1].append(line)
    return [(heading, "\n".join(body).strip()) for heading, body in sections]


def _guess_title(first_page_text: str) -> str:
    for line in first_page_text.splitlines():
        line = line.strip()
        if len(line) > 15 and not _ADOPTION_DATE_RE.search(line):
            return line
    return "Untitled EDPB Guideline"


def _guess_adoption_date(text: str) -> str:
    match = _ADOPTION_DATE_RE.search(text)
    return match.group(1) if match else ""


def parse_edpb_pdf(
    path: str | Path,
    title: str | None = None,
    adoption_date: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Chunk]:
    """Parse an EDPB guideline PDF into section-level chunks."""
    path = Path(path)
    pages = _extract_pdf_text(path)
    full_text = "\n".join(pages)

    resolved_title = title or _guess_title(pages[0] if pages else "")
    resolved_date = adoption_date or _guess_adoption_date(full_text)

    chunks: list[Chunk] = []
    for section_idx, (heading, body) in enumerate(_split_into_sections(full_text)):
        if not body:
            continue
        for part_idx, part in enumerate(_split_long_body(body, max_tokens)):
            meta: dict[str, Any] = {
                "source_type": "edpb_guideline",
                "guideline_title": resolved_title,
                "adoption_date": resolved_date,
                "section_heading": heading,
                "source_file": path.name,
                "chunk_id": f"edpb-{path.stem}-sec{section_idx}-part{part_idx}",
            }
            chunks.append(Chunk(text=f"{heading}\n\n{part}", metadata=meta))
    return chunks


def _split_long_body(body: str, max_tokens: int) -> list[str]:
    """Split an overly long section body into paragraph-sized parts."""
    if estimate_tokens(body) <= max_tokens:
        return [body]

    paragraphs = [p for p in body.split("\n") if p.strip()]
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        if current and current_tokens + para_tokens > max_tokens:
            parts.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(para)
        current_tokens += para_tokens
    if current:
        parts.append("\n".join(current))
    return parts or [body]
