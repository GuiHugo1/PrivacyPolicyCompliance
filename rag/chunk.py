"""Shared chunk data structure used by all parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A single unit of text to embed and index, with source metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = self.metadata.get("chunk_id")


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) to avoid pulling in a tokenizer
    dependency just for chunk-size decisions."""
    return max(1, len(text) // 4)
