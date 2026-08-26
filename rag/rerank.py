"""Local cross-encoder reranker for re-scoring retrieval candidates.

A cross-encoder scores a (query, chunk) pair jointly rather than comparing
independently-computed embeddings, which is far more expensive per pair but
much better at judging fine-grained relevance -- the intended use here is to
re-score a small top-k/top-n pool from ``rag.retriever.retrieve`` (dense or
hybrid), not to score the whole corpus. This is what pulls a correct-but-
outranked candidate (right answer retrieved, just not on top) back to the
top of the final list.

Mirrors ``rag.embeddings.Embedder``: the heavy model import/load only
happens on first use (``Reranker.__init__``), and ``get_reranker`` caches one
instance per process so it isn't reloaded per query.
"""

from __future__ import annotations

import os
from functools import lru_cache

MODEL_NAME = os.environ.get("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")


class Reranker:
    """Thin wrapper around a local sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text in ``texts``, same order in,
        same order out (higher = more relevant). Caller is responsible for
        sorting by the returned scores."""
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


@lru_cache(maxsize=1)
def get_reranker(model_name: str = MODEL_NAME) -> Reranker:
    """Process-wide cached reranker instance (model load is expensive)."""
    return Reranker(model_name)
