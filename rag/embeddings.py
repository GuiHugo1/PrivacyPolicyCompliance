"""Local embedding model wrapper (sentence-transformers, no external API).

``MODEL_NAME`` defaults to the base ``bge-large-en-v1.5`` checkpoint but can
be overridden -- via the ``RAG_EMBEDDING_MODEL`` env var, or by passing
``model_name``/``--embedding-model`` explicitly to ``Embedder``/the CLIs --
with either a different off-the-shelf base model or a fine-tuned checkpoint,
without any code change. This is the intended integration point for a
domain-tuned embedding model: fine-tune ``bge-large-en-v1.5`` (e.g. with
``sentence-transformers``' ``MultipleNegativesRankingLoss``) on
(statutory-wording, plain-English-paraphrase) pairs -- the eval set's
``clause`` text paired with its gold article's text is a natural source of
such pairs, using only the ``train`` split so ``held_out`` stays untouched --
then point ``RAG_EMBEDDING_MODEL`` (or ``--embedding-model``) at the
resulting checkpoint and compare ``recall@k`` on the hard-difficulty subset
(see ``eval/scripts/eval_retrieval.py``) against the base model.
"""

from __future__ import annotations

import os
from functools import lru_cache

MODEL_NAME = os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")


class Embedder:
    """Thin wrapper around a local sentence-transformers model."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        # bge models recommend an instruction prefix for queries (not documents).
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        embedding = self._model.encode(
            [prefixed], normalize_embeddings=True, show_progress_bar=False
        )
        return embedding[0].tolist()


@lru_cache(maxsize=1)
def get_embedder(model_name: str = MODEL_NAME) -> Embedder:
    """Process-wide cached embedder instance (model load is expensive)."""
    return Embedder(model_name)
