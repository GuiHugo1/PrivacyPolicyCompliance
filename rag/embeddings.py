"""Local embedding model wrapper (sentence-transformers, no external API)."""

from __future__ import annotations

from functools import lru_cache

MODEL_NAME = "BAAI/bge-large-en-v1.5"


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
