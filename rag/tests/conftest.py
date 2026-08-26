import hashlib
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class FakeEmbedder:
    """Deterministic, dependency-free stand-in for the real bge embedder.

    Avoids downloading/loading sentence-transformers in unit tests; produces
    a fixed-size vector derived from a hash of the text so identical/similar
    texts get identical/similar embeddings.
    """

    dim = 32

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.dim]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


class FakeReranker:
    """Deterministic stand-in for a cross-encoder: scores a (query, text)
    pair by term overlap, so hybrid/rerank tests can check that reranking
    actually reorders candidates without downloading a real cross-encoder
    model."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        query_terms = set(query.lower().split())
        scores = []
        for text in texts:
            text_terms = set(text.lower().split())
            scores.append(float(len(query_terms & text_terms)))
        return scores


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def fake_gdpr_path() -> Path:
    return FIXTURES_DIR / "fake_gdpr.json"


@pytest.fixture
def in_memory_collection():
    import uuid

    import chromadb

    client = chromadb.Client()
    return client.get_or_create_collection(name=f"test_collection_{uuid.uuid4().hex}")
