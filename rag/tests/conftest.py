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


@pytest.fixture
def fake_gdpr_path() -> Path:
    return FIXTURES_DIR / "fake_gdpr.json"


@pytest.fixture
def in_memory_collection():
    import uuid

    import chromadb

    client = chromadb.Client()
    return client.get_or_create_collection(name=f"test_collection_{uuid.uuid4().hex}")
