import hashlib
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class FakeEmbedder:
    """Deterministic, dependency-free stand-in for the real bge embedder.

    Mirrors rag/tests/conftest.py's FakeEmbedder so eval integration tests
    don't need to download the real sentence-transformers model.
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
    return (
        Path(__file__).resolve().parent.parent.parent.parent
        / "rag"
        / "tests"
        / "fixtures"
        / "fake_gdpr.json"
    )


@pytest.fixture
def mini_eval_set_path() -> Path:
    return FIXTURES_DIR / "mini_eval_set.jsonl"


@pytest.fixture
def in_memory_collection():
    import uuid

    import chromadb

    client = chromadb.Client()
    return client.get_or_create_collection(name=f"test_collection_{uuid.uuid4().hex}")
