"""Chroma-backed vector store for GDPR/EDPB chunks, persisted to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.chunk import Chunk

DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "chroma"
DEFAULT_COLLECTION_NAME = "gdpr_compliance"


def get_client(persist_dir: str | Path = DEFAULT_PERSIST_DIR):
    import chromadb

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_or_create_collection(
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    client = get_client(persist_dir)
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def reset_collection(
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    client = get_client(persist_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts str/int/float/bool metadata values."""
    return {k: v for k, v in metadata.items() if v is not None and v != ""}


def add_chunks(
    collection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    if not chunks:
        return
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")

    ids = [c.id or f"chunk-{i}" for i, c in enumerate(chunks)]
    documents = [c.text for c in chunks]
    metadatas = [_sanitize_metadata(c.metadata) for c in chunks]

    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
