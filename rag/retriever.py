"""Retrieval module over the persisted Chroma collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.embeddings import get_embedder
from rag.store import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR, get_or_create_collection


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float
    id: str


def retrieve(
    query: str,
    k: int = 5,
    filter: dict[str, Any] | None = None,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    collection=None,
    embedder=None,
) -> list[RetrievedChunk]:
    """Retrieve the top-k chunks most similar to ``query``.

    Args:
        query: natural-language query text.
        k: number of results to return.
        filter: optional Chroma ``where`` clause, e.g. {"source_type": "gdpr_article"}
            or {"article_number": "5"}.
        persist_dir / collection_name: where the index lives (ignored if
            ``collection`` is passed directly).
        collection / embedder: inject pre-built instances (mainly for tests).

    Returns:
        A list of RetrievedChunk, ordered by descending similarity.
    """
    if collection is None:
        collection = get_or_create_collection(persist_dir, collection_name)
    if embedder is None:
        embedder = get_embedder()

    query_embedding = embedder.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=filter,
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: list[RetrievedChunk] = []
    for id_, doc, meta, dist in zip(ids, documents, metadatas, distances, strict=False):
        # Chroma's cosine "distance" is 1 - cosine_similarity for normalized vectors.
        similarity = 1.0 - dist
        retrieved.append(
            RetrievedChunk(text=doc, metadata=dict(meta or {}), score=similarity, id=id_)
        )
    return retrieved
