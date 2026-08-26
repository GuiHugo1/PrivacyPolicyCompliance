"""Retrieval module over the persisted Chroma collection.

Three modes, controlled by ``hybrid``/``rerank`` (both default off, so the
plain dense-cosine ``retrieve(query, k=5)`` call keeps its original
behavior):

- **dense-only** (default): the original behavior -- top-k by cosine
  similarity from the embedding model alone.
- **hybrid** (``hybrid=True``): dense cosine search and a BM25 lexical pass
  (``rag.lexical``) are run independently over the same candidate pool and
  merged with reciprocal rank fusion (``rag.fusion``). This targets the
  easy/hard gap: BM25 still catches near-verbatim ("easy") queries that
  share exact statutory vocabulary with their gold chunk even when the dense
  embedding lands slightly off, without weakening the dense signal
  paraphrased ("hard") queries depend on.
- **rerank** (``rerank=True``, combinable with ``hybrid``): the top
  ``rerank_top_n`` fused/dense candidates are re-scored by a cross-encoder
  (``rag.rerank``) and re-sorted before taking the final top-k. This is what
  fixes "right answer is a candidate but outranked" failures -- a cheap
  final pass over a small pool rather than scoring the whole corpus.

In both extra modes, ``fetch_k`` controls the size of the candidate pool
pulled *before* fusion/reranking trims it down to ``k`` -- retrieval always
casts a wider net internally than it returns, so a plausible answer isn't
forced to already be in the raw top-k before it has a chance to be promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.embeddings import get_embedder
from rag.fusion import reciprocal_rank_fusion
from rag.lexical import BM25Index
from rag.store import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR, get_or_create_collection

DEFAULT_FETCH_K = 50
DEFAULT_RERANK_TOP_N = 20


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float
    id: str


def _dense_search(
    collection, embedder, query: str, n: int, filter: dict[str, Any] | None
) -> tuple[list[str], dict[str, RetrievedChunk]]:
    """Run the dense cosine query and return (ranked ids, id -> RetrievedChunk)."""
    query_embedding = embedder.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=n, where=filter)

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    ranked_ids: list[str] = []
    by_id: dict[str, RetrievedChunk] = {}
    for id_, doc, meta, dist in zip(ids, documents, metadatas, distances, strict=False):
        # Chroma's cosine "distance" is 1 - cosine_similarity for normalized vectors.
        similarity = 1.0 - dist
        by_id[id_] = RetrievedChunk(text=doc, metadata=dict(meta or {}), score=similarity, id=id_)
        ranked_ids.append(id_)
    return ranked_ids, by_id


def retrieve(
    query: str,
    k: int = 5,
    filter: dict[str, Any] | None = None,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    collection=None,
    embedder=None,
    hybrid: bool = False,
    bm25_index: BM25Index | None = None,
    fetch_k: int | None = None,
    rerank: bool = False,
    reranker=None,
    rerank_top_n: int = DEFAULT_RERANK_TOP_N,
) -> list[RetrievedChunk]:
    """Retrieve the top-k chunks most relevant to ``query``.

    Args:
        query: natural-language query text.
        k: number of results to return.
        filter: optional Chroma ``where`` clause, e.g. {"source_type": "gdpr_article"}
            or {"article_number": "5"}.
        persist_dir / collection_name: where the index lives (ignored if
            ``collection`` is passed directly).
        collection / embedder: inject pre-built instances (mainly for tests).
        hybrid: if True, fuse dense cosine search with a BM25 lexical pass
            (reciprocal rank fusion) instead of dense-only ranking.
        bm25_index: a pre-built ``rag.lexical.BM25Index`` to reuse across
            calls (building it scans the whole collection, so a caller doing
            many queries -- e.g. the eval harness -- should build it once and
            pass it in rather than let every call rebuild it). Only used
            when ``hybrid=True``; built on the fly from ``collection`` if
            omitted.
        fetch_k: candidate pool size fetched/fused before trimming to ``k``
            (and before reranking, if ``rerank=True``). Defaults to
            ``max(k, 50)`` whenever ``hybrid`` or ``rerank`` is on, and to
            plain ``k`` otherwise (matching the original dense-only
            behavior).
        rerank: if True, re-score the top ``rerank_top_n`` candidates with a
            cross-encoder (``rag.rerank``) and re-sort before taking the
            final top-k.
        reranker: a pre-built ``rag.rerank.Reranker`` (or test fake exposing
            ``.score(query, texts) -> list[float]``) to reuse across calls;
            built via ``rag.rerank.get_reranker()`` if omitted.
        rerank_top_n: how many fused/dense candidates to feed the reranker.

    Returns:
        A list of RetrievedChunk, ordered by descending relevance under
        whichever mode was requested. ``.score`` reflects that mode's final
        ranking score (cosine similarity for dense-only, fused RRF score for
        hybrid, cross-encoder score if reranked) -- it is not comparable
        across modes.
    """
    if collection is None:
        collection = get_or_create_collection(persist_dir, collection_name)
    if embedder is None:
        embedder = get_embedder()

    casting_wide_net = hybrid or rerank
    candidate_n = fetch_k or (max(k, DEFAULT_FETCH_K) if casting_wide_net else k)

    dense_ids, by_id = _dense_search(collection, embedder, query, candidate_n, filter)

    if hybrid:
        if bm25_index is None:
            bm25_index = BM25Index.from_collection(collection, where=filter)
        lexical_ids = [id_ for id_, _ in bm25_index.search(query, candidate_n)]

        missing_ids = [id_ for id_ in lexical_ids if id_ not in by_id]
        if missing_ids:
            fetched = collection.get(ids=missing_ids, where=filter)
            fetched_ids = fetched.get("ids", []) or []
            fetched_docs = fetched.get("documents", []) or []
            fetched_metas = fetched.get("metadatas", []) or []
            for id_, doc, meta in zip(fetched_ids, fetched_docs, fetched_metas, strict=True):
                by_id[id_] = RetrievedChunk(text=doc, metadata=dict(meta or {}), score=0.0, id=id_)

        fused_scores = reciprocal_rank_fusion([dense_ids, lexical_ids])
        ranked_ids = sorted(
            (id_ for id_ in fused_scores if id_ in by_id),
            key=lambda id_: fused_scores[id_],
            reverse=True,
        )
        for id_ in ranked_ids:
            by_id[id_].score = fused_scores[id_]
    else:
        ranked_ids = dense_ids

    pool_ids = ranked_ids[: rerank_top_n if rerank else k]

    if rerank:
        if reranker is None:
            from rag.rerank import get_reranker

            reranker = get_reranker()
        texts = [by_id[id_].text for id_ in pool_ids]
        scores = reranker.score(query, texts)
        for id_, score in zip(pool_ids, scores, strict=True):
            by_id[id_].score = score
        pool_ids = sorted(pool_ids, key=lambda id_: by_id[id_].score, reverse=True)

    final_ids = pool_ids[:k]
    return [by_id[id_] for id_ in final_ids]
