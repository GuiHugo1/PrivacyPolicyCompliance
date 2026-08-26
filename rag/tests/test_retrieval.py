from rag.parsers.gdpr import parse_gdpr_file
from rag.retriever import retrieve
from rag.store import add_chunks


def _build_test_index(in_memory_collection, fake_embedder, fake_gdpr_path):
    chunks = parse_gdpr_file(fake_gdpr_path)
    embeddings = fake_embedder.embed_documents([c.text for c in chunks])
    add_chunks(in_memory_collection, chunks, embeddings)
    return chunks


def test_retrieve_returns_k_results(in_memory_collection, fake_embedder, fake_gdpr_path):
    _build_test_index(in_memory_collection, fake_embedder, fake_gdpr_path)

    results = retrieve(
        "what is required for valid consent",
        k=3,
        collection=in_memory_collection,
        embedder=fake_embedder,
    )

    assert len(results) == 3
    for r in results:
        assert r.text
        assert isinstance(r.metadata, dict)
        assert isinstance(r.score, float)


def test_retrieve_with_metadata_filter_does_not_crash(
    in_memory_collection, fake_embedder, fake_gdpr_path
):
    _build_test_index(in_memory_collection, fake_embedder, fake_gdpr_path)

    results = retrieve(
        "data breach notification",
        k=5,
        filter={"article_number": "33"},
        collection=in_memory_collection,
        embedder=fake_embedder,
    )

    assert all(r.metadata.get("article_number") == "33" for r in results)


def test_retrieve_on_empty_collection_does_not_crash(in_memory_collection, fake_embedder):
    results = retrieve(
        "anything",
        k=5,
        collection=in_memory_collection,
        embedder=fake_embedder,
    )
    assert results == []


def test_retrieve_k_larger_than_corpus_does_not_crash(
    in_memory_collection, fake_embedder, fake_gdpr_path
):
    chunks = _build_test_index(in_memory_collection, fake_embedder, fake_gdpr_path)

    results = retrieve(
        "personal data",
        k=1000,
        collection=in_memory_collection,
        embedder=fake_embedder,
    )
    assert len(results) == len(chunks)


def test_hybrid_retrieval_uses_bm25_signal_for_exact_term_match(
    in_memory_collection, fake_embedder, fake_gdpr_path
):
    # "72 hours" only appears in Article 33's chunk. FakeEmbedder's
    # hash-derived vectors carry no real semantic signal, so dense-only
    # ranking for this query is arbitrary -- but BM25 uniquely matches
    # Article 33 on these terms, and RRF guarantees any chunk with a BM25
    # contribution outranks every chunk that only has a dense-only score
    # (see rag/fusion.py). This is a fusion-correctness check, not a claim
    # that FakeEmbedder is semantically meaningful.
    _build_test_index(in_memory_collection, fake_embedder, fake_gdpr_path)

    results = retrieve(
        "72 hours",
        k=1,
        hybrid=True,
        collection=in_memory_collection,
        embedder=fake_embedder,
    )

    assert len(results) == 1
    assert results[0].metadata.get("article_number") == "33"


def test_hybrid_retrieval_on_empty_collection_does_not_crash(in_memory_collection, fake_embedder):
    results = retrieve(
        "anything",
        k=5,
        hybrid=True,
        collection=in_memory_collection,
        embedder=fake_embedder,
    )
    assert results == []


def test_rerank_reorders_candidates_by_cross_encoder_score(
    in_memory_collection, fake_embedder, fake_reranker, fake_gdpr_path
):
    _build_test_index(in_memory_collection, fake_embedder, fake_gdpr_path)

    results = retrieve(
        "personal data breach notification supervisory authority",
        k=1,
        rerank=True,
        rerank_top_n=10,
        collection=in_memory_collection,
        embedder=fake_embedder,
        reranker=fake_reranker,
    )

    assert len(results) == 1
    assert results[0].metadata.get("article_number") == "33"


def test_rerank_score_reflects_reranker_not_dense_similarity(
    in_memory_collection, fake_embedder, fake_reranker, fake_gdpr_path
):
    _build_test_index(in_memory_collection, fake_embedder, fake_gdpr_path)

    results = retrieve(
        "personal data breach notification supervisory authority",
        k=3,
        rerank=True,
        rerank_top_n=10,
        collection=in_memory_collection,
        embedder=fake_embedder,
        reranker=fake_reranker,
    )

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # FakeReranker's score is a plain term-overlap count, not a cosine
    # similarity, so a score > 1.0 proves it's not silently falling back to
    # the dense score.
    assert results[0].score > 1.0
