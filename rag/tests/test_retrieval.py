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
