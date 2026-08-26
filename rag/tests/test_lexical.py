from rag.lexical import BM25Index, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Consent, freely given!") == ["consent", "freely", "given"]


def test_search_ranks_by_term_overlap_and_excludes_zero_scores():
    ids = ["a", "b", "c"]
    texts = [
        "the data subject has given consent to processing",
        "notification of a personal data breach to the supervisory authority",
        "consent consent consent freely given specific informed unambiguous",
    ]
    index = BM25Index.build(ids, texts)

    results = index.search("consent freely given", n=3)

    result_ids = [r[0] for r in results]
    # "b" shares no query terms at all, so it must not appear.
    assert result_ids == ["c", "a"]


def test_search_returns_empty_for_empty_query_or_empty_index():
    index = BM25Index.build(["a"], ["some document text"])
    assert index.search("", n=5) == []

    empty_index = BM25Index.build([], [])
    assert empty_index.search("consent", n=5) == []


def test_search_respects_n():
    ids = [f"id{i}" for i in range(10)]
    texts = ["consent appears in every document here" for _ in range(10)]
    index = BM25Index.build(ids, texts)

    assert len(index.search("consent", n=3)) == 3


def test_search_scores_are_descending():
    ids = ["a", "b", "c"]
    texts = [
        "consent",
        "consent consent consent",
        "consent consent",
    ]
    index = BM25Index.build(ids, texts)

    results = index.search("consent", n=3)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_from_collection_builds_index_from_all_documents(in_memory_collection):
    in_memory_collection.upsert(
        ids=["x", "y"],
        documents=["consent must be freely given", "data breach notification duties"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
    )

    index = BM25Index.from_collection(in_memory_collection)
    results = index.search("consent", n=5)

    assert [id_ for id_, _ in results] == ["x"]


def test_from_collection_respects_where_filter(in_memory_collection):
    in_memory_collection.upsert(
        ids=["x", "y"],
        documents=["consent must be freely given", "consent forms part of this too"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[{"article_number": "7"}, {"article_number": "6"}],
    )

    index = BM25Index.from_collection(in_memory_collection, where={"article_number": "7"})

    assert index.ids == ["x"]
