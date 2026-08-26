"""CLI for manually testing retrieval against the built index.

Usage:
    python -m rag.query_index "what is required for valid consent"
    python -m rag.query_index "data breach notification" --k 3 --filter source_type=gdpr_article
"""

from __future__ import annotations

import argparse

from rag.retriever import retrieve
from rag.store import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR


def parse_filter(pairs: list[str]) -> dict[str, str] | None:
    if not pairs:
        return None
    filter_dict: dict[str, str] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if not value:
            raise ValueError(f"Invalid filter '{pair}', expected key=value")
        filter_dict[key] = value
    return filter_dict


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Query text")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Metadata filter as key=value (repeatable)",
    )
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument(
        "--hybrid", action="store_true", help="Fuse dense cosine search with a BM25 lexical pass."
    )
    parser.add_argument(
        "--rerank", action="store_true", help="Rerank candidates with a cross-encoder."
    )
    parser.add_argument("--fetch-k", type=int, default=None)
    parser.add_argument("--rerank-top-n", type=int, default=20)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--reranker-model", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    filter_dict = parse_filter(args.filter)

    retrieve_kwargs: dict = {
        "k": args.k,
        "filter": filter_dict,
        "persist_dir": args.persist_dir,
        "collection_name": args.collection,
        "hybrid": args.hybrid,
        "rerank": args.rerank,
    }
    if args.fetch_k:
        retrieve_kwargs["fetch_k"] = args.fetch_k
    if args.rerank:
        retrieve_kwargs["rerank_top_n"] = args.rerank_top_n
    if args.embedding_model:
        from rag.embeddings import Embedder

        retrieve_kwargs["embedder"] = Embedder(args.embedding_model)
    if args.rerank:
        from rag.rerank import get_reranker

        reranker_kwargs = {"model_name": args.reranker_model} if args.reranker_model else {}
        retrieve_kwargs["reranker"] = get_reranker(**reranker_kwargs)

    results = retrieve(args.query, **retrieve_kwargs)

    if not results:
        print("No results.")
        return 0

    for rank, chunk in enumerate(results, start=1):
        print(f"\n[{rank}] score={chunk.score:.4f} id={chunk.id}")
        print(f"    metadata: {chunk.metadata}")
        preview = chunk.text.replace("\n", " ")[:300]
        print(f"    text: {preview}{'...' if len(chunk.text) > 300 else ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
