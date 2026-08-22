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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    filter_dict = parse_filter(args.filter)

    results = retrieve(
        args.query,
        k=args.k,
        filter=filter_dict,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )

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
