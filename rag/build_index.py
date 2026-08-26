"""CLI to (re)build the Chroma index from GDPR/EDPB source documents.

Usage:
    python -m rag.build_index --gdpr data/raw/gdpr.json --edpb-dir data/raw/edpb_pdfs
    python -m rag.build_index --gdpr data/raw/gdpr.json --edpb data/raw/edpb_pdfs/guideline1.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.chunk import Chunk
from rag.embeddings import get_embedder
from rag.parsers.edpb import parse_edpb_pdf
from rag.parsers.gdpr import parse_gdpr_file
from rag.store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIR,
    add_chunks,
    get_or_create_collection,
    reset_collection,
)

BATCH_SIZE = 64


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdpr", type=Path, help="Path to GDPR articles/recitals JSON or XML")
    parser.add_argument(
        "--edpb", type=Path, nargs="*", default=[], help="One or more EDPB guideline PDF paths"
    )
    parser.add_argument(
        "--edpb-dir", type=Path, help="Directory of EDPB guideline PDFs (all *.pdf ingested)"
    )
    parser.add_argument(
        "--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR, help="Chroma persist directory"
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument(
        "--reset", action="store_true", help="Drop and recreate the collection before indexing"
    )
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override the embedding model (default: BAAI/bge-large-en-v1.5, or "
        "$RAG_EMBEDDING_MODEL). A fine-tuned checkpoint or alternate base model must be "
        "built with this flag and then queried with the matching --embedding-model on the "
        "query/eval side -- see rag/embeddings.py.",
    )
    return parser.parse_args(argv)


def collect_chunks(args: argparse.Namespace) -> list[Chunk]:
    chunks: list[Chunk] = []

    if args.gdpr:
        print(f"Parsing GDPR source: {args.gdpr}")
        chunks.extend(parse_gdpr_file(args.gdpr, max_tokens=args.max_tokens))

    pdf_paths = list(args.edpb)
    if args.edpb_dir:
        pdf_paths.extend(sorted(args.edpb_dir.glob("*.pdf")))

    for pdf_path in pdf_paths:
        print(f"Parsing EDPB guideline PDF: {pdf_path}")
        chunks.extend(parse_edpb_pdf(pdf_path, max_tokens=args.max_tokens))

    return chunks


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.gdpr and not args.edpb and not args.edpb_dir:
        print("Nothing to index: pass --gdpr and/or --edpb/--edpb-dir", file=sys.stderr)
        return 1

    chunks = collect_chunks(args)
    if not chunks:
        print("No chunks produced from the given sources.", file=sys.stderr)
        return 1
    print(f"Produced {len(chunks)} chunks total.")

    if args.reset:
        collection = reset_collection(args.persist_dir, args.collection)
    else:
        collection = get_or_create_collection(args.persist_dir, args.collection)

    if args.embedding_model:
        from rag.embeddings import Embedder

        embedder = Embedder(args.embedding_model)
    else:
        embedder = get_embedder()

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        embeddings = embedder.embed_documents([c.text for c in batch])
        add_chunks(collection, batch, embeddings)
        print(f"Indexed {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"Done. Collection '{args.collection}' persisted to {args.persist_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
