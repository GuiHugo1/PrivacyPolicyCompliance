# rag

Local RAG ingestion + retrieval pipeline over GDPR article/recital text and
EDPB guideline PDFs. No external embedding API — everything runs locally via
`sentence-transformers` (`BAAI/bge-large-en-v1.5`), persisted to a local
Chroma collection.

## Layout

- `chunk.py` — shared `Chunk` dataclass + token estimator.
- `parsers/gdpr.py` — parses structured GDPR JSON/XML into article/recital
  chunks. An article stays as one chunk unless it exceeds `max_tokens`
  (~500), in which case it splits one-chunk-per-paragraph, always retaining
  the article number in metadata.
- `parsers/edpb.py` — parses EDPB guideline PDFs into section-level chunks
  (via a numbered/all-caps heading heuristic), tagged with guideline title,
  adoption date, and section heading.
- `embeddings.py` — local `sentence-transformers` wrapper (bge-large-en-v1.5).
- `store.py` — Chroma persistent client/collection helpers.
- `retriever.py` — `retrieve(query, k=5, filter=None)`.
- `build_index.py` — CLI to (re)build the index from source documents.
- `query_index.py` — CLI for manual retrieval testing.

## Usage

```bash
uv sync --group rag

# Build the index
python -m rag.build_index --gdpr data/raw/gdpr.json --edpb-dir data/raw/edpb_pdfs --reset

# Query it
python -m rag.query_index "what is required for valid consent" --k 5
python -m rag.query_index "data breach notification" --filter article_number=33
```

## GDPR source format

```json
{
  "articles": [
    {
      "number": "5",
      "title": "Principles relating to processing of personal data",
      "chapter": "II",
      "paragraphs": [{"number": "1", "text": "..."}]
    }
  ],
  "recitals": [{"number": "1", "text": "..."}]
}
```

An equivalent `<article number="5" title="..." chapter="II"><paragraph
number="1">...</paragraph></article>` XML shape is also accepted.

## Tests

```bash
pytest rag/tests
```

Tests use a 5-article fake GDPR fixture (`rag/tests/fixtures/fake_gdpr.json`)
and a deterministic `FakeEmbedder` (no model download) against an in-memory
Chroma collection, so they run fast and offline.
