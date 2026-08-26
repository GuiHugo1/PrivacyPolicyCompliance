# rag

Local RAG ingestion + retrieval pipeline over GDPR article/recital text and
EDPB guideline PDFs. No external embedding API — everything runs locally via
`sentence-transformers` (`BAAI/bge-large-en-v1.5`), persisted to a local
Chroma collection.

## Layout

- `chunk.py` — shared `Chunk` dataclass + token estimator.
- `parsers/gdpr.py` — parses structured GDPR JSON/XML into article/recital
  chunks. An article stays as one chunk unless it exceeds `max_tokens`
  (~500) *or* one of its paragraphs looks like an enumerated sub-point list
  (e.g. Art 6(1)'s six lettered legal bases), in which case it splits
  one-chunk-per-paragraph, splitting further into one chunk per lettered
  point where applicable. Also generates cross-referenced "concept" chunks
  (`source_type: "gdpr_concept"`) linking pairs of articles that are
  frequently invoked together (e.g. Art 6 legitimate interest ↔ Art 21 right
  to object) — see `CONCEPT_LINKS`. The article number (or, for a concept
  chunk, `concept_articles`) is always retained in metadata.
- `parsers/edpb.py` — parses EDPB guideline PDFs into section-level chunks
  (via a numbered/all-caps heading heuristic), tagged with guideline title,
  adoption date, and section heading.
- `embeddings.py` — local `sentence-transformers` wrapper. Model defaults to
  `bge-large-en-v1.5` but is overridable via `$RAG_EMBEDDING_MODEL` or
  `Embedder(model_name=...)`/`--embedding-model`, the integration point for
  a domain-tuned or fine-tuned checkpoint (see "Domain-tuned embeddings"
  below).
- `lexical.py` — dependency-free Okapi BM25 index (`BM25Index`) over the
  same chunk texts, the lexical leg of hybrid retrieval.
- `fusion.py` — `reciprocal_rank_fusion`, merges the dense and BM25 ranked
  lists into one fused score per chunk.
- `rerank.py` — cross-encoder reranker (`Reranker`, default
  `BAAI/bge-reranker-base`, overridable via `$RAG_RERANKER_MODEL`) that
  re-scores a small top-n candidate pool.
- `store.py` — Chroma persistent client/collection helpers.
- `retriever.py` — `retrieve(query, k=5, filter=None, hybrid=False,
  rerank=False, ...)`. See "Retrieval modes" below.
- `build_index.py` — CLI to (re)build the index from source documents.
- `query_index.py` — CLI for manual retrieval testing.

## Usage

```bash
uv sync --group rag

# Build the index
python -m rag.build_index --gdpr data/raw/gdpr.json --edpb-dir data/raw/edpb_pdfs --reset

# Query it (dense-only, the default)
python -m rag.query_index "what is required for valid consent" --k 5
python -m rag.query_index "data breach notification" --filter article_number=33

# Hybrid (dense + BM25) and/or cross-encoder reranking
python -m rag.query_index "72 hour breach notification deadline" --hybrid
python -m rag.query_index "controller accountability measures" --rerank
```

## Retrieval modes

`retrieve()` supports three modes, controlled by the `hybrid`/`rerank`
flags (both default off, so plain `retrieve(query, k=5)` keeps its original
dense-only behavior):

- **dense-only** (default) — top-k by cosine similarity from the embedding
  model alone.
- **hybrid** (`hybrid=True`) — dense cosine search and a BM25 lexical pass
  (`rag.lexical`) are run independently and merged with reciprocal rank
  fusion (`rag.fusion`). Targets the easy/hard gap: BM25 still catches
  near-verbatim ("easy") queries that share exact statutory vocabulary with
  their gold chunk even when the dense embedding lands slightly off,
  without weakening the dense signal paraphrased ("hard") queries rely on.
- **rerank** (`rerank=True`, combinable with `hybrid`) — the top
  `rerank_top_n` fused/dense candidates are re-scored by a cross-encoder
  (`rag.rerank`) and re-sorted before taking the final top-k. This is what
  fixes "right answer is a candidate but outranked" failures, at the cost
  of loading and running a cross-encoder model.

Both extra modes pull a wider `fetch_k` candidate pool (default
`max(k, 50)`) before trimming down to `k`, so a plausible answer isn't
forced to already be in the raw top-k before it has a chance to be
promoted by fusion or reranking.

A caller doing many queries against the same index (e.g. the eval harness)
should build `BM25Index`/`Reranker` once and pass them in via
`bm25_index=`/`reranker=`, rather than let every `retrieve()` call rebuild
the BM25 index or reload the cross-encoder.

## Domain-tuned embeddings

`rag/embeddings.py`'s `MODEL_NAME` and `rag/rerank.py`'s `MODEL_NAME` are
overridable (`$RAG_EMBEDDING_MODEL` / `$RAG_RERANKER_MODEL`, or
`--embedding-model`/`--reranker-model` on the CLIs) precisely so a
fine-tuned checkpoint or a different base model can be swapped in without a
code change. To fine-tune `bge-large-en-v1.5` on legal-paraphrase pairs:
train with `sentence-transformers`' `MultipleNegativesRankingLoss` (or a
similar contrastive/triplet loss) on (statutory-wording,
plain-English-paraphrase) pairs — the eval set's `clause` text paired with
its gold article's text is a natural source of such pairs, using only the
`train` split so `held_out` stays untouched. Then point
`RAG_EMBEDDING_MODEL`/`--embedding-model` at the resulting checkpoint,
rebuild the index with the same model (`build_index.py --embedding-model`),
and compare `recall@k` on the hard-difficulty subset (see
`eval/scripts/eval_retrieval.py`) against the base model. This repo does not
ship a fine-tuned checkpoint — the plumbing above is what a fine-tuning run
would plug into.

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
and a deterministic `FakeEmbedder`/`FakeReranker` (no model download)
against an in-memory Chroma collection, so they run fast and offline. This
means the real `Reranker`/cross-encoder model download is never exercised
in CI — only `retrieve()`'s hybrid/rerank orchestration is, via the fakes.
