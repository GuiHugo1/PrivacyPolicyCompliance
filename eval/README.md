# eval

Evaluation scripts and benchmark data for the `rag` retrieval pipeline (and,
over time, the `judge`/`scoring` stages). The first benchmark here targets a
narrow but load-bearing question: **before trusting a compliance verdict
grounded in retrieved GDPR text, does retrieval actually surface the right
article(s) for a given policy clause?**

## Layout

- `benchmarks/gdpr_retrieval_eval_set.jsonl` — hand-labeled retrieval eval
  set: policy clauses paired with the GDPR article(s) that should be
  retrieved for them.
- `scripts/eval_retrieval.py` — CLI that runs the eval set against a built
  index and reports recall/hit-rate/MRR.
- `scripts/retrieval_metrics.py` — the underlying scoring functions
  (recall@k, hit@k, MRR), kept dependency-free so they're cheap to unit test.
- `scripts/tests/` — unit tests for the scoring logic plus an integration
  test against a small in-memory index, and structural regression checks on
  the hand-labeled eval set itself (unique ids, plausible article numbers,
  topic coverage).

## The eval set

Each line in `gdpr_retrieval_eval_set.jsonl` is one JSON object:

```json
{
  "id": "eval-031",
  "topic": "breach_notification_authority",
  "clause": "In the event of a personal data breach, we notify the competent supervisory authority without undue delay and, where feasible, within 72 hours of becoming aware of it.",
  "gold_articles": ["33"],
  "notes": "Art 33 — notification of a personal data breach to the supervisory authority."
}
```

- `clause` — a representative sentence in the style of an actual privacy
  policy (written from scratch for this eval set, not copied from any real
  company's policy).
- `gold_articles` — the GDPR article number(s) that should ground a
  compliance judgment about this clause, hand-labeled against the regulation
  text. Most items have one gold article; a few genuinely span two (e.g. a
  legitimate-interest marketing clause that engages both Art 6 and Art 21).
- `topic` / `notes` — for humans auditing or extending the set, not consumed
  by the scoring code beyond grouping the by-topic breakdown.

45 clauses currently span the provisions most likely to appear in a
real privacy policy: lawfulness/consent (Art 6–10), the core principles
(Art 5), transparency and notice (Art 12–14), the data-subject rights
chapter (Art 15–22), controller/processor obligations (Art 24–30), security
and breach notification (Art 32–34), DPIA/DPO (Art 35–39), international
transfers (Art 44–46), and remedies (Art 77, 82). It is a *sample*, not a
census — extend it (see below) as gaps in coverage or retrieval failures
turn up.

**Why hand-labeled, not scraped:** existing public corpora (OPP-115,
APP-350, PolicyIE — see `reports/sota_review.md` §3) use law-agnostic or
loosely GDPR-aligned category schemes, not article-level GDPR labels, so
they can't directly ground a "did we retrieve the right article" check.

## Running it

Requires a built index (see `rag/README.md`):

```bash
uv sync --group rag
python -m rag.build_index --gdpr data/raw/gdpr.json --reset

python eval/scripts/eval_retrieval.py
python eval/scripts/eval_retrieval.py --k 1,3,5,10 --output eval/benchmarks/results.json
```

The report includes:

- **recall@k / hit_rate@k** for each requested k — recall@k is the fraction
  of an item's gold articles found in the top-k retrieved chunks
  (macro-averaged across the eval set); hit_rate@k is the simpler "was at
  least one gold article retrieved" fraction.
- **MRR** — mean reciprocal rank of the first correctly retrieved gold
  article.
- **a by-topic breakdown** at the largest requested k, to spot systematically
  weak topics rather than just an aggregate number.
- **the full list of complete misses** (no gold article retrieved at all) for
  manual triage — these are the clauses to look at first when deciding
  whether to trust the pipeline.

## Extending the eval set

Append a line to `benchmarks/gdpr_retrieval_eval_set.jsonl` with a unique
`id`, the clause text, and `gold_articles` checked against the actual GDPR
article text (not against what the retriever currently returns — the point
is to catch retrieval gaps, not confirm them). `eval/scripts/tests/
test_eval_retrieval.py::TestGoldEvalSetIsWellFormed` will catch duplicate
ids, missing topics, and implausible article numbers.
