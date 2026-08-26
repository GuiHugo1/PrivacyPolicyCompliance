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
  "id": "eval-025b",
  "topic": "controller_accountability_measures",
  "clause": "Because the risk profile of different processing activities varies, we tailor our compliance measures accordingly...",
  "gold_articles": [
    {"article": "24", "role": "primary"},
    {"article": "32", "role": "secondary"}
  ],
  "difficulty": "hard",
  "split": "train",
  "notes": "Variant of eval-025 (controller_accountability_measures) -- same gold grounding, different phrasing/structure."
}
```

- `clause` — a representative sentence in the style of an actual privacy
  policy (written from scratch for this eval set, not copied from any real
  company's policy).
- `gold_articles` — **multi-label**: a list of `{"article": "<number>",
  "role": "primary"|"secondary"}` objects, hand-labeled against the
  regulation text. At least one entry must be `"primary"`.
  - `"primary"` articles are the required grounding for **strict** credit.
    Items with more than one co-primary article (e.g. a cookie-consent
    clause needing both the Art 6 legal basis and the Art 7 conditions) are
    AND-required: strict recall needs all of them.
  - `"secondary"` articles are a legitimately overlapping alternative (e.g.
    Art 24's general accountability duty vs. Art 32's security-specific
    instance of the same duty) accepted for **lenient** credit only — see
    "Strict vs. lenient scoring" below. Most items have no secondary
    article at all.
- `difficulty` — `"easy"` if the clause closely mirrors statute wording,
  `"hard"` if it's paraphrased, vague, bundled with other clauses, or
  written in typical privacy-policy legalese. Reported as a separate
  breakdown (see below) so a single blended number can't hide a retriever
  that only works on statute-mirroring language.
- `split` — `"train"` or `"held_out"`. ~20% of items are `"held_out"` and
  must not be looked at while tuning chunking/retrieval/prompts — see
  "Held-out set" below.
- `topic` / `notes` — for humans auditing or extending the set, not consumed
  by the scoring code beyond grouping the by-topic breakdown.

An item with more than one co-primary article is a **compound** item (see
"By gold-article count" below); one with a single primary article is
**single**.

142 clauses across 45 topics (every topic has at least 3 examples) span the
provisions most likely to appear in a real privacy policy: lawfulness/consent
(Art 6–10), the core principles (Art 5), transparency and notice (Art 12–14),
the data-subject rights chapter (Art 15–22), controller/processor obligations
(Art 24–30), security and breach notification (Art 32–34), DPIA/DPO (Art
35–39), international transfers (Art 44–46), and remedies (Art 77, 82). It is
a *sample*, not a census — extend it (see below) as gaps in coverage or
retrieval failures turn up.

### Strict vs. lenient scoring

Every recall/hit-rate/MRR number is reported twice:

- **strict** — uses only `"primary"` gold articles. A strict pass requires
  every primary article to be found (fractional credit if only some of a
  multi-primary item's articles are found).
- **lenient** — a hit-test over primary ∪ secondary: did retrieval find
  *any* acceptable article? This is deliberately a boolean-style rate, not a
  fractional recall over the larger set — requiring every secondary article
  too would make "lenient" stricter than "strict" for any item with more
  than one secondary, which would defeat the point.

### Held-out set

~20% of items (one per topic, for roughly 28 of 142, chosen so every topic
keeps at least 2–3 train items) are tagged `split: "held_out"`. These must
not be used while tuning chunking, retrieval, or prompts — only run once, at
the end, for the number that goes in the report. `--split` defaults to
`train`; running with `--split held_out` or `--split all` logs a timestamp to
`eval/benchmarks/.held_out_eval_log` and prints a warning if that log already
has an entry, since re-running the held-out set defeats its purpose as a
one-time, untouched check.

### By gold-article count (single vs. compound)

Alongside the by-difficulty breakdown, every report also splits items by
whether they have one AND-required primary gold article ("single") or more
than one ("compound", e.g. a clause naming both its Art 6 legal basis and
the Art 21 objection right). A compound item structurally needs a bigger
top-k article budget than a single-gold item to earn full strict credit, so
this breakdown is what shows whether a strategy aimed at compound queries
specifically -- concept-linked chunks (`rag.parsers.gdpr.CONCEPT_LINKS`), a
wider candidate pool before fusion/reranking -- is actually working on the
group it targets, rather than any effect being hidden inside one blended
number. See `retrieval_metrics.aggregate_by_gold_count`.

### Retrieval modes: hybrid and reranking

`eval_retrieval.py` can drive `rag.retriever.retrieve`'s hybrid (dense +
BM25, reciprocal rank fusion) and cross-encoder reranking modes instead of
plain dense-only search, so their effect on recall/MRR -- overall, by
difficulty, and by gold-article count -- can be measured directly rather
than assumed:

```bash
# Hybrid retrieval
python eval/scripts/eval_retrieval.py --hybrid

# Cross-encoder reranking (top-20 candidates by default)
python eval/scripts/eval_retrieval.py --rerank

# Both, plus a wider candidate pool before fusion/reranking
python eval/scripts/eval_retrieval.py --hybrid --rerank --fetch-k 100

# Compare an alternate/fine-tuned embedding model on the hard subset --
# rebuild the index with the same --embedding-model first (see rag/README.md).
python eval/scripts/eval_retrieval.py --embedding-model my-org/bge-legal-ft
```

The BM25 index and/or cross-encoder are built/loaded once per run (not once
per eval item) and reused across the whole eval set. A `gdpr_concept` chunk
(see `rag.parsers.gdpr.CONCEPT_LINKS`) credits *every* article it links at
the rank it's retrieved, not just one -- so a compound item can get full
strict credit from a single concept chunk, without both of its articles
needing to independently break into the top-k on their own.

### Wide-k diagnostic pass

`--diagnostic` overrides `--k` with `3,5,10,20,50` and reports recall/hit-rate
at each, so a ranking problem (gold present, just outside the normal top-k)
can be told apart from a coverage problem (gold absent even at k=50). Meant
for the tuning phase — like the standard report, it defaults to `--split
train`, and running it against `held_out`/`all` prints a note that it's
spending the one-time held-out check on exploratory k values.

### Retrieval vs. judge metrics

This harness only ever measures retrieval — "did we find the right
article(s)" — never "was the resulting compliance judgment correct". The
JSON report (`--output`) namespaces everything under a `"retrieval_metrics"`
key specifically so a future judge-verdict eval can report its own numbers
under a sibling `"judge_metrics"` key without the two ever being averaged
into one combined accuracy score. Retrieval recall and judge-verdict
accuracy answer different questions.

**Why hand-labeled, not scraped:** existing public corpora (OPP-115,
APP-350, PolicyIE — see `reports/sota_review.md` §3) use law-agnostic or
loosely GDPR-aligned category schemes, not article-level GDPR labels, so
they can't directly ground a "did we retrieve the right article" check.

## Running it

Requires a built index (see `rag/README.md`):

```bash
uv sync --group rag
python -m rag.build_index --gdpr data/raw/gdpr.json --reset

# Standard report, train split only.
python eval/scripts/eval_retrieval.py
python eval/scripts/eval_retrieval.py --k 1,3,5,10 --output eval/benchmarks/results.json

# Wide-k diagnostic pass (still train split by default).
python eval/scripts/eval_retrieval.py --diagnostic

# Held-out set -- once, at the end. See "Held-out set" above.
python eval/scripts/eval_retrieval.py --split held_out --output eval/benchmarks/held_out_results.json
```

The report includes:

- **recall_strict@k / recall_lenient@k / hit_rate_strict@k** for each
  requested k — see "Strict vs. lenient scoring" above.
- **MRR strict / MRR lenient** — mean reciprocal rank of the first correctly
  retrieved gold article, under each scoring mode.
- **a by-difficulty breakdown** (easy vs. hard), **a by-gold-count
  breakdown** (single vs. compound-primary), and **a by-topic breakdown** at
  the largest requested k, to spot systematically weak topics, a
  paraphrase-robustness gap, or a compound-query gap rather than just an
  aggregate number.
- **the full list of complete strict misses** (no primary gold article
  retrieved at all) for manual triage — these are the clauses to look at
  first when deciding whether to trust the pipeline.

## Extending the eval set

Append a line to `benchmarks/gdpr_retrieval_eval_set.jsonl` with a unique
`id`, the clause text, `gold_articles` (as `{"article", "role"}` objects,
checked against the actual GDPR article text — not against what the
retriever currently returns, since the point is to catch retrieval gaps, not
confirm them), a `difficulty` tag, and a `split` (new items should almost
always be `"train"` — only add to `"held_out"` deliberately, and sparingly,
since it's meant to stay untouched). `eval/scripts/tests/
test_eval_retrieval.py::TestGoldEvalSetIsWellFormed` will catch duplicate
ids, missing topics, implausible article numbers, topics with fewer than 3
examples, topics with no train items left, and a held-out fraction that's
drifted far from ~20%.
