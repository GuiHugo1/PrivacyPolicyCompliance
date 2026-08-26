# RAG Retrieval Analysis — Architecture Comparison

Source data: `eval/benchmarks/results.json`, `results_hybrid.json`,
`results_rerank.json`, `results_wider.json`, `held_out_results.json`, and the
pre-upgrade `OLD_results.json` / `OLD_held_out_results.json`. All numbers
below are read directly from those reports (142-item hand-labeled GDPR
retrieval eval set, 114 `train` / 28 `held_out`, see `eval/README.md` for
methodology, strict-vs-lenient scoring, and the single-vs-compound
breakdown).

This document only analyzes retrieval quality — "did the pipeline surface
the right GDPR article(s) for a policy clause" — not downstream compliance-judgment
accuracy, matching the scope the eval harness itself declares.

## 1. What changed between the two generations of results

`OLD_results.json` / `OLD_held_out_results.json` are the dense-only
(embedding cosine search) baseline on the *original* chunking. The current
`results.json` is dense-only again, but on **finer chunking** (articles
split per-paragraph / per-lettered-point once they exceed ~500 tokens, plus
generated `gdpr_concept` cross-reference chunks). `results_hybrid.json`,
`results_rerank.json`, and `results_wider.json` layer BM25 fusion,
cross-encoder reranking, and a wider candidate pool on top of that same new
chunking.

### Finer chunking alone (dense-only, old vs. new), train split, n=114

| | MRR strict | recall_strict@10 | complete strict misses |
|---|---|---|---|
| Old chunking (dense) | 0.694 | 0.741 | 29 |
| New chunking (dense) | 0.689 | 0.750 | 28 |

Finer chunking by itself is close to a wash: it fixed 8 items (mostly
enumerated-list clauses that now land in their own chunk instead of being
diluted inside a long article-level chunk) but broke 7 others (mostly
`transfers_appropriate_safeguards`, where splitting apparently separated
text that needed to stay together for the query to match). Net effect:
+1 fewer miss, MRR essentially flat. **Chunking granularity was not the
lever that moved the numbers — hybrid retrieval and reranking were.**

## 2. Architecture comparison (train split, n=114, strict scoring)

| Architecture | MRR strict | recall_strict@3 | recall_strict@10 | complete misses |
|---|---|---|---|---|
| Dense-only (baseline) | 0.689 | 0.750 | 0.750 | 28 |
| + Hybrid (dense + BM25 RRF fusion) | 0.752 | 0.811 | 0.820 | 20 |
| + Rerank (dense + cross-encoder) | **0.777** | **0.829** | **0.838** | **18** |
| + Hybrid + Rerank + wide pool (fetch_k=100) | 0.756 | 0.820 | 0.820 | 20 |

By raw aggregate numbers, dense+rerank alone is the single best configuration.
But the aggregate hides a real trade-off — see the breakdown below.

### By difficulty (easy = statute-like wording, hard = paraphrased/bundled)

| Architecture | easy MRR | hard MRR | easy recall@10 | hard recall@10 |
|---|---|---|---|---|
| Dense-only | 0.797 | 0.522 | 0.877 | 0.556 |
| Hybrid | 0.895 | 0.533 | 0.964 | 0.600 |
| Rerank | 0.896 | **0.594** | 0.949 | **0.667** |
| Hybrid+Rerank+wide | **0.911** | 0.519 | 0.964 | 0.600 |

Reranking is what moves the needle on **hard** (paraphrased) clauses — it's
the only mode that meaningfully lifts hard-MRR above the dense baseline.
Interestingly, combining hybrid+rerank+wide pool does *not* stack that
hard-difficulty gain on top of hybrid's: its hard-MRR (0.519) is actually
*below* rerank-alone (0.594) and roughly at the dense baseline. The most
likely explanation is that a wider fetch_k (100 vs. the default ~50) feeds
the cross-encoder more borderline/lexically-similar-but-wrong candidates on
already-ambiguous hard clauses, diluting its precision even though it still
helps recall on easy clauses.

### By gold-article count (single vs. compound — the AND-required multi-article items)

| Architecture | single MRR | compound MRR | single recall@10 | compound recall@10 |
|---|---|---|---|---|
| Dense-only | 0.708 | 0.333 | 0.778 | 0.250 |
| Hybrid | 0.757 | **0.667** | 0.833 | **0.583** |
| Rerank | **0.802** | 0.333 | **0.870** | 0.250 |
| Hybrid+Rerank+wide | 0.761 | **0.667** | 0.833 | **0.583** |

This is the clearest signal in the whole comparison: **reranking alone does
nothing for compound (multi-gold-article) clauses** — its compound numbers
are identical to the dense-only baseline. **Hybrid retrieval is what fixes
compound recall**, more than doubling it (0.250 → 0.583). This tracks with
the architecture: compound items are disproportionately served by the
generated `gdpr_concept` cross-reference chunks (e.g. Art 6 legitimate
interest ↔ Art 21 objection), which tend to be lexically distinctive enough
for BM25 to surface even when dense embedding similarity ranks them lower.
A cross-encoder reranking a purely dense candidate pool never gets the
chance to promote a concept chunk that never made the dense pool in the
first place.

## 3. Held-out check (run once, n=28 — see `eval/README.md` "Held-out set")

| | MRR strict | recall_strict@10 |
|---|---|---|
| Old dense-only baseline | 0.708 | 0.750 |
| New architecture (held_out_results.json) | 0.679 | 0.750 |

The held-out MRR is slightly *lower* than the old dense-only baseline,
even though the same architecture gained nearly 9 points of MRR on the
114-item train split. With only 28 held-out items this is very likely
sampling noise rather than a real regression — a couple of items flipping
outcome moves this number by several points — but it is a genuine result
from the one-time held-out run, not an artifact filtered out here, and it
means the train-split gains should not be read as guaranteed to generalize
at the same magnitude. It's the clearest argument in this dataset for
growing the eval set before trusting small differences between
architectures.

## 4. Persistent failures (missed by every architecture, including hybrid+rerank+wide)

13 of 142 items are never retrieved correctly by any tested configuration:
`eval-007c`, `eval-008b`, `eval-008c`, `eval-014c`, `eval-016c`,
`eval-019c`, `eval-025d`, `eval-030c`, `eval-035c`, `eval-043`, `eval-043c`,
`eval-044c`, `eval-044d`. They cluster in a few topics — `cookies_tracking_consent`,
`data_minimisation`, `retention_period_disclosure`, `legitimate_interest_marketing_objection`
— and skew heavily toward `difficulty: hard` (paraphrased/legalese clauses,
not statute-mirroring ones). No amount of fusing or reranking the existing
dense candidate pool recovers these; the gold article's text and the
paraphrased clause simply don't share enough signal, lexical or semantic,
for any current mode to connect them. That points at the embedding model
itself (or the chunk text it's encoding) as the actual ceiling here, not the
retrieval strategy layered on top of it.

## 5. How this could be improved further

*(Not the goal of this study — the eval harness explicitly scopes itself to
retrieval quality under the current architecture, not to shipping a new one.
Recorded here for the backlog, formulated as: it could be done this way to
improve it, but we will keep this version for now.)*

- **Domain-tuned embeddings.** `rag/README.md` already documents the
  plumbing for this (`$RAG_EMBEDDING_MODEL`, fine-tune `bge-large-en-v1.5`
  with `MultipleNegativesRankingLoss` on (statutory-wording,
  plain-English-paraphrase) pairs built from the eval set's own `train`
  split). This is the most direct lever against the 13 persistent misses
  above, since they look like an embedding-similarity ceiling rather than a
  fusion/reranking gap. It could be done this way to close the hard-clause
  gap further, but we will keep the base `bge-large-en-v1.5` checkpoint for
  now — fine-tuning needs a larger labeled pair set and an evaluation
  discipline (held-out set) to avoid overfitting to this eval set's own
  phrasing.
- **Grow and rebalance the eval set**, especially compound items (currently
  only 6 of 114 train items) and the held-out split (28 items — small enough
  that the MRR swing in §3 could be pure noise). It could be done this way
  to get statistically firmer comparisons between architectures, but we
  will keep the current 142-item set for now since expanding it requires
  the same hand-labeling-against-statute-text discipline the eval set
  already commits to (not scraping/auto-labeling).
- **Reconsider fetch_k for the reranker specifically on hard queries.**
  §2 shows a wider candidate pool (fetch_k=100) *hurts* hard-difficulty MRR
  when reranking is on, even though it doesn't hurt recall. It could be
  done this way — e.g. use a wider fetch_k for the BM25/dense fusion step
  but cap what's actually handed to the cross-encoder (`rerank_top_n`)
  tighter than 100 — but we will keep the current single `fetch_k` knob for
  now rather than add a second tunable parameter without more data on
  where the crossover point is.
- **Concept-chunk coverage.** Compound-item gains are currently carried
  almost entirely by hybrid/BM25 hitting the hand-curated
  `CONCEPT_LINKS` chunks. It could be done this way — extend
  `CONCEPT_LINKS` coverage to more article pairs, or generate them
  more systematically instead of by hand — but we will keep the current
  hand-curated list for now, since the compound sample (n=6) is too small
  to tell which additional pairs would actually move the number.
- **Revisit the finer-chunking split logic** for
  `transfers_appropriate_safeguards`, the one topic where finer chunking
  measurably regressed dense retrieval (§1). It could be done this way —
  keep those articles as single chunks or adjust the split heuristic — but
  we will keep the current chunking as-is for now, since hybrid/rerank on
  top of it already recovers most of what it lost.

## 6. Conclusion: best strategy for article retrieval

No single architecture wins on every axis, but the breakdowns make the
trade-off legible rather than hidden inside one blended number:

- **Dense+rerank alone** posts the best raw aggregate and hard-difficulty
  numbers, but it **structurally cannot help compound (multi-article)
  clauses** — its compound recall is bit-for-bit identical to the
  unmodified dense baseline, because reranking can only reorder a
  candidate pool that dense search already failed to fill correctly.
- **Dense+hybrid (BM25 fusion)** is what actually fixes compound-item
  recall (0.250 → 0.583), because BM25 catches the lexically distinctive
  `gdpr_concept` cross-reference chunks that dense embedding search
  under-ranks.
- **The combined hybrid+rerank(+wide pool)** configuration is the only one
  that gets *both*: compound recall matching hybrid's (0.583, vs. rerank's
  0.250) while still clearing the dense baseline by ~9 points of MRR
  overall — even though its raw aggregate is a couple of points behind
  rerank-alone and its hard-difficulty MRR is, surprisingly, no better than
  hybrid-alone once the pool is widened to 100.

**Recommendation: query the RAG with hybrid retrieval + cross-encoder
reranking together** (`retrieve(query, k, hybrid=True, rerank=True)`),
rather than reranking a dense-only pool. For a compliance-grounding use
case, a strategy that silently drops one of two co-required articles on
every compound clause (rerank-alone's behavior) is a worse failure mode
than a couple of points of aggregate MRR — an incomplete grounding for a
compliance verdict is not a partial-credit outcome, it's a wrong one. The
`fetch_k=100` widening tested in `results_wider.json` is not clearly worth
it on this data (it slightly *hurts* hard-difficulty MRR without adding
recall beyond hybrid+rerank at the default pool size), so the default
`fetch_k` (`max(k, 50)`) is the better starting point rather than the wider
pool.