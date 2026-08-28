# GDPR Privacy-Policy Compliance Assessment: Technical Report

**Project:** Privacy Policy Compliance
**Report date:** 2026-08-28
**Status:** Draft — sections marked `[DATA NEEDED]` await results not yet committed to the repository; `[FIGURE: ...]` marks a diagram/chart placeholder.

---

## 1. Problem statement and scope

### 1.1 Motivation

Privacy policies are the primary public artifact through which a controller discloses how it processes personal data, yet verifying that a given policy actually satisfies the General Data Protection Regulation (GDPR) is a slow, expert-driven task: it requires reading the policy clause by clause, holding each clause against the relevant article(s) of the regulation (and, in practice, the European Data Protection Board's (EDPB) interpretive guidance), and forming a judgment about whether the disclosure is present, adequate, and correctly grounded. This does not scale to the number of policies that exist, changes frequently as products and vendors change, and — as the state-of-the-art review in `reports/sota_review.md` documents — is not solved transparently by either the academic literature or the commercial compliance-tooling market today. Academic systems either predate GDPR-specific framing (Polisis, PolicyGPT), classify policies into law-agnostic categories rather than judging them against specific articles (OPP-115-style corpora), or, where they do use LLMs as compliance judges, rely on prompting closed commercial models rather than a fine-tuned, open-weight judge (§1.5 and §5 of `reports/sota_review.md`). Commercial platforms (OneTrust, TrustArc, Osano, Termly, iubenda) do not publish their scoring rubric, model, or validation methodology at all.

This project builds and evaluates a system that takes a real privacy policy as input and produces a structured, article-level compliance assessment: which GDPR articles the policy addresses, how it addresses them, and an aggregate compliance score, with every verdict traceable back to the specific policy clause and GDPR text it was grounded in.

### 1.2 Scope

The system is deliberately scoped narrowly, on the premise that a system that is transparent and evaluated within a narrow scope is more useful and more trustworthy than one that claims broad coverage without evidence:

- **GDPR only.** The judge model, the scoring engine's chapter/article structure, and the retrieval corpus are all built exclusively against Regulation (EU) 2016/679 (`data/raw/gdpr.json`) plus EDPB guidance documents that interpret it (`data/raw/edpb_pdfs/`). No other privacy regime (CCPA/CPRA, UK GDPR as a distinct instrument, LGPD, PIPEDA, etc.) is modeled, mapped, or scored. `reports/sota_review.md` documents this as a deliberate differentiator from multi-jurisdictional systems (e.g. Xie et al., USENIX Security 2025) rather than a limitation to be resolved incidentally — see §8 for what a multi-jurisdiction extension would require.
- **Article-level granularity.** Compliance is assessed and scored per GDPR article (e.g. Art. 13, Art. 17, Art. 32), not at the level of the whole document, nor at the finer level of individual sub-paragraphs or lettered points (e.g. Art. 6(1)(a)–(f)), except where the underlying grounding text itself is chunked at that granularity (§3.1). The judge's structured output schema (`judge/judge_schema.json`) always resolves its `article` field to a base article number for aggregation, even when the retrieved grounding chunk or the mapping config cites a pinpoint sub-provision.
- **A privacy-policy *document* is the unit of assessment**, not a company's actual data-processing practices, vendor contracts, or technical implementation. The system judges what the policy *says*, against what GDPR requires it to say and (to the extent inferable from text alone) do — it cannot verify that the controller's real-world processing matches its own disclosure.
- **In-scope chapters.** Of GDPR's eleven chapters, only Chapters II–V (Principles; Data Subject Rights; Controller and Processor Obligations; International Transfers — Art. 5–50) are in scope for scoring, per `scoring/config/article_weights.yaml`. Chapter I (definitions/scope of the regulation itself) and Chapters VI–XI (supervisory authorities, cooperation/consistency, remedies/penalties, specific processing situations, delegated acts, final provisions) concern the regulation's own operation, not obligations a privacy-policy document itself discharges — a policy cannot "comply" or "fail to comply" with, say, the composition of a supervisory authority. These out-of-scope articles are still retrieved, judged, and reported for explainability; they are simply excluded from the score itself (§5).
- **English-language policies only** (§7.4).
- **Not legal advice.** The system produces a technical, evidence-linked assessment intended to support a compliance review, not a substitute for legal counsel or a certification of compliance.

## 2. System architecture

[FIGURE: end-to-end system architecture diagram — policy input (PDF/text/URL) → clause segmentation → RAG retrieval over the GDPR+EDPB Chroma index (hybrid dense+BM25, cross-encoder rerank) → per-(clause, reference) judge inference (QLoRA-fine-tuned Qwen2.5) → per-article aggregation → scoring engine (chapter/overall rollup) → Streamlit UI, with the GDPR/EDPB ingestion pipeline and OPP-115 training-data pipeline shown as offline/build-time paths feeding the index and the judge adapter respectively]

The system is organized as four pipeline stages plus a presentation layer, implemented as four Python packages (`rag/`, `judge/`, `scoring/`, `ui/`) that are composed by import, not by shelling out between CLIs — `ui/app.py` calls `judge.pipeline` and `scoring.score` directly, and `judge/pipeline.py` calls `rag.retriever` directly.

### 2.1 Offline build-time paths

Two pipelines run ahead of any assessment and produce artifacts the runtime path depends on:

1. **RAG ingestion** (`rag/build_index.py`). GDPR article/recital text (`data/raw/gdpr.json`) and EDPB guideline PDFs (`data/raw/edpb_pdfs/`) are parsed into chunks, embedded, and persisted to a local Chroma collection. This is a one-time (or on-source-change) build step; the resulting index is what every runtime retrieval call queries.
2. **Judge fine-tuning** (`judge/build_sft_dataset.py` → `judge/train_qlora.py`). OPP-115 annotations are turned into supervised fine-tuning examples via the OPP-115→GDPR mapping config, and a QLoRA adapter is trained on top of a base Qwen2.5 model. The resulting adapter checkpoint (`judge/checkpoints/`) is what the runtime judge loads.

### 2.2 Runtime assessment path

Given one privacy-policy document, `judge/pipeline.py` orchestrates the following stages end to end, producing one JSON object matching `judge/output_schema.json`:

1. **Load.** `load_policy_text` reads a `.pdf` (via `pypdf`'s layout-preserving extraction, which is needed to keep paragraph breaks intact — see §7.1) or plain text.
2. **Segment.** `segment_clauses` splits the policy into paragraph/sentence-level clauses, capped at `--max-clause-chars` (default 2000, matching what the judge was trained on).
3. **Retrieve.** For each clause, `RetrieverContext.retrieve` queries the Chroma index via `rag.retriever.retrieve`, by default with hybrid (dense + BM25) retrieval and cross-encoder reranking both on — the configuration `eval/RAG_ANALYSIS.md` recommends (§3.2) — returning the top-`k` (default 3) most relevant GDPR/EDPB chunks.
4. **Judge.** Every (clause, retrieved-chunk) pair is submitted individually to the fine-tuned judge model, in exactly the prompt format used during training (system prompt + `"Clause:\n...\n\nRetrieved GDPR Article ...:\n..."`), and the model's JSON output is parsed and validated against `judge/judge_schema.json`. A schema-invalid response gets one automatic repair retry (the bad output plus an error description, asking the model to correct itself); a pair that still fails falls back to a `needs_review` verdict rather than aborting the run.
5. **Aggregate.** `aggregate_articles` groups every clause-verdict by GDPR article and picks a `best_compliance_status` per article. Articles that exist in the RAG index but that no clause ever retrieved are recorded as `not_addressed` — distinct from `not_applicable`, which means a clause *was* checked against the article and the judge decided it doesn't apply (§5, §7.5).
6. **Score.** `scoring.score.score_judge_output` consumes the pipeline's `articles` array and produces a `ComplianceReport`: an overall 0–100 score, a per-chapter breakdown, and a per-article breakdown carrying evidence/rationale through unmodified.
7. **Present.** `ui/app.py` (Streamlit) accepts a policy URL or file upload, runs the above end to end, and renders the overall score, a chapter-level bar chart, and an expandable per-article breakdown with evidence, plus JSON/PDF report export. The judge model, RAG index, embedding model, and reranker are all expensive to load, so they are cached per settings-combination (`st.cache_resource`) rather than reloaded per interaction; long-running operations (model load, per-clause judging) run under an explicit timeout so the UI stays responsive rather than hanging.

A structural point worth making explicit: retrieval draws from one shared Chroma collection covering GDPR articles, recitals, *and* EDPB guideline sections, but the judge was fine-tuned only on GDPR-article-shaped grounding text (`judge/gdpr_source.py` never loads recitals or EDPB PDFs into the SFT data). When retrieval surfaces a recital or EDPB chunk, the pipeline still submits it to the judge for a best-effort verdict — it records the result in `clause_verdicts` but excludes it from the per-article `articles` rollup (`article_number: null`), since there's no article to aggregate it under. This is a known coverage gap, not a silent failure — see §7.6 and §8.

## 3. RAG design

### 3.1 Chunking strategy

`rag/parsers/gdpr.py` and `rag/parsers/edpb.py` chunk the two source corpora differently, matching their structure:

- **GDPR text.** An article is kept as a single chunk by default. It is split further — one chunk per paragraph, and further to one chunk per lettered sub-point where applicable — only when it exceeds ~500 tokens *or* one of its paragraphs is an enumerated list (the canonical case being Art. 6(1)'s six lettered legal bases). In addition, the parser generates synthetic **`gdpr_concept` cross-reference chunks** that link pairs of articles frequently invoked together (e.g. Art. 6 legitimate interest ↔ Art. 21 right to object), via a hand-curated `CONCEPT_LINKS` table. Every chunk retains its source article number (or, for a concept chunk, both linked article numbers) in metadata, so retrieval hits can always be traced back to a specific provision.
- **EDPB guidance.** Guideline PDFs are chunked at section granularity via a numbered/all-caps heading heuristic, tagged with guideline title, adoption date, and section heading.

An ablation isolating chunking granularity from the retrieval-strategy changes below found finer chunking alone close to a wash on its own (train split, dense-only, n=114):

| | MRR strict | recall_strict@10 | complete strict misses |
|---|---|---|---|
| Original (article-level) chunking | 0.694 | 0.741 | 29 |
| Finer chunking (per-paragraph/lettered-point) | 0.689 | 0.750 | 28 |

Finer chunking fixed 8 items (mostly enumerated-list clauses that now land in their own chunk instead of being diluted inside a long article-level chunk) but broke 7 others (mostly `transfers_appropriate_safeguards` clauses, where splitting apparently separated text that needed to stay together for the query to match) — net one fewer miss, MRR essentially flat. **Chunking granularity was not the lever that moved retrieval quality; hybrid retrieval and reranking were** (§3.3).

### 3.2 Embedding model

Retrieval runs entirely locally, with no external embedding API: `rag/embeddings.py` wraps `sentence-transformers`, defaulting to **`BAAI/bge-large-en-v1.5`**, persisted to a local Chroma collection (`rag/store.py`). The model is overridable via `$RAG_EMBEDDING_MODEL` / `--embedding-model`, which is the deliberate integration point for a future domain-tuned checkpoint (§8). A parallel dependency-free Okapi BM25 index (`rag/lexical.py`) provides the lexical leg of hybrid retrieval, and a cross-encoder reranker (`rag/rerank.py`, default `BAAI/bge-reranker-base`, also overridable) re-scores a top-n candidate pool for the rerank mode.

### 3.3 Retrieval evaluation

**Eval set.** `eval/benchmarks/gdpr_retrieval_eval_set.jsonl` is a hand-labeled set of 142 policy-style clauses across 45 topics, each paired with one or more gold GDPR articles labeled `primary` (required for strict credit) or `secondary` (accepted only for lenient credit). Items are tagged `difficulty: easy` (statute-mirroring wording) or `hard` (paraphrased/bundled/legalese) and split `train` (114 items, used for tuning) or `held_out` (28 items, ~20%, run once and never used to tune chunking/retrieval/prompts). The set was hand-labeled from scratch against the regulation text rather than derived from an existing corpus, because existing public corpora (OPP-115, APP-350, PolicyIE) use law-agnostic or loosely GDPR-aligned category schemes and cannot ground an article-level "did we retrieve the right article" check (`eval/README.md`).

**Scoring.** Every metric is reported strict (all `primary` gold articles required — fractional credit for a partially-found multi-primary item) and lenient (a hit against `primary ∪ secondary`, boolean-style). Reports are additionally broken out by difficulty and by gold-article count (single- vs. compound-primary, i.e. clauses that structurally need more than one co-required article to earn full strict credit).

**Architecture comparison** (train split, n=114, strict scoring):

| Architecture | MRR strict | recall_strict@3 | recall_strict@10 | complete misses |
|---|---|---|---|---|
| Dense-only (baseline) | 0.689 | 0.750 | 0.750 | 28 |
| + Hybrid (dense + BM25, RRF fusion) | 0.752 | 0.811 | 0.820 | 20 |
| + Rerank (dense + cross-encoder) | **0.777** | **0.829** | **0.838** | **18** |
| + Hybrid + Rerank + wide pool (fetch_k=100) | 0.756 | 0.820 | 0.820 | 20 |

By raw aggregate numbers, dense+rerank alone is the best single configuration — but the aggregate hides a real trade-off, visible once the results are broken out:

*By difficulty:*

| Architecture | easy MRR | hard MRR | easy recall@10 | hard recall@10 |
|---|---|---|---|---|
| Dense-only | 0.797 | 0.522 | 0.877 | 0.556 |
| Hybrid | 0.895 | 0.533 | 0.964 | 0.600 |
| Rerank | 0.896 | **0.594** | 0.949 | **0.667** |
| Hybrid+Rerank+wide | **0.911** | 0.519 | 0.964 | 0.600 |

*By gold-article count (single vs. compound):*

| Architecture | single MRR | compound MRR | single recall@10 | compound recall@10 |
|---|---|---|---|---|
| Dense-only | 0.708 | 0.333 | 0.778 | 0.250 |
| Hybrid | 0.757 | **0.667** | 0.833 | **0.583** |
| Rerank | **0.802** | 0.333 | **0.870** | 0.250 |
| Hybrid+Rerank+wide | 0.761 | **0.667** | 0.833 | **0.583** |

The clearest signal in the comparison: **reranking alone does nothing for compound (multi-gold-article) clauses** — its compound numbers are bit-for-bit identical to the dense-only baseline, because a cross-encoder can only reorder a candidate pool that dense search already failed to fill correctly. **Hybrid retrieval is what fixes compound recall**, more than doubling it (0.250 → 0.583), because compound items are disproportionately served by the generated `gdpr_concept` cross-reference chunks, which BM25 surfaces via their lexically distinctive vocabulary even when dense cosine similarity under-ranks them.

**Held-out check** (n=28, run once):

| | MRR strict | recall_strict@10 |
|---|---|---|
| Old dense-only baseline | 0.708 | 0.750 |
| New architecture (hybrid+rerank) | 0.679 | 0.750 |

The held-out MRR is slightly *lower* than the old baseline, despite the same architecture gaining nearly 9 points of MRR on the 114-item train split. With only 28 held-out items this is very likely sampling noise — a couple of items flipping outcome moves the number by several points — but it is a genuine result from the one-time held-out run, not filtered out here, and it means the train-split gains should not be assumed to generalize at the same magnitude without a larger held-out set (§7.7, §8).

**Persistent failures.** 13 of 142 items (`eval-007c`, `eval-008b`, `eval-008c`, `eval-014c`, `eval-016c`, `eval-019c`, `eval-025d`, `eval-030c`, `eval-035c`, `eval-043`, `eval-043c`, `eval-044c`, `eval-044d`) are never retrieved correctly by any tested configuration, including hybrid+rerank+wide. They cluster around `cookies_tracking_consent`, `data_minimisation`, `retention_period_disclosure`, and `legitimate_interest_marketing_objection`, and skew heavily `hard`-difficulty. No amount of fusing or reranking the existing dense candidate pool recovers them — the gold article's text and the paraphrased clause simply don't share enough signal for any current mode to connect them, pointing at the embedding model (or the chunk text it encodes) as the actual ceiling, not the retrieval strategy layered on top of it (§7.2, §8).

**Recommendation adopted by the runtime pipeline:** query with hybrid retrieval + cross-encoder reranking together (`retrieve(query, k, hybrid=True, rerank=True)`), which is `judge/pipeline.py`'s default. Reranking-alone posts the best raw aggregate and hard-difficulty numbers, but it structurally cannot help compound clauses; for a compliance-grounding use case, silently dropping one of two co-required articles on every compound clause is judged a worse failure mode than a couple of points of aggregate MRR — an incomplete grounding for a compliance verdict is not a partial-credit outcome, it is a wrong one.

## 4. Judge model

### 4.1 Base model and fine-tuning method

The judge is **Qwen2.5-7B-Instruct**, fine-tuned with **QLoRA**: the base model is loaded in 4-bit (`bitsandbytes`, NF4 quantization, double quantization enabled, bfloat16 compute dtype), and a LoRA adapter is trained on top via `peft`. Only the LoRA adapter is saved (`PeftModel.save_pretrained`), not a merged model, keeping the fine-tuning artifact small and the base model swappable. Training uses the Hugging Face `Trainer` with a custom collator (`judge.qlora_data.JudgeSFTCollator`) that tokenizes prompt and response separately and masks the prompt tokens to `-100`, so loss is computed only on the JSON-verdict tokens the model actually needs to learn to produce.

A smaller **Qwen2.5-0.5B-Instruct** config (`judge/config/qlora_judge_0.5b_gpu.yaml`) and a CPU-only, non-quantized variant of the same model (`judge/config/qlora_judge_cpu.yaml`) exist as fast smoke-test paths for validating the training pipeline (data loading, collator, LoRA wiring, eval callback) without requiring the hardware the 7B run needs — not as substitutes for the production judge (§7.3).

### 4.2 Training data construction: OPP-115 → GDPR mapping

Training examples are built by `judge/build_sft_dataset.py` from two inputs:

1. **OPP-115** (Wilson et al., ACL 2016) — 115 privacy policies annotated by law students into 10 practice categories (First Party Collection/Use, Third Party Sharing/Collection, Data Retention, Data Security, User Choice/Control, User Access/Edit/Deletion, Policy Change, International/Specific Audiences, Do Not Track, Other) with fine-grained attribute values per category.
2. **A hand-authored mapping config** (`judge/config/opp115_gdpr_mapping.yaml`) from each OPP-115 category/attribute-value combination to GDPR article(s) — marked `primary` (used to ground a training example) or `secondary` (recorded for traceability only) — plus a heuristic rule for the resulting `compliance_status` and `confidence`.

One SFT example is generated per (policy, segment, category, primary GDPR article): the annotated segment becomes the `clause`, the mapped article's canonical text from `data/raw/gdpr.json` becomes the "retrieved GDPR article" grounding (gold grounding by default — the judge is trained against correct context rather than the RAG pipeline's own retrieval noise; `--use-retriever` can instead run the real retriever, but per §3.3 that bakes in retrieval mistakes and is meant to contribute a *minority* of imperfect-retrieval examples, not the primary grounding source), and the heuristic rule resolves the target `compliance_status`. Multiple annotators labeling the same practice are collapsed via majority vote before the mapping rule is applied.

**This is weak supervision, not ground truth**, by explicit design decision and label: every generated example carries `meta.weak_label: true`, because OPP-115 predates GDPR and records *whether a practice is disclosed*, not whether that disclosure satisfies a specific GDPR article. `judge/coverage_report.py` quantifies where this weak signal is thin or entirely absent, run against the currently loaded OPP-115 corpus:

| Category | Annotations | Policies | Segments | Generatable |
|---|---:|---:|---:|---:|
| Data Retention | 370 | 76 | 156 | 156 |
| Data Security | 1,008 | 102 | 375 | 375 |
| Do Not Track | 90 | 31 | 32 | 32 |
| First Party Collection/Use | 8,935 | 114 | 1,522 | 1,522 |
| International and Specific Audiences | 939 | 90 | 353 | 353 |
| Other | 3,548 | 114 | 1,763 | 287 |
| Policy Change | 548 | 93 | 192 | 192 |
| Third Party Sharing/Collection | 5,221 | 114 | 1,186 | 1,186 |
| User Access, Edit and Deletion | 746 | 90 | 231 | 231 |
| User Choice/Control | 1,789 | 106 | 632 | 632 |

Beyond data-volume thinness, nine GDPR requirements have **no** OPP-115 category or attribute that records them at all, regardless of volume, and therefore need genuinely new hand-labeled data rather than more OPP-115 annotations: legal-basis granularity (which Art. 6(1)(a)–(f) applies), special-category data conditions (Art. 9), DPO designation (Art. 37–39), the specific cross-border transfer mechanism (Art. 44–49), DPIA (Art. 35), privacy by design/default (Art. 25), automated decision-making (Art. 22), restriction of processing (Art. 18), breach-notification specifics (Art. 33–34), and portability (Art. 20) — see `reports/opp115_gdpr_coverage.md` and `judge/README.md`'s "Coverage gaps" for the full detail and rationale per gap. `Do Not Track` and `Policy Change` are additionally flagged `weak_alignment`, since GDPR has no article that natively governs either.

### 4.3 Hyperparameters

Production config (`judge/config/qlora_judge.yaml`, `Qwen2.5-7B-Instruct`):

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Max sequence length | 2048 |
| Quantization | 4-bit NF4, double quantization, bfloat16 compute |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| LoRA bias | none |
| Epochs | 3 |
| Per-device batch size (train / eval) | 2 / 2 |
| Gradient accumulation steps | 8 |
| Effective batch size | 16 |
| Learning rate | 2.0e-4, cosine schedule, 3% warmup |
| Weight decay | 0.0 |
| Max grad norm | 0.3 |
| Optimizer | `paged_adamw_8bit` |
| Precision | bf16, gradient checkpointing on |
| Seed | 42 |
| Generation (inference) | greedy decoding (`do_sample: false`), max 256 new tokens |

The 0.5B GPU smoke-test config (`judge/config/qlora_judge_0.5b_gpu.yaml`) shares the same schema with a lower LoRA rank (r=8, alpha=16), 1 epoch, gradient accumulation of 4, and a 1024-token max sequence length, sized to fit a free-tier GPU (e.g. a Colab T4). Only the LoRA adapter differs structurally between configs; base model and batch/accumulation settings scale with available VRAM.

### 4.4 Evaluation metrics

`judge/eval_metrics.py` defines two metric families, shared between the training-time eval callback and the standalone test-split report (`judge/eval_qlora.py`):

- **JSON-validity rate** — the fraction of generated verdicts that are both syntactically valid JSON and pass `judge/judge_schema.json` validation. This is measured by actually re-generating (not teacher-forcing) a capped sample of validation examples at each eval step, since teacher-forced loss does not tell you whether the model can produce a schema-valid object unprompted.
- **Per-class precision/recall/F1** over the four `compliance_status` classes the judge is trained to emit (`compliant`, `partial`, `non_compliant`, `not_applicable`; `needs_review` is the pipeline's own fallback for unparseable output, never a class the judge itself is trained to produce — see §5).

Held-out test-set results for the trained adapter:

**[DATA NEEDED — `judge/eval_qlora.py`'s test-split report is not committed to this repo (`judge/metrics/` is gitignored).** Run:

```bash
python -m judge.eval_qlora \
    --config judge/config/qlora_judge.yaml \
    --adapter judge/checkpoints/qwen2.5-7b-qlora-judge \
    --output judge/metrics/test_eval_report.json
```

and drop the resulting numbers into the table below.]

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `compliant` | | | | |
| `partial` | | | | |
| `non_compliant` | | | | |
| `not_applicable` | | | | |
| **Macro avg** | | | | |
| **JSON-validity rate** | | | *(not per-class)* | |

## 5. Scoring methodology

### 5.1 Weighting scheme

`scoring/score.py` turns one judge-pipeline output into a `ComplianceReport` via a deterministic, fully config-driven aggregation (`scoring/config/article_weights.yaml`):

1. Every article carries a numeric `best_compliance_status` mapped to a score in `[0, 1]` via `status_scores`: `compliant → 1.0`, `partial → 0.5`, `non_compliant → 0.0`, `not_addressed → 0.0`, `needs_review → 0.0` (the pipeline's own parse-failure fallback, scored conservatively in the same bucket as a confirmed failure so a policy cannot improve its score by producing clauses the judge fails to parse). `not_applicable` is excluded entirely from the denominator — it contributes neither a score nor a weight, rather than counting as a zero.
2. Each article carries a weight (default 1.0, override-able per article in `article_weights.overrides` — e.g. weighting Art. 17 right-to-erasure higher within its chapter).
3. **Chapter score** = weighted average of its scorable (non-excluded) member articles.
4. **Overall score** = weighted average of chapter scores, restricted to chapters marked `in_scope: true` (Chapters II–V, per §1.2).
5. A chapter or article with no scorable members (e.g. every article in it came back `not_applicable`, or the judge output simply never touched it) is excluded from its parent's weighted average — weights are renormalized over whatever *is* scorable, never silently treated as a zero. This means the overall score is never artificially depressed by an article the judge correctly determined doesn't apply, or by a chapter absent from a given run's input.

`non_compliant` and `not_addressed` are scored identically (0.0) today, on purpose but not by necessity: they represent two different findings — a *confirmed violation* (a clause was checked against the article and the judge found it failing) versus *silence* (no clause ever retrieved the article at all, which could mean the article genuinely doesn't apply to this controller, or that retrieval/segmentation missed a real gap). The code path never branches on which of the two it is; changing their relative weight is a one-line config edit (§5.2 exercises exactly this).

### 5.2 Sensitivity analysis

`scoring/sensitivity_analysis.py` recomputes one judge-pipeline output's overall and per-chapter score under a fixed set of preset config variants, to quantify how much of the reported score is a property of the underlying compliance findings versus a property of the scoring methodology's own choices:

| Preset | What it changes |
|---|---|
| `baseline` | Default config, unmodified. |
| `principles_heavy` | Ch. II (Principles) weight ×3. |
| `rights_heavy` | Ch. III (Data subject rights) weight ×3. |
| `transfers_heavy` | Ch. V (International transfers) weight ×3. |
| `strict_partial_credit` | `partial` scored 0.0 instead of 0.5. |
| `lenient_partial_credit` | `partial` scored 0.75 instead of 0.5. |
| `silence_penalized_less` | `not_addressed` scored 0.25 (silence weighted below a confirmed violation). |
| `silence_penalized_more` | `non_compliant` scored 0.25 (confirmed violation weighted below silence). |

**[DATA NEEDED — results below are not yet generated.** Run against a representative judge-pipeline output (ideally the same held-out set used in §6):

```bash
python -m scoring.sensitivity_analysis --input <judge_output>.json
```

and populate the table.]

| Config | Description | Overall | Ch. II | Ch. III | Ch. IV | Ch. V |
|---|---|---|---|---|---|---|
| `baseline` | Default config, unmodified. | | | | | |
| `principles_heavy` | Ch. II weight ×3. | | | | | |
| `rights_heavy` | Ch. III weight ×3. | | | | | |
| `transfers_heavy` | Ch. V weight ×3. | | | | | |
| `strict_partial_credit` | `partial` → 0.0. | | | | | |
| `lenient_partial_credit` | `partial` → 0.75. | | | | | |
| `silence_penalized_less` | `not_addressed` → 0.25. | | | | | |
| `silence_penalized_more` | `non_compliant` → 0.25. | | | | | |

[FIGURE: sensitivity-analysis comparison — overall score by preset config, as a bar or dot plot, to make the spread introduced by scoring choices visually legible against the baseline]

Interpreting this table once populated: the spread between `principles_heavy`/`rights_heavy`/`transfers_heavy` and `baseline` quantifies how much a stakeholder's chapter-weighting priorities alone move the headline number, independent of any change in the underlying judge findings; the spread between `strict_partial_credit`/`lenient_partial_credit` quantifies sensitivity to how "partial" compliance is valued; and the `silence_penalized_*` pair quantifies the practical consequence of the `non_compliant` vs. `not_addressed` design choice flagged in §5.1 — a large gap here is itself evidence for resolving that conflation rather than leaving both at 0.0 by default (§8).

## 6. End-to-end evaluation

### 6.1 Methodology

The evaluations in §3 (retrieval) and §4 (judge verdicts) are deliberately kept separate and reported under distinct metric namespaces, matching the eval harness's own design (`eval/README.md`'s "Retrieval vs. judge metrics") — "did we retrieve the right article" and "was the resulting compliance judgment correct" answer different questions, and blending them into one score would hide which stage is responsible for an error. End-to-end evaluation is the complementary, outermost check: running the full pipeline (`judge.pipeline` → `scoring.score`) on real, previously unseen privacy policies and comparing its output against independently produced ground truth for those same policies, so that retrieval errors, judge errors, aggregation behavior, and scoring choices are all captured in one pass exactly as a real user would experience them.

A held-out set of real privacy policies — disjoint from any policy used to build or tune the RAG index, the OPP-115 training split, or the retrieval eval set — is the appropriate instrument for this: each policy needs an independent, article-level compliance annotation (produced the same hand-labeling-against-statute-text way the retrieval eval set was, per `eval/README.md`) against which the pipeline's `articles[].best_compliance_status` output and the scoring engine's overall/chapter scores can be compared.

### 6.2 Results

**[DATA NEEDED — no end-to-end evaluation report is currently committed to this repository.** Populate this section with, at minimum:

- the held-out policy set's size and composition (industry mix, policy length distribution, source);
- per-article agreement between the pipeline's `best_compliance_status` and the ground-truth label (accuracy and macro-F1 over the same four classes as §4.4, plus the `not_addressed` case where ground truth says the article doesn't apply but the pipeline never retrieved it, or vice versa);
- agreement on the overall 0–100 score against an independently produced human/expert overall assessment (e.g. mean absolute error, or a correlation coefficient, plus a scatter plot);
- a qualitative breakdown of failure modes observed on this set, and whether they trace back to segmentation, retrieval, or judge error (cross-referencing §7.1–§7.3);
- wall-clock latency per policy, since this determines whether the pipeline is viable for interactive (UI) use vs. batch-only use.]

| Metric | Value |
|---|---|
| Held-out policies (n) | |
| Per-article status accuracy | |
| Per-article status macro-F1 | |
| Overall-score MAE vs. ground truth | |
| Median / p95 latency per policy | |

[FIGURE: predicted vs. ground-truth overall compliance score scatter plot across the held-out policy set, with the y=x reference line]

[FIGURE: confusion matrix — pipeline `best_compliance_status` vs. ground-truth status, aggregated across all held-out policies' articles]

## 7. Limitations

### 7.1 Clause segmentation quality

`segment_clauses` is a deliberately simple paragraph/sentence heuristic, not a legal-clause-boundary parser. Its own docstring already flags plain-text hard-wrapping (a policy where paragraphs are wrapped to a fixed line width with no blank-line separator) as a case it does not currently handle well — the wrapped lines may be merged or split incorrectly, which propagates downstream: a clause that gets merged with an unrelated neighboring clause dilutes the retrieval query and can cause the judge to see (and verdict on) more than one actual legal statement at once; a clause split mid-sentence can retrieve on an incomplete thought. PDF extraction adds a second layer of the same risk: `load_policy_text` relies on `pypdf`'s layout-preserving mode to keep blank-line paragraph gaps intact, but an unusually laid-out source PDF (multi-column layout, tables, or no vertical whitespace between paragraphs) can still lose or merge paragraph breaks before segmentation ever runs. The documented workaround — extracting to `.txt` by hand or with a different tool for a policy that segments poorly from its PDF — is a manual mitigation, not a fix to the underlying heuristic.

### 7.2 OPP-115 → GDPR mapping subjectivity

The judge's entire training signal passes through a hand-authored mapping from OPP-115's law-agnostic category scheme to GDPR articles (§4.2). This mapping is explicitly labeled weak supervision (`meta.weak_label: true` on every generated example) for a specific, non-cosmetic reason: OPP-115 records *whether a practice is disclosed*, not whether the disclosure *satisfies* a GDPR article, and a companion mapping study (cited in `reports/sota_review.md` §3.1) independently documents a structural mismatch between OPP-115's category boundaries and GDPR's own structure (e.g. OPP-115 separates "First Party" from "Third Party" collection as distinct categories, while GDPR's Art. 5 principles apply uniformly regardless of party). Beyond that structural mismatch, nine specific GDPR requirements have no OPP-115 signal at all (§4.2's schema-gap list), and two categories (`Do Not Track`, `Policy Change`) are flagged `weak_alignment` because GDPR has no article that natively governs either. Every judge verdict the pipeline produces inherits this mapping's editorial judgment calls, which were made by this project's own review of the OPP-115 codebook, not independently validated by a second annotator or legal reviewer.

### 7.3 Model size trade-offs

The production config targets Qwen2.5-7B-Instruct; the only checkpoint actually committed to this repository (`judge/checkpoints/qwen2.5-0.5b-gpu-qlora-judge/`) is the 0.5B smoke-test model, trained and sized specifically to validate the training pipeline mechanics (data loading, collator, LoRA wiring, 4-bit quantization, the eval callback) on modest hardware (a single consumer/free-tier GPU), not to serve as a production judge. Its documented purpose in `judge/README.md` is explicit that it is "far weaker" than the 7B target and "should not be treated as a final model." Any assessment run through the sidebar's default selection in the current UI build is therefore running on the weaker model unless a 7B adapter has been separately trained and selected. §4.4's evaluation-metrics table should be populated per adapter (0.5B vs. 7B) so this trade-off is quantified, not just asserted.

### 7.4 English-only scope

Every component — the GDPR source text, the EDPB guideline corpus, the embedding and reranker models, the OPP-115 training data, and the retrieval eval set — is English-only. GDPR applies across all EU/EEA member states and official policy text is frequently published in multiple languages; a non-English policy, or an English policy from a jurisdiction where the operative legal text a court would reference is a non-English translation, is entirely out of scope for this system as built. There is no multilingual embedding model, no translated GDPR/EDPB corpus, and no non-English training data anywhere in the pipeline.

### 7.5 Retrieval ceiling and the `non_compliant`/`not_addressed` conflation

Two limitations compound each other in a way worth calling out together. First, §3.3's 13 persistent retrieval misses and the held-out MRR regression (0.708 → 0.679, likely noise but not filtered out) mean that some fraction of "not addressed" article findings are actually retrieval failures, not genuine policy silence. Second, `not_addressed` and `non_compliant` are scored identically (§5.1) by design today. The practical consequence is that a retrieval miss on an article the policy *does* actually address correctly will surface, indistinguishably from a genuine violation, as a 0.0-scored `not_addressed` article in the final report — the compliance score cannot currently be fully trusted to separate "the pipeline failed to find this" from "the policy failed to comply with this" without a human reading the underlying clause list. The sensitivity analysis in §5.2 is precisely the instrument for quantifying how much this conflation actually moves the headline score once real data is run through it.

### 7.6 Judge coverage of non-article grounding

Retrieval draws from GDPR articles, recitals, and EDPB guideline sections in one shared index, but the judge was fine-tuned exclusively on GDPR-article-shaped grounding text. When retrieval surfaces a recital or an EDPB chunk (which happens whenever it is genuinely the most relevant result), the pipeline still submits it to the judge and records a best-effort verdict, but that verdict is excluded from the per-article aggregation entirely (`article_number: null`) rather than folded into any article's rollup. This means EDPB interpretive guidance — despite being ingested and retrievable — never actually influences a reported compliance score today, and any signal it could add (EDPB guidance is often more specific and more current than the bare statutory text) is currently discarded rather than degraded gracefully.

### 7.7 Eval set size

Both the retrieval eval set's held-out split (28 items) and, pending §6, the end-to-end held-out policy set are small relative to the granularity of the breakdowns reported against them (by-topic, by-difficulty, by-gold-count). §3.3's held-out MRR swing is the concrete illustration: a difference plausibly attributable to 1–2 items flipping outcome. Architecture and configuration comparisons in this report should be read as directionally informative, not as statistically established at fine granularity, until the eval sets grow (§8).

## 8. Future work

- **Domain-tuned embeddings.** `rag/README.md` already documents the plumbing (`$RAG_EMBEDDING_MODEL`, fine-tuning `bge-large-en-v1.5` with `MultipleNegativesRankingLoss` on (statutory-wording, plain-English-paraphrase) pairs drawn from the eval set's own `train` split). This is the most direct lever against the 13 persistent retrieval misses (§3.3, §7.5), which look like an embedding-similarity ceiling rather than a fusion/reranking gap.
- **Grow and rebalance the retrieval eval set**, especially compound items (currently 6 of 114 train items) and the held-out split (28 items), to get statistically firmer architecture comparisons and shrink the noise band in §3.3 and §7.7.
- **Revisit `fetch_k` specifically for the reranker on hard queries** — a wider candidate pool (100) measurably hurts hard-difficulty MRR under reranking even though it doesn't hurt recall (§3.3); decoupling the fusion pool size from what's actually handed to the cross-encoder is a plausible fix worth testing with more data.
- **Extend `CONCEPT_LINKS` coverage**, either by hand or more systematically, since compound-item gains today are carried almost entirely by hybrid/BM25 hitting the current hand-curated set, and the compound sample (n=6) is currently too small to tell which additional article pairs would move the number.
- **Revisit the finer-chunking split logic for `transfers_appropriate_safeguards`**, the one topic where finer chunking measurably regressed dense retrieval (§3.1) — keeping those articles as single chunks, or adjusting the split heuristic, is worth testing now that hybrid/rerank on top of the current chunking already recovers most of what it lost.
- **Fill the nine GDPR schema gaps** identified in §4.2 / `reports/opp115_gdpr_coverage.md` with purpose-built, hand-labeled data (legal-basis granularity, special-category data, DPO designation, transfer mechanism, DPIA, privacy by design/default, automated decision-making, restriction of processing, breach-notification specifics, portability) rather than more OPP-115 volume, since OPP-115 structurally cannot express any of them.
- **Train and evaluate the production 7B judge** (`judge/config/qlora_judge.yaml`) on real GPU hardware and populate §4.4's evaluation table, replacing the currently-committed 0.5B smoke-test adapter as the default in the UI once validated.
- **Resolve the `non_compliant` vs. `not_addressed` scoring conflation** (§5.1, §7.5) once real sensitivity-analysis and end-to-end data (§5.2, §6) show how much it actually matters in practice — the fix is a one-line config change (`scoring/config/article_weights.yaml`'s `status_scores`), the open question is what value is defensible, which needs evidence rather than a guess.
- **Extend judge grounding to recitals and EDPB guidance** (§7.6) rather than discarding those verdicts from the article rollup, so retrieved interpretive guidance can actually influence a reported score.
- **Improve clause segmentation** (§7.1) beyond the current paragraph/sentence heuristic, particularly for hard-wrapped plain text and multi-column/table-heavy PDF layouts.
- **Multilingual support** (§7.4): a translated or multilingual GDPR/EDPB corpus, a multilingual embedding model, and non-English training/eval data would be required before any non-English policy could be assessed; this is a substantial scope expansion, not an incremental change, and should be scoped as its own project phase.
- **Other jurisdictions**, as a longer-horizon, explicitly separate track from GDPR-exclusivity (§1.2): the same RAG-over-statute-text + fine-tuned-judge architecture generalizes to another regulation (e.g. UK GDPR, CCPA/CPRA) only if paired with that regulation's own source corpus, article-level eval set, and mapping/training data — none of which transfer from the GDPR-specific artifacts built here.

---

*Companion documents: `reports/sota_review.md` (literature and commercial-market review), `reports/opp115_gdpr_coverage.md` (live OPP-115→GDPR coverage report), `eval/RAG_ANALYSIS.md` (full retrieval architecture comparison), `eval/README.md`, `rag/README.md`, `judge/README.md` (module-level documentation this report draws from).*
