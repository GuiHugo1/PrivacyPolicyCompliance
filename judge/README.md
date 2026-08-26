# judge

Fine-tuning data and (eventually) inference code for the GDPR compliance-judge
LLM. The first piece here is a pipeline that turns OPP-115 (Wilson et al.
2016) privacy-policy annotations into supervised fine-tuning (SFT) examples
for a judge that, given `(clause, retrieved GDPR article text)`, outputs a
structured compliance verdict.

## Why this is weak supervision, not ground truth

OPP-115 predates GDPR and records **whether a privacy practice is
disclosed**, not whether that disclosure satisfies a specific GDPR article --
see `eval/README.md`'s "Why hand-labeled, not scraped" note, which makes the
same point about why the RAG retrieval eval set had to be hand-labeled from
scratch instead of reusing OPP-115. Every example this pipeline generates
carries `meta.weak_label: true` for exactly that reason. Treat the generated
dataset as a bootstrap to review and refine, not a finished training set --
see "Coverage gaps" below for where it's weakest.

## Layout

- `config/opp115_gdpr_mapping.yaml` -- editable mapping from OPP-115's 10
  categories (and their attribute value vocabulary) to GDPR articles, plus
  heuristic `compliance_status`/`confidence` rules and the static list of
  GDPR requirements OPP-115 cannot express at all (`gdpr_schema_gaps`).
  **Review this file before generating data** -- see the header comment
  inside it for what to check.
- `mapping.py` -- loads/validates the YAML config and resolves a heuristic
  rule for one (category, attributes) pair.
- `opp115.py` -- loads OPP-115 annotation CSVs, collapses multiple
  annotators' independent labeling of the same practice via majority vote,
  and best-effort reconstructs segment text from annotation highlight spans.
- `gdpr_source.py` -- loads full GDPR article text from the same
  `data/raw/gdpr.json`/`.xml` source `rag.build_index` indexes (see
  `rag/README.md`'s "GDPR source format"), so grounding text can't drift out
  of sync with what the RAG pipeline actually retrieves at inference time.
- `build_sft_dataset.py` -- CLI: OPP-115 annotations + mapping + GDPR source
  -> chat-format SFT JSONL train/val/test splits.
- `coverage_report.py` -- CLI: flags where OPP-115 coverage is thin for
  GDPR-specific requirements, both from data volume and from the mapping's
  documented schema gaps.

## Usage

```bash
uv sync --group judge

python -m judge.build_sft_dataset \
    --opp115-dir /data/opp115/annotations \
    --gdpr data/raw/gdpr.json \
    --out-dir data/processed/judge_sft

python -m judge.coverage_report \
    --opp115-dir /data/opp115/annotations \
    --markdown-output reports/opp115_gdpr_coverage.md
```

One SFT example is generated per (policy, segment, category, primary GDPR
article) -- see the module docstring in `build_sft_dataset.py` for exactly
how attribute values are collapsed across annotators and how an OPP-115
category's multiple primary articles are (or aren't) all grounded from the
same clause.

## Alignment with the RAG pipeline

Grounding text defaults to the mapped article's **canonical** text (gold
grounding), not retrieval output, so the judge trains against correct
context rather than the RAG pipeline's own retrieval noise. Pass
`--use-retriever` to instead run each clause through the real
`rag.retriever.retrieve` and use its top-hit chunk -- per
`eval/RAG_ANALYSIS.md`'s conclusion, this project's recommended retrieval
configuration is `hybrid=True, rerank=True` (best joint recall across
single- and compound-article clauses), so that's what `--use-retriever`
uses. That mode needs `uv sync --group rag` and a pre-built index (see
`rag/README.md`), and will bake in whatever retrieval mistakes
`eval/RAG_ANALYSIS.md` documents -- notably that reranking alone does
nothing for compound (multi-article) clauses, and that 13 hard/paraphrased
eval items are never retrieved correctly by any tested configuration. It's
best used to add a *minority* of imperfect-retrieval examples (teaching the
judge to say `not_applicable`/low-confidence when the retrieved article
doesn't actually match the clause), not as the primary grounding source.

`eval/README.md` and this module intentionally keep retrieval-quality
metrics (`eval/benchmarks/`) and judge-verdict data/metrics (`judge/`)
separate, for the same reason `eval/scripts/retrieval_metrics.py` never
blends "found the right article" and "judged the clause correctly" into one
score: they answer different questions.

## Coverage gaps

Run `python -m judge.coverage_report` for a live report. Two kinds of gap
are called out separately (see `coverage_report.py`'s module docstring):

- **Schema-level gaps** -- GDPR requirements no OPP-115 category/attribute
  records at all, regardless of volume: legal basis granularity (which
  Art 6(1)(a)-(f) applies), DPO designation (Art 37-39), the specific
  cross-border transfer mechanism (Art 44-49: adequacy/SCCs/BCRs), DPIA
  (Art 35), privacy by design/default (Art 25), automated decision-making
  (Art 22), restriction of processing (Art 18), breach-notification
  specifics (Art 33-34), and portability (Art 20). These need genuinely new
  labeled data (hand-labeled against the specific requirement, the way
  `eval/benchmarks/gdpr_retrieval_eval_set.jsonl` is hand-labeled against
  article text), not more OPP-115 annotations.
- **Data-volume thinness** -- an OPP-115 signal that *could* ground a
  requirement but has too few examples in the corpus actually loaded. This
  is corpus-dependent, so it's computed live rather than hardcoded; see
  `coverage_report.py`'s `THIN_EXAMPLE_THRESHOLD`/`THIN_VALUE_THRESHOLD`.

`Do Not Track` and `Policy Change` are additionally flagged
`coverage: weak_alignment` in the mapping config -- GDPR simply has no
article that natively governs either (DNT is ePrivacy Directive territory;
"policy change" is approximated via the Art 13(3) further-processing notice
duty), so their generated examples should be treated as lower-confidence
even where data volume is adequate.

## Tests

```bash
pytest judge/tests
```

Uses a small hand-built OPP-115 fixture (`judge/tests/fixtures/`, two
policies, multiple annotators per segment to exercise majority voting) and
a 16-article GDPR fixture, so tests run fast and don't need the real
`/data/opp115` corpus.
