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

## QLoRA fine-tuning

Once SFT JSONL splits exist (either from `build_sft_dataset.py` above, or
any dataset following the schema below), `judge/train_qlora.py` QLoRA
fine-tunes `Qwen2.5-7B-Instruct` on them and `judge/eval_qlora.py` reports
held-out test-set metrics for the resulting adapter.

- `judge/judge_schema.json` -- JSON Schema for the verdict object the judge
  must emit (`article`, `requirement_present`, `compliance_status`,
  `evidence_span`, `rationale`, `confidence`); mirrors `JUDGE_SYSTEM_PROMPT`
  in `build_sft_dataset.py`.
- `judge/schema_utils.py` -- dependency-free JSON extraction + schema
  validation for model output (no `jsonschema` package needed for this
  flat schema).
- `judge/eval_metrics.py` -- JSON-validity rate and per-`compliance_status`
  precision/recall/F1, shared by the training-time eval callback and
  `eval_qlora.py`.
- `judge/metrics_logger.py` -- appends metrics to a local JSONL file every
  eval step; optionally also logs to a local `mlflow` run
  (`file:judge/mlruns`, no tracking server required).
- `judge/qlora_data.py` -- loads SFT JSONL records (accepts either this
  repo's own `messages`-chat-format records, or a plain
  `instruction`/`input`/`output` shape), and `JudgeSFTCollator`, a data
  collator that tokenizes prompt + response separately and masks the
  prompt tokens to `-100` so the loss is computed only on the JSON-verdict
  tokens.
- `judge/train_qlora.py` -- config-driven (`judge/config/qlora_judge.yaml`)
  QLoRA training script: 4-bit `bitsandbytes` quantization, `peft` LoRA
  adapter, HF `Trainer`. Only the LoRA adapter is saved
  (`PeftModel.save_pretrained`), not a merged model. `resolve_device_backend`
  resolves the config's `device.backend` (`auto`/`cuda`/`cpu`) to a concrete
  device -- see "Choosing a device backend" below -- and gates
  quantization/bf16/fp16/`dataloader_pin_memory` on it.
- `judge/eval_qlora.py` -- loads a saved adapter, generates verdicts for
  the test split, and writes a JSON report of per-class precision/recall/F1
  + JSON-validity rate.

### Expected SFT data

```
/data/judge_sft/train.jsonl
/data/judge_sft/val.jsonl
/data/judge_sft/test.jsonl
```

Each line is either:

```jsonc
// this repo's own build_sft_dataset.py output
{"messages": [{"role": "system", ...}, {"role": "user", "content": "Clause:\n...\n\nRetrieved GDPR Article ...:\n..."}, {"role": "assistant", "content": "<json verdict>"}]}
// or the plain instruction/output shape
{"instruction": "...", "input": "...", "output": "<json verdict>"}
```

where `<json verdict>` matches `judge/judge_schema.json`.

### Usage

```bash
uv sync --group judge
```

`judge/train_qlora.py` and `judge/eval_qlora.py` always take `--config`, so
which config you pass picks both the base model and the hardware path. Every
possibility:

**Full judge model, CUDA GPU (`judge/config/qlora_judge.yaml`).** The real
target: `Qwen2.5-7B-Instruct`, 4-bit QLoRA, bf16/fp16. Needs a real GPU
(e.g. an A100/H100, or multi-GPU) -- see "Choosing a device backend" below.

```bash
python -m judge.train_qlora --config judge/config/qlora_judge.yaml

python -m judge.eval_qlora \
    --config judge/config/qlora_judge.yaml \
    --adapter judge/checkpoints/qwen2.5-7b-qlora-judge \
    --output judge/metrics/test_eval_report.json
```

**Fast GPU smoke test (`judge/config/qlora_judge_0.5b_gpu.yaml`).** Same
QLoRA pipeline as above but on `Qwen2.5-0.5B-Instruct`, so a training run
completes in minutes instead of hours -- use this to validate the pipeline
(data, collator, LoRA wiring, eval callback) end-to-end, or to iterate on
config/data changes quickly, before committing to a full 7B run. Works on
any CUDA GPU, including small/free ones (e.g. a Colab T4 -- see "Training on
Google Colab" below). Not a substitute for the 7B judge; bigger base models
need bigger hardware, per the table below.

```bash
python -m judge.train_qlora --config judge/config/qlora_judge_0.5b_gpu.yaml

python -m judge.eval_qlora \
    --config judge/config/qlora_judge_0.5b_gpu.yaml \
    --adapter judge/checkpoints/qwen2.5-0.5b-gpu-qlora-judge \
    --output judge/metrics/test_eval_report_0.5b_gpu.json
```

**CPU-only (`judge/config/qlora_judge_cpu.yaml`).** No GPU at all --
`Qwen2.5-0.5B-Instruct` at full precision. Slower than the GPU smoke test
above but needs no CUDA hardware; see "CPU training (limited)" below.

```bash
python -m judge.train_qlora --config judge/config/qlora_judge_cpu.yaml

python -m judge.eval_qlora \
    --config judge/config/qlora_judge_cpu.yaml \
    --adapter judge/checkpoints/qwen2.5-0.5b-cpu-lora-judge \
    --output judge/metrics/test_eval_report_cpu.json
```

All hyperparameters -- LoRA rank/alpha/target modules, quantization,
learning rate, epochs, batch size, gradient accumulation -- live in each
config's YAML file; edit that file rather than the scripts to sweep them.
Training metrics (train/val loss + JSON-validity rate) are appended to
`judge/metrics/<config's metrics.metrics_file>` every eval step.

### Choosing a device backend

`train_qlora.py`/`eval_qlora.py` pick a compute device from the config's
`device.backend` key rather than assuming CUDA:

| `device.backend` | Hardware | Quantization / bf16-fp16 | Config |
| --- | --- | --- | --- |
| `auto` (default) | tries CUDA, then CPU | depends on what's found | `qlora_judge.yaml` |
| `cuda` | NVIDIA GPU | 4-bit QLoRA + bf16/fp16 | `qlora_judge.yaml` (7B), `qlora_judge_0.5b_gpu.yaml` (0.5B, fast test) |
| `cpu` | no GPU | neither (full precision, `adamw_torch`) | `qlora_judge_cpu.yaml` |

An explicitly requested backend that isn't actually available fails fast
with a clear error (e.g. `backend: cuda` on a machine with no CUDA GPU),
rather than silently training on the wrong device.

`TrainingArguments.dataloader_pin_memory` is only enabled for the `cuda`
backend, since pinned host memory only speeds up transfers to a CUDA device
and does nothing on CPU (this also avoids PyTorch's `UserWarning:
'pin_memory' argument is set as true but no accelerator is found`).

### Training on Google Colab

`judge/config/qlora_judge_0.5b_gpu.yaml` (see above) is sized to fit on a
free Colab GPU runtime (e.g. a T4), so you don't need local GPU hardware to
smoke-test the training pipeline:

1. Open a new Colab notebook and set **Runtime -> Change runtime type ->
   T4 GPU**.
2. Clone this repo and install the judge dependency group:

   ```bash
   !git clone <this-repo-url>
   %cd PrivacyPolicyCompliance
   !pip install uv && uv sync --group judge
   ```

3. Upload or generate the SFT JSONL splits (`judge/build_sft_dataset.py`,
   see "Usage" above) under `data/processed/judge_sft/`, matching the paths
   in `qlora_judge_0.5b_gpu.yaml`'s `data` section (or edit that section to
   point at wherever you put them in the Colab filesystem/Drive).
4. Run training:

   ```bash
   !uv run python -m judge.train_qlora --config judge/config/qlora_judge_0.5b_gpu.yaml
   ```

5. Download the resulting adapter from `judge/checkpoints/qwen2.5-0.5b-gpu-qlora-judge`
   (or `%cp` it to a mounted Google Drive) before the runtime recycles --
   Colab storage is ephemeral.

This is still the 0.5B smoke-test path, not the final judge: once real GPU
hardware (an A100/H100, or a paid Colab/Kaggle tier with enough VRAM) is
available, switch to `qlora_judge.yaml` for the actual `Qwen2.5-7B-Instruct`
judge model.

### CPU training (limited)

`judge/config/qlora_judge_cpu.yaml` runs the same `train_qlora.py`/
`eval_qlora.py` scripts with no GPU:

```bash
python -m judge.train_qlora --config judge/config/qlora_judge_cpu.yaml
```

It sets `quantization.enabled: false` (bitsandbytes 4-bit quantization
needs a CUDA GPU, so `build_bnb_config` returns `None` and the base model
loads at full precision instead) and swaps the 7B base model for
`Qwen/Qwen2.5-0.5B-Instruct`, small enough to train on CPU in reasonable
time. `bf16`/`fp16` are both disabled (`resolve_mixed_precision` falls
back to plain fp32 whenever no CUDA device is present) and `optim` is
`adamw_torch` instead of `paged_adamw_8bit`.

This is a documented limitation of the current architecture, not a
drop-in replacement: the 0.5B model is far weaker than the 7B QLoRA judge
and this path exists to smoke-test the training pipeline (data loading,
collator, LoRA wiring, eval callback) on hardware without a GPU. Train on
`judge/config/qlora_judge.yaml` with a GPU for an actual judge model --
swapping `model.base_model` to a larger checkpoint and re-enabling
`quantization.enabled` is the way back to that once GPU hardware is
available.

Note: `judge/tests/test_qlora_data.py`, `test_schema_utils.py`,
`test_eval_metrics.py`, and `test_metrics_logger.py` are dependency-free
and always run with `pytest judge/tests`. `test_eval_qlora_report.py`
additionally exercises `eval_qlora.build_test_report` and is skipped
unless `uv sync --group judge` has installed `torch`. The training and
model-loading code paths in `train_qlora.py`/`eval_qlora.py` themselves
need the actual base-model weights (and, for the default GPU config, a
GPU), so they aren't covered by `pytest` -- validate them by running an
actual (even tiny/smoke) training job, e.g. against the CPU config above.
