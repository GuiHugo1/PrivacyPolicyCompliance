# Privacy Policy Compliance

GDPR privacy-policy compliance assessment project. Given a company's privacy
policy, the system retrieves relevant GDPR articles and EDPB guidance,
judges the policy's compliance using a fine-tuned open-source LLM, and
produces a structured compliance score and report.

> This README is a placeholder — architecture details will be filled in as
> the project develops.

## Architecture (high level)

```
policy text --> [rag] --> relevant GDPR/EDPB context
                              |
                              v
                         [judge] (Qwen, fine-tuned)
                              |
                              v
                         [scoring] --> compliance score + findings
                              |
                              v
                          [ui] / [reports]
```

- **rag/** — retrieval pipeline over GDPR articles and EDPB guidelines;
  indexes source text and serves relevant context for a given policy.
- **judge/** — fine-tuning and inference code for the LLM judge, built on
  an open-source Qwen model. See `judge/README.md`'s "End-to-end judge
  pipeline" section to run the judge on a privacy policy (PDF or text) and
  "Try it out" for a runnable example that needs no trained model or GPU.
- **scoring/** — compliance scoring engine that turns judge output into
  structured scores and findings.
- **ui/** — user-facing app (Streamlit, or FastAPI backend + React
  frontend) for submitting policies and viewing results.
- **reports/** — LaTeX/Markdown sources for the technical report and
  white paper.
- **data/** — raw and processed datasets (GDPR text, OPP-115, scraped
  privacy policies).
- **eval/** — evaluation scripts and benchmark results.

## Development

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/) using
dependency groups defined in `pyproject.toml`: `rag`, `judge`, `scoring`,
`ui`, `dev`.

```bash
uv sync --group dev
pre-commit install
```

## Running the UI

`ui/app.py` is a Streamlit app: paste a privacy policy URL or upload a
`.txt`/`.pdf`/`.html` file, run it through the `judge`/`scoring` pipeline, and
view the resulting compliance score, chapter breakdown, and per-article
findings (with JSON/PDF report downloads).

```bash
uv sync --group rag --group judge --group scoring --group ui
streamlit run ui/app.py
```

Running a real assessment additionally needs a built RAG index and a trained
judge adapter -- see `rag/README.md` and `judge/README.md`'s "QLoRA
fine-tuning" section (the committed `qwen2.5-0.5b-gpu-qlora-judge` checkpoint
plus `judge/config/qlora_judge_0.5b_gpu.yaml`, both selected by default in the
app's sidebar, are enough to exercise it end to end on a CUDA GPU). Both are
configurable from the sidebar without editing code.
