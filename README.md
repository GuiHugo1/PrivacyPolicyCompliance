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
  an open-source Qwen model.
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
