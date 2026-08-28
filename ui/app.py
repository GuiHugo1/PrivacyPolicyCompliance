"""Streamlit UI for the GDPR privacy-policy compliance assessor.

Takes a privacy policy -- either a URL or an uploaded file -- runs it through
``judge.pipeline`` (clause segmentation, RAG retrieval, LLM-judge verdicts)
and ``scoring.score`` (per-article/chapter/overall compliance scoring), and
renders the result: an overall 0-100 score, a chapter-level bar chart, and
an expandable per-article breakdown with evidence and rationale. Both
modules are imported and called directly -- nothing here shells out to the
``judge.pipeline``/``scoring.score`` CLIs.

Run with::

    uv sync --group rag --group judge --group scoring --group ui
    streamlit run ui/app.py

The judge model and RAG index are expensive to load (a local LLM + LoRA
adapter, an embedding model, a cross-encoder reranker), so both are built
once per (settings) combination via ``st.cache_resource`` -- see
``get_retriever_context``/``get_judge_model`` below -- rather than reloaded
on every button click or widget interaction.
"""

from __future__ import annotations

import json
import tempfile
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from html import escape as esc
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st

from judge.pipeline import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_MAX_CLAUSE_CHARS,
    DEFAULT_PERSIST_DIR,
    JudgeModel,
    RetrieverContext,
    aggregate_articles,
    build_output,
    build_retriever_context,
    judge_clauses,
    known_articles,
    load_config,
    load_policy_text,
    segment_clauses,
)
from judge.schema_utils import DEFAULT_SCHEMA_PATH, load_schema
from scoring.score import DEFAULT_CONFIG_PATH as DEFAULT_SCORING_CONFIG_PATH
from scoring.score import ScoringConfig, score_judge_output

REPO_ROOT = Path(__file__).resolve().parent.parent
GDPR_SOURCE_PATH = REPO_ROOT / "data" / "raw" / "gdpr.json"
JUDGE_CONFIG_DIR = REPO_ROOT / "judge" / "config"
JUDGE_CHECKPOINTS_DIR = REPO_ROOT / "judge" / "checkpoints"

HTTP_TIMEOUT_S = 20
USER_AGENT = "Mozilla/5.0 (compatible; PrivacyPolicyComplianceAssessor/1.0)"
MIN_EXTRACTED_CHARS = 50

STATUS_STYLE: dict[str, tuple[str, str, str]] = {
    # status -> (label, text color, background color)
    "compliant": ("Compliant", "#1a7f37", "#e6f4ea"),
    "partial": ("Partial", "#9a6700", "#fff8e1"),
    "non_compliant": ("Non-compliant", "#cf222e", "#fdecea"),
    "not_addressed": ("Not addressed", "#57606a", "#eef1f4"),
    "not_applicable": ("Not applicable", "#57606a", "#eef1f4"),
    "needs_review": ("Needs review", "#8250df", "#f5eeff"),
}


class OperationTimeout(Exception):
    """Raised when a slow operation (model/index load, one clause's judge
    call) exceeds its configured budget -- see ``run_with_timeout``."""


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------


def run_with_timeout(fn, timeout_s: float, *args: Any, **kwargs: Any) -> Any:
    """Runs ``fn(*args, **kwargs)`` in a worker thread, raising
    ``OperationTimeout`` if it doesn't finish within ``timeout_s`` seconds.

    A synchronous model-generation call can't actually be cancelled
    mid-flight, so on timeout the worker thread is abandoned (the pool is
    shut down with ``wait=False``) rather than blocked on -- this keeps the
    UI responsive instead of hanging on the very timeout it's trying to
    enforce.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, *args, **kwargs)
    try:
        result = future.result(timeout=timeout_s)
    except FutureTimeoutError as e:
        pool.shutdown(wait=False)
        raise OperationTimeout(f"Timed out after {timeout_s:.0f}s") from e
    pool.shutdown(wait=False)
    return result


# ---------------------------------------------------------------------------
# Text extraction: URL fetch and file upload, both boiled down to plain text
# ---------------------------------------------------------------------------


def fetch_url_html(url: str) -> str:
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Please enter a full URL starting with http:// or https://.")
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT})
    except requests.exceptions.Timeout as e:
        raise ValueError(f"Fetching {url} timed out. Check the address and try again.") from e
    except requests.exceptions.ConnectionError as e:
        raise ValueError(
            f"Could not reach {url}. Check the address and your network connection."
        ) from e
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Could not fetch {url}: {e}") from e

    if response.status_code >= 400:
        raise ValueError(f"The server returned HTTP {response.status_code} for {url}.")
    response.encoding = response.encoding or "utf-8"
    return response.text


def extract_main_text(html: str, url: str | None = None) -> str:
    """Strips nav/footer/boilerplate from a raw HTML document, trying
    ``trafilatura`` first (generally the more accurate extractor) and
    falling back to ``readability-lxml`` + BeautifulSoup's plain text if
    trafilatura can't find a main content block."""
    text: str | None = None
    try:
        import trafilatura

        text = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=False, favor_recall=True
        )
    except Exception:
        text = None

    if not text or len(text.strip()) < MIN_EXTRACTED_CHARS:
        try:
            from bs4 import BeautifulSoup
            from readability import Document

            summary_html = Document(html).summary()
            text = BeautifulSoup(summary_html, "lxml").get_text("\n\n")
        except Exception:
            text = None

    if not text or len(text.strip()) < MIN_EXTRACTED_CHARS:
        raise ValueError(
            "Could not extract readable article text from this page -- it may not contain a "
            "text-based privacy policy, or the page requires JavaScript to render."
        )
    return text.strip()


def extract_text_from_upload(uploaded_file: Any) -> str:
    name = uploaded_file.name
    suffix = Path(name).suffix.lower()
    data = uploaded_file.getvalue()
    if not data:
        raise ValueError(f"{name} is empty.")

    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(data)
            tmp.flush()
            try:
                text = load_policy_text(tmp.name)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                # PDF-parsing native libraries (pypdf's optional crypto
                # backends included) occasionally fail with exceptions that
                # don't subclass Exception -- caught broadly here (this one
                # call site only) so a malformed/unusual PDF can't take the
                # whole app down.
                raise ValueError(
                    f"Could not extract text from {name} -- it may be a scanned/image-only "
                    "PDF or a corrupted file."
                ) from e
        if len(text.strip()) < MIN_EXTRACTED_CHARS:
            raise ValueError(
                f"Almost no extractable text was found in {name} -- it may be a "
                "scanned/image-only PDF (no embedded text layer)."
            )
        return text

    if suffix in (".html", ".htm"):
        try:
            html = data.decode("utf-8")
        except UnicodeDecodeError:
            html = data.decode("latin-1")
        return extract_main_text(html)

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("latin-1")
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Could not decode {name} as text. Please upload a UTF-8 plain-text file."
            ) from e


# ---------------------------------------------------------------------------
# Cached resource loading -- built once per (settings), not per interaction
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_retriever_context(
    persist_dir: str, collection_name: str, hybrid: bool, rerank: bool
) -> RetrieverContext:
    return build_retriever_context(
        persist_dir=persist_dir,
        collection_name=collection_name,
        hybrid=hybrid,
        rerank=rerank,
    )


@st.cache_resource(show_spinner=False)
def get_judge_model(config_path: str, adapter_path: str) -> JudgeModel:
    cfg = load_config(config_path)
    schema = load_schema(DEFAULT_SCHEMA_PATH)
    return JudgeModel(cfg, adapter_path, schema)


@st.cache_resource(show_spinner=False)
def get_scoring_config(path: str = str(DEFAULT_SCORING_CONFIG_PATH)) -> ScoringConfig:
    return ScoringConfig.load(path)


@st.cache_data(show_spinner=False)
def load_article_titles() -> dict[str, str]:
    """Best-effort ``{article_number: title}`` lookup from
    ``data/raw/gdpr.json``, used only to make the per-article section
    headings more readable. Not required -- degrades to no titles if the
    file isn't present in a given deployment."""
    try:
        raw = json.loads(GDPR_SOURCE_PATH.read_text(encoding="utf-8"))
        return {
            str(article["number"]): article["title"]
            for article in raw.get("articles", [])
            if article.get("number") and article.get("title")
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Pipeline orchestration with clause-by-clause progress
# ---------------------------------------------------------------------------


def run_full_assessment(
    policy_text: str, policy_source: str, settings: dict[str, Any], status, progress_bar
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Runs retrieval + judge + scoring end to end, reporting clause-by-clause
    progress via ``status``/``progress_bar``. Mirrors ``judge.pipeline.run_pipeline``
    exactly, but drives ``judge_clauses`` one clause at a time (instead of the
    whole list in one call) so progress can be reported, and wraps each
    clause's judge call in a timeout."""
    status.update(label="Segmenting policy text into clauses...")
    clauses = segment_clauses(policy_text, max_clause_chars=DEFAULT_MAX_CLAUSE_CHARS)
    if not clauses:
        raise ValueError("No text could be segmented from this document -- it may be empty.")

    status.update(label="Loading judge model and RAG index (first run may take a while)...")
    retriever_ctx = run_with_timeout(
        get_retriever_context,
        settings["load_timeout"],
        settings["persist_dir"],
        settings["collection_name"],
        settings["hybrid"],
        settings["rerank"],
    )
    judge_model = run_with_timeout(
        get_judge_model, settings["load_timeout"], settings["config_path"], settings["adapter_path"]
    )

    known = known_articles(retriever_ctx.collection)
    if not known:
        raise ValueError(
            f"The RAG index at '{settings['persist_dir']}' (collection "
            f"'{settings['collection_name']}') has no GDPR articles indexed. Build it with "
            "`python -m rag.build_index` (see rag/README.md), or point Settings at an "
            "existing built index."
        )

    total = len(clauses)
    clause_verdicts: list[dict[str, Any]] = []
    n_timed_out = 0
    for i, clause in enumerate(clauses, start=1):
        status.update(label=f"Judging clause {i} of {total}...")
        try:
            verdicts = run_with_timeout(
                judge_clauses,
                settings["clause_timeout"],
                [clause],
                retriever_ctx,
                judge_model,
                k=settings["k"],
            )
        except OperationTimeout:
            n_timed_out += 1
            verdicts = [
                {
                    "clause_id": clause.id,
                    "article": "timeout",
                    "article_number": None,
                    "source_type": "gdpr_article",
                    "chunk_id": "timeout",
                    "retrieval_rank": 1,
                    "retrieval_score": 0.0,
                    "requirement_present": False,
                    "compliance_status": "needs_review",
                    "evidence_span": "",
                    "rationale": (
                        f"Judge model timed out after {settings['clause_timeout']:.0f}s on "
                        "this clause."
                    ),
                    "confidence": 0.0,
                    "retry_used": True,
                    "error": "timeout",
                }
            ]
        clause_verdicts.extend(verdicts)
        progress_bar.progress(i / total)

    status.update(label="Aggregating results per GDPR article...")
    articles = aggregate_articles(clause_verdicts, known)
    judge_output = build_output(
        policy_source=policy_source,
        clauses=clauses,
        clause_verdicts=clause_verdicts,
        articles=articles,
        judge_config_path=settings["config_path"],
        adapter_path=settings["adapter_path"],
        retrieval_meta={
            "k": settings["k"],
            "hybrid": settings["hybrid"],
            "rerank": settings["rerank"],
            "fetch_k": retriever_ctx.fetch_k,
            "rerank_top_n": retriever_ctx.rerank_top_n if settings["rerank"] else None,
        },
    )

    status.update(label="Scoring compliance...")
    report = score_judge_output(judge_output, get_scoring_config())

    if n_timed_out:
        st.warning(
            f"{n_timed_out} of {total} clause(s) timed out during judging and were marked "
            "'needs review'."
        )

    return judge_output, report.to_dict()


# ---------------------------------------------------------------------------
# Error messages -- no raw stack traces in the primary UI
# ---------------------------------------------------------------------------


def humanize_error(exc: Exception) -> str:
    if isinstance(exc, OperationTimeout):
        return (
            f"{exc} -- the model or index took too long to respond. Try increasing the "
            "timeout in Settings, or check that the configured hardware (e.g. a GPU) is "
            "actually available."
        )
    if isinstance(exc, (requests.exceptions.MissingSchema, requests.exceptions.InvalidURL)):
        return "That doesn't look like a valid URL. Make sure it starts with http:// or https://."
    if isinstance(exc, requests.exceptions.RequestException):
        return f"There was a problem fetching that URL: {exc}"
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        missing = getattr(exc, "name", None) or "a required package"
        return (
            f"A required package ('{missing}') is not installed. Run "
            "`uv sync --group rag --group judge --group scoring --group ui` and restart the app."
        )
    if isinstance(exc, FileNotFoundError):
        return f"Could not find a required file: {exc.filename or exc}"
    if isinstance(exc, ValueError):
        return str(exc)
    return f"An unexpected error occurred while assessing this policy: {exc}"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def score_color(score: float | None) -> str:
    if score is None:
        return "#57606a"
    if score < 40:
        return "#cf222e"
    if score <= 70:
        return "#9a6700"
    return "#1a7f37"


def status_badge_html(status: str | None) -> str:
    label, fg, bg = STATUS_STYLE.get(status or "", (esc(status or "Unknown"), "#57606a", "#eef1f4"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;border-radius:12px;'
        f'font-size:0.85em;font-weight:600;white-space:nowrap;">{label}</span>'
    )


def make_gauge_figure(score: float | None) -> go.Figure:
    color = score_color(score)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score if score is not None else 0,
            number={"suffix": " / 100", "font": {"size": 44}, "valueformat": ".0f"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 40], "color": "#fdecea"},
                    {"range": [40, 70], "color": "#fff8e1"},
                    {"range": [70, 100], "color": "#e6f4ea"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin={"l": 30, "r": 30, "t": 30, "b": 10})
    return fig


def make_chapter_chart(chapters: list[dict[str, Any]]) -> go.Figure | None:
    rows = [c for c in chapters if c.get("articles") and c.get("score_pct") is not None]
    if not rows:
        return None
    rows = list(reversed(rows))  # top-to-bottom in chapter order
    names = [f"Ch. {c['id']} — {c['name']}" for c in rows]
    values = [c["score_pct"] for c in rows]
    colors = [score_color(v) for v in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}" for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        xaxis={"range": [0, 105], "title": "Score (0–100)"},
        height=70 + 45 * len(rows),
        margin={"l": 10, "r": 30, "t": 10, "b": 40},
    )
    return fig


def render_gauge_and_summary(report: dict[str, Any]) -> None:
    overall = report.get("overall_score")
    all_articles = [a for ch in report["chapters"] for a in ch["articles"]]
    counts = Counter(a["compliance_status"] for a in all_articles)

    col_gauge, col_stats = st.columns([1, 1])
    with col_gauge:
        st.plotly_chart(make_gauge_figure(overall), width="stretch")
        if overall is None:
            st.caption("Not enough in-scope, scorable articles to compute an overall score.")
    with col_stats:
        st.caption("Article findings, by compliance status")
        metric_cols = st.columns(2)
        labels = [
            ("compliant", "Compliant"),
            ("partial", "Partial"),
            ("non_compliant", "Non-compliant"),
            ("not_addressed", "Not addressed"),
        ]
        for i, (key, label) in enumerate(labels):
            metric_cols[i % 2].metric(label, counts.get(key, 0))


def render_chapter_chart(report: dict[str, Any]) -> None:
    st.subheader("Chapter-level breakdown")
    fig = make_chapter_chart(report["chapters"])
    if fig is None:
        st.caption("No in-scope chapter had a scorable article.")
        return
    st.plotly_chart(fig, width="stretch")
    st.caption("Red < 40 · Yellow 40–70 · Green > 70")


def render_article_sections(report: dict[str, Any], judge_output: dict[str, Any]) -> None:
    st.subheader("Article-level findings")
    titles = load_article_titles()
    clause_text_by_id = {c["id"]: c["text"] for c in judge_output.get("clauses", [])}

    for chapter in report["chapters"]:
        if not chapter["articles"]:
            continue
        scope_note = "" if chapter["in_scope"] else " (out of scope for scoring)"
        st.markdown(f"**Chapter {chapter['id']} — {chapter['name']}{scope_note}**")

        for article in chapter["articles"]:
            number = article["article"]
            title = titles.get(number)
            heading = f"Article {number}" + (f" — {title}" if title else "")
            n_clauses = len(article["clauses_addressing_it"])
            status = article["compliance_status"]

            with st.expander(heading, expanded=False):
                st.markdown(status_badge_html(status), unsafe_allow_html=True)
                if n_clauses:
                    st.caption(f"{n_clauses} policy clause(s) addressed this article.")
                else:
                    st.caption("No clause in the policy addressed this article.")

                if article.get("evidence"):
                    st.markdown("**Evidence**")
                    st.markdown(
                        '<div style="background:#fff8c5;border-left:4px solid #d4a72c;'
                        'padding:8px 12px;border-radius:4px;margin-bottom:8px;">'
                        f"{esc(article['evidence'])}</div>",
                        unsafe_allow_html=True,
                    )

                if article.get("rationale"):
                    st.markdown("**Rationale**")
                    st.write(article["rationale"])

                if n_clauses:
                    st.markdown("**Clauses checked against this article**")
                    table_rows = []
                    for c in article["clauses_addressing_it"]:
                        clause_text = clause_text_by_id.get(c["clause_id"], "")
                        table_rows.append(
                            {
                                "Clause": (
                                    clause_text[:160] + "..."
                                    if len(clause_text) > 160
                                    else clause_text
                                ),
                                "Status": c["compliance_status"],
                                "Confidence": round(c["confidence"], 2),
                            }
                        )
                    st.dataframe(table_rows, width="stretch", hide_index=True)

    if report.get("unmapped_articles"):
        with st.expander("Other findings (articles outside the configured chapters)"):
            for article in report["unmapped_articles"]:
                st.markdown(
                    f"Article {article['article']}: "
                    + status_badge_html(article["compliance_status"]),
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------


def build_pdf_report(
    policy_source: str, judge_output: dict[str, Any], report: dict[str, Any]
) -> bytes:
    import io

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    story: list[Any] = []

    story.append(Paragraph("GDPR Privacy Policy Compliance Report", styles["Title"]))
    story.append(Paragraph(f"Source: {esc(policy_source)}", styles["Normal"]))
    generated_at = (judge_output.get("meta") or {}).get("generated_at", "")
    story.append(Paragraph(f"Generated: {esc(generated_at)}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    overall = report.get("overall_score")
    overall_str = "N/A" if overall is None else f"{overall:.1f} / 100"
    story.append(Paragraph(f"Overall compliance score: {overall_str}", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * inch))

    chapter_rows = [["Chapter", "In scope", "Score"]]
    for chapter in report.get("chapters", []):
        if not chapter.get("articles"):
            continue
        score_str = "N/A" if chapter.get("score_pct") is None else f"{chapter['score_pct']:.1f}"
        chapter_rows.append(
            [
                f"Ch. {chapter['id']} - {chapter['name']}",
                "Yes" if chapter["in_scope"] else "No",
                score_str,
            ]
        )
    table = Table(chapter_rows, colWidths=[3.8 * inch, 1 * inch, 1 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f1f2")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Article-level findings", styles["Heading2"]))
    for chapter in report.get("chapters", []):
        for article in chapter.get("articles", []):
            status = (article.get("compliance_status") or "unknown").replace("_", " ").title()
            story.append(
                Paragraph(f"Article {esc(article['article'])} — {esc(status)}", styles["Heading4"])
            )
            if article.get("evidence"):
                story.append(
                    Paragraph(f"<i>Evidence:</i> {esc(article['evidence'])}", styles["Normal"])
                )
            if article.get("rationale"):
                story.append(
                    Paragraph(f"<i>Rationale:</i> {esc(article['rationale'])}", styles["Normal"])
                )
            story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def default_judge_config() -> Path | None:
    candidates = sorted(JUDGE_CONFIG_DIR.glob("qlora_judge*.yaml"))
    if not candidates:
        return None
    preferred = JUDGE_CONFIG_DIR / "qlora_judge_0.5b_gpu.yaml"
    return preferred if preferred in candidates else candidates[0]


def default_adapter_path() -> str:
    candidate = JUDGE_CHECKPOINTS_DIR / "qwen2.5-0.5b-gpu-qlora-judge"
    return str(candidate) if candidate.is_dir() else ""


def render_sidebar_settings() -> dict[str, Any]:
    st.sidebar.header("Model & retrieval settings")

    config_candidates = sorted(JUDGE_CONFIG_DIR.glob("qlora_judge*.yaml"))
    default_config = default_judge_config()
    if config_candidates:
        config_path = st.sidebar.selectbox(
            "Judge config",
            options=[str(p) for p in config_candidates],
            index=config_candidates.index(default_config) if default_config else 0,
            help="Picks the base model + generation settings (judge/config/qlora_judge*.yaml).",
        )
    else:
        config_path = st.sidebar.text_input(
            "Judge config path", value="judge/config/qlora_judge.yaml"
        )

    adapter_path = st.sidebar.text_input(
        "LoRA adapter directory",
        value=default_adapter_path(),
        help="Trained adapter matching the judge config above.",
    )
    persist_dir = st.sidebar.text_input("RAG index directory", value=str(DEFAULT_PERSIST_DIR))
    collection_name = st.sidebar.text_input("Chroma collection name", value=DEFAULT_COLLECTION_NAME)
    k = st.sidebar.number_input("Top-k references per clause", min_value=1, max_value=10, value=3)
    hybrid = st.sidebar.checkbox("Hybrid retrieval (dense + BM25)", value=True)
    rerank = st.sidebar.checkbox("Cross-encoder reranking", value=True)

    with st.sidebar.expander("Advanced"):
        clause_timeout = st.number_input(
            "Per-clause judge timeout (s)", min_value=5, max_value=600, value=90
        )
        load_timeout = st.number_input(
            "Model/index load timeout (s)", min_value=10, max_value=1800, value=300
        )

    return {
        "config_path": config_path,
        "adapter_path": adapter_path,
        "persist_dir": persist_dir,
        "collection_name": collection_name,
        "k": int(k),
        "hybrid": hybrid,
        "rerank": rerank,
        "clause_timeout": float(clause_timeout),
        "load_timeout": float(load_timeout),
    }


def main() -> None:
    st.set_page_config(
        page_title="GDPR Privacy Policy Compliance Assessor", page_icon="🛡️", layout="wide"
    )
    st.title("GDPR Privacy Policy Compliance Assessor")
    st.caption(
        "Retrieves relevant GDPR articles for each clause of a privacy policy, judges "
        "compliance with a fine-tuned LLM, and scores the result per GDPR chapter."
    )

    settings = render_sidebar_settings()

    input_method = st.radio(
        "Input", ["Paste a URL", "Upload a file (.txt / .pdf / .html)"], horizontal=True
    )
    url = ""
    uploaded_file = None
    if input_method == "Paste a URL":
        url = st.text_input("Privacy policy URL", placeholder="https://example.com/privacy")
    else:
        uploaded_file = st.file_uploader(
            "Upload a privacy policy", type=["txt", "pdf", "html", "htm"]
        )

    run_clicked = st.button("Run assessment", type="primary")

    if run_clicked:
        st.session_state.pop("result", None)
        st.session_state.pop("error", None)

        try:
            with st.spinner("Reading input..."):
                if input_method == "Paste a URL":
                    if not url.strip():
                        raise ValueError("Enter a URL first.")
                    policy_text = extract_main_text(fetch_url_html(url), url=url)
                    policy_source = url.strip()
                else:
                    if uploaded_file is None:
                        raise ValueError("Upload a file first.")
                    policy_text = extract_text_from_upload(uploaded_file)
                    policy_source = uploaded_file.name
        except Exception as e:
            st.session_state.error = humanize_error(e)
            st.session_state.traceback = traceback.format_exc()
        else:
            st.caption(f"Extracted {len(policy_text):,} characters from:")
            st.text(policy_source)
            with st.expander("Preview extracted text"):
                st.text(policy_text[:3000] + ("..." if len(policy_text) > 3000 else ""))

            with st.status("Running assessment...", expanded=True) as status:
                progress_bar = st.progress(0.0)
                try:
                    judge_output, report_dict = run_full_assessment(
                        policy_text, policy_source, settings, status, progress_bar
                    )
                except Exception as e:
                    status.update(label="Assessment failed", state="error")
                    st.session_state.error = humanize_error(e)
                    st.session_state.traceback = traceback.format_exc()
                else:
                    status.update(label="Assessment complete", state="complete")
                    st.session_state.result = {
                        "judge_output": judge_output,
                        "report": report_dict,
                        "policy_source": policy_source,
                    }

    if st.session_state.get("error"):
        st.error(st.session_state["error"])
        with st.expander("Technical details"):
            st.code(st.session_state.get("traceback", ""))

    result = st.session_state.get("result")
    if result:
        st.divider()
        st.subheader("Results")
        st.text(result["policy_source"])
        report = result["report"]
        judge_output = result["judge_output"]

        render_gauge_and_summary(report)
        render_chapter_chart(report)
        render_article_sections(report, judge_output)

        st.divider()
        col_json, col_pdf = st.columns(2)
        with col_json:
            full_report = {"judge_output": judge_output, "compliance_report": report}
            st.download_button(
                "📥 Download full report (JSON)",
                data=json.dumps(full_report, indent=2, ensure_ascii=False),
                file_name="compliance_report.json",
                mime="application/json",
                width="stretch",
            )
        with col_pdf:
            pdf_bytes = build_pdf_report(result["policy_source"], judge_output, report)
            st.download_button(
                "📥 Download full report (PDF)",
                data=pdf_bytes,
                file_name="compliance_report.pdf",
                mime="application/pdf",
                width="stretch",
            )


if __name__ == "__main__":
    main()
