"""Tests for the dependency-free parts of ui/app.py: text extraction, error
message translation, score/status styling, chart building, and PDF report
generation. The real judge/RAG pipeline (needs chromadb/torch installed and
a built index/trained adapter) and Streamlit widget wiring (needs a live
script run) aren't exercised here -- see ui/app.py's module docstring for
how to run the app itself, and its "Running the UI" section in README.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import requests

from scoring.score import ScoringConfig, score_judge_output
from ui.app import (
    OperationTimeout,
    build_pdf_report,
    extract_main_text,
    extract_text_from_upload,
    humanize_error,
    make_chapter_chart,
    make_gauge_figure,
    run_with_timeout,
    score_color,
    status_badge_html,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_OUTPUT_PATH = REPO_ROOT / "judge" / "examples" / "sample_output.json"


class _FakeUploadedFile:
    """Duck-types Streamlit's UploadedFile just enough for
    extract_text_from_upload (``.name``/``.getvalue()``)."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


# ---------------------------------------------------------------------------
# run_with_timeout
# ---------------------------------------------------------------------------


def test_run_with_timeout_returns_result():
    assert run_with_timeout(lambda x: x + 1, 5, 41) == 42


def test_run_with_timeout_raises_operation_timeout_and_does_not_block():
    start = time.monotonic()
    with pytest.raises(OperationTimeout):
        run_with_timeout(time.sleep, 0.05, 5)
    # Must return promptly once the budget is up, not wait for the
    # abandoned worker thread's full 5s sleep to finish.
    assert time.monotonic() - start < 2


# ---------------------------------------------------------------------------
# HTML/URL text extraction
# ---------------------------------------------------------------------------

_SAMPLE_HTML = """
<html><body>
<nav>Home | About | Contact</nav>
<article>
<h1>Privacy Policy</h1>
<p>We collect your name and email address to operate your account and will
never sell it to third parties without your explicit consent under GDPR.</p>
<p>You may request access to, correction of, or deletion of your personal
data at any time by contacting our data protection officer.</p>
</article>
<footer>Copyright 2026 Example Co. | Terms | Sitemap</footer>
</body></html>
"""


def test_extract_main_text_strips_nav_and_footer_boilerplate():
    text = extract_main_text(_SAMPLE_HTML)
    assert "Privacy Policy" in text
    assert "data protection officer" in text
    assert "Home | About | Contact" not in text
    assert "Sitemap" not in text


def test_extract_main_text_raises_friendly_error_on_empty_page():
    with pytest.raises(ValueError, match="readable article text"):
        extract_main_text("<html><body></body></html>")


# ---------------------------------------------------------------------------
# File upload extraction
# ---------------------------------------------------------------------------


def test_extract_text_from_upload_plain_txt():
    upload = _FakeUploadedFile("policy.txt", b"Hello world.")
    assert extract_text_from_upload(upload) == "Hello world."


def test_extract_text_from_upload_html_strips_boilerplate():
    upload = _FakeUploadedFile("policy.html", _SAMPLE_HTML.encode())
    text = extract_text_from_upload(upload)
    assert "Privacy Policy" in text
    assert "Sitemap" not in text


def test_extract_text_from_upload_empty_file_raises_friendly_error():
    upload = _FakeUploadedFile("policy.txt", b"")
    with pytest.raises(ValueError, match="empty"):
        extract_text_from_upload(upload)


# ---------------------------------------------------------------------------
# Error message translation
# ---------------------------------------------------------------------------


def test_humanize_error_operation_timeout_mentions_timeout():
    message = humanize_error(OperationTimeout("Timed out after 90s"))
    assert "took too long" in message


def test_humanize_error_missing_url_schema():
    message = humanize_error(requests.exceptions.MissingSchema("no scheme"))
    assert "valid URL" in message


def test_humanize_error_missing_package_names_it_and_suggests_uv_sync():
    try:
        import definitely_not_a_real_module  # noqa: F401
    except ImportError as e:
        message = humanize_error(e)
    assert "definitely_not_a_real_module" in message
    assert "uv sync" in message


def test_humanize_error_value_error_passes_through_verbatim():
    assert humanize_error(ValueError("custom message")) == "custom message"


# ---------------------------------------------------------------------------
# Score/status styling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,expected_color",
    [
        (0, "#cf222e"),
        (39.9, "#cf222e"),
        (40, "#9a6700"),
        (70, "#9a6700"),
        (70.1, "#1a7f37"),
        (100, "#1a7f37"),
    ],
)
def test_score_color_thresholds(score, expected_color):
    assert score_color(score) == expected_color


def test_score_color_none_is_neutral_gray():
    assert score_color(None) == "#57606a"


def test_status_badge_html_known_status_uses_its_style():
    html = status_badge_html("compliant")
    assert "Compliant" in html
    assert "#1a7f37" in html


def test_status_badge_html_unknown_status_falls_back_without_crashing():
    html = status_badge_html("something_new")
    assert "something_new" in html


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def test_make_gauge_figure_carries_the_score_value():
    fig = make_gauge_figure(82.5)
    assert fig.data[0].value == 82.5


def test_make_chapter_chart_skips_chapters_with_no_scorable_articles():
    chapters = [
        {"id": "I", "name": "General", "articles": [], "score_pct": None},
        {"id": "III", "name": "Rights", "articles": [{"article": "13"}], "score_pct": 75.0},
    ]
    fig = make_chapter_chart(chapters)
    assert fig is not None
    assert list(fig.data[0].x) == [75.0]


def test_make_chapter_chart_returns_none_when_nothing_scorable():
    chapters = [{"id": "I", "name": "General", "articles": [], "score_pct": None}]
    assert make_chapter_chart(chapters) is None


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------


def test_build_pdf_report_produces_valid_pdf_bytes():
    judge_output = json.loads(SAMPLE_OUTPUT_PATH.read_text(encoding="utf-8"))
    report = score_judge_output(judge_output, ScoringConfig.load()).to_dict()

    pdf_bytes = build_pdf_report(judge_output["policy"]["source"], judge_output, report)

    assert pdf_bytes.startswith(b"%PDF")


def test_build_pdf_report_escapes_special_characters_in_evidence():
    judge_output = json.loads(SAMPLE_OUTPUT_PATH.read_text(encoding="utf-8"))
    report = score_judge_output(judge_output, ScoringConfig.load()).to_dict()
    chapter_with_articles = next(c for c in report["chapters"] if c["articles"])
    chapter_with_articles["articles"][0]["evidence"] = "We share data with <Acme> & partners"

    # Must not raise -- reportlab's Paragraph parser would choke on
    # unescaped '<'/'&' in the evidence text.
    pdf_bytes = build_pdf_report(judge_output["policy"]["source"], judge_output, report)
    assert pdf_bytes.startswith(b"%PDF")
