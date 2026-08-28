"""Sensitivity analysis for ``scoring/score.py``'s compliance scoring.

Recomputes the same judge-pipeline output's overall (and per-chapter) score
under N different ``ScoringConfig`` weight configurations and renders a
comparison table -- for the technical report's "how much does the scoring
methodology's choices actually move the number" section.

By default this compares a fixed set of built-in preset variants derived
from ``config/article_weights.yaml`` (chapter-weight tilts, and both
directions of the ``non_compliant`` vs. ``not_addressed`` split
``scoring/score.py``'s module docstring documents as a one-line config
change -- this script is exactly what demonstrates that in practice).
Pass ``--configs`` with your own list of ``article_weights.yaml``-shaped
files to compare specific configurations instead.

Usage::

    python -m scoring.sensitivity_analysis --input judge/examples/sample_output.json
    python -m scoring.sensitivity_analysis --input path/to/output.json \\
        --configs cfg_a.yaml cfg_b.yaml cfg_c.yaml --format csv --output comparison.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from scoring.score import (
    DEFAULT_CONFIG_PATH,
    ComplianceReport,
    ScoringConfig,
    load_judge_output,
    score_judge_output,
)


def _pct(score: float | None) -> float | None:
    return None if score is None else round(score * 100, 2)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


# ---------------------------------------------------------------------------
# Built-in preset variants
# ---------------------------------------------------------------------------


def _scale_chapter_weight(raw: dict[str, Any], chapter_id: str, factor: float) -> dict[str, Any]:
    raw = copy.deepcopy(raw)
    for chapter in raw.get("chapters", []):
        if str(chapter["id"]) == chapter_id:
            chapter["weight"] = float(chapter.get("weight", 1.0)) * factor
    return raw


def _set_status_score(raw: dict[str, Any], status: str, value: float) -> dict[str, Any]:
    raw = copy.deepcopy(raw)
    raw.setdefault("status_scores", {})[status] = value
    return raw


def default_presets(base_raw: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Named ``(name, description, raw_config_dict)`` variants built by
    mutating a copy of the base config's already-loaded raw YAML.
    """
    return [
        ("baseline", "Default config, unmodified.", copy.deepcopy(base_raw)),
        (
            "principles_heavy",
            "Ch. II (Principles) chapter weight x3.",
            _scale_chapter_weight(base_raw, "II", 3.0),
        ),
        (
            "rights_heavy",
            "Ch. III (Data subject rights) chapter weight x3.",
            _scale_chapter_weight(base_raw, "III", 3.0),
        ),
        (
            "transfers_heavy",
            "Ch. V (International transfers) chapter weight x3.",
            _scale_chapter_weight(base_raw, "V", 3.0),
        ),
        (
            "strict_partial_credit",
            "'partial' scored 0.0 instead of 0.5.",
            _set_status_score(base_raw, "partial", 0.0),
        ),
        (
            "lenient_partial_credit",
            "'partial' scored 0.75 instead of 0.5.",
            _set_status_score(base_raw, "partial", 0.75),
        ),
        (
            "silence_penalized_less",
            "'not_addressed' scored 0.25 (silence weighted below a confirmed violation).",
            _set_status_score(base_raw, "not_addressed", 0.25),
        ),
        (
            "silence_penalized_more",
            "'non_compliant' scored 0.25 (confirmed violation weighted below silence).",
            _set_status_score(base_raw, "non_compliant", 0.25),
        ),
    ]


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


@dataclass
class SensitivityRow:
    name: str
    description: str
    overall_score: float | None
    chapter_scores: dict[str, float | None] = field(default_factory=dict)


def run_sensitivity_analysis(
    judge_output: dict[str, Any],
    named_configs: list[tuple[str, str, ScoringConfig]],
) -> tuple[list[SensitivityRow], list[str]]:
    """Scores ``judge_output`` once per config in ``named_configs``.

    Returns the comparison rows plus the ordered list of chapter ids worth
    showing as columns -- every chapter that had at least one scored article
    under any of the configs, in first-seen order, so a table stays legible
    even when ``named_configs`` mixes configs with different chapter sets.
    """
    rows: list[SensitivityRow] = []
    reports: list[ComplianceReport] = []
    chapter_ids: list[str] = []

    for name, description, config in named_configs:
        report = score_judge_output(judge_output, config)
        reports.append(report)
        rows.append(
            SensitivityRow(
                name=name,
                description=description,
                overall_score=report.overall_score,
                chapter_scores={chapter.id: _pct(chapter.score) for chapter in report.chapters},
            )
        )
        for chapter in report.chapters:
            if chapter.articles and chapter.id not in chapter_ids:
                chapter_ids.append(chapter.id)

    return rows, chapter_ids


def render_markdown_table(rows: list[SensitivityRow], chapter_ids: list[str]) -> str:
    header = ["Config", "Description", "Overall"] + [f"Ch. {cid}" for cid in chapter_ids]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        cells = [row.name, row.description, _fmt(row.overall_score)]
        cells += [_fmt(row.chapter_scores.get(cid)) for cid in chapter_ids]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_csv_table(rows: list[SensitivityRow], chapter_ids: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["config", "description", "overall"] + [f"ch_{cid}" for cid in chapter_ids])
    for row in rows:
        writer.writerow(
            [row.name, row.description, row.overall_score]
            + [row.chapter_scores.get(cid) for cid in chapter_ids]
        )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="judge/output_schema.json-shaped JSON file."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Base config the built-in preset variants are derived from "
            "(ignored if --configs is given)."
        ),
    )
    parser.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Explicit list of article_weights.yaml-shaped config files to compare instead of "
            "the built-in presets -- one column-comparable row per file, named by its stem."
        ),
    )
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional path to write the rendered table."
    )
    args = parser.parse_args(argv)

    judge_output = load_judge_output(args.input)

    if args.configs:
        named_configs = [(path.stem, str(path), ScoringConfig.load(path)) for path in args.configs]
    else:
        base_raw = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
        named_configs = [
            (name, description, ScoringConfig.from_dict(raw))
            for name, description, raw in default_presets(base_raw)
        ]

    rows, chapter_ids = run_sensitivity_analysis(judge_output, named_configs)
    table = (
        render_markdown_table(rows, chapter_ids)
        if args.format == "markdown"
        else render_csv_table(rows, chapter_ids)
    )

    print(table)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")
        print(f"\nComparison table written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
