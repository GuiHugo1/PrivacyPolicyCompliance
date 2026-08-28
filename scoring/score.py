"""GDPR compliance scoring engine.

Consumes one judge-pipeline output JSON (``judge/pipeline.py``, schema at
``judge/output_schema.json``) and produces a ``ComplianceReport``: an overall
0-100 compliance score, a per-GDPR-chapter breakdown, and a per-article
breakdown that carries the underlying evidence/rationale through from the
judge output for explainability -- nothing here re-derives or overrides a
judge verdict, it only aggregates ``articles[].best_compliance_status``
entries into a score.

Methodology
-----------
1. Every GDPR article has a weight (default: equal within its chapter,
   override-able per article in ``config/article_weights.yaml``'s
   ``article_weights.overrides``).
2. Per-article score, from ``best_compliance_status``:
   ``compliant`` = 1.0, ``partial`` = 0.5, ``non_compliant`` = 0.0,
   ``not_addressed`` = 0.0 (see "non_compliant vs. not_addressed" below);
   ``not_applicable`` is excluded from the denominator entirely -- it
   contributes neither a score nor a weight, rather than counting as 0.
3. Chapter score = weighted average of its (non-excluded) articles' scores.
4. Overall score = weighted average of chapter scores (chapter weights also
   configurable), restricted to chapters marked ``in_scope: true`` -- see
   ``config/article_weights.yaml``'s chapter docstring for why Ch. I and
   VI-XI (regulation scope, supervisory authorities, enforcement, procedure)
   are out of scope by default for scoring a privacy-policy *document*.
5. A chapter/article that ends up with no scorable members (e.g. every
   article in it is ``not_applicable``, or it has none of the judge output's
   articles at all) is simply excluded from its parent's weighted average --
   weights are renormalized over whatever is actually scorable, never
   silently treated as a zero.

All of this is driven by ``ScoringConfig``, loaded from
``config/article_weights.yaml`` by default; nothing in this module hardcodes
a status-to-score mapping, a chapter's articles, or a weight.

non_compliant vs. not_addressed
--------------------------------
These are two different findings that this module currently *scores*
identically (both 0.0), on purpose, but they are not the same thing and are
kept as distinct status strings everywhere (never collapsed into one) so
that changing this is purely a config edit, not a rewrite:

- ``non_compliant``: at least one clause was checked against the article and
  the judge determined the policy actively fails it -- a confirmed finding.
- ``not_addressed``: no clause in the policy ever retrieved this article at
  all (see ``judge/pipeline.py``'s ``aggregate_articles`` docstring) -- the
  policy is silent on it. This could mean the article doesn't apply to this
  controller's processing (arguably closer to ``not_applicable``), or it
  could mean a real gap retrieval/segmentation failed to surface -- the
  judge pipeline itself cannot tell these apart from silence alone.

Because both map to the *same* ``status_scores`` value, this module never
branches on ``compliance_status == "non_compliant"`` vs.
``"not_addressed"`` anywhere in the aggregation code path (``score_article``,
``_weighted_average``, chapter/overall rollups) -- both simply flow through
one dict lookup, ``config.status_scores[status]``. To weight them
differently later (e.g. penalize a confirmed violation harder than silence,
or the reverse), change their two entries in
``config/article_weights.yaml``'s ``status_scores`` -- e.g.::

    status_scores:
      non_compliant: 0.0    # confirmed violation
      not_addressed: 0.25   # silence, penalized less than a confirmed violation

No code in this file needs to change for that; ``score_article`` re-reads
whatever ``status_scores`` says on every call.

Usage::

    python -m scoring.score --input judge/examples/sample_output.json
    python -m scoring.score --input path/to/output.json \\
        --config path/to/article_weights.yaml --output report.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "article_weights.yaml"

_ARTICLE_NUMBER_RE = re.compile(r"\d+")


def _article_sort_key(article: str) -> tuple[int, str]:
    match = _ARTICLE_NUMBER_RE.match(article)
    return (int(match.group()), article) if match else (10**9, article)


def _weighted_average(pairs: list[tuple[float | None, float]]) -> float | None:
    """Weighted average of ``(score, weight)`` pairs, renormalized over
    whatever is passed in. A ``None`` score (caller didn't already filter it
    out) and any ``weight <= 0`` are both dropped rather than treated as a
    zero. Returns ``None`` (not 0.0) when nothing is left to average -- "no
    scorable members" and "scored 0" must stay distinguishable all the way
    up to the overall score.
    """
    scored = [(score, weight) for score, weight in pairs if score is not None and weight > 0]
    total_weight = sum(weight for _, weight in scored)
    if total_weight <= 0:
        return None
    return sum(score * weight for score, weight in scored) / total_weight


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChapterConfig:
    id: str
    name: str
    in_scope: bool
    weight: float
    articles: tuple[str, ...]


@dataclass(frozen=True)
class ScoringConfig:
    status_scores: dict[str, float]
    excluded_statuses: frozenset[str]
    article_weight_default: float
    article_weight_overrides: dict[str, float]
    chapters: tuple[ChapterConfig, ...]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> ScoringConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ScoringConfig:
        chapters = tuple(
            ChapterConfig(
                id=str(chapter["id"]),
                name=chapter["name"],
                in_scope=bool(chapter.get("in_scope", True)),
                weight=float(chapter.get("weight", 1.0)),
                articles=tuple(str(a) for a in chapter.get("articles", [])),
            )
            for chapter in raw.get("chapters", [])
        )
        article_weights = raw.get("article_weights") or {}
        config = cls(
            status_scores={
                str(status): float(value)
                for status, value in (raw.get("status_scores") or {}).items()
            },
            excluded_statuses=frozenset(str(s) for s in raw.get("excluded_statuses") or []),
            article_weight_default=float(article_weights.get("default", 1.0)),
            article_weight_overrides={
                str(article): float(weight)
                for article, weight in (article_weights.get("overrides") or {}).items()
            },
            chapters=chapters,
        )
        config._validate()
        return config

    def _validate(self) -> None:
        overlap = set(self.status_scores) & self.excluded_statuses
        if overlap:
            raise ValueError(
                f"status(es) {sorted(overlap)} cannot be listed in both "
                "status_scores and excluded_statuses"
            )
        seen: dict[str, str] = {}
        for chapter in self.chapters:
            for article in chapter.articles:
                if article in seen:
                    raise ValueError(
                        f"article {article!r} listed under both chapter "
                        f"{seen[article]!r} and {chapter.id!r}"
                    )
                seen[article] = chapter.id

    def article_weight(self, article: str) -> float:
        return self.article_weight_overrides.get(article, self.article_weight_default)

    def chapter_for_article(self, article: str) -> ChapterConfig | None:
        for chapter in self.chapters:
            if article in chapter.articles:
                return chapter
        return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class ArticleScore:
    """Per-article result. Carries ``evidence``/``rationale``/
    ``clauses_addressing_it`` straight through from the judge output's
    ``article_summary`` entry, unmodified, for explainability -- this module
    never rewrites or discards them, including for excluded articles.
    """

    article: str
    chapter_id: str | None
    chapter_name: str | None
    in_scope: bool
    compliance_status: str | None
    score: float | None
    weight: float
    included_in_score: bool
    exclusion_reason: str | None
    evidence: str
    rationale: str
    clauses_addressing_it: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "article": self.article,
            "chapter_id": self.chapter_id,
            "chapter_name": self.chapter_name,
            "in_scope": self.in_scope,
            "compliance_status": self.compliance_status,
            "score": self.score,
            "weight": self.weight,
            "included_in_score": self.included_in_score,
            "exclusion_reason": self.exclusion_reason,
            "evidence": self.evidence,
            "rationale": self.rationale,
            "clauses_addressing_it": self.clauses_addressing_it,
        }


@dataclass
class ChapterScore:
    id: str
    name: str
    in_scope: bool
    weight: float
    score: float | None
    included_in_overall: bool
    articles: list[ArticleScore]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "in_scope": self.in_scope,
            "weight": self.weight,
            "score": self.score,
            "score_pct": None if self.score is None else round(self.score * 100, 2),
            "included_in_overall": self.included_in_overall,
            "articles": [article.to_dict() for article in self.articles],
        }


@dataclass
class ComplianceReport:
    overall_score: float | None  # 0-100; None if nothing was scorable
    chapters: list[ChapterScore]
    unmapped_articles: list[ArticleScore]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "chapters": [chapter.to_dict() for chapter in self.chapters],
            "unmapped_articles": [article.to_dict() for article in self.unmapped_articles],
            "meta": self.meta,
        }


def score_article(entry: dict[str, Any], config: ScoringConfig) -> ArticleScore:
    """Scores one ``judge/output_schema.json`` ``article_summary`` entry.

    ``in_scope`` (a chapter-level property, carried onto the article here
    for display) only ever gates whether the article's *chapter* feeds into
    the *overall* score -- it does not affect whether the article counts
    toward its own chapter's score, so an out-of-scope chapter's breakdown
    still shows real per-article scores, not a wall of exclusions. Only the
    article's own status decides ``included_in_score``: ``not_applicable``
    (or whatever ``excluded_statuses`` says) is excluded, as is any status
    not recognized by ``status_scores`` -- see ``score_judge_output`` for
    where ``in_scope`` actually gets applied (at chapter aggregation).

    Never raises on an unrecognized article or status -- an article outside
    every configured chapter, or a status not in ``config.status_scores``
    and not in ``config.excluded_statuses``, is recorded with
    ``included_in_score=False`` and an ``exclusion_reason`` rather than
    crashing the whole report, since a scoring-config mismatch on one
    article shouldn't hide every other article's result.
    """
    article = str(entry["article"])
    status = entry.get("best_compliance_status")
    chapter = config.chapter_for_article(article)
    in_scope = bool(chapter and chapter.in_scope)

    score: float | None = None
    included = False
    exclusion_reason: str | None = None

    if chapter is None:
        exclusion_reason = "unmapped_article"
    elif status in config.excluded_statuses:
        exclusion_reason = status
    elif status not in config.status_scores:
        exclusion_reason = "unknown_status"
    else:
        score = config.status_scores[status]
        included = True

    return ArticleScore(
        article=article,
        chapter_id=chapter.id if chapter else None,
        chapter_name=chapter.name if chapter else None,
        in_scope=in_scope,
        compliance_status=status,
        score=score,
        weight=config.article_weight(article),
        included_in_score=included,
        exclusion_reason=exclusion_reason,
        evidence=entry.get("evidence", ""),
        rationale=entry.get("rationale", ""),
        clauses_addressing_it=entry.get("clauses_addressing_it", []),
    )


def score_judge_output(judge_output: dict[str, Any], config: ScoringConfig) -> ComplianceReport:
    """Scores a full judge-pipeline output JSON (``judge/output_schema.json``)
    against ``config``, producing the overall/chapter/article breakdown.

    Iterates ``judge_output["articles"]`` only -- per ``output_schema.json``,
    that array already covers every article known to the run that produced
    it (including ``not_addressed`` placeholders for articles no clause ever
    retrieved), so nothing here needs to independently enumerate "which
    articles should exist".
    """
    article_scores = [score_article(entry, config) for entry in judge_output.get("articles", [])]

    by_chapter: dict[str, list[ArticleScore]] = defaultdict(list)
    unmapped: list[ArticleScore] = []
    for article_score in article_scores:
        if article_score.chapter_id is None:
            unmapped.append(article_score)
        else:
            by_chapter[article_score.chapter_id].append(article_score)

    chapter_scores: list[ChapterScore] = []
    for chapter in config.chapters:
        members = sorted(by_chapter.get(chapter.id, []), key=lambda a: _article_sort_key(a.article))
        chapter_score_value = _weighted_average(
            [(a.score, a.weight) for a in members if a.included_in_score]
        )
        chapter_scores.append(
            ChapterScore(
                id=chapter.id,
                name=chapter.name,
                in_scope=chapter.in_scope,
                weight=chapter.weight,
                score=chapter_score_value,
                included_in_overall=chapter.in_scope and chapter_score_value is not None,
                articles=members,
            )
        )

    overall = _weighted_average(
        [(c.score, c.weight) for c in chapter_scores if c.included_in_overall]
    )
    overall_pct = None if overall is None else round(overall * 100, 2)

    return ComplianceReport(
        overall_score=overall_pct,
        chapters=chapter_scores,
        unmapped_articles=sorted(unmapped, key=lambda a: _article_sort_key(a.article)),
        meta={
            "policy": judge_output.get("policy"),
            "generated_at": (judge_output.get("meta") or {}).get("generated_at"),
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_judge_output(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_summary(report: ComplianceReport) -> str:
    lines = []
    overall = "n/a" if report.overall_score is None else f"{report.overall_score:.2f}"
    lines.append(f"Overall compliance score: {overall} / 100")
    lines.append("")
    lines.append("By chapter:")
    for chapter in report.chapters:
        if not chapter.articles:
            continue
        score_str = "n/a" if chapter.score is None else f"{chapter.score * 100:.2f}"
        scope_str = "in scope" if chapter.in_scope else "out of scope"
        lines.append(
            f"  Ch. {chapter.id} {chapter.name} ({scope_str}): {score_str} / 100 "
            f"[{len(chapter.articles)} article(s)]"
        )
    if report.unmapped_articles:
        lines.append("")
        unmapped = ", ".join(a.article for a in report.unmapped_articles)
        lines.append(f"Unmapped articles (not in any configured chapter): {unmapped}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="judge/output_schema.json-shaped JSON file."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional path to write the full JSON report."
    )
    args = parser.parse_args(argv)

    config = ScoringConfig.load(args.config)
    judge_output = load_judge_output(args.input)
    report = score_judge_output(judge_output, config)

    print(render_summary(report))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nFull report written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
