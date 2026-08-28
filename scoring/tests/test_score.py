"""Unit tests for scoring/score.py, built around a small hand-crafted judge
output where every expected score is checked against a value computed by
hand in these test docstrings/comments -- not against the module's own
output, so a bug shared between the implementation and the test can't hide.

Hand-computed baseline (see BASE_CONFIG_DICT / BASE_JUDGE_OUTPUT below):

    Ch. II  (5=compliant, 6=partial):
        (1.0 + 0.5) / 2 = 0.75
    Ch. III (13=non_compliant, 15=not_addressed, 17=not_applicable):
        17 is excluded entirely (not_applicable) -> denominator is just 13, 15
        (0.0 + 0.0) / 2 = 0.0
    Ch. V   (44=compliant):
        1.0
    Ch. I   (1=compliant, but in_scope=False):
        1.0, but excluded from the overall score.

    Overall (equal chapter weights, II/III/V in scope):
        (0.75 + 0.0 + 1.0) / 3 = 1.75 / 3 = 0.58333... -> 58.33 / 100
"""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from scoring.score import (
    DEFAULT_CONFIG_PATH,
    ScoringConfig,
    load_judge_output,
    main,
    render_summary,
    score_article,
    score_judge_output,
)

BASE_CONFIG_DICT = {
    "status_scores": {
        "compliant": 1.0,
        "partial": 0.5,
        "non_compliant": 0.0,
        "not_addressed": 0.0,
        "needs_review": 0.0,
    },
    "excluded_statuses": ["not_applicable"],
    "article_weights": {"default": 1.0, "overrides": {}},
    "chapters": [
        {
            "id": "I",
            "name": "General provisions",
            "in_scope": False,
            "weight": 1.0,
            "articles": ["1"],
        },
        {"id": "II", "name": "Principles", "in_scope": True, "weight": 1.0, "articles": ["5", "6"]},
        {
            "id": "III",
            "name": "Data subject rights",
            "in_scope": True,
            "weight": 1.0,
            "articles": ["13", "15", "17"],
        },
        {
            "id": "V",
            "name": "International transfers",
            "in_scope": True,
            "weight": 1.0,
            "articles": ["44"],
        },
    ],
}


def _article(article: str, status: str, **extra):
    entry = {
        "article": article,
        "best_compliance_status": status,
        "evidence": f"evidence for {article}",
        "rationale": f"rationale for {article}",
        "clauses_addressing_it": [],
    }
    entry.update(extra)
    return entry


def _base_judge_output():
    return {
        "policy": {"source": "test.txt", "n_clauses": 6},
        "meta": {"generated_at": "2026-01-01T00:00:00+00:00"},
        "articles": [
            _article("1", "compliant"),
            _article("5", "compliant"),
            _article("6", "partial"),
            _article("13", "non_compliant"),
            _article("15", "not_addressed"),
            _article("17", "not_applicable"),
            _article("44", "compliant"),
        ],
    }


def _config(**chapter_overrides) -> ScoringConfig:
    raw = copy.deepcopy(BASE_CONFIG_DICT)
    raw.update(chapter_overrides)
    return ScoringConfig.from_dict(raw)


class TestScoringConfig:
    def test_article_weight_default_and_override(self):
        raw = copy.deepcopy(BASE_CONFIG_DICT)
        raw["article_weights"]["overrides"] = {"6": 3.0}
        config = ScoringConfig.from_dict(raw)
        assert config.article_weight("5") == 1.0
        assert config.article_weight("6") == 3.0

    def test_chapter_for_article(self):
        config = _config()
        chapter = config.chapter_for_article("13")
        assert chapter is not None
        assert chapter.id == "III"
        assert config.chapter_for_article("999") is None

    def test_rejects_article_in_two_chapters(self):
        raw = copy.deepcopy(BASE_CONFIG_DICT)
        raw["chapters"][0]["articles"] = ["5"]  # "5" already lives in chapter II
        with pytest.raises(ValueError, match="listed under both chapter"):
            ScoringConfig.from_dict(raw)

    def test_rejects_status_both_scored_and_excluded(self):
        raw = copy.deepcopy(BASE_CONFIG_DICT)
        raw["excluded_statuses"] = ["not_applicable", "compliant"]
        with pytest.raises(ValueError, match="cannot be listed in both"):
            ScoringConfig.from_dict(raw)

    def test_loads_shipped_default_config(self):
        config = ScoringConfig.load(DEFAULT_CONFIG_PATH)
        ids = {c.id for c in config.chapters}
        assert ids == {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"}
        in_scope_ids = {c.id for c in config.chapters if c.in_scope}
        assert in_scope_ids == {"II", "III", "IV", "V"}
        # every article 1-99 appears in exactly one chapter
        all_articles = [a for c in config.chapters for a in c.articles]
        assert sorted(all_articles, key=int) == [str(n) for n in range(1, 100)]
        assert len(all_articles) == len(set(all_articles))


class TestScoreArticle:
    def test_compliant_scores_one(self):
        result = score_article(_article("5", "compliant"), _config())
        assert result.score == 1.0
        assert result.included_in_score is True
        assert result.chapter_id == "II"
        assert result.evidence == "evidence for 5"

    def test_partial_scores_half(self):
        result = score_article(_article("6", "partial"), _config())
        assert result.score == 0.5
        assert result.included_in_score is True

    def test_non_compliant_and_not_addressed_score_identically_by_default(self):
        non_compliant = score_article(_article("13", "non_compliant"), _config())
        not_addressed = score_article(_article("15", "not_addressed"), _config())
        assert non_compliant.score == not_addressed.score == 0.0
        assert non_compliant.included_in_score is True
        assert not_addressed.included_in_score is True
        # they remain distinguishable as *findings* even though the score matches
        assert non_compliant.compliance_status == "non_compliant"
        assert not_addressed.compliance_status == "not_addressed"

    def test_not_applicable_is_excluded_not_zero(self):
        result = score_article(_article("17", "not_applicable"), _config())
        assert result.score is None
        assert result.included_in_score is False
        assert result.exclusion_reason == "not_applicable"

    def test_out_of_scope_chapter_article_still_scores_normally(self):
        # in_scope only gates the *chapter's* contribution to the overall
        # score (see score_judge_output) -- the article itself still scores
        # and counts toward its own (out-of-scope) chapter's score.
        result = score_article(_article("1", "compliant"), _config())
        assert result.chapter_id == "I"
        assert result.in_scope is False
        assert result.score == 1.0
        assert result.included_in_score is True
        assert result.exclusion_reason is None

    def test_unmapped_article_does_not_crash(self):
        result = score_article(_article("999", "compliant"), _config())
        assert result.chapter_id is None
        assert result.included_in_score is False
        assert result.exclusion_reason == "unmapped_article"

    def test_unknown_status_does_not_crash(self):
        result = score_article(_article("5", "some_future_status"), _config())
        assert result.included_in_score is False
        assert result.exclusion_reason == "unknown_status"

    def test_evidence_and_rationale_pass_through_even_when_excluded(self):
        result = score_article(_article("17", "not_applicable"), _config())
        assert result.evidence == "evidence for 17"
        assert result.rationale == "rationale for 17"


class TestScoreJudgeOutput:
    def test_hand_computed_baseline(self):
        report = score_judge_output(_base_judge_output(), _config())

        by_id = {c.id: c for c in report.chapters}
        assert by_id["II"].score == pytest.approx(0.75)
        assert by_id["III"].score == pytest.approx(0.0)
        assert by_id["V"].score == pytest.approx(1.0)
        assert by_id["I"].score == pytest.approx(1.0)
        assert by_id["I"].included_in_overall is False  # out of scope

        assert report.overall_score == pytest.approx(58.33, abs=0.01)

    def test_article_weight_override_shifts_chapter_score(self):
        raw = copy.deepcopy(BASE_CONFIG_DICT)
        raw["article_weights"]["overrides"] = {"6": 3.0}
        config = ScoringConfig.from_dict(raw)

        report = score_judge_output(_base_judge_output(), config)
        chapter_ii = next(c for c in report.chapters if c.id == "II")
        # (1.0*1 + 0.5*3) / (1 + 3) = 2.5 / 4 = 0.625
        assert chapter_ii.score == pytest.approx(0.625)

    def test_chapter_weight_override_shifts_overall_score(self):
        raw = copy.deepcopy(BASE_CONFIG_DICT)
        raw["chapters"][2]["weight"] = 2.0  # chapter III
        config = ScoringConfig.from_dict(raw)

        report = score_judge_output(_base_judge_output(), config)
        # (0.75*1 + 0.0*2 + 1.0*1) / (1 + 2 + 1) = 1.75 / 4 = 0.4375
        assert report.overall_score == pytest.approx(43.75)

    def test_reweighting_not_addressed_only_changes_that_status(self):
        raw = copy.deepcopy(BASE_CONFIG_DICT)
        raw["status_scores"]["not_addressed"] = 0.5
        config = ScoringConfig.from_dict(raw)

        report = score_judge_output(_base_judge_output(), config)
        chapter_iii = next(c for c in report.chapters if c.id == "III")
        # 13=non_compliant (0.0), 15=not_addressed (now 0.5), 17 still excluded
        assert chapter_iii.score == pytest.approx(0.25)

        # non_compliant's own scoring is untouched by that change
        non_compliant_article = next(a for a in chapter_iii.articles if a.article == "13")
        assert non_compliant_article.score == 0.0

    def test_chapter_with_all_articles_not_applicable_excluded_from_overall(self):
        judge_output = _base_judge_output()
        for entry in judge_output["articles"]:
            if entry["article"] == "44":
                entry["best_compliance_status"] = "not_applicable"

        report = score_judge_output(judge_output, _config())
        chapter_v = next(c for c in report.chapters if c.id == "V")
        assert chapter_v.score is None
        assert chapter_v.included_in_overall is False

        # overall renormalizes over II and III only: (0.75 + 0.0) / 2 = 0.375
        assert report.overall_score == pytest.approx(37.5)

    def test_unmapped_articles_are_reported_separately(self):
        judge_output = _base_judge_output()
        judge_output["articles"].append(_article("999", "compliant"))

        report = score_judge_output(judge_output, _config())
        assert [a.article for a in report.unmapped_articles] == ["999"]
        # doesn't perturb the otherwise-identical baseline overall score
        assert report.overall_score == pytest.approx(58.33, abs=0.01)

    def test_empty_articles_yields_no_score_anywhere(self):
        report = score_judge_output({"articles": []}, _config())
        assert report.overall_score is None
        assert all(c.score is None for c in report.chapters)

    def test_evidence_and_rationale_carried_through_per_article(self):
        report = score_judge_output(_base_judge_output(), _config())
        chapter_ii = next(c for c in report.chapters if c.id == "II")
        article_5 = next(a for a in chapter_ii.articles if a.article == "5")
        assert article_5.evidence == "evidence for 5"
        assert article_5.rationale == "rationale for 5"

    def test_to_dict_round_trips_through_json(self):
        report = score_judge_output(_base_judge_output(), _config())
        # should not raise, and should be plain JSON-serializable data
        dumped = json.dumps(report.to_dict())
        reloaded = json.loads(dumped)
        assert reloaded["overall_score"] == report.overall_score


class TestRenderSummaryAndCli:
    def test_render_summary_mentions_overall_and_chapters(self):
        report = score_judge_output(_base_judge_output(), _config())
        text = render_summary(report)
        assert "Overall compliance score" in text
        assert "Ch. II Principles" in text

    def test_cli_end_to_end(self, tmp_path):
        input_path = tmp_path / "judge_output.json"
        input_path.write_text(json.dumps(_base_judge_output()), encoding="utf-8")

        config_path = tmp_path / "article_weights.yaml"
        config_path.write_text(yaml.dump(BASE_CONFIG_DICT), encoding="utf-8")

        output_path = tmp_path / "report.json"
        exit_code = main(
            [
                "--input",
                str(input_path),
                "--config",
                str(config_path),
                "--output",
                str(output_path),
            ]
        )

        assert exit_code == 0
        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert report["overall_score"] == pytest.approx(58.33, abs=0.01)

    def test_load_judge_output_reads_json_file(self, tmp_path):
        path = tmp_path / "output.json"
        path.write_text(json.dumps({"articles": []}), encoding="utf-8")
        assert load_judge_output(path) == {"articles": []}
