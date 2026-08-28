"""Unit tests for scoring/sensitivity_analysis.py.

Reuses the same small hand-crafted judge output shape as
scoring/tests/test_score.py (kept as an independent local copy here rather
than a cross-module import, so this test file stands alone), with expected
overall scores worked out by hand for a handful of the built-in presets:

    baseline:                (0.75 + 0.0 + 1.0) / 3            = 58.33
    strict_partial_credit:   (0.50 + 0.0 + 1.0) / 3             = 50.00
    principles_heavy (Ch.II x3): (0.75*3 + 0.0 + 1.0) / 5       = 65.00
    rights_heavy (Ch.III x3):    (0.75 + 0.0*3 + 1.0) / 5       = 35.00
    transfers_heavy (Ch.V x3):   (0.75 + 0.0 + 1.0*3) / 5       = 75.00
    silence_penalized_less (not_addressed=0.25): (0.75+0.125+1.0)/3 = 62.50

(Ch. II here = 5:compliant, 6:partial -> 0.75; Ch. III = 13:non_compliant,
15:not_addressed, 17:not_applicable(excluded) -> 0.0; Ch. V = 44:compliant
-> 1.0 -- see BASE_CONFIG_DICT / _base_judge_output below.)
"""

from __future__ import annotations

import copy
import csv
import io
import json

import pytest
import yaml

from scoring.score import ScoringConfig
from scoring.sensitivity_analysis import (
    default_presets,
    main,
    render_csv_table,
    render_markdown_table,
    run_sensitivity_analysis,
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


def _article(article: str, status: str):
    return {
        "article": article,
        "best_compliance_status": status,
        "evidence": f"evidence for {article}",
        "rationale": f"rationale for {article}",
        "clauses_addressing_it": [],
    }


def _base_judge_output():
    return {
        "articles": [
            _article("5", "compliant"),
            _article("6", "partial"),
            _article("13", "non_compliant"),
            _article("15", "not_addressed"),
            _article("17", "not_applicable"),
            _article("44", "compliant"),
        ],
    }


class TestDefaultPresets:
    def test_baseline_preset_is_unmodified_copy(self):
        presets = default_presets(BASE_CONFIG_DICT)
        names = [name for name, _desc, _raw in presets]
        assert names[0] == "baseline"
        baseline_raw = presets[0][2]
        assert baseline_raw == BASE_CONFIG_DICT
        assert baseline_raw is not BASE_CONFIG_DICT  # deep-copied, not shared

    def test_presets_do_not_mutate_the_base_dict(self):
        original = copy.deepcopy(BASE_CONFIG_DICT)
        default_presets(BASE_CONFIG_DICT)
        assert BASE_CONFIG_DICT == original

    def test_preset_names_are_unique(self):
        presets = default_presets(BASE_CONFIG_DICT)
        names = [name for name, _desc, _raw in presets]
        assert len(names) == len(set(names))


class TestRunSensitivityAnalysis:
    def _named_configs(self, preset_names: set[str]) -> list:
        presets = default_presets(BASE_CONFIG_DICT)
        return [
            (name, desc, ScoringConfig.from_dict(raw))
            for name, desc, raw in presets
            if name in preset_names
        ]

    def test_hand_computed_overall_scores(self):
        wanted = {
            "baseline": 58.33,
            "strict_partial_credit": 50.0,
            "principles_heavy": 65.0,
            "rights_heavy": 35.0,
            "transfers_heavy": 75.0,
            "silence_penalized_less": 62.5,
        }
        rows, _chapter_ids = run_sensitivity_analysis(
            _base_judge_output(), self._named_configs(set(wanted))
        )
        by_name = {row.name: row.overall_score for row in rows}
        for name, expected in wanted.items():
            assert by_name[name] == pytest.approx(expected, abs=0.01), name

    def test_chapter_ids_only_include_chapters_with_scored_articles(self):
        rows, chapter_ids = run_sensitivity_analysis(
            _base_judge_output(), self._named_configs({"baseline"})
        )
        assert chapter_ids == ["II", "III", "V"]
        assert rows[0].chapter_scores["II"] == pytest.approx(75.0)
        assert rows[0].chapter_scores["III"] == pytest.approx(0.0)
        assert rows[0].chapter_scores["V"] == pytest.approx(100.0)


class TestRenderTables:
    def test_markdown_table_has_expected_columns_and_rows(self):
        rows, chapter_ids = run_sensitivity_analysis(
            _base_judge_output(),
            [
                (name, desc, ScoringConfig.from_dict(raw))
                for name, desc, raw in default_presets(BASE_CONFIG_DICT)
                if name in {"baseline", "strict_partial_credit"}
            ],
        )
        table = render_markdown_table(rows, chapter_ids)
        lines = table.splitlines()
        assert lines[0].startswith("| Config | Description | Overall | Ch. II | Ch. III | Ch. V |")
        assert any(line.startswith("| baseline |") for line in lines)
        assert any(line.startswith("| strict_partial_credit |") for line in lines)
        assert "58.33" in table
        assert "50.00" in table

    def test_csv_table_round_trips(self):
        rows, chapter_ids = run_sensitivity_analysis(
            _base_judge_output(),
            [
                (name, desc, ScoringConfig.from_dict(raw))
                for name, desc, raw in default_presets(BASE_CONFIG_DICT)
                if name == "baseline"
            ],
        )
        csv_text = render_csv_table(rows, chapter_ids)
        reader = csv.DictReader(io.StringIO(csv_text))
        parsed = list(reader)
        assert len(parsed) == 1
        assert parsed[0]["config"] == "baseline"
        assert float(parsed[0]["overall"]) == pytest.approx(58.33, abs=0.01)


class TestCli:
    def test_cli_end_to_end_with_default_presets(self, tmp_path):
        input_path = tmp_path / "judge_output.json"
        input_path.write_text(json.dumps(_base_judge_output()), encoding="utf-8")

        base_config_path = tmp_path / "article_weights.yaml"
        base_config_path.write_text(yaml.dump(BASE_CONFIG_DICT), encoding="utf-8")

        output_path = tmp_path / "comparison.md"
        exit_code = main(
            [
                "--input",
                str(input_path),
                "--base-config",
                str(base_config_path),
                "--output",
                str(output_path),
            ]
        )

        assert exit_code == 0
        text = output_path.read_text(encoding="utf-8")
        assert "baseline" in text
        assert "58.33" in text

    def test_cli_with_explicit_configs(self, tmp_path):
        input_path = tmp_path / "judge_output.json"
        input_path.write_text(json.dumps(_base_judge_output()), encoding="utf-8")

        config_a = tmp_path / "config_a.yaml"
        config_a.write_text(yaml.dump(BASE_CONFIG_DICT), encoding="utf-8")

        strict = copy.deepcopy(BASE_CONFIG_DICT)
        strict["status_scores"]["partial"] = 0.0
        config_b = tmp_path / "config_b.yaml"
        config_b.write_text(yaml.dump(strict), encoding="utf-8")

        output_path = tmp_path / "comparison.csv"
        exit_code = main(
            [
                "--input",
                str(input_path),
                "--configs",
                str(config_a),
                str(config_b),
                "--format",
                "csv",
                "--output",
                str(output_path),
            ]
        )

        assert exit_code == 0
        text = output_path.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(text))
        rows = {row["config"]: row for row in reader}
        assert rows["config_a"]["overall"] == "58.33"
        assert float(rows["config_b"]["overall"]) == pytest.approx(50.0)
