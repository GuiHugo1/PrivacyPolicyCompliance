from judge import opp115
from judge.coverage_report import build_report, compute_category_stats, render_markdown
from judge.mapping import DEFAULT_CONFIG_PATH, Opp115GdprMapping
from judge.tests.conftest import OPP115_FIXTURE_DIR


def _load():
    annotations = opp115.load_annotations_dir(OPP115_FIXTURE_DIR)
    mapping = Opp115GdprMapping.load(DEFAULT_CONFIG_PATH)
    return annotations, mapping


class TestComputeCategoryStats:
    def test_counts_annotations_and_policies_per_category(self):
        annotations, mapping = _load()
        stats = compute_category_stats(annotations, mapping)
        first_party = stats["First Party Collection/Use"]
        # 3 annotator rows on policy 1001/segment 0, plus 1 row on policy 1002/segment 0
        assert first_party["n_annotations"] == 4
        assert first_party["n_policies"] == 2
        assert first_party["n_segments"] == 2

    def test_excluded_category_has_zero_generated(self):
        annotations, mapping = _load()
        stats = compute_category_stats(annotations, mapping)
        other = stats["Other"]
        # Introductory/Generic excluded, Privacy contact information generated
        assert other["n_generated"] == 1
        assert other["n_excluded"] == 1


class TestBuildReport:
    def test_flags_do_not_track_as_not_gdpr_native(self):
        annotations, mapping = _load()
        report = build_report(annotations, mapping)
        assert any(
            f["category"] == "Do Not Track" and f["kind"] == "not_gdpr_native"
            for f in report["flags"]
        )

    def test_includes_schema_gaps_from_mapping(self):
        annotations, mapping = _load()
        report = build_report(annotations, mapping)
        gap_ids = {g["id"] for g in report["gdpr_schema_gaps"]}
        assert "legal_basis_granularity" in gap_ids
        assert "dpo_designation" in gap_ids
        assert "cross_border_transfer_mechanism" in gap_ids

    def test_small_fixture_categories_flagged_data_volume_thin(self):
        annotations, mapping = _load()
        report = build_report(annotations, mapping)
        # every category in this tiny fixture has far fewer than
        # THIN_EXAMPLE_THRESHOLD generatable examples
        thin_categories = {
            f["category"] for f in report["flags"] if f["kind"] == "data_volume_thin"
        }
        assert "First Party Collection/Use" in thin_categories

    def test_markdown_renders_without_error(self):
        annotations, mapping = _load()
        report = build_report(annotations, mapping)
        markdown = render_markdown(report)
        assert "# OPP-115 -> GDPR Coverage Report" in markdown
        assert "legal_basis_granularity" in markdown
