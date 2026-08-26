from judge import opp115
from judge.tests.conftest import OPP115_FIXTURE_DIR


def _load():
    return opp115.load_annotations_dir(OPP115_FIXTURE_DIR)


class TestLoadAnnotations:
    def test_loads_all_fixture_rows(self):
        annotations = _load()
        # 15 rows in 1001.csv + 2 rows in 1002.csv
        assert len(annotations) == 17

    def test_parses_attributes_json(self):
        annotations = _load()
        first_party = next(
            a
            for a in annotations
            if a.category == "First Party Collection/Use" and a.policy_id == "1001"
        )
        assert "Does/Does Not" in first_party.attributes
        assert first_party.attributes["Does/Does Not"]["value"] in {"Does", "Does Not"}

    def test_missing_dir_returns_empty(self, tmp_path):
        assert opp115.load_annotations_dir(tmp_path) == []


class TestGrouping:
    def test_group_by_segment_category_collapses_annotators(self):
        annotations = _load()
        groups = opp115.group_by_segment_category(annotations)
        key = ("1001", "0", "First Party Collection/Use")
        assert key in groups
        assert len(groups[key]) == 3  # three annotators labeled this segment/category

    def test_group_by_segment_spans_categories(self):
        annotations = _load()
        groups = opp115.group_by_segment(annotations)
        # segment 0 only has First Party Collection/Use annotations in the fixture
        assert len(groups[("1001", "0")]) == 3


class TestMajorityAttributes:
    def test_majority_vote_breaks_ties_by_frequency(self):
        annotations = _load()
        groups = opp115.group_by_segment_category(annotations)
        group = groups[("1001", "0", "First Party Collection/Use")]
        resolved = opp115.majority_attributes(group)
        # 2 annotators said "Does", 1 said "Does Not" -> majority is "Does"
        assert resolved["Does/Does Not"]["value"] == "Does"
        assert resolved["Collection Mode"]["value"] == "Explicit"

    def test_prefers_longest_selected_text_among_winning_value(self):
        annotations = _load()
        groups = opp115.group_by_segment_category(annotations)
        group = groups[("1001", "0", "First Party Collection/Use")]
        resolved = opp115.majority_attributes(group)
        assert resolved["Does/Does Not"]["selectedText"] == "collect your name and email address"

    def test_single_annotator_group_passthrough(self):
        annotations = _load()
        groups = opp115.group_by_segment_category(annotations)
        group = groups[("1001", "1", "Third Party Sharing/Collection")]
        resolved = opp115.majority_attributes(group)
        assert resolved["Choice Type"]["value"] == "Opt-in"


class TestReconstructSegmentText:
    def test_stitches_spans_in_offset_order(self):
        annotations = _load()
        groups = opp115.group_by_segment(annotations)
        text = opp115.reconstruct_segment_text(groups[("1001", "1")])
        assert "share your usage data with analytics partners" in text
        assert "opt in to this sharing" in text
        # offset order preserved: the earlier span appears first
        assert text.index("share your usage data") < text.index("opt in to this sharing")

    def test_falls_back_to_unique_texts_without_offsets(self):
        annotations = _load()
        groups = opp115.group_by_segment(annotations)
        # segment 5 (Data Retention, Unspecified) has no usable offsets in the fixture
        text = opp115.reconstruct_segment_text(groups[("1001", "5")])
        assert text == ""

    def test_empty_group_returns_empty_string(self):
        assert opp115.reconstruct_segment_text([]) == ""
