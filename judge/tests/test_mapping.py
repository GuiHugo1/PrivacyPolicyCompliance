import pytest

from judge.mapping import (
    ALLOWED_COMPLIANCE_STATUS,
    DEFAULT_CONFIG_PATH,
    Opp115GdprMapping,
)

OPP115_CATEGORIES = {
    "First Party Collection/Use",
    "Third Party Sharing/Collection",
    "User Choice/Control",
    "User Access, Edit and Deletion",
    "Data Retention",
    "Data Security",
    "Policy Change",
    "Do Not Track",
    "International and Specific Audiences",
    "Other",
}


@pytest.fixture(scope="module")
def mapping() -> Opp115GdprMapping:
    return Opp115GdprMapping.load(DEFAULT_CONFIG_PATH)


class TestConfigStructure:
    def test_loads_without_error(self, mapping):
        assert mapping.version == 1

    def test_covers_all_ten_opp115_categories(self, mapping):
        assert set(mapping.known_categories()) == OPP115_CATEGORIES

    def test_every_category_has_a_primary_article(self, mapping):
        for category in mapping.known_categories():
            assert mapping.primary_articles(category), f"{category} has no primary article"

    def test_schema_gaps_have_required_fields(self, mapping):
        assert len(mapping.gdpr_schema_gaps) >= 5
        for gap in mapping.gdpr_schema_gaps:
            assert gap["id"]
            assert gap["articles"]
            assert gap["description"]
            assert gap["why_opp115_cant_capture"]

    def test_rejects_bad_role(self):
        raw = {
            "categories": {
                "X": {
                    "gdpr_articles": [{"article": "6", "role": "tertiary"}],
                    "default_rule": {"compliance_status": "partial", "confidence": 0.5},
                }
            }
        }
        with pytest.raises(ValueError, match="invalid role"):
            Opp115GdprMapping(raw)

    def test_rejects_no_primary(self):
        raw = {
            "categories": {
                "X": {
                    "gdpr_articles": [{"article": "6", "role": "secondary"}],
                    "default_rule": {"compliance_status": "partial", "confidence": 0.5},
                }
            }
        }
        with pytest.raises(ValueError, match="no primary entry"):
            Opp115GdprMapping(raw)

    def test_rejects_bad_compliance_status(self):
        raw = {
            "categories": {
                "X": {
                    "gdpr_articles": [{"article": "6", "role": "primary"}],
                    "default_rule": {"compliance_status": "very compliant", "confidence": 0.5},
                }
            }
        }
        with pytest.raises(ValueError, match="invalid compliance_status"):
            Opp115GdprMapping(raw)

    def test_rejects_confidence_out_of_range(self):
        raw = {
            "categories": {
                "X": {
                    "gdpr_articles": [{"article": "6", "role": "primary"}],
                    "default_rule": {"compliance_status": "partial", "confidence": 1.5},
                }
            }
        }
        with pytest.raises(ValueError, match=r"out of \[0, 1\]"):
            Opp115GdprMapping(raw)

    def test_rejects_implausible_article_cite(self):
        raw = {
            "categories": {
                "X": {
                    "gdpr_articles": [{"article": "not-an-article", "role": "primary"}],
                    "default_rule": {"compliance_status": "partial", "confidence": 0.5},
                }
            }
        }
        with pytest.raises(ValueError, match="implausible article cite"):
            Opp115GdprMapping(raw)


class TestResolve:
    def test_first_party_does_not_collect_is_not_applicable(self, mapping):
        rule = mapping.resolve(
            "First Party Collection/Use", {"Does/Does Not": {"value": "Does Not"}}
        )
        assert rule.requirement_present is False
        assert rule.compliance_status == "not_applicable"
        assert not rule.exclude

    def test_other_introductory_generic_is_excluded(self, mapping):
        rule = mapping.resolve("Other", {"Other Type": {"value": "Introductory/Generic"}})
        assert rule.exclude is True

    def test_other_privacy_contact_information_is_not_excluded(self, mapping):
        rule = mapping.resolve("Other", {"Other Type": {"value": "Privacy contact information"}})
        assert rule.exclude is False
        assert rule.compliance_status in ALLOWED_COMPLIANCE_STATUS

    def test_unknown_category_is_excluded(self, mapping):
        rule = mapping.resolve("Not A Real Category", {})
        assert rule.exclude is True

    def test_unmatched_attribute_value_falls_back_to_default_rule(self, mapping):
        rule = mapping.resolve(
            "Data Retention", {"Retention Period": {"value": "Some Never-Seen Value"}}
        )
        assert rule.compliance_status in ALLOWED_COMPLIANCE_STATUS

    def test_do_not_track_is_flagged_non_native(self, mapping):
        assert mapping.is_gdpr_native("Do Not Track") is False
