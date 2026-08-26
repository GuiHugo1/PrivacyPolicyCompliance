import json

from judge import gdpr_source, opp115
from judge.build_sft_dataset import (
    build_example,
    generate_examples,
    split_for_policy,
    write_jsonl,
)
from judge.mapping import (
    ALLOWED_COMPLIANCE_STATUS,
    DEFAULT_CONFIG_PATH,
    GdprArticleRef,
    Opp115GdprMapping,
    ResolvedRule,
)
from judge.tests.conftest import MINI_GDPR_PATH, OPP115_FIXTURE_DIR


def _generate(val_frac=0.0, test_frac=0.0):
    annotations = opp115.load_annotations_dir(OPP115_FIXTURE_DIR)
    mapping = Opp115GdprMapping.load(DEFAULT_CONFIG_PATH)
    article_texts = gdpr_source.load_article_texts(MINI_GDPR_PATH)
    return generate_examples(
        annotations, mapping, article_texts, val_frac=val_frac, test_frac=test_frac
    )


class TestBuildExample:
    def test_produces_valid_chat_record(self):
        article_ref = GdprArticleRef(article="15", role="primary", note="access")
        rule = ResolvedRule(
            requirement_present=True,
            compliance_status="compliant",
            confidence=0.5,
            note="Access mechanism disclosed.",
            matched_attribute="Access Type",
            matched_value="View",
        )
        example = build_example(
            example_id="ex1",
            policy_id="1001",
            segment_id="3",
            category="User Access, Edit and Deletion",
            clause="You may review your account information at any time.",
            article_ref=article_ref,
            article_text="Article 15 — Right of access\n\n(1) ...",
            rule=rule,
            attributes={
                "Access Type": {"value": "View", "selectedText": "review your account information"}
            },
        )

        assert example["messages"][0]["role"] == "system"
        assert example["messages"][1]["role"] == "user"
        assert "Clause:" in example["messages"][1]["content"]
        assert "Article 15" in example["messages"][1]["content"]

        assistant_msg = example["messages"][2]
        assert assistant_msg["role"] == "assistant"
        verdict = json.loads(assistant_msg["content"])
        assert verdict == example["target"]
        assert set(verdict) == {
            "article",
            "requirement_present",
            "compliance_status",
            "evidence_span",
            "rationale",
            "confidence",
        }
        assert verdict["article"] == "15"
        assert verdict["compliance_status"] in ALLOWED_COMPLIANCE_STATUS
        assert 0.0 <= verdict["confidence"] <= 1.0
        assert example["meta"]["weak_label"] is True
        assert example["meta"]["opp115_category"] == "User Access, Edit and Deletion"


class TestSplitForPolicy:
    def test_deterministic(self):
        assert split_for_policy("policy-42", 0.1, 0.1) == split_for_policy("policy-42", 0.1, 0.1)

    def test_zero_fractions_always_train(self):
        for policy_id in ["a", "b", "c", "1001", "1002", "xyz123"]:
            assert split_for_policy(policy_id, 0.0, 0.0) == "train"

    def test_val_frac_one_always_val(self):
        for policy_id in ["a", "b", "c", "1001", "1002"]:
            assert split_for_policy(policy_id, 1.0, 0.0) == "val"

    def test_test_frac_one_always_test(self):
        for policy_id in ["a", "b", "c", "1001", "1002"]:
            assert split_for_policy(policy_id, 0.0, 1.0) == "test"


class TestGenerateExamples:
    def test_counts_match_fixture_expectations(self):
        splits, stats = _generate()
        assert stats.total_groups == 15
        assert stats.excluded == 2  # Data Retention/Unspecified (empty clause) + Other/Introductory
        assert stats.missing_article_text == 0
        assert stats.generated == 14
        assert len(splits["train"]) == 14
        assert splits["val"] == []
        assert splits["test"] == []

    def test_excludes_introductory_generic_other(self):
        splits, _stats = _generate()
        assert not any(
            e["meta"]["opp115_category"] == "Other"
            and e["meta"]["matched_value"] == "Introductory/Generic"
            for e in splits["train"]
        )

    def test_access_type_view_grounds_only_article_15(self):
        splits, _stats = _generate()
        access_examples = [
            e
            for e in splits["train"]
            if e["meta"]["opp115_category"] == "User Access, Edit and Deletion"
        ]
        assert len(access_examples) == 1
        assert access_examples[0]["target"]["article"] == "15"

    def test_data_retention_grounds_both_primary_articles(self):
        splits, _stats = _generate()
        retention_examples = [
            e for e in splits["train"] if e["meta"]["opp115_category"] == "Data Retention"
        ]
        articles = sorted(e["target"]["article"] for e in retention_examples)
        assert articles == ["13(2)(a)", "5(1)(e)"]

    def test_international_audience_children_grounds_article_8_only(self):
        splits, _stats = _generate()
        children_examples = [
            e for e in splits["train"] if e["meta"].get("matched_value") == "Children"
        ]
        assert len(children_examples) == 1
        assert children_examples[0]["target"]["article"] == "8"

    def test_majority_vote_wins_over_dissenting_annotator(self):
        splits, _stats = _generate()
        first_party = [
            e
            for e in splits["train"]
            if e["meta"]["opp115_category"] == "First Party Collection/Use"
            and e["meta"]["policy_id"] == "1001"
        ]
        assert len(first_party) == 1
        # 2 of 3 annotators said "Does" with Collection Mode "Explicit" -- the
        # single dissenting "Does Not"/"Implicit" annotator should not win.
        assert first_party[0]["meta"]["opp115_attributes"]["Does/Does Not"] == "Does"
        assert first_party[0]["target"]["compliance_status"] == "compliant"

    def test_every_generated_verdict_is_well_formed(self):
        splits, _stats = _generate()
        for example in splits["train"]:
            verdict = example["target"]
            assert verdict["compliance_status"] in ALLOWED_COMPLIANCE_STATUS
            assert isinstance(verdict["requirement_present"], bool)
            assert 0.0 <= verdict["confidence"] <= 1.0
            assert verdict["evidence_span"]
            assert verdict["rationale"]

    def test_same_policy_examples_share_one_split(self):
        splits, _stats = _generate(val_frac=0.5, test_frac=0.0)
        policy_to_splits: dict[str, set[str]] = {}
        for split_name, records in splits.items():
            for example in records:
                policy_id = example["meta"]["policy_id"]
                policy_to_splits.setdefault(policy_id, set()).add(split_name)
        assert all(len(split_names) == 1 for split_names in policy_to_splits.values())


class TestWriteJsonl:
    def test_round_trips(self, tmp_path):
        records = [{"id": "a", "x": 1}, {"id": "b", "x": 2}]
        out_path = tmp_path / "out" / "train.jsonl"
        write_jsonl(records, out_path)
        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in lines] == records
