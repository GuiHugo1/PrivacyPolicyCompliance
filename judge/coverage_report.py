"""Flags where OPP-115 coverage is thin for GDPR-specific requirements, so
supplementary labeled data can be prioritized before trusting the judge on
those areas.

Two distinct kinds of gap are reported, and kept separate rather than
blended into one number (same principle as eval/README.md's retrieval-vs-
judge-metrics separation):

- **Schema-level gaps** (``judge.mapping.Opp115GdprMapping.gdpr_schema_gaps``):
  GDPR requirements no OPP-115 category/attribute records at all (legal
  basis granularity, DPO designation, transfer mechanism, DPIA, Art 22
  automated decision-making, etc.) -- these need genuinely new labeled data,
  not more OPP-115 annotations, since no volume of existing-schema data adds
  the missing signal.
- **Data-volume thinness**: an OPP-115 category/attribute-value combination
  that *could* ground a requirement but has too few annotated examples in
  the corpus actually loaded to trust a judge trained on it. This is
  reported per category and cross-checked against each category's
  ``coverage:`` field in the mapping config (a mismatch -- e.g. YAML says
  ``adequate`` but the loaded corpus has few examples -- is flagged
  explicitly, since it means the config's editorial judgment and the actual
  data have drifted apart).

Usage::

    python -m judge.coverage_report --opp115-dir /data/opp115/annotations
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from judge import opp115
from judge.mapping import DEFAULT_CONFIG_PATH, Opp115GdprMapping

# Below this many generated (non-excluded) examples for a category, flag it
# as thin on data volume regardless of what the mapping config's `coverage`
# field claims. Adjust once /data/opp115's real scale is known -- this is a
# conservative starting default, not a validated statistical threshold.
THIN_EXAMPLE_THRESHOLD = 20
# Below this many occurrences of a specific attribute value within a
# category, flag that value's compliance_status rule as backed by too few
# examples to trust in isolation.
THIN_VALUE_THRESHOLD = 5


def compute_category_stats(
    annotations: list[opp115.Annotation], mapping: Opp115GdprMapping
) -> dict[str, dict[str, Any]]:
    groups = opp115.group_by_segment_category(annotations)

    by_category: dict[str, dict[str, Any]] = {}
    for category in mapping.known_categories():
        by_category[category] = {
            "n_annotations": 0,
            "n_policies": set(),
            "n_segments": 0,
            "n_generated": 0,
            "n_excluded": 0,
            "attribute_value_counts": {},
        }

    for (policy_id, _segment_id, category), group in groups.items():
        if category not in by_category:
            by_category.setdefault(
                category,
                {
                    "n_annotations": 0,
                    "n_policies": set(),
                    "n_segments": 0,
                    "n_generated": 0,
                    "n_excluded": 0,
                    "attribute_value_counts": {},
                },
            )
        stats = by_category[category]
        stats["n_annotations"] += len(group)
        stats["n_policies"].add(policy_id)
        stats["n_segments"] += 1

        attributes = opp115.majority_attributes(group)
        for attr_name, attr in attributes.items():
            counts: Counter = stats["attribute_value_counts"].setdefault(attr_name, Counter())
            counts[attr.get("value")] += 1

        if category in mapping.categories:
            rule = mapping.resolve(category, attributes)
            if rule.exclude:
                stats["n_excluded"] += 1
            else:
                stats["n_generated"] += 1

    for stats in by_category.values():
        stats["n_policies"] = len(stats["n_policies"])
        stats["attribute_value_counts"] = {
            attr: dict(counts) for attr, counts in stats["attribute_value_counts"].items()
        }

    return by_category


def build_report(
    annotations: list[opp115.Annotation], mapping: Opp115GdprMapping
) -> dict[str, Any]:
    category_stats = compute_category_stats(annotations, mapping)

    flags: list[dict[str, Any]] = []
    for category, stats in sorted(category_stats.items()):
        declared_coverage = mapping.coverage(category)
        gdpr_native = mapping.is_gdpr_native(category)

        if not gdpr_native:
            flags.append(
                {
                    "category": category,
                    "kind": "not_gdpr_native",
                    "detail": (
                        f"{category} is mapped via a weak analogy, not a genuine GDPR "
                        "requirement (see mapping config note)."
                    ),
                }
            )
        elif stats["n_generated"] < THIN_EXAMPLE_THRESHOLD:
            flags.append(
                {
                    "category": category,
                    "kind": "data_volume_thin",
                    "detail": (
                        f"only {stats['n_generated']} generatable example(s) across "
                        f"{stats['n_policies']} polic(y/ies) in the loaded corpus "
                        f"(threshold: {THIN_EXAMPLE_THRESHOLD})."
                    ),
                }
            )
        elif declared_coverage == "thin":
            notes = ", ".join(mapping.thin_notes(category)) or "see thin_notes"
            flags.append(
                {
                    "category": category,
                    "kind": "config_declared_thin",
                    "detail": f"mapping config declares coverage: thin ({notes}).",
                }
            )

        for attr_name, counts in stats["attribute_value_counts"].items():
            for value, count in counts.items():
                if count < THIN_VALUE_THRESHOLD:
                    flags.append(
                        {
                            "category": category,
                            "kind": "attribute_value_thin",
                            "detail": (
                                f"attribute {attr_name!r} value {value!r}: only {count} "
                                f"occurrence(s) (threshold: {THIN_VALUE_THRESHOLD})."
                            ),
                        }
                    )

    return {
        "category_stats": category_stats,
        "flags": flags,
        "gdpr_schema_gaps": mapping.gdpr_schema_gaps,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# OPP-115 -> GDPR Coverage Report", ""]

    lines.append("## Category stats (loaded corpus)")
    lines.append("")
    lines.append("| Category | Annotations | Policies | Segments | Generatable | Excluded |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for category, stats in sorted(report["category_stats"].items()):
        lines.append(
            f"| {category} | {stats['n_annotations']} | {stats['n_policies']} | "
            f"{stats['n_segments']} | {stats['n_generated']} | {stats['n_excluded']} |"
        )
    lines.append("")

    lines.append("## Coverage flags")
    lines.append("")
    if not report["flags"]:
        lines.append("(none)")
    for flag in report["flags"]:
        lines.append(f"- **{flag['category']}** [{flag['kind']}]: {flag['detail']}")
    lines.append("")

    lines.append("## GDPR schema gaps (need new labeled data, not more OPP-115)")
    lines.append("")
    for gap in report["gdpr_schema_gaps"]:
        articles = ", ".join(gap["articles"])
        lines.append(f"### {gap['id']} (Art. {articles})")
        lines.append("")
        lines.append(gap["description"])
        lines.append("")
        lines.append(f"*Why OPP-115 can't capture this:* {gap['why_opp115_cant_capture']}")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opp115-dir", required=True, type=Path)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("reports/opp115_gdpr_coverage.md")
    )
    parser.add_argument(
        "--json-output", type=Path, default=Path("data/processed/opp115_gdpr_coverage.json")
    )
    args = parser.parse_args(argv)

    mapping = Opp115GdprMapping.load(args.mapping)
    annotations = opp115.load_annotations_dir(args.opp115_dir)
    report = build_report(annotations, mapping)

    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(
        f"{len(report['flags'])} coverage flag(s), {len(report['gdpr_schema_gaps'])} schema gap(s)."
    )
    print(f"Markdown report: {args.markdown_output}")
    print(f"JSON report: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
