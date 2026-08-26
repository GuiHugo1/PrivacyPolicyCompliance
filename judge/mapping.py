"""Loads and validates ``judge/config/opp115_gdpr_mapping.yaml`` and resolves
a heuristic verdict rule for one (OPP-115 category, attributes) pair.

Kept deliberately free of any OPP-115-loading or dataset-generation logic
(see ``judge.opp115`` / ``judge.build_sft_dataset``) so the mapping config's
structure and validation can be tested in isolation, and so the config can
be loaded and inspected (e.g. by ``judge/coverage_report.py``) without
pulling in the annotation-loading machinery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ALLOWED_COMPLIANCE_STATUS = {"compliant", "partial", "non_compliant", "not_applicable"}
ALLOWED_ROLES = {"primary", "secondary"}

# Pinpoint cites like "5(1)(e)" or "13(1)(a)" as well as bare article numbers.
_ARTICLE_CITE_RE = re.compile(r"^\d{1,3}(\(\d+\))?(\([a-z]+\))?$")

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "opp115_gdpr_mapping.yaml"


@dataclass(frozen=True)
class GdprArticleRef:
    article: str
    role: str
    note: str = ""


@dataclass(frozen=True)
class ResolvedRule:
    """The heuristic verdict-fields rule resolved for one annotation, plus
    which attribute (if any) drove the match -- kept for traceability in
    each generated example's ``meta``.

    ``articles``, when set, overrides which of the category's primary
    ``gdpr_articles`` this particular resolved rule actually grounds (e.g.
    Access Type "View" grounds only Art 15, not also Art 16/17) -- see
    ``Opp115GdprMapping.target_articles``. ``None`` means "every primary
    article for the category", the mapping's default when a rule doesn't
    discriminate between them.
    """

    requirement_present: bool
    compliance_status: str
    confidence: float
    note: str
    exclude: bool = False
    matched_attribute: str | None = None
    matched_value: str | None = None
    articles: tuple[str, ...] | None = None


class Opp115GdprMapping:
    def __init__(self, raw: dict[str, Any]):
        self._raw = raw
        self.version = raw.get("version")
        self.categories: dict[str, dict[str, Any]] = raw.get("categories", {})
        self.gdpr_schema_gaps: list[dict[str, Any]] = raw.get("gdpr_schema_gaps", [])
        self._validate()

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> Opp115GdprMapping:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(raw)

    # -- validation ---------------------------------------------------

    def _validate(self) -> None:
        if not self.categories:
            raise ValueError("mapping config has no 'categories' section")
        for name, cfg in self.categories.items():
            self._validate_category(name, cfg)

    def _validate_category(self, name: str, cfg: dict[str, Any]) -> None:
        articles = cfg.get("gdpr_articles") or []
        if not articles:
            raise ValueError(f"category {name!r}: no gdpr_articles")

        roles: set[str] = set()
        seen_cites: set[str] = set()
        for a in articles:
            if a.get("role") not in ALLOWED_ROLES:
                raise ValueError(f"category {name!r}: invalid role {a.get('role')!r}")
            cite = a.get("article", "")
            if not _ARTICLE_CITE_RE.match(cite):
                raise ValueError(f"category {name!r}: implausible article cite {cite!r}")
            if cite in seen_cites:
                raise ValueError(f"category {name!r}: duplicate article cite {cite!r}")
            seen_cites.add(cite)
            roles.add(a["role"])
        if "primary" not in roles:
            raise ValueError(f"category {name!r}: gdpr_articles has no primary entry")

        default_rule = cfg.get("default_rule")
        if not default_rule:
            raise ValueError(f"category {name!r}: missing default_rule")
        self._validate_rule(name, "default_rule", default_rule)
        self._validate_articles_override(name, "default_rule", default_rule, seen_cites)

        for attr_name, value_rules in cfg.get("attribute_rules", {}).items():
            for value, rule in value_rules.items():
                where = f"{attr_name}={value!r}"
                self._validate_rule(name, where, rule)
                self._validate_articles_override(name, where, rule, seen_cites)

    def _validate_articles_override(
        self, category: str, where: str, rule: dict[str, Any], known_cites: set[str]
    ) -> None:
        articles = rule.get("articles")
        if articles is None:
            return
        for cite in articles:
            if cite not in known_cites:
                raise ValueError(
                    f"category {category!r} rule {where}: articles override {cite!r} is not "
                    f"one of this category's gdpr_articles {sorted(known_cites)}"
                )

    def _validate_rule(self, category: str, where: str, rule: dict[str, Any]) -> None:
        if rule.get("exclude"):
            return
        status = rule.get("compliance_status")
        if status is not None and status not in ALLOWED_COMPLIANCE_STATUS:
            raise ValueError(
                f"category {category!r} rule {where}: invalid compliance_status {status!r}"
            )
        confidence = rule.get("confidence")
        if confidence is not None and not (0.0 <= float(confidence) <= 1.0):
            raise ValueError(
                f"category {category!r} rule {where}: confidence {confidence!r} out of [0, 1]"
            )

    # -- lookups --------------------------------------------------------

    def known_categories(self) -> list[str]:
        return list(self.categories.keys())

    def primary_articles(self, category: str) -> list[GdprArticleRef]:
        cfg = self.categories[category]
        return [GdprArticleRef(**a) for a in cfg["gdpr_articles"] if a["role"] == "primary"]

    def secondary_articles(self, category: str) -> list[GdprArticleRef]:
        cfg = self.categories[category]
        return [GdprArticleRef(**a) for a in cfg["gdpr_articles"] if a["role"] == "secondary"]

    def article_ref(self, category: str, cite: str) -> GdprArticleRef:
        cfg = self.categories[category]
        for a in cfg["gdpr_articles"]:
            if a["article"] == cite:
                return GdprArticleRef(**a)
        raise KeyError(f"{cite!r} is not one of {category!r}'s gdpr_articles")

    def target_articles(self, category: str, rule: ResolvedRule) -> list[GdprArticleRef]:
        """The article(s) one resolved rule should ground SFT examples on:
        ``rule.articles`` if the matched attribute-value rule overrode it
        (e.g. Access Type "View" -> only Art 15), else every primary
        article declared for the category."""
        if rule.articles is not None:
            return [self.article_ref(category, cite) for cite in rule.articles]
        return self.primary_articles(category)

    def coverage(self, category: str) -> str | None:
        return self.categories.get(category, {}).get("coverage")

    def thin_notes(self, category: str) -> list[str]:
        return self.categories.get(category, {}).get("thin_notes", [])

    def is_gdpr_native(self, category: str) -> bool:
        return self.categories.get(category, {}).get("gdpr_native", True)

    def resolve(self, category: str, attributes: dict[str, dict[str, Any]]) -> ResolvedRule:
        """Resolves the heuristic verdict-fields rule for one annotation's
        (majority-collapsed) attributes. Attributes are tried in the YAML's
        declared order; the first one present on the annotation with either
        an exact value match or a ``default`` entry wins. Falls back to the
        category's own ``default_rule`` if no attribute_rules entry matches."""
        cfg = self.categories.get(category)
        if cfg is None:
            return ResolvedRule(
                requirement_present=False,
                compliance_status="not_applicable",
                confidence=0.0,
                note=f"No mapping entry for OPP-115 category {category!r}.",
                exclude=True,
            )

        default_rule = cfg["default_rule"]
        for attr_name, value_rules in cfg.get("attribute_rules", {}).items():
            attr = attributes.get(attr_name)
            if attr is None:
                continue
            value = attr.get("value")
            rule = value_rules.get(value)
            if rule is None:
                rule = value_rules.get("default")
            if rule is None:
                continue
            merged = {**default_rule, **rule}
            articles = merged.get("articles")
            return ResolvedRule(
                requirement_present=bool(merged.get("requirement_present", True)),
                compliance_status=merged.get("compliance_status", "partial"),
                confidence=float(merged.get("confidence", 0.5)),
                note=merged.get("note", ""),
                # `exclude` is deliberately NOT inherited from default_rule --
                # it must come from the specific matched rule (or False), so a
                # category-level `default_rule.exclude: true` (e.g. Other's,
                # for its unmapped-value fallback) doesn't leak into a value
                # that specifically matched and should NOT be excluded (e.g.
                # Other Type "Privacy contact information").
                exclude=bool(rule.get("exclude", False)),
                matched_attribute=attr_name,
                matched_value=value,
                articles=tuple(articles) if articles is not None else None,
            )

        articles = default_rule.get("articles")
        return ResolvedRule(
            requirement_present=bool(default_rule.get("requirement_present", True)),
            compliance_status=default_rule.get("compliance_status", "partial"),
            confidence=float(default_rule.get("confidence", 0.5)),
            note=default_rule.get("note", ""),
            exclude=bool(default_rule.get("exclude", False)),
            articles=tuple(articles) if articles is not None else None,
        )
