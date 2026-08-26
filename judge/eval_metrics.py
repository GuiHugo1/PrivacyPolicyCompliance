"""Pure-python scoring for the judge's generated verdicts: JSON-validity
rate and per-``compliance_status``-class precision/recall/F1. No ML
dependencies, so it's shared as-is between the training-time eval callback
(``judge.train_qlora.JsonValidityCallback``) and the standalone test-split
report (``judge.eval_qlora``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from judge.schema_utils import extract_json_object, validate_against_schema

COMPLIANCE_CLASSES = ("compliant", "partial", "non_compliant", "not_applicable")


@dataclass
class ParsedVerdict:
    raw_text: str
    json_valid: bool
    verdict: dict[str, Any] | None


def parse_verdict(text: str, schema: dict[str, Any]) -> ParsedVerdict:
    """Extracts and schema-validates one generated verdict. ``verdict`` is
    ``None`` whenever the text isn't valid JSON or doesn't match the
    schema -- that's the "invalid" case the JSON-validity rate counts."""
    obj = extract_json_object(text)
    if obj is None or validate_against_schema(obj, schema):
        return ParsedVerdict(raw_text=text, json_valid=False, verdict=None)
    return ParsedVerdict(raw_text=text, json_valid=True, verdict=obj)


def json_validity_rate(parsed: list[ParsedVerdict]) -> float:
    if not parsed:
        return 0.0
    return sum(p.json_valid for p in parsed) / len(parsed)


@dataclass
class ClassMetrics:
    label: str
    support: int
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def per_class_prf1(
    y_true: list[str],
    y_pred: list[str | None],
    labels: tuple[str, ...] = COMPLIANCE_CLASSES,
) -> dict[str, ClassMetrics]:
    """Computes precision/recall/F1 per ``compliance_status`` class.

    A ``None`` prediction (unparseable/invalid-schema model output) never
    contributes a true/false positive for any class -- it can't be
    "correct" -- but it does cost recall on its example's true label,
    which is the correct way to charge a judge that failed to produce a
    verdict at all rather than silently excluding it from scoring.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    support = Counter(y_true)
    metrics: dict[str, ClassMetrics] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if p == label and t != label)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p != label)
        metrics[label] = ClassMetrics(
            label=label, support=support.get(label, 0), tp=tp, fp=fp, fn=fn
        )
    return metrics


def macro_f1(metrics: dict[str, ClassMetrics]) -> float:
    if not metrics:
        return 0.0
    return sum(m.f1 for m in metrics.values()) / len(metrics)
