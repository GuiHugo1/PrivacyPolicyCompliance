"""Loader for OPP-115 (Wilson et al. 2016) annotation CSVs.

Expected columns, matching the public OPP-115 "annotations" release (one CSV
per policy)::

    annotation_id, batch_id, annotator_id, policy_id, segment_id,
    category, attributes_json, date, policy_url

``attributes_json`` is a JSON object of ``{attribute_name: {"value": ...,
"selectedText": ..., "startIndexInSegment": ..., "endIndexInSegment": ...}}``.

OPP-115's public "annotations" release records highlighted sub-spans per
attribute, not each segment's full text (that lives in a separate
"sanitized_policies" release, keyed by the same ``segment_id``, which this
loader does not assume is available). ``reconstruct_segment_text`` is a
best-effort fallback that stitches a segment back together from the
recorded highlight offsets across every annotator/category that touched it,
for use when the sanitized-policy text file isn't available; pass real
segment text directly to ``build_sft_dataset.py`` instead when you have it.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_COLUMNS = [
    "annotation_id",
    "batch_id",
    "annotator_id",
    "policy_id",
    "segment_id",
    "category",
    "attributes_json",
    "date",
    "policy_url",
]

_GAP_PLACEHOLDER = " […] "


@dataclass(frozen=True)
class Annotation:
    annotation_id: str
    batch_id: str
    annotator_id: str
    policy_id: str
    segment_id: str
    category: str
    attributes: dict[str, dict[str, Any]]
    date: str
    policy_url: str


def load_annotations_csv(path: str | Path) -> list[Annotation]:
    path = Path(path)
    out: list[Annotation] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for i, row in enumerate(reader):
            if not row:
                continue
            if len(row) != len(_COLUMNS):
                raise ValueError(
                    f"{path}:{i + 1}: expected {len(_COLUMNS)} columns, got {len(row)}"
                )
            rec = dict(zip(_COLUMNS, row, strict=True))
            try:
                attributes = json.loads(rec["attributes_json"])
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{i + 1}: bad attributes JSON in annotation {rec['annotation_id']}"
                ) from e
            out.append(
                Annotation(
                    annotation_id=rec["annotation_id"],
                    batch_id=rec["batch_id"],
                    annotator_id=rec["annotator_id"],
                    policy_id=rec["policy_id"],
                    segment_id=rec["segment_id"],
                    category=rec["category"],
                    attributes=attributes,
                    date=rec["date"],
                    policy_url=rec["policy_url"],
                )
            )
    return out


def load_annotations_dir(path: str | Path) -> list[Annotation]:
    """Loads every ``*.csv`` file in ``path`` (one per policy, the shape of
    the OPP-115 ``annotations/`` directory)."""
    path = Path(path)
    out: list[Annotation] = []
    for csv_path in sorted(path.glob("*.csv")):
        out.extend(load_annotations_csv(csv_path))
    return out


def group_by_segment(annotations: list[Annotation]) -> dict[tuple[str, str], list[Annotation]]:
    """Groups annotations by (policy_id, segment_id) across all categories
    and annotators -- the unit ``reconstruct_segment_text`` operates on."""
    groups: dict[tuple[str, str], list[Annotation]] = defaultdict(list)
    for a in annotations:
        groups[(a.policy_id, a.segment_id)].append(a)
    return dict(groups)


def group_by_segment_category(
    annotations: list[Annotation],
) -> dict[tuple[str, str, str], list[Annotation]]:
    """Groups annotations by (policy_id, segment_id, category) -- the unit
    one SFT example is generated from, collapsing multiple annotators'
    independent labeling of the same practice via ``majority_attributes``."""
    groups: dict[tuple[str, str, str], list[Annotation]] = defaultdict(list)
    for a in annotations:
        groups[(a.policy_id, a.segment_id, a.category)].append(a)
    return dict(groups)


def reconstruct_segment_text(segment_annotations: list[Annotation]) -> str:
    """Best-effort reconstruction of a policy segment's text from the
    ``selectedText`` highlight spans recorded across every annotation
    (any annotator, any category) for that (policy_id, segment_id).

    Spans are placed at their recorded (startIndexInSegment,
    endIndexInSegment) offsets, overlapping spans are merged (keeping the
    longer end), and a placeholder marks any gap between non-adjacent
    spans -- so the result recovers most of a well-annotated segment but
    can be partial where no attribute happened to highlight some part of
    it. Falls back to the unique highlighted strings (offset-order lost)
    when no annotation carries usable offsets at all.
    """
    spans: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for ann in segment_annotations:
        for attr in ann.attributes.values():
            start = attr.get("startIndexInSegment")
            end = attr.get("endIndexInSegment")
            text = attr.get("selectedText")
            if start is None or end is None or start < 0 or end <= start or not text:
                continue
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            spans.append((start, end, text))

    if not spans:
        texts: list[str] = []
        seen_text: set[str] = set()
        for ann in segment_annotations:
            for attr in ann.attributes.values():
                text = attr.get("selectedText")
                if text and text not in seen_text:
                    seen_text.add(text)
                    texts.append(text)
        return " ".join(texts).strip()

    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    merged: list[tuple[int, int, str]] = []
    for start, end, text in spans:
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end, merged[-1][2])
            continue
        merged.append((start, end, text))

    pieces: list[str] = []
    prev_end: int | None = None
    for start, end, text in merged:
        if prev_end is not None and start > prev_end:
            pieces.append(_GAP_PLACEHOLDER)
        pieces.append(text)
        prev_end = end
    return "".join(pieces).strip()


def majority_attributes(
    segment_category_annotations: list[Annotation],
) -> dict[str, dict[str, Any]]:
    """Collapses multiple annotators' independent labeling of the same
    (policy_id, segment_id, category) into one attribute set, by majority
    vote per attribute name on its ``"value"`` (ties broken by first-seen
    order), keeping the longest available ``selectedText`` among the rows
    that agree on the winning value."""
    by_attr: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in segment_category_annotations:
        for attr_name, attr in ann.attributes.items():
            by_attr[attr_name].append(attr)

    resolved: dict[str, dict[str, Any]] = {}
    for attr_name, occurrences in by_attr.items():
        values = [o.get("value") for o in occurrences]
        winner = Counter(values).most_common(1)[0][0]
        candidates = [o for o in occurrences if o.get("value") == winner]
        best = max(candidates, key=lambda o: len(o.get("selectedText") or ""))
        resolved[attr_name] = best
    return resolved
