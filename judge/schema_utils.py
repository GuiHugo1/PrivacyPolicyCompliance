"""JSON extraction and validation against ``judge/judge_schema.json`` (the
per-verdict schema the fine-tuned judge must emit) and ``judge/output_schema.json``
(the nested schema ``judge/pipeline.py`` assembles those verdicts into).

Deliberately dependency-free (no ``jsonschema`` package): both schemas this
project needs are covered by a small hand-rolled subset validator --
``type`` (including a JSON-Schema-style union list like ``["integer",
"null"]``), ``enum``, ``required``, ``minimum``/``maximum``,
``additionalProperties``, nested ``object``/``array`` (``properties``/
``items``), and ``$ref``/``$defs`` -- without adding a dependency the rest
of the ``judge`` group doesn't otherwise need. See
``judge/tests/test_schema_utils.py``'s ``output_schema.json``-based tests
for this validating the nested/``$ref`` shape, not just the flat
``judge_schema.json`` shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SCHEMA_PATH = Path(__file__).parent / "judge_schema.json"

_SIMPLE_TYPE_MAP: dict[str, type] = {
    "string": str,
    "object": dict,
    "array": list,
    "null": type(None),
}


def load_schema(path: Path | str = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_json_object(text: str) -> Any | None:
    """Best-effort extraction of a JSON value from model output text.

    Tries a strict ``json.loads`` first (the expected case, since the judge
    is prompted to emit ONLY a JSON object), then falls back to slicing the
    first ``{`` .. last ``}`` span to tolerate stray leading/trailing text
    (e.g. a chatty preamble or trailing whitespace/markdown fence).
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type in ("number", "integer"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return expected_type == "number" or isinstance(value, int)
    py_type = _SIMPLE_TYPE_MAP.get(expected_type)
    return py_type is None or isinstance(value, py_type)


def _check_type(label: str, value: Any, expected_type: str | list[str] | None) -> str | None:
    """``expected_type`` may be a single JSON-Schema type name, or a union
    list like ``["integer", "null"]`` (``judge/output_schema.json`` uses this
    for nullable fields, e.g. ``fetch_k``) -- valid if ``value`` matches any
    one of them."""
    if expected_type is None:
        return None
    types = expected_type if isinstance(expected_type, list) else [expected_type]
    if any(_matches_type(value, t) for t in types):
        return None
    return f"{label}: expected {' or '.join(types)}, got {type(value).__name__}"


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _resolve_schema(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Follows a ``{"$ref": "#/$defs/name"}`` indirection to the actual
    subschema in ``root["$defs"]`` -- the only ``$ref`` shape
    ``judge/output_schema.json`` uses -- so callers always see a schema with
    real ``type``/``properties``/``enum``/etc. keys."""
    seen: set[str] = set()
    while "$ref" in schema:
        ref = schema["$ref"]
        if ref in seen:
            raise ValueError(f"circular $ref: {ref}")
        seen.add(ref)
        if not ref.startswith("#/$defs/"):
            raise ValueError(f"unsupported $ref (only '#/$defs/<name>' is): {ref}")
        schema = root["$defs"][ref.removeprefix("#/$defs/")]
    return schema


def validate_against_schema(
    obj: Any,
    schema: dict[str, Any],
    *,
    root: dict[str, Any] | None = None,
    path: str = "",
) -> list[str]:
    """Validates ``obj`` against a JSON-schema subset -- flat or nested.
    Returns a list of human-readable error strings, each prefixed with its
    dotted/indexed location (e.g. ``"articles[2].best_compliance_status"``);
    an empty list means valid.

    Handles ``type`` (a single name or a union list), ``enum``,
    ``required``, ``minimum``/``maximum``, ``additionalProperties``, nested
    ``object``/``array`` (recursing into ``properties``/``items``), and
    ``$ref``/``$defs`` indirection -- covering both the flat
    ``judge/judge_schema.json`` and the nested ``judge/output_schema.json``.

    ``root``/``path`` are for internal recursion (``root`` is the top-level
    schema doc, needed to resolve ``$ref`` at any depth; ``path`` is the
    dotted location built up so far) -- callers validating a whole object
    against its own schema only ever need to pass ``obj``/``schema``.
    """
    root = schema if root is None else root
    resolved = _resolve_schema(schema, root)
    label = path or "value"
    schema_type = resolved.get("type")

    is_array = schema_type == "array" or (schema_type is None and "items" in resolved)
    is_object = schema_type == "object" or (
        schema_type is None and ("properties" in resolved or "required" in resolved)
    )

    if is_array:
        if not isinstance(obj, list):
            return [f"{label}: expected array, got {type(obj).__name__}"]
        errors: list[str] = []
        item_schema = resolved.get("items")
        if item_schema is not None:
            for i, item in enumerate(obj):
                errors.extend(
                    validate_against_schema(item, item_schema, root=root, path=f"{path}[{i}]")
                )
        return errors

    if is_object:
        if not isinstance(obj, dict):
            return [f"{label}: expected a JSON object, got {type(obj).__name__}"]

        errors = []
        for key in resolved.get("required", []):
            if key not in obj:
                errors.append(f"{_join(path, key)}: missing required key")

        properties = resolved.get("properties", {})
        if resolved.get("additionalProperties") is False:
            for key in obj:
                if key not in properties:
                    errors.append(f"{_join(path, key)}: unexpected key")

        for key, subschema in properties.items():
            if key not in obj:
                continue
            errors.extend(
                validate_against_schema(obj[key], subschema, root=root, path=_join(path, key))
            )
        return errors

    # Leaf value: string / number / integer / boolean / null, or a union of
    # these (e.g. output_schema.json's `["integer", "null"]`).
    type_error = _check_type(label, obj, schema_type)
    if type_error:
        return [type_error]

    errors = []
    if "enum" in resolved and obj not in resolved["enum"]:
        errors.append(f"{label}: {obj!r} not in enum {resolved['enum']}")
    if "minimum" in resolved and isinstance(obj, (int, float)) and obj < resolved["minimum"]:
        errors.append(f"{label}: {obj} < minimum {resolved['minimum']}")
    if "maximum" in resolved and isinstance(obj, (int, float)) and obj > resolved["maximum"]:
        errors.append(f"{label}: {obj} > maximum {resolved['maximum']}")
    return errors


def is_valid_verdict(text: str, schema: dict[str, Any]) -> bool:
    obj = extract_json_object(text)
    if obj is None:
        return False
    return not validate_against_schema(obj, schema)
