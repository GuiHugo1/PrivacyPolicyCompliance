"""JSON-verdict extraction and validation against ``judge/judge_schema.json``.

Deliberately dependency-free (no ``jsonschema`` package): the schema this
project needs is one flat object, so a small hand-rolled subset validator
(``type``/``enum``/``required``/``minimum``/``maximum``/
``additionalProperties``) covers it without adding a dependency the rest of
the ``judge`` group doesn't otherwise need.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SCHEMA_PATH = Path(__file__).parent / "judge_schema.json"

_SIMPLE_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "object": dict,
    "array": list,
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


def _check_type(key: str, value: Any, expected_type: str | None) -> str | None:
    if expected_type is None:
        return None
    if expected_type == "boolean":
        if not isinstance(value, bool):
            return f"{key!r}: expected boolean, got {type(value).__name__}"
        return None
    if expected_type in ("number", "integer"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{key!r}: expected {expected_type}, got {type(value).__name__}"
        if expected_type == "integer" and not isinstance(value, int):
            return f"{key!r}: expected integer, got float"
        return None
    py_type = _SIMPLE_TYPE_MAP.get(expected_type)
    if py_type is not None and not isinstance(value, py_type):
        return f"{key!r}: expected {expected_type}, got {type(value).__name__}"
    return None


def validate_against_schema(obj: Any, schema: dict[str, Any]) -> list[str]:
    """Validates ``obj`` against a flat JSON-schema-subset. Returns a list of
    human-readable error strings; an empty list means valid."""
    if not isinstance(obj, dict):
        return [f"expected a JSON object, got {type(obj).__name__}"]

    errors: list[str] = []
    for key in schema.get("required", []):
        if key not in obj:
            errors.append(f"missing required key: {key!r}")

    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        for key in obj:
            if key not in properties:
                errors.append(f"unexpected key: {key!r}")

    for key, subschema in properties.items():
        if key not in obj:
            continue
        value = obj[key]

        type_error = _check_type(key, value, subschema.get("type"))
        if type_error:
            errors.append(type_error)
            continue

        if "enum" in subschema and value not in subschema["enum"]:
            errors.append(f"{key!r}: {value!r} not in enum {subschema['enum']}")
        if (
            "minimum" in subschema
            and isinstance(value, (int, float))
            and value < subschema["minimum"]
        ):
            errors.append(f"{key!r}: {value} < minimum {subschema['minimum']}")
        if (
            "maximum" in subschema
            and isinstance(value, (int, float))
            and value > subschema["maximum"]
        ):
            errors.append(f"{key!r}: {value} > maximum {subschema['maximum']}")

    return errors


def is_valid_verdict(text: str, schema: dict[str, Any]) -> bool:
    obj = extract_json_object(text)
    if obj is None:
        return False
    return not validate_against_schema(obj, schema)
