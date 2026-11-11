"""Results files match their folder's schema.json (build2 Section 15: results with a schema).

A schema is a JSON Schema document; its `x-files` maps file-name patterns to definitions
(the first matching pattern applies). Only the keywords used in this repository's schemas are
checked here (type, required, properties, additionalProperties, items, minItems, maxItems,
minimum, maximum, $ref), so the schemas must not rely on others; a test checks that too.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

import pytest

from aerialrange.utils.config import REPO_ROOT

SCHEMAS = sorted((REPO_ROOT / "results").glob("*/schema.json"))
SCHEMAS = [p for p in SCHEMAS if "x-files" in json.loads(p.read_text(encoding="utf-8"))]
KNOWN = {
    "$schema", "$id", "$defs", "$ref", "title", "description", "x-files", "type", "required",
    "properties", "additionalProperties", "items", "minItems", "maxItems", "minimum", "maximum",
}  # fmt: skip
TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _is(value: object, t: str) -> bool:
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, TYPES[t])


def check(value: object, schema: dict, root: dict, where: str) -> list[str]:
    if "$ref" in schema:
        node = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            node = node[part]
        return check(value, node, root, where)
    errs = []
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is(value, t) for t in types):
            return [f"{where}: expected {types}, got {type(value).__name__}"]
    if isinstance(value, dict):
        errs += [f"{where}: missing '{k}'" for k in schema.get("required", []) if k not in value]
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties")
        for k, v in value.items():
            if k in props:
                errs += check(v, props[k], root, f"{where}.{k}")
            elif isinstance(extra, dict):
                errs += check(v, extra, root, f"{where}.{k}")
            elif extra is False:
                errs.append(f"{where}: unexpected '{k}'")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 1 << 62):
            errs.append(f"{where}: {len(value)} items")
        if "items" in schema:
            for i, v in enumerate(value):
                errs += check(v, schema["items"], root, f"{where}[{i}]")
    if _is(value, "number"):
        if value < schema.get("minimum", float("-inf")) or value > schema.get(
            "maximum", float("inf")
        ):
            errs.append(f"{where}: {value} out of range")
    return errs


def _schema_keys(node: dict, out: set[str]) -> None:
    """Keys used at schema positions (not property or definition names)."""
    out.update(node)
    for sub in node.get("$defs", {}).values():
        _schema_keys(sub, out)
    for sub in node.get("properties", {}).values():
        _schema_keys(sub, out)
    for k in ("additionalProperties", "items"):
        if isinstance(node.get(k), dict):
            _schema_keys(node[k], out)


def _files(schema_path: Path) -> list[tuple[Path, str]]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    out = []
    for f in sorted(schema_path.parent.rglob("*.json")):
        rel = f.relative_to(schema_path.parent).as_posix()  # patterns match the path inside
        if rel == "schema.json":
            continue
        ref = next((r for pat, r in schema["x-files"].items() if fnmatch.fnmatch(rel, pat)), None)
        out.append((f, ref))
    return out


@pytest.mark.parametrize("schema_path", SCHEMAS, ids=lambda p: p.parent.name)
def test_schema_uses_only_checked_keywords(schema_path: Path) -> None:
    used: set[str] = set()
    _schema_keys(json.loads(schema_path.read_text(encoding="utf-8")), used)
    assert used <= KNOWN, f"unchecked keywords {used - KNOWN}"


@pytest.mark.parametrize("schema_path", SCHEMAS, ids=lambda p: p.parent.name)
def test_results_match_schema(schema_path: Path) -> None:
    root = json.loads(schema_path.read_text(encoding="utf-8"))
    errs = []
    for f, ref in _files(schema_path):
        if ref is None:
            errs.append(f"{f.relative_to(schema_path.parent).as_posix()}: no pattern in x-files")
            continue
        value = json.loads(f.read_text(encoding="utf-8"))
        errs += check(value, {"$ref": ref}, root, f.relative_to(schema_path.parent).as_posix())
    assert not errs, "\n".join(errs[:20])


def test_checker_catches_errors() -> None:
    root = {"$defs": {"p": {"type": "number", "minimum": 0, "maximum": 1}}}
    s = {
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"$ref": "#/$defs/p"}, "b": {"type": "array", "maxItems": 1}},
    }
    assert check({"a": 0.5, "b": [1]}, s, root, "x") == []
    assert check({}, s, root, "x") == ["x: missing 'a'"]
    assert check({"a": 2}, s, root, "x") == ["x.a: 2 out of range"]
    assert check({"a": True}, s, root, "x")  # a bool is not a number
    assert check({"a": 0.1, "b": [1, 2]}, s, root, "x") == ["x.b: 2 items"]
