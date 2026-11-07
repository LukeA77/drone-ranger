"""Fill {{RESULT:...}} tokens in documents from results files.

Token syntax: {{RESULT:<path under results/>#<dotted.key.path>|<format spec>}}
  e.g. {{RESULT:midair/test/summary.json#data.oracle_plane.land.abs_rel_err_median|.1%}}
List elements are addressed by index (ci.0). The format spec is Python's (optional).

Usage:
  python scripts/fill_docs.py SRC.md DEST.md     fill one document
  python scripts/fill_docs.py --check FILE...    exit 1 if any {{...}} remains (release check)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
TOKEN = re.compile(r"\{\{RESULT:([^#}]+)#([^|}]+)(?:\|([^}]*))?\}\}")
ANY_PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")


class FillError(KeyError):
    """A token refers to a file or key that does not exist."""


def _lookup(obj: Any, dotted: str, where: str) -> Any:
    """Follow a dotted key path. A dictionary key may itself contain dots (for example
    `range_coverage_0.90`): when a part is not a key, the shortest run of following
    parts that joins into one is used."""
    parts = dotted.split(".")
    i = 0
    while i < len(parts):
        if isinstance(obj, list):
            try:
                obj = obj[int(parts[i])]
            except (IndexError, ValueError) as e:
                raise FillError(f"{where}: no '{parts[i]}' in {dotted}") from e
            i += 1
            continue
        for j in range(i + 1, len(parts) + 1):
            key = ".".join(parts[i:j])
            if isinstance(obj, dict) and key in obj:
                obj, i = obj[key], j
                break
        else:
            raise FillError(f"{where}: no '{parts[i]}' in {dotted}")
    return obj


def fill(text: str, results_dir: Path = RESULTS) -> str:
    """Replace every RESULT token; raise FillError if any cannot be resolved."""
    cache: dict[str, Any] = {}

    def repl(m: re.Match) -> str:
        rel, key, spec = m.group(1), m.group(2), m.group(3)
        if rel not in cache:
            p = results_dir / rel
            if not p.exists():
                raise FillError(f"results file not found: {rel}")
            cache[rel] = json.loads(p.read_text(encoding="utf-8"))
        val = _lookup(cache[rel], key, rel)
        return format(val, spec) if spec else str(val)

    return TOKEN.sub(repl, text)


def unresolved(text: str) -> list[str]:
    """All remaining {{...}} placeholders."""
    return ANY_PLACEHOLDER.findall(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args(argv)
    if args.check:
        bad = {str(f): unresolved(f.read_text(encoding="utf-8")) for f in args.files}
        bad = {k: v for k, v in bad.items() if v}
        for k, v in bad.items():
            print(f"{k}: {len(v)} unresolved: {sorted(set(v))[:10]}")
        return 1 if bad else 0
    if len(args.files) != 2:
        ap.error("give SRC and DEST")
    src, dest = args.files
    dest.write_text(fill(src.read_text(encoding="utf-8")), encoding="utf-8")
    left = unresolved(dest.read_text(encoding="utf-8"))
    print(f"wrote {dest}; {len(left)} placeholders remain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
