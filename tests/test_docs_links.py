"""Every file a published document links to or names must exist in the repository.

Covers markdown links and back-quoted repository paths in README.md and docs/*.md. Paths under
folders that are never committed (cache/, Data/, weights/), URLs, GitHub pages such as
../../releases, placeholders such as
<frames-or-video> and glob patterns are skipped.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOP = ("src/", "scripts/", "results/", "docs/", "configs/", "tests/", "gpu/")
SKIP = ("cache/", "Data/", "weights/")
GITHUB_PAGES = ("../../releases", "../../issues", "../../actions")  # repository pages, not files
DOCS = [
    ROOT / "README.md",
    *sorted((ROOT / "docs").glob("*.md")),
    *sorted((ROOT / "docs/src").glob("*.md")),
]


def _ignored(path: Path) -> bool:
    """True for files git ignores (local working files that are never published)."""
    import subprocess

    r = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=ROOT, capture_output=True)
    return r.returncode == 0


def referenced_paths(text: str) -> set[str]:
    links = re.findall(r"\]\(([^)\s]+)\)", text)
    links += re.findall(r'<img src="([^"]+)"', text)
    ticks = re.findall(r"`([^`\s]+)`", text)
    out = set()
    for p in links + ticks:
        p = p.split("#")[0].rstrip(".,;:")
        if (
            not p
            or "://" in p
            or any(c in p for c in "<>*{}$|")
            or p.startswith(SKIP + GITHUB_PAGES)
        ):
            continue
        if p.startswith(TOP) or (p in links and "/" in p):
            out.add(p)
    return out


@pytest.mark.parametrize("doc", [d for d in DOCS if not _ignored(d)], ids=lambda d: d.name)
def test_referenced_files_exist(doc: Path) -> None:
    base = ROOT if doc.parent in (ROOT, ROOT / "docs" / "src") else doc.parent
    missing = []
    for p in sorted(referenced_paths(doc.read_text(encoding="utf-8"))):
        candidates = [ROOT / p, base / p]
        if not any(c.exists() for c in candidates):
            missing.append(p)
    assert not missing, f"{doc.relative_to(ROOT)} names files that do not exist: {missing}"
