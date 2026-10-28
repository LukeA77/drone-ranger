"""Parse Manual clip labels from docs/auair_clip_notes.md.

Usage: python scripts/auair_labels.py

Writes results/auair/clip_labels.csv: clip, frames, view (down, oblique or mixed), tilt
below the horizon in degrees as labelled (one value for oblique; min and max for mixed; 90
for down), curvature, and the label as written. Labelled by eye from contact sheets, so tilts
are rough (D-504). A misspelt view is normalised and listed.
"""

from __future__ import annotations

import csv
import re
import sys

from aerialrange.utils.config import REPO_ROOT

FIX = {"obique": "oblique"}


def main() -> int:
    rows, fixed = [], []
    text = (REPO_ROOT / "docs" / "auair_clip_notes.md").read_text(encoding="utf-8")
    for line in text.splitlines():
        c = [x.strip().strip("`") for x in line.strip().strip("|").split("|")]
        if len(c) != 10 or not c[0].isdigit():
            continue
        label = c[8]
        m = re.fullmatch(r"(\w+)\s*(?:~?(\d+)(?:\s*-\s*(\d+))?)?", label)
        if m is None:
            raise ValueError(f"clip {c[0]}: cannot parse view label {label!r}")
        view = FIX.get(m.group(1), m.group(1))
        if view != m.group(1):
            fixed.append(f"clip {c[0]}: {m.group(1)} -> {view}")
        if view not in ("down", "oblique", "mixed"):
            raise ValueError(f"clip {c[0]}: unknown view {view!r}")
        lo = 90.0 if view == "down" else (float(m.group(2)) if m.group(2) else None)
        hi = float(m.group(3)) if m.group(3) else lo
        rows.append(
            {
                "clip": int(c[0]),
                "frames": int(c[2]),
                "view": view,
                "tilt_min_deg": lo,
                "tilt_max_deg": hi,
                "curvature": c[9],
                "label_as_written": label,
            }
        )
    dest = REPO_ROOT / "results" / "auair" / "clip_labels.csv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} clips; normalised: {fixed or 'none'}; wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
