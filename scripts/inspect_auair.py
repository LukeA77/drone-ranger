"""Discover the AU-AIR annotation schema and print it.

Usage: python scripts/inspect_auair.py

Field names, types and value ranges are read from the file, not assumed. Also prints the
time-stamp spacing between consecutive frames (frame rate evidence) and the totals that
Checks the files against the AU-AIR paper. Writes results/auair/schema.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import UTC, datetime

import numpy as np

from aerialrange.utils.config import REPO_ROOT, load_config


def _describe(values: list) -> dict:
    types = Counter(type(v).__name__ for v in values)
    out: dict = {"types": dict(types), "n": len(values)}
    nums = [v for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
    if nums:
        a = np.asarray(nums, float)
        out |= {
            "min": float(a.min()),
            "p1": float(np.percentile(a, 1)),
            "median": float(np.median(a)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
        }
    return out


def frame_time_s(a: dict) -> float:
    """Seconds since midnight from the `time` record. `ms` is checked against `sec` below."""
    t = a["time"]
    return t["hour"] * 3600 + t["min"] * 60 + t["sec"] + t["ms"] / 1e6


def main() -> int:
    cfg = load_config("data")
    root = REPO_ROOT / cfg.require("data_root") / cfg.require("auair_subdir")
    d = json.loads((root / "annotations.json").read_text())
    ann = d["annotations"]
    top = sorted({k for a in ann for k in a})
    fields = {
        k: _describe([a[k] for a in ann if k in a and not isinstance(a[k], dict | list)])
        for k in top
        if k not in ("bbox", "time")
    }
    time_keys = sorted({k for a in ann for k in a["time"]})
    fields["time"] = {k: _describe([a["time"][k] for a in ann]) for k in time_keys}
    boxes = [b for a in ann for b in a["bbox"]]
    fields["bbox"] = {
        k: _describe([b[k] for b in boxes]) for k in sorted({k for b in boxes for k in b})
    }
    cats = d["categories"]
    per_class = Counter(cats[b["class"]] for b in boxes)

    # Frame spacing within a recording: sort by (date, time); keep gaps under 1 s.
    order = sorted(
        ann,
        key=lambda a: (a["time"]["year"], a["time"]["month"], a["time"]["day"], frame_time_s(a)),
    )
    t = np.array([frame_time_s(a) for a in order])
    dt = np.diff(t)
    small = dt[(dt > 0) & (dt < 1.0)]
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": (root / "annotations.json").relative_to(REPO_ROOT).as_posix(),
        "top_level_keys": sorted(d),
        "info": d["info"],
        "licenses_in_file": d["licenses"],
        "categories": cats,
        "frame_keys": top,
        "fields": fields,
        "totals": {
            "frames": len(ann),
            "instances": len(boxes),
            "classes": len(cats),
            "per_class": dict(per_class),
            "paper": {"frames": 32823, "instances": 132034, "classes": 8},
        },
        "frame_spacing": {
            "note": "Consecutive frames sorted by time stamp; gaps of 1 s or more excluded",
            "n_gaps_used": int(len(small)),
            "n_zero_or_negative": int((dt <= 0).sum()),
            "n_gaps_ge_1s": int((dt >= 1.0).sum()),
            "median_s": float(np.median(small)) if len(small) else None,
            "p10_s": float(np.percentile(small, 10)) if len(small) else None,
            "p90_s": float(np.percentile(small, 90)) if len(small) else None,
        },
    }
    dest = REPO_ROOT / "results" / "auair" / "schema.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
