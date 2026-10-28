"""Build the canonical AU-AIR frame table.

Usage: python scripts/auair_frames.py

Writes cache/auair/frames.csv (one row per frame: clip, recording, part, index, image path,
time, GPS, altitude in metres, angles, velocities, box count, local NED) and
cache/auair/boxes.jsonl (boxes per frame); both are too large to commit and are rebuilt by
this script. Saves per-clip totals to results/auair/clips.csv and the dataset totals, checked
against the paper (differences logged, not corrected), to results/auair/totals.json.
"""

from __future__ import annotations

import json
import sys

from aerialrange.data.auair import frames_table
from aerialrange.utils.config import REPO_ROOT, load_config

PAPER = {"frames": 32823, "instances": 132034, "classes": 8}


def main() -> int:
    cfg = load_config("data")
    root = REPO_ROOT / cfg.require("data_root") / cfg.require("auair_subdir")
    au = load_config("auair")
    df = frames_table(root, au.require("clip_gap_ms"))
    out = REPO_ROOT / "results" / "auair"
    out.mkdir(parents=True, exist_ok=True)
    cache = REPO_ROOT / "cache" / "auair"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "boxes.jsonl").open("w") as fh:
        for name, boxes in zip(df["image_name"], df["boxes"], strict=True):
            fh.write(json.dumps({"image_name": name, "boxes": json.loads(boxes)}) + "\n")
    df.drop(columns="boxes").to_csv(cache / "frames.csv", index=False, float_format="%.7f")
    clips = df.groupby("clip").agg(
        recording=("recording", "first"),
        parts=("part", lambda p: ",".join(str(x) for x in sorted(set(p)))),
        frames=("clip", "size"),
        t_start_s=("t_ms", lambda t: t.min() / 1000),
        duration_s=("t_ms", lambda t: (t.max() - t.min()) / 1000),
        alt_min_m=("alt_m", "min"),
        alt_median_m=("alt_m", "median"),
        alt_max_m=("alt_m", "max"),
        boxes=("n_boxes", "sum"),
    )
    clips.to_csv(out / "clips.csv", float_format="%.2f")
    totals = {
        "frames": int(len(df)),
        "instances": int(df["n_boxes"].sum()),
        "recordings": int(df["recording"].nunique()),
        "clips": int(len(clips)),
        "clip_gap_ms": au.require("clip_gap_ms"),
        "paper": PAPER,
        "differences": {
            k: int(v) - PAPER[k]
            for k, v in (("frames", len(df)), ("instances", df["n_boxes"].sum()))
        },
    }
    (out / "totals.json").write_text(json.dumps(totals, indent=1) + "\n")
    print(clips.to_string())
    print(json.dumps(totals, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
