"""Collect every SynDrone actor's footprint-centre track for one town.

Reads the per-frame box files (local cache first) and writes
`cache/syndrone/actors/<town>.npz` with, for each actor (non-level box): id, tag, frames,
footprint centres (Unreal world) and box yaw (degrees, from the box x axis).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from aerialrange.data.syndrone import N_FRAMES, SynDroneSequence, parse_boxes
from aerialrange.data.syndrone_archive import MemberStore
from aerialrange.evaluate.syndrone_truth import box_frames
from aerialrange.utils.config import REPO_ROOT, load_config

OUT = REPO_ROOT / "cache" / "syndrone" / "actors"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", required=True)
    args = ap.parse_args(argv)
    d = load_config("data")
    store = MemberStore(
        REPO_ROOT / d.require("data_root") / d.require("syndrone_subdir"),
        d.require("syndrone_base_url"),
    )
    seq = SynDroneSequence(store, args.town, 80)
    names = [seq.member("bboxes", k) for k in range(N_FRAMES)]
    rows: dict[int, list] = {}
    tags: dict[int, int] = {}
    n_level = set()
    for name, raw in store.iter_read(names):
        k = int(name.rsplit("/", 1)[1].split(".")[0])
        b = parse_boxes(raw)
        n_level.add(int(b.level.sum()))
        a = b.select(~b.level)
        _, axes, _ = box_frames(a.corners)
        yaw = np.degrees(np.arctan2(axes[:, 0, 1], axes[:, 0, 0]))
        for i, fc, y, t in zip(a.ids, a.footprint_centre, yaw, a.tags, strict=True):
            rows.setdefault(int(i), []).append((k, *fc, y))
            tags[int(i)] = int(t)
    ids = np.array(sorted(rows))
    out = {
        "ids": ids,
        "tags": np.array([tags[i] for i in ids]),
        "n_level": np.array(sorted(n_level)),
    }
    for i in ids:
        r = np.array(sorted(rows[i]))
        out[f"a{i}"] = r  # columns: frame, x, y, z, yaw_deg
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{args.town}.npz", **out)
    print(f"{len(ids)} actors; level box counts seen: {sorted(n_level)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
