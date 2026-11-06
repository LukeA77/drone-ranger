"""Build SynDrone per-frame base height (z0) and per-object truth for one town.

Streams depth from the remote archive (not kept on disk, D-702); boxes, cameras and semantic
labels are read from the local cache if present (scripts/syndrone_fetch.py), else streamed.
Work is split into chunks of frames run in parallel processes; each chunk writes
`cache/syndrone/truth/<town>/chunk_<a>_<b>.json.gz` atomically, so a rerun resumes.

Every frame's z0 is fitted jointly from all three cameras (`estimate_z0`), and truth rows are
computed with that frame's z0. Geometry that depends on z0 (ranges, projected boxes) is
recomputed later from the smoothed z0 (scripts/syndrone_z0_check.py); instance masks and
visibility do not depend on z0.
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import sys
import time

from aerialrange.data.remote_zip import write_atomic
from aerialrange.data.syndrone import N_FRAMES, SynDroneSequence
from aerialrange.data.syndrone_archive import HEIGHTS, MemberStore, free_gb
from aerialrange.evaluate.syndrone_truth import View, estimate_z0, object_truth
from aerialrange.utils.config import REPO_ROOT, load_config

OUT_ROOT = REPO_ROOT / "cache" / "syndrone" / "truth"


def _store() -> MemberStore:
    d = load_config("data")
    root = REPO_ROOT / d.require("data_root") / d.require("syndrone_subdir")
    return MemberStore(root, d.require("syndrone_base_url"))


def run_chunk(args: tuple[str, int, int]) -> tuple[int, int, float]:
    town, a, b = args
    out = OUT_ROOT / town / f"chunk_{a:05d}_{b:05d}.json.gz"
    if out.exists():
        return a, b, 0.0
    t0 = time.time()
    store = _store()
    seqs = {h: SynDroneSequence(store, town, h) for h in HEIGHTS}
    frames = range(a, b)
    depth = {}
    for h, s in seqs.items():  # one merged range request per camera for the whole chunk
        names = [s.member("depth", k) for k in frames]
        got = dict(store.iter_read(names))
        depth[h] = {k: got[n] for k, n in zip(frames, names, strict=True)}
    from aerialrange.data.syndrone import decode_depth

    recs = []
    for k in frames:
        boxes = seqs[80].boxes(k)
        views = [
            View(h, s.camera_record(k), decode_depth(depth[h][k]), s.semantic(k), s.camera)
            for h, s in seqs.items()
        ]
        fit = estimate_z0(views, boxes)
        z0_used = fit.z0 if fit.n_boxes else 0.0
        objs = {v.height: object_truth(v, boxes, z0_used) for v in views}
        recs.append(
            {
                "frame": k,
                "z0": fit.z0,
                "z0_se": fit.se,
                "z0_scale": fit.scale,
                "z0_n": fit.n_boxes,
                "z0_used": z0_used,
                "per_view": {str(h): list(v) for h, v in fit.per_view.items()},
                "offsets": fit.offsets.tolist(),
                "cameras": {str(v.height): vars(v.record) for v in views},
                "objects": {str(h): rows for h, rows in objs.items()},
            }
        )
    write_atomic(out, gzip.compress(json.dumps(recs).encode("utf-8")))
    return a, b, time.time() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=N_FRAMES)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args(argv)
    floor = float(load_config("data").require("disk_floor_gb"))
    if free_gb(REPO_ROOT) < floor:
        print(f"free space below {floor} GB; stopping")
        return 2
    (OUT_ROOT / args.town).mkdir(parents=True, exist_ok=True)
    _store()._zip(args.town, "depth")  # cache the depth index once, before the workers start
    jobs = [
        (args.town, a, min(a + args.chunk, args.stop))
        for a in range(args.start, args.stop, args.chunk)
    ]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (a, b, dt) in enumerate(pool.imap_unordered(run_chunk, jobs), 1):
            el = time.time() - t0
            print(
                f"[{i}/{len(jobs)}] frames {a}-{b} in {dt:.0f}s; elapsed {el / 60:.1f} min",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
