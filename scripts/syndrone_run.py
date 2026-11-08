"""Run `aerialrange run` on one SynDrone sequence (town, camera height), frames streamed.

Frames come from the local cache or the remote archive (not kept, D-702), read
ahead in merged spans by a background thread. The pipeline gets the CARLA camera (90 deg
horizontal field of view) and, per frame, the camera's Unreal x, y as north, east, its
heading (yaw), tilt (-pitch) and height above the floor `h` (20, 50 or 80 m; SynDrone's
base height z0 is the floor height, `data_handling.py` keyframes_loader). That is version 1's
flat-ground ranging given the true height. With propagated measurement errors (D-735) the
pose error terms (tilt, height, heading) are set to zero, since the poses are exact.
Writes `cache/syndrone/runs/<name>/<town>_h<h>/`:
tracks.csv, detections.csv (raw, before tracking) and timing.json.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
from collections.abc import Iterator

import numpy as np
import pandas as pd

from aerialrange.data.syndrone import FPS, N_FRAMES, SynDroneSequence, decode_rgb, syndrone_camera
from aerialrange.data.syndrone_archive import MemberStore
from aerialrange.pipeline.intervals import config_from_calibration
from aerialrange.pipeline.range_speed import exact_pose_overrides
from aerialrange.pipeline.run import run
from aerialrange.utils.config import REPO_ROOT, load_config

OUT = REPO_ROOT / "cache" / "syndrone" / "runs"


def stream_frames(
    store: MemberStore, seq: SynDroneSequence, frames: range, ahead: int = 4, chunk: int = 50
) -> Iterator[np.ndarray]:
    """Yield decoded frames in order; a thread fetches `chunk` frames per request."""
    q: queue.Queue = queue.Queue(maxsize=ahead)

    def producer() -> None:
        try:
            for a in range(frames.start, frames.stop, chunk):
                ks = list(range(a, min(a + chunk, frames.stop)))
                names = [seq.member("rgb", k) for k in ks]
                got = dict(store.iter_read(names))
                q.put([got[n] for n in names])
        except Exception as e:  # surfaced in the consumer
            q.put(e)
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    while (item := q.get()) is not None:
        if isinstance(item, Exception):
            raise item
        for raw in item:
            yield decode_rgb(raw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--config", default="run_accurate")
    ap.add_argument("--name", default="v1_accurate")
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument(
        "--weights",
        default=None,
        help="detector weights replacing the config's (e.g. a fine-tuning fold's model)",
    )
    ap.add_argument("--kind", default=None, help="detector kind with --weights (yolo, rfdetr)")
    ap.add_argument("--imgsz", type=int, default=None, help="detector input size with --weights")
    ap.add_argument(
        "--intervals", help="interval calibration result to apply (D-737; SynDrone conditions)"
    )
    args = ap.parse_args(argv)
    splits = json.loads((REPO_ROOT / "results" / "splits.json").read_text(encoding="utf-8"))
    if args.town in splits["syndrone"]["splits"]["test"] and not args.allow_test:
        print(f"{args.town} is a test town: run once, after freezing; pass --allow-test")
        return 2
    d = load_config("data")
    store = MemberStore(
        REPO_ROOT / d.require("data_root") / d.require("syndrone_subdir"),
        d.require("syndrone_base_url"),
    )
    seq = SynDroneSequence(store, args.town, args.height)
    recs = seq.camera_records(range(args.frames))
    meta = pd.DataFrame(
        {
            "frame": range(1, args.frames + 1),  # the pipeline counts frames from 1
            "t_s": [k / FPS for k in range(args.frames)],
            "n_m": [r.x for r in recs],
            "e_m": [r.y for r in recs],
            "h_m": [r.h for r in recs],
            "yaw_rad": [math.radians(r.yaw) for r in recs],
            "tilt_deg": [-r.pitch for r in recs],
        }
    )
    cam = syndrone_camera()
    out = OUT / args.name / f"{args.town}_h{args.height}"
    dets: list[dict] = []
    overrides: dict = {"camera": {"focal_px": cam.fx, "cx": cam.cx, "cy": cam.cy}}
    overrides |= exact_pose_overrides(load_config(args.config).require("kalman"))
    if args.intervals:
        overrides["intervals"] = config_from_calibration(REPO_ROOT / args.intervals)
    if args.weights:
        if not (REPO_ROOT / args.weights).is_file():
            print(f"weights not found: {args.weights}")
            return 1
        overrides["detector"] = {"weights": args.weights}
        if args.kind:
            overrides["detector"]["kind"] = args.kind
        if args.imgsz:
            overrides["detector"]["imgsz"] = args.imgsz
    timing = run(
        stream_frames(store, seq, range(args.frames)),
        meta,
        out,
        config=args.config,
        overrides=overrides,
        detections_out=dets,
        write_video=False,
    )
    (out / "detector.json").write_text(
        json.dumps(
            {
                "weights": args.weights or "config",
                "kind": args.kind or "config",
                "imgsz": args.imgsz or "config",
                "config": args.config,
            }
        )
        + "\n"
    )
    pd.DataFrame(dets, columns=["frame", "left", "top", "width", "height", "score", "cls"]).to_csv(
        out / "detections.csv", index=False
    )
    e2e = timing["stages"].get("end_to_end", {})
    print(
        f"{args.town} h{args.height}: {timing['frames']} frames, "
        f"median {e2e.get('median_ms', float('nan')):.0f} ms per frame"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
