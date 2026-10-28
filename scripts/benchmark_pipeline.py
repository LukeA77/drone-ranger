"""End-to-end CPU latency of read, detect and track.

Usage: python scripts/benchmark_pipeline.py --run B2_yolo26n_960 --weights <best.pt> --imgsz 960
       [--sequence uav0000086_00000_v] [--note "what else was running"]

Stages: `read` (JPEG decode from disk), `detect` (Ultralytics predict: preprocess, inference
and post-processing, score floor 0.1 as in tracking), and `track` (BoT-SORT with camera-motion
compensation and the tuned thresholds for this detector, configs/tracker.yaml `tuned`). The
library's own preprocess / inference / postprocess split of `detect` is reported beside the
stage timings. Ranging and the Kalman filter are not yet in the pipeline and are
not timed here; this is stated in the output. Frames come from one MOT val sequence, in order.
Protocol sizes from configs/edge.yaml. Output: results/edge/benchmark_<run>.json, with the
budget verdict at {{TARGET_FPS}} and at the reference rate. Run with nothing else heavy running
, and say in --note what else was running.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from aerialrange.edge.benchmark import (
    BenchmarkProtocol,
    background_cpu,
    budget_verdict,
    run_benchmark,
)
from aerialrange.track.runner import DetSet, coast, make_tracker
from aerialrange.utils.config import REPO_ROOT, latency_budget_ms, load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--class-map", default="visdrone10")
    ap.add_argument("--sequence", default="uav0000086_00000_v")
    ap.add_argument("--note", default="", help="other processes running during timing")
    ap.add_argument("--smoke", action="store_true", help="tiny protocol to test the script")
    ap.add_argument("--tuned-from", default=None, help="tracker.yaml `tuned` key (default --run)")
    ap.add_argument("--gmc-downscale", type=int, default=None, help="camera-motion downscale")
    ap.add_argument("--high", type=float, default=None, help="override track_high_thresh")
    ap.add_argument("--new", type=float, default=None, help="override new_track_thresh")
    ap.add_argument("--tag", default=None, help="output name (default --run)")
    ap.add_argument("--tracker", default="botsort_cmc", help="configs/trackers/<name>.yaml")
    ap.add_argument("--det-stride", type=int, default=1, help="detect every n-th frame (D-610)")
    args = ap.parse_args(argv)
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    edge, det, trk = load_config("edge"), load_config("detector"), load_config("tracker")
    proto = (
        BenchmarkProtocol(2, 5, 1)
        if args.smoke
        else BenchmarkProtocol(
            edge.require("n_warmup"), edge.require("n_timed_frames"), edge.require("n_repeats")
        )
    )
    cmap = {int(k): v for k, v in det.require("class_maps")[args.class_map].items()}
    overrides = dict(trk.require("tuned")[args.tuned_from or args.run])
    if args.gmc_downscale is not None:
        overrides["gmc_downscale"] = args.gmc_downscale
    if args.high is not None:
        overrides["track_high_thresh"] = args.high
    if args.new is not None:
        overrides["new_track_thresh"] = args.new
    tracker_cfg = str(REPO_ROOT / "configs" / "trackers" / f"{args.tracker}.yaml")
    model = YOLO(args.weights)
    info = json.loads(
        (
            REPO_ROOT / "cache" / "visdrone" / "mot" / "val" / args.sequence / "seqinfo.json"
        ).read_text()
    )
    frames = sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))
    tracker = make_tracker(tracker_cfg, overrides)
    lib_split: list[dict[str, float]] = []

    def read(p: Path) -> np.ndarray:
        return cv2.imread(str(p))

    counter = {"k": 0}

    def detect(img: np.ndarray) -> tuple[np.ndarray, DetSet | None]:
        k = counter["k"]
        counter["k"] += 1
        if k % args.det_stride:
            return img, None  # between detections: the tracker coasts
        r = model.predict(
            img,
            imgsz=args.imgsz,
            classes=sorted(cmap),
            max_det=det.require("max_det"),
            conf=0.1,
            verbose=False,
        )[0]
        lib_split.append(dict(r.speed))
        b = r.boxes
        return img, DetSet(b.xywh.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy())

    def track(x: tuple[np.ndarray, DetSet | None]) -> object:
        img, d = x
        return coast(tracker, img) if d is None else tracker.update(d, img)

    background = background_cpu()
    res = run_benchmark([("read", read), ("detect", detect), ("track", track)], frames, proto)
    split = {
        k: float(np.median([s[k] for s in lib_split[proto.n_warmup // args.det_stride :]]))
        for k in ("preprocess", "inference", "postprocess")
    }
    verdicts = {
        f"{edge.require('target_fps')}_fps": budget_verdict(res["stages"], latency_budget_ms(edge)),
        f"{edge.require('reference_fps')}_fps": budget_verdict(
            res["stages"], 1000.0 / edge.require("reference_fps")
        ),
    }
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "tag": args.tag or args.run,
        "weights": Path(args.weights).name,
        "imgsz": args.imgsz,
        "tracker": {"config": Path(tracker_cfg).name, "overrides": overrides},
        "det_stride": args.det_stride,
        "frames": f"MOT val {args.sequence} ({info['width']}x{info['height']}), in order; "
        "cycled per repeat",
        "not_timed": "ranging and Kalman filter (not yet integrated); overlay drawing",
        "smoke": args.smoke,
        "machine": {
            "cpu": platform.processor(),
            "logical_cores": __import__("os").cpu_count(),
            "os": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
        },
        "other_processes": args.note,
        "background_measured": background,
        **res,
        "detect_library_split_median_ms": split,
        "budget": verdicts,
    }
    dest = (
        REPO_ROOT
        / "results"
        / "edge"
        / f"benchmark_{args.tag or args.run}{'_smoke' if args.smoke else ''}.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    e = res["stages"]["end_to_end"]
    print(
        f"{args.run}: end-to-end median {e['median_ms']:.1f} ms, p99 {e['p99_ms']:.1f} ms, "
        f"{res['throughput_fps']:.1f} fps; "
        + ", ".join(f"{k} {res['stages'][k]['median_ms']:.1f}" for k in ("read", "detect", "track"))
        + f"; wrote {dest}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
