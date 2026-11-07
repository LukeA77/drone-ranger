"""End-to-end CPU latency of the version 2 pipeline's read, detect and track stages.

Usage: python scripts/benchmark_v2.py [--config run] [--tag laptop] [--frames-dir DIR]
       [--threads N] [--note "what else was running"] [--smoke]
       [--kind rfdetr --weights weights/rfdetr_nano_640.onnx --imgsz 640]

The version 1 benchmark protocol (configs/edge.yaml: warm-up, timed frames, repeats; stage
medians and percentiles; budget verdicts at the target and reference rates), applied to the
version 2 pipeline as configured (configs/<config>.yaml): the detector through the project's
own ONNX runner (`detect.factory`, no training library), and the project's tracker with
camera-motion compensation, coasting between detections at the configured stride. `detect` is
split into pre-processing, inference and post-processing. Nothing here needs PyTorch or
Ultralytics, so it runs unchanged on a Raspberry Pi 5 (`make pi-benchmark`).

Frames: a VisDrone MOT val sequence, in order (default uav0000086_00000_v), or any folder of
JPEG frames (--frames-dir), which must be stated with the result. --kind, --weights and
--imgsz replace the configured detector (to time each detector candidate the same way).
Ranging, the Kalman filter and overlay drawing are not timed. Run with nothing else heavy running.
Output: results/edge/benchmark_v2_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from aerialrange.detect.factory import DETR_NUM_SELECT
from aerialrange.detect.onnx_detectors import DFINEOnnx, RFDETROnnx, YoloOnnx
from aerialrange.edge.benchmark import (
    BenchmarkProtocol,
    background_cpu,
    budget_verdict,
    run_benchmark,
)
from aerialrange.track.runner import DetSet, coast, make_tracker
from aerialrange.utils.config import REPO_ROOT, latency_budget_ms, load_config

RUNNERS = {"yolo": YoloOnnx, "rfdetr": RFDETROnnx, "dfine": DFINEOnnx}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="run")
    ap.add_argument("--tag", default="laptop")
    ap.add_argument("--sequence", default="uav0000086_00000_v")
    ap.add_argument("--frames-dir", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=None, help="ONNX Runtime intra-op threads")
    ap.add_argument("--note", default="", help="other processes running during timing")
    ap.add_argument("--smoke", action="store_true", help="tiny protocol to test the script")
    ap.add_argument("--kind", choices=sorted(RUNNERS), default=None)
    ap.add_argument("--weights", default=None, help="ONNX model, relative to the repository")
    ap.add_argument("--imgsz", type=int, default=None)
    args = ap.parse_args(argv)
    import cv2

    edge, det_cfg, run_cfg = load_config("edge"), load_config("detector"), load_config(args.config)
    det, trk = dict(run_cfg.require("detector")), run_cfg.require("tracker")
    for key in ("kind", "weights", "imgsz"):
        if getattr(args, key) is not None:
            det[key] = getattr(args, key)
    proto = (
        BenchmarkProtocol(2, 5, 1)
        if args.smoke
        else BenchmarkProtocol(
            edge.require("n_warmup"), edge.require("n_timed_frames"), edge.require("n_repeats")
        )
    )
    cmap = {int(k): v for k, v in det_cfg.require("class_maps")["visdrone10"].items()}
    keep = np.array(sorted(cmap))
    kind = det.get("kind", "yolo")
    weights = REPO_ROOT / det["weights"]
    if weights.suffix != ".onnx":
        print("the version 2 benchmark runs ONNX models only")
        return 1
    max_det = int(det_cfg.require("max_det"))
    if kind != "yolo":
        max_det = min(max_det, DETR_NUM_SELECT)  # as in detect.factory
    runner = RUNNERS[kind](weights, float(det["score_floor"]), max_det, threads=args.threads)
    if int(det["imgsz"]) not in (runner.w, runner.h):
        print(f"{weights.name} is exported for {runner.w} px, not {det['imgsz']}")
        return 1
    if args.frames_dir:
        frames = sorted(args.frames_dir.glob("*.jpg"))
        source = f"{args.frames_dir.name} ({len(frames)} JPEG frames)"
    else:
        seq = REPO_ROOT / "cache" / "visdrone" / "mot" / "val" / args.sequence
        info = json.loads((seq / "seqinfo.json").read_text())
        frames = sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))
        source = f"MOT val {args.sequence} ({info['width']}x{info['height']})"
    if not frames:
        print("no frames found")
        return 1
    tracker = make_tracker(
        str(REPO_ROOT / trk["config"]),
        {"gmc_downscale": trk["gmc_downscale"]} if trk.get("gmc_downscale") else None,
    )
    stride = int(trk.get("det_stride", 1))
    split: list[tuple[float, float, float]] = []
    counter = {"k": 0}

    def read(p: Path) -> np.ndarray:
        return cv2.imread(str(p))

    def detect(img: np.ndarray) -> tuple[np.ndarray, DetSet | None]:
        k = counter["k"]
        counter["k"] += 1
        if k % stride:
            return img, None  # between detections the tracker coasts (D-610)
        t0 = time.perf_counter()
        if kind == "yolo":
            x, geom = runner.letterbox(img)
        else:
            x = runner.preprocess(img)
        t1 = time.perf_counter()
        out = runner.session.run(
            None if kind == "yolo" else list(runner.output_names), {runner.input_name: x}
        )
        t2 = time.perf_counter()
        h, w = img.shape[:2]
        if kind == "yolo":
            d = runner.postprocess(out[0][0], geom, h, w)
        else:
            d = runner.postprocess(out[0][0], out[1][0], h, w)
        m = np.isin(d.cls, keep)
        t3 = time.perf_counter()
        split.append(((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000))
        return img, DetSet(d.xywh[m], d.score[m], d.cls[m].astype(float))

    def track(x: tuple[np.ndarray, DetSet | None]) -> object:
        img, d = x
        return coast(tracker, img) if d is None else tracker.update(d, img)

    background = background_cpu()
    res = run_benchmark([("read", read), ("detect", detect), ("track", track)], frames, proto)
    s = np.array(split[max(1, proto.n_warmup // stride) :])
    detect_split = {
        n: float(np.median(s[:, i]))
        for i, n in enumerate(("preprocess", "inference", "postprocess"))
    }
    verdicts = {
        f"{edge.require('target_fps')}_fps": budget_verdict(res["stages"], latency_budget_ms(edge)),
        f"{edge.require('reference_fps')}_fps": budget_verdict(
            res["stages"], 1000.0 / edge.require("reference_fps")
        ),
    }
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "tag": args.tag,
        "config": args.config,
        "detector": {**det, "runner": type(runner).__name__, "input_hw": [runner.h, runner.w]},
        "tracker": trk,
        "frames": source + ", in order; cycled per repeat",
        "not_timed": "ranging and Kalman filter; overlay drawing",
        "smoke": args.smoke,
        "machine": {
            "cpu": platform.processor() or platform.machine(),
            "machine": platform.machine(),
            "logical_cores": os.cpu_count(),
            "os": platform.platform(),
            "onnxruntime": version("onnxruntime"),
            "onnxruntime_threads": args.threads or "default",
            "opencv": cv2.__version__,
        },
        "other_processes": args.note,
        "background_measured": background,
        **res,
        "detect_split_median_ms": detect_split,
        "budget": verdicts,
    }
    dest = (
        REPO_ROOT
        / "results"
        / "edge"
        / f"benchmark_v2_{args.tag}{'_smoke' if args.smoke else ''}.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    e = res["stages"]["end_to_end"]
    print(
        f"{args.tag}: end-to-end median {e['median_ms']:.1f} ms, p99 {e['p99_ms']:.1f} ms, "
        f"{res['throughput_fps']:.1f} fps; "
        + ", ".join(f"{k} {res['stages'][k]['median_ms']:.1f}" for k in ("read", "detect", "track"))
        + f"; wrote {dest}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
