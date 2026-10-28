"""Overlapped pipeline benchmark: detection and tracking on separate threads (Phase 5, D-610).

Usage: python scripts/benchmark_threaded.py --tag p5_mt_onnx640_bot8_s2
       --weights weights/B2_yolo26n_960w_640_rect.onnx --imgsz 640 --gmc-downscale 8
       --det-stride 2 [--note "..."]

A reader/detector thread reads frame k+1 (and detects on detection frames) while the main
thread tracks frame k; a queue of `QUEUE` frames joins them. On frames between detections
the tracker coasts (`track.runner.coast`, D-610). Measured per frame: latency from the start
of reading to the tracker's output, and throughput over the timed frames. Protocol sizes from
configs/edge.yaml (D-601); the tracker restarts for each repeat. Verdicts at the target and
reference rates use latency (median, p99) and throughput. ONNX Runtime and OpenCV release
Python's global lock while computing, so the threads overlap. Output:
results/edge/benchmark_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import queue
import sys
import threading
import time
from datetime import UTC, datetime
from itertools import cycle, islice
from pathlib import Path

import numpy as np

from aerialrange.edge.benchmark import background_cpu, peak_rss_mb
from aerialrange.track.runner import DetSet, coast, make_tracker
from aerialrange.utils.config import REPO_ROOT, latency_budget_ms, load_config

QUEUE = 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--class-map", default="visdrone10")
    ap.add_argument("--tuned-from", default="B2_yolo26n_960")
    ap.add_argument("--tracker", default="botsort_cmc")
    ap.add_argument("--gmc-downscale", type=int, default=None)
    ap.add_argument("--high", type=float, default=None)
    ap.add_argument("--new", type=float, default=None)
    ap.add_argument("--det-stride", type=int, default=1)
    ap.add_argument("--sequence", default="uav0000086_00000_v")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    import cv2
    import ultralytics
    from ultralytics import YOLO

    edge, det, trk = load_config("edge"), load_config("detector"), load_config("tracker")
    n_warm, n_timed, n_rep = (
        edge.require("n_warmup"),
        edge.require("n_timed_frames"),
        edge.require("n_repeats"),
    )
    cmap = {int(k): v for k, v in det.require("class_maps")[args.class_map].items()}
    overrides = dict(trk.require("tuned")[args.tuned_from])
    for k, v in (
        ("gmc_downscale", args.gmc_downscale),
        ("track_high_thresh", args.high),
        ("new_track_thresh", args.new),
    ):
        if v is not None:
            overrides[k] = v
    tracker_cfg = str(REPO_ROOT / "configs" / "trackers" / f"{args.tracker}.yaml")
    model = YOLO(args.weights, task="detect")
    info = json.loads(
        (
            REPO_ROOT / "cache" / "visdrone" / "mot" / "val" / args.sequence / "seqinfo.json"
        ).read_text()
    )
    frames = sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))

    def detect(img: np.ndarray) -> DetSet:
        r = model.predict(
            img,
            imgsz=args.imgsz,
            classes=sorted(cmap),
            max_det=det.require("max_det"),
            conf=0.1,
            verbose=False,
        )[0]
        b = r.boxes
        return DetSet(b.xywh.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy())

    def run(n: int) -> tuple[np.ndarray, float]:
        tracker = make_tracker(tracker_cfg, overrides)
        q: queue.Queue = queue.Queue(maxsize=QUEUE)
        starts = np.zeros(n)

        def producer() -> None:
            for i, p in enumerate(islice(cycle(frames), n)):
                starts[i] = time.perf_counter()
                img = cv2.imread(str(p))
                q.put((i, img, detect(img) if i % args.det_stride == 0 else None))
            q.put(None)

        th = threading.Thread(target=producer, daemon=True)
        th.start()
        ends = np.zeros(n)
        while (item := q.get()) is not None:
            i, img, d = item
            if d is None:
                coast(tracker, img)
            else:
                tracker.update(d, img)
            ends[i] = time.perf_counter()
        th.join()
        return (ends - starts) * 1000, n / (ends[-1] - starts[0])

    background = background_cpu()
    run(n_warm)
    lat, thr = [], []
    for _ in range(n_rep):
        latency, throughput = run(n_timed)
        lat.append(latency)
        thr.append(throughput)
    lat_all = np.concatenate(lat)
    budget, ref = latency_budget_ms(edge), 1000.0 / edge.require("reference_fps")
    stats = {
        "median_ms": float(np.median(lat_all)),
        "p95_ms": float(np.percentile(lat_all, 95)),
        "p99_ms": float(np.percentile(lat_all, 99)),
        "max_ms": float(lat_all.max()),
    }
    fps = float(np.median(thr))

    def verdict(b: float) -> dict:
        return {
            "budget_ms": b,
            "throughput_keeps_up": fps >= 1000.0 / b,
            "latency_median_within": stats["median_ms"] <= b,
            "latency_p99_within": stats["p99_ms"] <= b,
        }

    import torch

    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "tag": args.tag,
        "mode": "threaded: read+detect thread, track on main thread",
        "weights": Path(args.weights).name,
        "imgsz": args.imgsz,
        "tracker": {"config": Path(tracker_cfg).name, "overrides": overrides},
        "det_stride": args.det_stride,
        "queue": QUEUE,
        "protocol": {"n_warmup": n_warm, "n_timed": n_timed, "n_repeats": n_rep},
        "frames": f"MOT val {args.sequence}, in order, cycled",
        "machine": {
            "cpu": platform.processor(),
            "os": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "ultralytics": ultralytics.__version__,
        },
        "other_processes": args.note,
        "background_measured": background,
        "latency": stats,
        "throughput_fps": fps,
        "throughput_by_repeat": [float(x) for x in thr],
        "peak_rss_mb": peak_rss_mb(),
        "not_timed": "ranging and Kalman filter; overlay drawing",
        "budget": {
            f"{1000 / budget:.0f}_fps": verdict(budget),
            f"{1000 / ref:.0f}_fps": verdict(ref),
        },
    }
    dest = REPO_ROOT / "results" / "edge" / f"benchmark_{args.tag}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(
        f"{args.tag}: throughput {fps:.1f} fps; latency median {stats['median_ms']:.1f} ms, "
        f"p99 {stats['p99_ms']:.1f} ms; wrote {dest}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
