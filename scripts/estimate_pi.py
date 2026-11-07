"""Estimate Raspberry Pi 5 detector latency from laptop timings and one published Pi 5 timing.

Usage: python scripts/estimate_pi.py --model NAME weights/x.onnx [--model ...] [--repeats 200]

ESTIMATED, NOT MEASURED. Method (D-723): Arm publishes YOLO26n at 640 x 640 on a
Raspberry Pi 5 with ONNX Runtime at 126.9 ms per image, inference only, batch 1, Raspberry Pi
OS Trixie (huggingface.co/Arm/Arm-Runtime-for-YOLO26-on-Raspberry-Pi-5, read 2026-11-07).
The same model (weights/yolo26n_640.onnx, the library's COCO YOLO26n) is timed here the same
way: ONNX Runtime on the CPU, 4 intra-op threads (the Pi 5 has 4 cores), batch 1, a fixed
real frame, inference only (session.run; no pre- or post-processing), median of `repeats`
runs after 20 warm-up runs. Each model's Pi estimate is its laptop median times the ratio
(126.9 ms / laptop median of the reference). The ratio is measured on a convolutional
network; attention-heavy detectors may scale differently between x86 and ARM, and no
published Pi 5 CPU timing of a DETR-style detector exists to narrow that, so the stated range
is 0.75 to 1.5 times the point estimate (an assumption, to be replaced by `make pi-benchmark`
on a device). Run on an otherwise idle machine. Writes results/edge/pi_estimate.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version

import numpy as np

from aerialrange.utils.config import REPO_ROOT

PI5_REF_MS = 126.9
PI5_REF_SOURCE = (
    "Arm, 'Arm Runtime for YOLO26 on Raspberry Pi 5' model card, benchmark table: YOLO26n, "
    "640x640, batch 1, ONNX Runtime 126.90 ms/image, inference only, Raspberry Pi OS Trixie "
    "(huggingface.co/Arm/Arm-Runtime-for-YOLO26-on-Raspberry-Pi-5, read 2026-11-07)"
)
REFERENCE = "weights/yolo26n_640.onnx"
THREADS = 4
RANGE = (0.75, 1.5)


def time_model(path: str, repeats: int) -> dict:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(str(REPO_ROOT / path), opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [int(v) for v in inp.shape]
    # A fixed input: content barely affects these networks' run time (no NMS inside).
    x = np.random.default_rng(0).random(shape, dtype=np.float32)
    for _ in range(20):
        sess.run(None, {inp.name: x})
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        t.append((time.perf_counter() - t0) * 1000)
    return {
        "input_shape": shape,
        "median_ms": statistics.median(t),
        "p90_ms": float(np.percentile(t, 90)),
        "repeats": repeats,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=2, action="append", required=True, metavar=("NAME", "ONNX"))
    ap.add_argument("--repeats", type=int, default=200)
    args = ap.parse_args(argv)
    ref = time_model(REFERENCE, args.repeats)
    ratio = PI5_REF_MS / ref["median_ms"]
    models = {}
    for name, path in args.model:
        m = time_model(path, args.repeats)
        est = m["median_ms"] * ratio
        models[name] = {
            "onnx": path,
            "laptop": m,
            "pi5_estimate_ms": est,
            "pi5_estimate_range_ms": [est * RANGE[0], est * RANGE[1]],
            "pi5_estimate_fps": 1000.0 / est,
        }
        print(f"{name}: laptop {m['median_ms']:.1f} ms -> Pi 5 about {est:.0f} ms (estimated)")
    out = {
        "label": "ESTIMATED, NOT MEASURED",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": "laptop median x (published Pi 5 time / laptop median) of the same reference",
        "reference": {"onnx": REFERENCE, "pi5_ms": PI5_REF_MS, "source": PI5_REF_SOURCE},
        "reference_laptop": ref,
        "ratio_pi5_over_laptop": ratio,
        "range_factor": list(RANGE),
        "range_note": "assumption: the ratio is measured on a convolutional network; "
        "attention-heavy networks may scale differently between x86 and ARM",
        "threads": THREADS,
        "scope": "inference only (session.run), batch 1, no pre- or post-processing",
        "laptop_cpu": platform.processor(),
        "laptop_logical_cpus": os.cpu_count(),
        "onnxruntime": version("onnxruntime"),
        "models": models,
    }
    dest = REPO_ROOT / "results" / "edge" / "pi_estimate.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"ratio {ratio:.2f}; wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
