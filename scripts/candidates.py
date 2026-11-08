"""Detector candidates side by side: accuracy, tracking and speed, for the detector decision.

Usage: python scripts/candidates.py

For each candidate in CANDIDATES that has been scored (scripts/queues/score_candidate.sh),
collects from saved results only:
- detection AP50 and AP50:95 per group on MOT val frames (results/visdrone/val/detect_*.json,
  same protocol and score floor for every detector, D-719), and small-person detection: AP50
  and recall at the score floor for people 16 to 32 px tall;
- tracking with the project's tracker tuned for that detector on version 1's grid and rule
  (results/visdrone/val/grid_<dets>__motion.json, D-319): mean IDF1 and MOTA, per group, and
  person recall at the selected thresholds;
- speed: the laptop pipeline benchmark (results/edge/benchmark_v2_cand_<name>.json,
  scripts/benchmark_v2.py, the run configuration's tracker and stride) and the Raspberry Pi 5
  estimate of the detector network (results/edge/pi_estimate.json, ESTIMATED, D-723).
Writes results/edge/candidates.json and results/edge/figures/candidates.png (mean IDF1
against the Pi 5 estimate, the speed that decides deployment) with a CSV beside it.
Validation numbers only: the test-dev run comes after the decision, once.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from aerialrange.utils.config import REPO_ROOT, load_config
from aerialrange.viz.plots import pareto_chart, pareto_frontier

VAL = REPO_ROOT / "results" / "visdrone" / "val"
EDGE = REPO_ROOT / "results" / "edge"
# name: (detection run, cached detections run, kind, ONNX model, input size, label)
CANDIDATES = {
    "v1_960": (
        "B2_yolo26n_960_onnx",
        "B2_yolo26n_960",
        "yolo",
        "weights/B2_yolo26n_960_rect.onnx",
        960,
        "Version 1 YOLO26n 960",
    ),
    "v1_640": (
        "B2_yolo26n_960w_640",
        "B2_yolo26n_960w_640",
        "yolo",
        "weights/B2_yolo26n_960w_640_rect.onnx",
        640,
        "Version 1 YOLO26n 640",
    ),
    "rfdetr_nano_640": (
        "rfdetr_nano_640",
        "rfdetr_nano_640",
        "rfdetr",
        "weights/rfdetr_nano_640.onnx",
        640,
        "RF-DETR Nano 640",
    ),
    "rfdetr_nano_960": (
        "rfdetr_nano_960",
        "rfdetr_nano_960",
        "rfdetr",
        "weights/rfdetr_nano_960.onnx",
        960,
        "RF-DETR Nano 960",
    ),
    "dfine_n_640": (
        "dfine_n_640",
        "dfine_n_640",
        "dfine",
        "weights/dfine_n_640.onnx",
        640,
        "D-FINE N 640",
    ),
}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    pi = _load(EDGE / "pi_estimate.json") or {"models": {}}
    edge = load_config("edge")
    rows = []
    for name, (det_run, dets_run, kind, onnx, imgsz, label) in CANDIDATES.items():
        det = _load(VAL / f"detect_{det_run}.json")
        grid = _load(VAL / f"grid_{dets_run}__motion.json")
        if det is None or grid is None:
            print(f"{name}: not scored yet")
            continue
        sel = grid["selected"]
        point = next(p for p in grid["points"] if p["overrides"] == sel)
        groups = sorted(point["metrics"])
        overall = {g: point["metrics"][g]["OVERALL"] for g in groups}
        bench = _load(EDGE / f"benchmark_v2_cand_{name}.json")
        est = pi["models"].get(name)
        res = det["results"]
        rows.append(
            {
                "name": name,
                "label": label,
                "kind": kind,
                "onnx": onnx,
                "imgsz": imgsz,
                "ap50": {g: res[g]["ap50"]["ap"] for g in groups},
                "ap50_95": {g: res[g]["ap50_95"] for g in groups},
                "det_max_recall": {g: res[g]["ap50"]["max_recall"] for g in groups},
                "person_16_32px": {
                    "ap50": res["person"]["by_height"]["16-32"]["ap"],
                    "recall": res["person"]["by_height"]["16-32"]["max_recall"],
                    "n_gt": res["person"]["by_height"]["16-32"]["n_gt"],
                },
                "tracker_thresholds": sel,
                "mean_idf1": point["mean_idf1"],
                "mean_mota": point["mean_mota"],
                "idf1": {g: overall[g]["idf1"] for g in groups},
                "mota": {g: overall[g]["mota"] for g in groups},
                "track_recall": {g: overall[g]["recall"] for g in groups},
                "laptop_ms_median": bench["stages"]["end_to_end"]["median_ms"] if bench else None,
                "laptop_fps": bench["throughput_fps"] if bench else None,
                "laptop_detect_inference_ms": (
                    bench["detect_split_median_ms"]["inference"] if bench else None
                ),
                "pi5_estimate_ms": est["pi5_estimate_ms"] if est else None,
                "pi5_estimate_range_ms": est["pi5_estimate_range_ms"] if est else None,
            }
        )
    if not rows:
        print("no candidate scored yet")
        return 1
    timed = [r for r in rows if r["pi5_estimate_ms"] is not None]
    front = pareto_frontier([(r["pi5_estimate_ms"], r["mean_idf1"]) for r in timed])
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "accuracy": "VisDrone MOT val: detection AP on every 10th frame; tracking with the "
        "project's tracker tuned per detector (D-319). Development numbers.",
        "speed": "laptop: version 2 pipeline benchmark (read, detect, track; run configuration's "
        "stride); Pi 5: ESTIMATED detector inference only (D-723)",
        "pi5_frontier": [timed[i]["name"] for i in front],
        "candidates": rows,
    }
    (EDGE / "candidates.json").write_text(json.dumps(out, indent=1) + "\n")
    if timed:
        budget = 1000.0 / edge.require("target_fps")
        pareto_chart(
            EDGE / "figures" / "candidates.png",
            [r["label"] for r in timed],
            [r["pi5_estimate_ms"] for r in timed],
            [r["mean_idf1"] for r in timed],
            {f"{edge.require('target_fps'):.0f} fps": budget},
            "Detector candidates: tracking accuracy against Pi 5 speed",
            "Raspberry Pi 5 detector time per frame, ESTIMATED (ms)",
            "Mean IDF1 on VisDrone val",
            note="Estimated, not measured: laptop timings scaled by a published Pi 5 timing of a "
            "reference model (D-723); range 0.75 to 1.5 times. Detector network only. "
            "Accuracy: development numbers, the project's tracker tuned per detector. "
            "Version 1's detector is AGPL-licensed and shown for reference only.",
            logx=True,
            reference=[i for i, r in enumerate(timed) if r["kind"] == "yolo"],
            label_all=True,
        )
    for r in rows:
        pi_ms = f"{r['pi5_estimate_ms']:.0f}" if r["pi5_estimate_ms"] else "-"
        print(
            f"{r['label']:24s} person AP50 {r['ap50']['person']:.3f}  vehicle AP50 "
            f"{r['ap50']['vehicle']:.3f}  person 16-32 px recall "
            f"{r['person_16_32px']['recall']:.3f}  mean IDF1 {r['mean_idf1']:.3f}  "
            f"Pi 5 ~{pi_ms} ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
