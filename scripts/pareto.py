"""Accuracy against latency for the Phase 5 configurations.

Usage: python scripts/pareto.py

Pairs each benchmark (results/edge/benchmark_p5_*.json; protocol D-601, quiet laptop) with the
tracking accuracy of the same configuration on MOT val (mean IDF1 over person and vehicle;
replayed detections with the tuned thresholds of B2 n 960, D-321). Writes
results/edge/pareto.json, results/edge/figures/pareto.png and a CSV beside it. Validation
accuracy only (development numbers); budgets from configs/edge.yaml.
"""

from __future__ import annotations

import json
import sys

from aerialrange.utils.config import REPO_ROOT, latency_budget_ms, load_config
from aerialrange.viz.plots import pareto_chart, pareto_frontier

# benchmark tag -> (tracking-accuracy source under cache/visdrone/runs, short label)
CONFIGS = {
    "p5_pt960_bot2": ("grid", "PyTorch 960, BoT-SORT"),
    "p5_onnx960_bot2": ("grid", "ONNX 960, BoT-SORT"),
    "p5_int8dyn960_bot2": ("p5_int8dyn960_bot2", "INT8-dyn 960, BoT-SORT"),
    "p5_int8st960_bot2": ("p5_int8st960_bot2", "INT8-static 960, BoT-SORT"),
    "p5_onnx960_bot4": ("p5_n960_bot4", "ONNX 960, BoT-SORT GMC/4"),
    "p5_onnx960_bot8": ("p5_n960_bot8", "ONNX 960, BoT-SORT GMC/8"),
    "p5_onnx960_byte": ("p5_n960_byte", "ONNX 960, ByteTrack"),
    "p5_int8dyn960_bot4": ("p5_int8dyn960_bot4", "INT8-dyn 960, GMC/4"),
    "p5_onnx640_bot4": ("p5_n960w640_bot4", "ONNX 640, GMC/4"),
    "p5_int8dyn640_bot4": ("p5_int8dyn640_bot4", "INT8-dyn 640, GMC/4"),
    "p5_int8dyn640_byte": ("p5_int8dyn640_byte", "INT8-dyn 640, ByteTrack"),
    # round 2 (D-610): camera-motion compensation at 1/8, detection stride, overlap
    "p5_onnx640_bot8_s1": ("p5_n960w640_bot8", "640, every frame"),
    "p5_onnx640_bot8_s2": ("p5_n960w640_bot8_s2", "640, every 2nd frame"),
    "p5_onnx640_bot8_s3": ("p5_n960w640_bot8_s3", "640, every 3rd frame"),
    "p5_onnx960_bot8_s2": ("p5_n960_bot8_s2", "960, every 2nd frame"),
    "p5_onnx960_bot8_s3": ("p5_n960_bot8_s3", "960, every 3rd frame"),
    "p5_mt_onnx640_bot8_s1": ("p5_n960w640_bot8", "640 overlapped, every frame"),
    "p5_mt_onnx640_bot8_s2": ("p5_n960w640_bot8_s2", "640 overlapped, every 2nd"),
    "p5_mt_onnx640_bot8_s3": ("p5_n960w640_bot8_s3", "640 overlapped, every 3rd"),
    "p5_mt_onnx960_bot8_s1": ("p5_n960_bot8", "960 overlapped, every frame"),
    "p5_mt_onnx960_bot8_s2": ("p5_n960_bot8_s2", "960 overlapped, every 2nd"),
    "p5_mt_onnx960_bot8_s3": ("p5_n960_bot8_s3", "960 overlapped, every 3rd"),
}


def accuracy(source: str) -> dict | None:
    if source == "grid":
        rows = json.loads((REPO_ROOT / "results/visdrone/val/grid_summary.json").read_text())[
            "rows"
        ]
        r = next(
            x
            for x in rows
            if x["dets"] == "B2_yolo26n_960"
            and x["tracker"] == "botsort_cmc.yaml"
            and x["point"] == "selected"
        )
        return {
            "mean_idf1": r["mean_idf1"],
            "person_idf1": r["person_idf1"],
            "vehicle_idf1": r["vehicle_idf1"],
        }
    p = REPO_ROOT / "cache" / "visdrone" / "runs" / source / "val" / "summary.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())["metrics"]
    return {
        "mean_idf1": (m["person"]["idf1"] + m["vehicle"]["idf1"]) / 2,
        "person_idf1": m["person"]["idf1"],
        "vehicle_idf1": m["vehicle"]["idf1"],
    }


def main() -> int:
    edge = load_config("edge")
    budget, ref = latency_budget_ms(edge), 1000.0 / edge.require("reference_fps")
    rows = []
    for tag, (src, label) in CONFIGS.items():
        b = REPO_ROOT / "results" / "edge" / f"benchmark_{tag}.json"
        acc = accuracy(src)
        if not b.exists() or acc is None:
            print(f"{tag}: missing {'benchmark' if not b.exists() else 'accuracy'}")
            continue
        bench = json.loads(b.read_text())
        threaded = "latency" in bench
        e = bench["latency"] if threaded else bench["stages"]["end_to_end"]
        rows.append(
            {
                "config": tag,
                "label": label,
                "mode": "overlapped" if threaded else "sequential",
                "det_stride": bench.get("det_stride", 1),
                "median_ms": e["median_ms"],
                "p99_ms": e["p99_ms"],
                "fps": bench["throughput_fps"],
                "ms_per_frame_sustained": 1000.0 / bench["throughput_fps"],
                "meets_target_throughput": bench["throughput_fps"] >= 1000.0 / budget,
                "meets_reference_throughput": bench["throughput_fps"] >= 1000.0 / ref,
                "background_cpu": bench.get("background_measured", {}).get(
                    "other_processes_cpu_percent_of_one_core"
                ),
                "stages_median_ms": {}
                if threaded
                else {k: v["median_ms"] for k, v in bench["stages"].items() if k != "end_to_end"},
                "meets_target_median": e["median_ms"] <= budget,
                "meets_target_p99": e["p99_ms"] <= budget,
                "meets_reference_median": e["median_ms"] <= ref,
                "meets_reference_p99": e["p99_ms"] <= ref,
                **acc,
            }
        )
    if not rows:
        print("no complete configurations yet")
        return 1
    front = pareto_frontier([(r["ms_per_frame_sustained"], r["mean_idf1"]) for r in rows])
    best = max(rows, key=lambda r: r["mean_idf1"])
    out = {
        "budget_ms": budget,
        "reference_ms": ref,
        "accuracy": "MOT val mean IDF1 (person, vehicle), development numbers",
        "speed": "x: time per frame sustained (1000 / throughput, D-601); per-frame latency "
        "median and p99 in each row (overlapped runs: start of reading to tracker output)",
        "frontier": [rows[i]["config"] for i in front],
        "best_accuracy": best["config"],
        "configs": rows,
    }
    for r in rows:
        r["idf1_cost_vs_best"] = best["mean_idf1"] - r["mean_idf1"]
    (REPO_ROOT / "results" / "edge" / "pareto.json").write_text(json.dumps(out, indent=1) + "\n")
    pareto_chart(
        REPO_ROOT / "results" / "edge" / "figures" / "pareto.png",
        [r["label"] for r in rows],
        [r["ms_per_frame_sustained"] for r in rows],
        [r["mean_idf1"] for r in rows],
        {f"{1000 / budget:.0f} fps": budget, f"{1000 / ref:.0f} fps": ref},
        "Tracking accuracy against speed (laptop CPU)",
        "Time per frame sustained, 1000 / throughput (ms, log scale)",
        "Mean IDF1 on VisDrone val",
        note="Development numbers: accuracy on MOT val with replayed detections; speed on this "
        "laptop (D-601, background load measured per run). In overlapped runs each frame's delay "
        "is longer than the time per frame sustained (pareto.json). Ranging not yet timed.",
        logx=True,
    )
    for r in sorted(rows, key=lambda r: r["ms_per_frame_sustained"]):
        print(
            f"{r['label']:28s} {r['median_ms']:6.1f} ms (p99 {r['p99_ms']:6.1f})  "
            f"{r['fps']:5.1f} fps  IDF1 {r['mean_idf1']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
