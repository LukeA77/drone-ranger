"""`aerialrange run`: detect, track, range, filter and annotate one flight.

Inputs: a video file or a folder of frames; a flight-data CSV with one row per frame
(`frame` 1-based, `t_s`, `n_m`, `e_m` local position in metres, `h_m` true height above the
ground, `yaw_rad` heading clockwise from north, `tilt_deg` camera tilt below the horizon, NaN if
unknown); and configs/run.yaml (or another file of the same form). Writes, in `out`:
`tracks.csv` (frame, time, track, class, box, range, range sigma, azimuth, relative azimuth,
depression, speed, speed sigma, validity, reason, reliability flag), `annotated.mp4`, and
`timing.json` (per-stage medians and percentiles; overlay drawing reported separately and
excluded from the latency budget).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from aerialrange.pipeline.filterbank import TrackFilterBank
from aerialrange.pipeline.overlay import draw_frame
from aerialrange.pipeline.ranging import CameraPose, RangingModel, range_boxes
from aerialrange.track.runner import DetSet, coast, make_tracker
from aerialrange.utils.config import REPO_ROOT, load_config

COLUMNS = [
    "frame",
    "t_s",
    "track_id",
    "class",
    "left",
    "top",
    "width",
    "height",
    "range_m",
    "range_sigma_m",
    "azimuth_deg",
    "relative_azimuth_deg",
    "depression_deg",
    "speed_ms",
    "speed_sigma_ms",
    "valid",
    "reason",
    "flag",
]


def _frames(source: Path) -> Iterator[np.ndarray]:
    if source.is_dir():
        for p in sorted(list(source.glob("*.jpg")) + list(source.glob("*.png"))):
            yield cv2.imread(str(p))
    else:
        cap = cv2.VideoCapture(str(source))
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield img
        cap.release()


def _stats(ms: list[float]) -> dict[str, float]:
    a = np.asarray(ms, float)
    if not len(a):
        return {}
    return {
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "n": int(len(a)),
    }


def run(
    source: Path,
    metadata: Path,
    out: Path,
    config: str = "run",
    boxes_csv: Path | None = None,
    overrides: dict[str, Any] | None = None,
    max_frames: int | None = None,
    credit: str = "",
) -> dict[str, Any]:
    """Run the pipeline; returns the timing summary. `overrides` replace config values
    (e.g. camera intrinsics for the synthetic demo); `boxes_csv` makes the detector read boxes
    from a CSV (frame, left, top, width, height, score, cls) instead of running a model."""
    cfg = load_config(config)

    def get(k: str) -> Any:
        return (
            {**cfg.require(k), **(overrides or {}).get(k, {})}
            if isinstance(cfg.require(k), dict)
            else (overrides or {}).get(k, cfg.require(k))
        )

    det, trk, cam = get("detector"), get("tracker"), get("camera")
    rs, km, ov = get("range_sigma_rel_by_depression"), get("kalman"), get("overlay")
    model = RangingModel(
        float(cam["focal_px"]),
        float(cam["cx"]),
        float(cam["cy"]),
        np.array(rs["edges_deg"], float),
        np.array(rs["rel"], float),
        float(get("reliable_max_rel_sigma")),
    )
    meas_rel = np.array(km["meas_rel"] if "meas_rel" in km else km["meas_rel_sigma"], float)
    cmap = {
        int(k): v for k, v in load_config("detector").require("class_maps")["visdrone10"].items()
    }
    meta = pd.read_csv(metadata).set_index("frame")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if boxes_csv is not None:
        bdf = pd.read_csv(boxes_csv)
        by_frame = {f: g for f, g in bdf.groupby("frame")}

        def detect(img: np.ndarray, k: int) -> DetSet:
            g = by_frame.get(k)
            if g is None:
                return DetSet(np.zeros((0, 4)), np.zeros(0), np.zeros(0))
            b = g[["left", "top", "width", "height"]].to_numpy(float)
            xywh = np.column_stack([b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2, b[:, 2], b[:, 3]])
            return DetSet(xywh, g["score"].to_numpy(float), g["cls"].to_numpy(float))
    else:
        from ultralytics import YOLO

        yolo = YOLO(str(REPO_ROOT / det["weights"]), task="detect")
        max_det = load_config("detector").require("max_det")

        def detect(img: np.ndarray, k: int) -> DetSet:
            r = yolo.predict(
                img,
                imgsz=int(det["imgsz"]),
                classes=sorted(cmap),
                max_det=max_det,
                conf=float(det["score_floor"]),
                verbose=False,
            )[0]
            b = r.boxes
            return DetSet(b.xywh.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy())

    tracker = make_tracker(
        str(REPO_ROOT / trk["config"]),
        {"gmc_downscale": trk["gmc_downscale"]} if trk.get("gmc_downscale") else None,
    )
    stride = int(trk.get("det_stride", 1))
    bank = TrackFilterBank(float(km["q"]), int(km["max_unseen_frames"]))
    times: dict[str, list[float]] = {
        k: [] for k in ("read", "detect", "track", "range_filter", "end_to_end", "overlay")
    }
    rows_all: list[dict] = []
    writer = None
    t_prev = None
    it = _frames(Path(source))
    k = 0
    while max_frames is None or k < max_frames:
        t0 = time.perf_counter()
        img = next(it, None)
        if img is None:
            break
        k += 1
        t1 = time.perf_counter()
        if (k - 1) % stride == 0:
            d = detect(img, k)
            t2 = time.perf_counter()
            res = np.asarray(tracker.update(d, img)).reshape(-1, 8)
            tracked = [(int(r[4]), int(r[6]), r[0], r[1], r[2] - r[0], r[3] - r[1]) for r in res]
        else:
            t2 = time.perf_counter()
            tracked = [(int(s.track_id), int(s.cls), *s.tlwh) for s in coast(tracker, img)]
        t3 = time.perf_counter()
        m = meta.loc[k] if k in meta.index else None
        pose = (
            CameraPose(*(float(m[c]) for c in ("n_m", "e_m", "h_m", "yaw_rad", "tilt_deg")))
            if m is not None
            else CameraPose(np.nan, np.nan, np.nan, 0.0, np.nan)
        )
        t_s = float(m["t_s"]) if m is not None else np.nan
        tracked = [t for t in tracked if t[1] in cmap]
        boxes = np.array([t[2:] for t in tracked], float).reshape(-1, 4)
        rg = range_boxes(boxes, pose, model)
        dt = (t_s - t_prev) if t_prev is not None and np.isfinite(t_s) else 0.2
        t_prev = t_s
        valid = rg["valid"]
        msig = meas_rel[model.band(np.nan_to_num(rg["depression_deg"]))] * np.nan_to_num(
            rg["range_m"]
        )
        ids = np.array([t[0] for t in tracked], int)
        fb = bank.step(
            max(dt, 1e-3),
            ids[valid],
            np.column_stack([rg["ground_n_m"], rg["ground_e_m"]])[valid],
            msig[valid],
        )
        speed = dict(
            zip(
                fb["id"].tolist(),
                zip(fb["speed_ms"].tolist(), fb["speed_sigma_ms"].tolist(), strict=True),
                strict=True,
            )
        )
        t4 = time.perf_counter()
        frame_rows = []
        for j, (tid, cls, x, y, w, h) in enumerate(tracked):
            sp, ss = speed.get(tid, (np.nan, np.nan))
            frame_rows.append(
                {
                    "frame": k,
                    "t_s": t_s,
                    "track_id": tid,
                    "class": cmap[cls],
                    "left": x,
                    "top": y,
                    "width": w,
                    "height": h,
                    "range_m": rg["range_m"][j],
                    "range_sigma_m": rg["range_sigma_m"][j],
                    "azimuth_deg": rg["azimuth_deg"][j],
                    "relative_azimuth_deg": rg["relative_azimuth_deg"][j],
                    "depression_deg": rg["depression_deg"][j],
                    "speed_ms": sp,
                    "speed_sigma_ms": ss,
                    "valid": bool(valid[j]),
                    "reason": rg["reason"][j],
                    "flag": rg["flag"][j],
                }
            )
        rows_all += frame_rows
        for name, a, b in (
            ("read", t0, t1),
            ("detect", t1, t2),
            ("track", t2, t3),
            ("range_filter", t3, t4),
            ("end_to_end", t0, t4),
        ):
            times[name].append((b - a) * 1000)
        t5 = time.perf_counter()
        draw_frame(
            img,
            k,
            t_s,
            frame_rows,
            pose,
            model.focal_px,
            model.cx,
            model.cy,
            ov["rings_m"],
            float(ov["font_scale"]),
            credit,
        )
        if writer is None:
            fps = 1.0 / dt if np.isfinite(dt) and dt > 0 else 5.0
            for code in ("avc1", "mp4v"):
                writer = cv2.VideoWriter(
                    str(out / "annotated.mp4"),
                    cv2.VideoWriter_fourcc(*code),
                    fps,
                    (img.shape[1], img.shape[0]),
                )
                if writer.isOpened():
                    break
        writer.write(img)
        times["overlay"].append((time.perf_counter() - t5) * 1000)
    if writer is not None:
        writer.release()
    pd.DataFrame(rows_all, columns=COLUMNS).to_csv(out / "tracks.csv", index=False)
    timing = {
        "frames": k,
        "det_stride": stride,
        "stages": {n: _stats(v) for n, v in times.items()},
        "note": "Sequential run on this machine; overlay drawing excluded from end_to_end (10.1). "
        "detect is near zero on frames between detections. For the protocol benchmark see "
        "scripts/benchmark_pipeline.py and scripts/benchmark_threaded.py.",
    }
    (out / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    return timing
