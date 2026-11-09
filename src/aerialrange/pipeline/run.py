"""`aerialrange run`: detect, track, range, filter and annotate one flight.

Inputs: a video file or a folder of frames; a flight-data CSV with one row per frame
(`frame` 1-based, `t_s`, `n_m`, `e_m` local position in metres, `h_m` true height above the
ground, `yaw_rad` heading clockwise from north, `tilt_deg` camera tilt below the horizon, NaN if
unknown); and configs/run.yaml (or another file of the same form). Writes, in `out`:
`tracks.csv` (frame, time, track, class, box, range, range sigma, azimuth, relative azimuth,
depression, speed, speed sigma, validity, reason, reliability flag), `annotated.mp4`, and
`timing.json` (per-stage medians and percentiles; overlay drawing reported separately and
excluded from the latency budget).

Version 2: `source` may also be any iterable of BGR frames and `metadata` a
DataFrame, so frames can be streamed (SynDrone). tracks.csv also carries heading, heading
sigma, the filter's age and recent mean NIS, and `speed_shown`, the display rule's verdict
(`pipeline/speed_display.py`); speed is drawn on the overlay only when it is true.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from aerialrange.detect.factory import make_detector
from aerialrange.pipeline.overlay import draw_frame
from aerialrange.pipeline.range_speed import DEFAULT_DT_S, RangeSpeedStage, pose_of
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
    "heading_deg",
    "heading_sigma_deg",
    "track_age",
    "nis_mean",
    "speed_shown",
    "range_half_68_m",
    "range_half_90_m",
    "range_half_95_m",
    "speed_half_68_ms",
    "speed_half_90_ms",
    "speed_half_95_ms",
]


def _frames(source: Path | Iterable[np.ndarray]) -> Iterator[np.ndarray]:
    if not isinstance(source, (str, Path)):
        yield from source
        return
    source = Path(source)
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
    source: Path | Iterable[np.ndarray],
    metadata: Path | pd.DataFrame,
    out: Path,
    config: str = "run",
    boxes_csv: Path | None = None,
    overrides: dict[str, Any] | None = None,
    max_frames: int | None = None,
    credit: str = "",
    detections_out: list[dict] | None = None,
    write_video: bool = True,
) -> dict[str, Any]:
    """Run the pipeline; returns the timing summary. `overrides` replace config values
    (e.g. camera intrinsics for the synthetic demo); `boxes_csv` makes the detector read boxes
    from a CSV (frame, left, top, width, height, score, cls) instead of running a model.
    `detections_out`, if given, receives every raw detection before tracking (frame, box,
    score, class), for detection evaluation; `write_video=False` skips the overlay video."""
    cfg = load_config(config)

    def get(k: str) -> Any:
        ov = overrides or {}
        try:
            base = cfg.require(k)
        except KeyError:  # a section given only as an override (e.g. intervals, D-737)
            if k in ov:
                return ov[k]
            raise
        return {**base, **ov.get(k, {})} if isinstance(base, dict) else ov.get(k, base)

    det, trk, ov = get("detector"), get("tracker"), get("overlay")
    stage = RangeSpeedStage.from_config(get)
    model = stage.model
    cmap = {
        int(k): v for k, v in load_config("detector").require("class_maps")["visdrone10"].items()
    }
    meta = (metadata if isinstance(metadata, pd.DataFrame) else pd.read_csv(metadata)).set_index(
        "frame"
    )
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
        model_detect = make_detector(
            det.get("kind", "yolo"),
            REPO_ROOT / det["weights"],
            int(det["imgsz"]),
            float(det["score_floor"]),
            int(load_config("detector").require("max_det")),
            sorted(cmap),
        )

        def detect(img: np.ndarray, k: int) -> DetSet:
            ltwh, score, cls = model_detect(img)
            xywh = np.column_stack([ltwh[:, :2] + ltwh[:, 2:] / 2, ltwh[:, 2:]])
            return DetSet(xywh, np.asarray(score, float), np.asarray(cls, float))

    tracker = make_tracker(
        str(REPO_ROOT / trk["config"]),
        {"gmc_downscale": trk["gmc_downscale"]} if trk.get("gmc_downscale") else None,
    )
    stride = int(trk.get("det_stride", 1))
    times: dict[str, list[float]] = {
        k: [] for k in ("read", "detect", "track", "range_filter", "end_to_end", "overlay")
    }
    rows_all: list[dict] = []
    writer = None
    t_prev = None
    it = _frames(source)
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
            if detections_out is not None:
                for (cx_, cy_, w_, h_), sc, cl in zip(d.xywh, d.conf, d.cls, strict=True):
                    detections_out.append(
                        {
                            "frame": k,
                            "left": cx_ - w_ / 2,
                            "top": cy_ - h_ / 2,
                            "width": w_,
                            "height": h_,
                            "score": float(sc),
                            "cls": int(cl),
                        }
                    )
            res = np.asarray(tracker.update(d, img)).reshape(-1, 8)
            tracked = [(int(r[4]), int(r[6]), r[0], r[1], r[2] - r[0], r[3] - r[1]) for r in res]
        else:
            t2 = time.perf_counter()
            tracked = [(int(s.track_id), int(s.cls), *s.tlwh) for s in coast(tracker, img)]
        t3 = time.perf_counter()
        pose, t_s, tilt_sigma = pose_of(meta, k)
        tracked = [(t[0], cmap[t[1]], *t[2:]) for t in tracked if t[1] in cmap]
        frame_rows = stage.step(k, t_s, pose, tracked, tilt_sigma)
        dt = (t_s - t_prev) if t_prev is not None and np.isfinite(t_s) else DEFAULT_DT_S
        t_prev = t_s
        t4 = time.perf_counter()
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
        if not write_video:
            continue
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
        "speed_display": "unresolved: no speed is displayed"
        if stage.rule is None
        else (
            {g: (vars(r) if r is not None else "not displayed") for g, r in stage.rule.items()}
            if isinstance(stage.rule, dict)
            else vars(stage.rule)
        ),
        "note": "Sequential run on this machine; overlay drawing excluded from end_to_end (10.1). "
        "detect is near zero on frames between detections. For the protocol benchmark see "
        "scripts/benchmark_pipeline.py and scripts/benchmark_threaded.py.",
    }
    (out / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    return timing
