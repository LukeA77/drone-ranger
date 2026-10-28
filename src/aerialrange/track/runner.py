"""Run a detector + tracker over a VisDrone sequence.

Uses the detector library's own trackers (ByteTrack, BoT-SORT) through `model.track`.
Appearance re-identification stays off (motion and overlap only). Each
detector class maps to an evaluation group through `class_to_group`; classes not in the map
are dropped before tracking.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def run_sequence(
    model: Any,
    frames: Sequence[Path],
    tracker_cfg: str,
    imgsz: int,
    class_to_group: Mapping[int, str],
    max_det: int = 300,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Track one sequence. Returns rows per group, (frame, id, left, top, width, height,
    score), with 1-based frame numbers as in VisDrone, and timing totals in seconds.
    `max_det` caps detections per frame (`max_det` in configs/detector.yaml, D-316)."""
    rows: dict[str, list] = {g: [] for g in set(class_to_group.values())}
    classes = sorted(class_to_group)
    t_total = 0.0
    for k, path in enumerate(frames, start=1):
        t0 = time.perf_counter()
        res = model.track(
            source=str(path),
            persist=k > 1,
            tracker=tracker_cfg,
            imgsz=imgsz,
            classes=classes,
            max_det=max_det,
            verbose=False,
        )[0]
        t_total += time.perf_counter() - t0
        b = res.boxes
        if b is None or b.id is None or len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy()
        ids = b.id.cpu().numpy().astype(int)
        cls = b.cls.cpu().numpy().astype(int)
        conf = b.conf.cpu().numpy()
        for (x1, y1, x2, y2), i, c, s in zip(xyxy, ids, cls, conf, strict=True):
            g = class_to_group.get(int(c))
            if g is not None:
                rows[g].append((k, i, x1, y1, x2 - x1, y2 - y1, s))
    out = {g: np.array(v, float).reshape(-1, 7) for g, v in rows.items()}
    return out, {"track_seconds": t_total, "frames": len(frames)}


@dataclass
class DetSet:
    """Detections in the form the library trackers accept: centre-format boxes (xywh, pixels),
    scores and class ids, with boolean indexing."""

    xywh: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return len(self.conf)

    @property
    def xyxy(self) -> np.ndarray:
        """Corner-format boxes; BoT-SORT's camera-motion compensation uses them to mask the
        detections (without it GMC silently falls back to identity)."""
        c, wh = self.xywh[:, :2], self.xywh[:, 2:4]
        return np.column_stack([c - wh / 2, c + wh / 2])

    def __getitem__(self, m: Any) -> DetSet:
        return DetSet(self.xywh[m], self.conf[m], self.cls[m])


def make_tracker(tracker_cfg: str, overrides: Mapping[str, Any] | None = None) -> Any:
    """A library tracker (ByteTrack, BoT-SORT, ...) built from a tracker YAML. `overrides`
    replace YAML values (the validation grid, `{{TRACKER_GRID}}`); unknown keys are refused.
    `gmc_downscale` sets BoT-SORT's camera-motion frame downscale (library default 2), which
    the library does not read from YAML (Phase 5 speed options)."""
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import YAML, IterableSimpleNamespace

    raw = YAML.load(tracker_cfg)
    ov = dict(overrides or {})
    gmc_downscale = ov.pop("gmc_downscale", None)  # not a YAML key in the library (Phase 5)
    unknown = set(ov) - set(raw)
    if unknown:
        raise KeyError(f"tracker overrides not in {tracker_cfg}: {sorted(unknown)}")
    cfg = IterableSimpleNamespace(**{**raw, **ov})
    tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)
    if gmc_downscale is not None:
        if not hasattr(tracker, "gmc"):
            raise KeyError(f"gmc_downscale given but {cfg.tracker_type} has no motion compensation")
        tracker.gmc.downscale = max(1, int(gmc_downscale))
    return tracker


def coast(tracker: Any, img: np.ndarray) -> list:
    """Advance a library tracker by one frame without detections and return its confirmed
    tracks (predicted). Mirrors the library's update up to matching: Kalman predict of the
    tracked and lost tracks, then, for BoT-SORT, the camera-motion warp of those and of the
    unconfirmed tracks. Tracks are neither confirmed, lost nor removed here."""
    from ultralytics.trackers.utils.stracks import joint_stracks, multi_gmc

    unconfirmed, tracked = tracker._split_tracked()
    pool = joint_stracks(tracked, tracker.lost_stracks)
    tracker.multi_predict(pool)
    if hasattr(tracker, "gmc") and tracker.gmc.method is not None:
        warp = tracker.gmc.apply(img, np.zeros((0, 4)))
        multi_gmc(pool, warp)
        multi_gmc(unconfirmed, warp)
    return [t for t in tracked if t.is_activated]


def run_sequence_custom(
    detect: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    frames: Sequence[Path | np.ndarray],
    tracker_cfg: str,
    class_to_group: Mapping[int, str],
    overrides: Mapping[str, Any] | None = None,
    det_stride: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Track one sequence with any detection function (for example tiled detection, B3).

    `detect(image)` returns (ltwh boxes, scores, class ids). Frames are image paths or arrays.
    Same output as `run_sequence`: rows per group (frame, id, left, top, width, height, score),
    1-based frames, and timing totals (detection plus tracking). `overrides` as in
    `make_tracker`.

    `det_stride` > 1 runs the detector on every `det_stride`-th frame only (Phase 5, D-610).
    On the frames in between, the tracker is not updated (an update without detections would
    delete new, not yet confirmed tracks); its motion model is advanced instead, exactly as
    the library does before matching (Kalman predict of tracked and lost tracks, then the
    camera-motion warp for BoT-SORT), and the predicted boxes of confirmed tracks are output.
    """
    import cv2

    tracker = make_tracker(tracker_cfg, overrides)
    rows: dict[str, list] = {g: [] for g in set(class_to_group.values())}
    t_total = 0.0
    for k, f in enumerate(frames, start=1):
        img = cv2.imread(str(f)) if not isinstance(f, np.ndarray) else f
        t0 = time.perf_counter()
        if (k - 1) % det_stride:
            if hasattr(detect, "skip"):  # keep frame-indexed detection sources in step
                detect.skip()
            for t in coast(tracker, img):
                g = class_to_group.get(int(t.cls))
                if g is not None:
                    x, y, w, h = t.tlwh
                    rows[g].append((k, int(t.track_id), x, y, w, h, float(t.score)))
            t_total += time.perf_counter() - t0
            continue
        b, s, c = detect(img)
        b, s, c = np.asarray(b, float).reshape(-1, 4), np.asarray(s, float), np.asarray(c, int)
        keep = np.isin(c, list(class_to_group))
        b, s, c = b[keep], s[keep], c[keep]
        xywh = np.column_stack([b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2, b[:, 2], b[:, 3]])
        out = tracker.update(DetSet(xywh, s, c.astype(float)), img)
        t_total += time.perf_counter() - t0
        for x1, y1, x2, y2, tid, score, cls, _ in np.asarray(out).reshape(-1, 8):
            g = class_to_group.get(int(cls))
            if g is not None:
                rows[g].append((k, int(tid), x1, y1, x2 - x1, y2 - y1, score))
    out_rows = {g: np.array(v, float).reshape(-1, 7) for g, v in rows.items()}
    return out_rows, {"track_seconds": t_total, "frames": len(frames)}
