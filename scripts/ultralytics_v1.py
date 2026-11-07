"""Version 1's Ultralytics pieces, kept outside the package to reproduce version 1's results.

Ultralytics is AGPL-3.0 and is not a dependency of the package: `src/aerialrange` never
imports it or this module. Install it only in a separate environment, from
requirements-v1.txt (version 1's full pinned environment), to rerun version 1's training,
ONNX export and library-tracker results. Everything here imports the library lazily, so
scripts that also handle the project's own tracker run without it.

- `make_tracker`: a tracker from a tracker YAML. `tracker_type: motion` builds the
  project's own tracker (`aerialrange.track.runner.make_tracker`); `bytetrack` or `botsort`
  builds the library's tracker, wrapped so it has the same interface (`update`, `coast`).
- `run_sequence`: detector and library tracker through the library's `model.track`
  (version 1's tracking path).
- `make_detector`: as `aerialrange.detect.factory.make_detector`, plus version 1's PyTorch
  checkpoints (.pt) through the library's predict.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerialrange.detect.factory import Detect
from aerialrange.detect.factory import make_detector as make_onnx_detector
from aerialrange.track.runner import make_tracker as make_own_tracker


class LibraryTracker:
    """A library tracker (ByteTrack, BoT-SORT) with the project tracker's interface.

    `update(dets, img)` is the library's update. `coast(img)` advances one frame without
    detections and returns the confirmed tracks (predicted): it mirrors the library's update
    up to matching (Kalman predict of the tracked and lost tracks, then, for BoT-SORT, the
    camera-motion warp of those and of the unconfirmed tracks). Tracks are neither confirmed,
    lost nor removed when coasting. Other attributes pass through to the library tracker."""

    def __init__(self, tracker: Any) -> None:
        self.inner = tracker

    def update(self, dets: Any, img: NDArray) -> Any:
        return self.inner.update(dets, img)

    def coast(self, img: NDArray) -> list:
        from ultralytics.trackers.utils.stracks import joint_stracks, multi_gmc

        t = self.inner
        unconfirmed, tracked = t._split_tracked()
        pool = joint_stracks(tracked, t.lost_stracks)
        t.multi_predict(pool)
        if hasattr(t, "gmc") and t.gmc.method is not None:
            warp = t.gmc.apply(img, np.zeros((0, 4)))
            multi_gmc(pool, warp)
            multi_gmc(unconfirmed, warp)
        return [s for s in tracked if s.is_activated]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def make_tracker(tracker_cfg: str, overrides: Mapping[str, Any] | None = None) -> Any:
    """The project's tracker for a `tracker_type: motion` YAML, otherwise the library's.

    For a library tracker, `overrides` replace YAML values (the validation grid); unknown keys
    are refused. `gmc_downscale` sets BoT-SORT's camera-motion frame downscale (library
    default 2), which the library does not read from YAML."""
    import yaml

    with open(tracker_cfg, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw.get("tracker_type") == "motion":
        return make_own_tracker(tracker_cfg, overrides)
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import IterableSimpleNamespace

    ov = dict(overrides or {})
    gmc_downscale = ov.pop("gmc_downscale", None)
    unknown = set(ov) - set(raw)
    if unknown:
        raise KeyError(f"tracker overrides not in {tracker_cfg}: {sorted(unknown)}")
    cfg = IterableSimpleNamespace(**{**raw, **ov})
    tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)
    if gmc_downscale is not None:
        if not hasattr(tracker, "gmc"):
            raise KeyError(f"gmc_downscale given but {cfg.tracker_type} has no motion compensation")
        tracker.gmc.downscale = max(1, int(gmc_downscale))
    return LibraryTracker(tracker)


def run_sequence(
    model: Any,
    frames: Sequence[Path],
    tracker_cfg: str,
    imgsz: int,
    class_to_group: Mapping[int, str],
    max_det: int = 300,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Track one sequence with the library's `model.track`. Returns rows per group (frame,
    id, left, top, width, height, score), with 1-based frame numbers as in VisDrone, and
    timing totals in seconds. `max_det` caps detections per frame (configs/detector.yaml)."""
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


def make_detector(
    kind: str,
    weights: str | Path,
    imgsz: int,
    score_floor: float | None,
    max_det: int,
    classes: Sequence[int] | None = None,
) -> Detect:
    """`aerialrange.detect.factory.make_detector`, plus "yolo" PyTorch checkpoints (.pt).

    `score_floor=None` keeps the library's predict default (0.25), as version 1 did."""
    if not (kind == "yolo" and Path(weights).suffix == ".pt"):
        return make_onnx_detector(kind, weights, imgsz, score_floor, max_det, classes)
    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    keep = None if classes is None else sorted(int(c) for c in classes)
    extra = {} if score_floor is None else {"conf": float(score_floor)}

    def detect_yolo(img: NDArray[np.uint8]) -> tuple[NDArray, NDArray, NDArray]:
        r = model.predict(img, imgsz=imgsz, classes=keep, max_det=max_det, verbose=False, **extra)[
            0
        ]
        xyxy = r.boxes.xyxy.cpu().numpy()
        ltwh = np.column_stack([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]])
        return ltwh, r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)

    return detect_yolo


def library_version(kind: str, weights: str | Path) -> dict[str, str]:
    """Versions of what runs this detector, for result files."""
    from importlib.metadata import version

    from aerialrange.detect.factory import library_version as onnx_version

    if kind == "yolo" and Path(weights).suffix == ".pt":
        return {"ultralytics": version("ultralytics")}
    return onnx_version(kind)
