"""Run a detector and a tracker over a sequence.

The tracker is the project's own (`track/tracker.py`), built from a tracker YAML with
`tracker_type: motion`. Any object with the same interface works in `run_sequence_custom`
(`update(DetSet, image)` returning rows of x1, y1, x2, y2, id, score, class, index, and
`coast(image)` returning predicted confirmed tracks): version 1's library trackers are
wrapped that way in scripts/ultralytics_v1.py, outside the package. Appearance
re-identification is not used. Each detector class maps to an evaluation group through
`class_to_group`; classes not in the map are dropped before tracking.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DetSet:
    """Detections in the form the trackers accept (the project's, and version 1's library
    trackers): centre-format boxes (xywh, pixels), scores and class ids, with boolean
    indexing."""

    xywh: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return len(self.conf)

    @property
    def xyxy(self) -> np.ndarray:
        """Corner-format boxes; camera-motion compensation uses them to mask the detections
        (without them version 1's BoT-SORT silently fell back to no compensation)."""
        c, wh = self.xywh[:, :2], self.xywh[:, 2:4]
        return np.column_stack([c - wh / 2, c + wh / 2])

    def __getitem__(self, m: Any) -> DetSet:
        return DetSet(self.xywh[m], self.conf[m], self.cls[m])


# Version 1's library tracker keys and the project's tracker settings they correspond to.
LIBRARY_KEYS = {
    "gmc_downscale": "cmc_downscale",
    "track_high_thresh": "high",
    "track_low_thresh": "low",
    "new_track_thresh": "new",
}


def make_tracker(tracker_cfg: str, overrides: Mapping[str, Any] | None = None) -> Any:
    """The project's tracker (`track/tracker.py`) from a tracker YAML with
    `tracker_type: motion`; its keys are `TrackerSettings` fields. `overrides` replace YAML
    values (the validation grid, `{{TRACKER_GRID}}`); unknown keys are refused. Version 1's
    library tracker names are accepted as overrides so the validation grid tunes every
    tracker the same way: `gmc_downscale` sets `cmc_downscale`, `track_high_thresh` sets
    `high`, `track_low_thresh` sets `low` and `new_track_thresh` sets `new`. Other tracker
    types (version 1's ByteTrack and BoT-SORT) are built by scripts/ultralytics_v1.py."""
    import yaml

    from aerialrange.track.tracker import MotionTracker, TrackerSettings

    with open(tracker_cfg, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw.get("tracker_type") != "motion":
        raise ValueError(
            f"{tracker_cfg}: tracker_type {raw.get('tracker_type')!r} is not the project's "
            "tracker ('motion'); version 1's library trackers are in scripts/ultralytics_v1.py"
        )
    ov = dict(overrides or {})
    for lib_key, own_key in LIBRARY_KEYS.items():
        if lib_key in ov:
            ov[own_key] = ov.pop(lib_key)
    unknown = set(ov) - set(raw)
    if unknown:
        raise KeyError(f"tracker overrides not in {tracker_cfg}: {sorted(unknown)}")
    merged = {k: v for k, v in {**raw, **ov}.items() if k != "tracker_type"}
    return MotionTracker(TrackerSettings.from_mapping(merged))


def coast(tracker: Any, img: np.ndarray) -> list:
    """Advance a tracker by one frame without detections and return its confirmed tracks
    (predicted). Tracks are neither confirmed, lost nor removed."""
    return tracker.coast(img)


def run_sequence_custom(
    detect: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    frames: Sequence[Path | np.ndarray],
    tracker_cfg: str,
    class_to_group: Mapping[int, str],
    overrides: Mapping[str, Any] | None = None,
    det_stride: int = 1,
    build_tracker: Callable[[str, Mapping[str, Any] | None], Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Track one sequence with any detection function (for example tiled detection, B3).

    `detect(image)` returns (ltwh boxes, scores, class ids). Frames are image paths or arrays.
    Returns rows per group (frame, id, left, top, width, height, score), with 1-based frame
    numbers as in VisDrone, and timing totals (detection plus tracking). `overrides` as in
    `make_tracker`; `build_tracker(tracker_cfg, overrides)` replaces `make_tracker` (for
    example scripts/ultralytics_v1.py's, for version 1's library trackers).

    `det_stride` > 1 runs the detector on every `det_stride`-th frame only (Phase 5, D-610).
    On the frames in between, the tracker is not updated (an update without detections would
    delete new, not yet confirmed tracks); it coasts instead (`coast`: motion model and
    camera-motion compensation advanced), and the predicted boxes of confirmed tracks are
    output.
    """
    import cv2

    tracker = (build_tracker or make_tracker)(tracker_cfg, overrides)
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
