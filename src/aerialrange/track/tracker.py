"""The project's multi-object tracker: motion and overlap only, no appearance features.

Method (two-stage association of ByteTrack, Zhang et al. 2022, with the camera-motion
compensation and box-size Kalman state of BoT-SORT, Aharon et al. 2022; written for this
project from the papers):

1. Every track is predicted one frame ahead (`BoxKalman`), then moved by the camera's
   frame-to-frame motion (`CameraMotion`).
2. Stage one: confirmed and lost tracks are matched to confident detections
   (score >= `high`) by minimum total cost, cost = 1 - IoU (optionally times the detection
   score), refusing any pair costing more than `match_cost`.
3. Stage two: tracks still unmatched that were seen last frame are matched to the
   remaining weak detections (`low` < score < `high`) by IoU alone (`second_cost`), which
   keeps a track through a frame where its object is detected with low confidence.
4. Tracks seen last frame and matched in neither stage become lost; lost tracks are kept for
   `max_lost` frames (so a briefly hidden object keeps its identity) and then dropped.
5. Tentative tracks (one detection so far) are matched to the confident detections left
   (`tentative_cost`); a match confirms them, no match drops them.
6. Confident detections still unmatched with score >= `new` start tentative tracks (on the
   first frame they are confirmed at once).
7. A tracked and a lost track overlapping at IoU > `duplicate_iou` are one object: the
   younger is dropped.

Options beyond the papers, each off by default and chosen on validation only:
`class_groups` forbids matching across groups (a person track never takes a vehicle
detection); `iou_buffer` enlarges both boxes by that fraction of their size on each side
before IoU (buffered IoU, Yang et al. 2023), so small fast objects that no longer overlap
their prediction can still match. `high_by_class` and `new_by_class` give a class its own
`high` and `new` thresholds (classes not listed use `high` and `new`), so that people and
vehicles can be tuned to different aims.

Output per frame: confirmed tracks matched in this frame, with the filtered box.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from aerialrange.track.box_kalman import BoxKalman
from aerialrange.track.camera_motion import CameraMotion

TRACKED, LOST = "tracked", "lost"


@dataclass(frozen=True)
class TrackerSettings:
    high: float = 0.5
    low: float = 0.1
    new: float = 0.6
    match_cost: float = 0.8
    second_cost: float = 0.5
    tentative_cost: float = 0.7
    fuse_score: bool = True
    max_lost: int = 30
    duplicate_iou: float = 0.85
    cmc: bool = True
    cmc_downscale: int = 2
    cmc_corners: int = 1000
    cmc_min_distance_px: float = 3.0
    cmc_window_px: int = 21
    cmc_levels: int = 3
    cmc_check_back: bool = True
    std_pos: float = 1 / 20
    std_vel: float = 1 / 160
    class_groups: tuple[tuple[int, ...], ...] | None = None
    iou_buffer: float = 0.0
    high_by_class: Mapping[int, float] | None = None
    new_by_class: Mapping[int, float] | None = None

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> TrackerSettings:
        names = {f.name for f in fields(cls)}
        unknown = set(m) - names
        if unknown:
            raise KeyError(f"unknown tracker settings: {sorted(unknown)}")
        kw = dict(m)
        if kw.get("class_groups") is not None:
            kw["class_groups"] = tuple(tuple(int(c) for c in g) for g in kw["class_groups"])
        for key in ("high_by_class", "new_by_class"):
            if kw.get(key) is not None:
                kw[key] = {int(c): float(v) for c, v in kw[key].items()}
        return cls(**kw)

    def thresholds(self, cls: NDArray) -> tuple[NDArray, NDArray]:
        """Per-detection `high` and `new` thresholds for an array of class ids."""
        hi = np.full(len(cls), self.high, float)
        new = np.full(len(cls), self.new, float)
        for table, out in ((self.high_by_class, hi), (self.new_by_class, new)):
            for c, v in (table or {}).items():
                out[cls == c] = v
        return hi, new


@dataclass(eq=False)  # identity, not field, comparison
class Track:
    track_id: int
    cls: int
    score: float
    mean: NDArray
    cov: NDArray
    start_frame: int
    last_frame: int
    confirmed: bool = False
    state: str = TRACKED
    hits: int = 1
    extra: dict = field(default_factory=dict)

    @property
    def tlwh(self) -> NDArray:
        cx, cy, w, h = self.mean[:4]
        return np.array([cx - w / 2, cy - h / 2, w, h])

    @property
    def xyxy(self) -> NDArray:
        cx, cy, w, h = self.mean[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def iou_xyxy(a: NDArray, b: NDArray, buffer: float = 0.0) -> NDArray:
    """IoU between box sets (corner format); `buffer` enlarges every box by that fraction of
    its width and height on each side first."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    if buffer:
        a, b = _grow(a, buffer), _grow(b, buffer)
    iw = np.clip(
        np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None
    )
    ih = np.clip(
        np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None
    )
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def _grow(b: NDArray, f: float) -> NDArray:
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    return np.column_stack([b[:, 0] - f * w, b[:, 1] - f * h, b[:, 2] + f * w, b[:, 3] + f * h])


def assign(cost: NDArray, max_cost: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Minimum-total-cost matching where leaving a row or column unmatched costs
    `max_cost` / 2, so no pair costing more than `max_cost` is ever used (the usual
    gated assignment of ByteTrack). Returns matches and unmatched rows and columns.
    Rows and columns with no allowed pair cannot be matched, so they are left out of the
    solve, which keeps it small in crowded frames without changing its answer."""
    n, m = cost.shape
    ok = np.isfinite(cost) & (cost <= max_cost)
    rs, cs = np.flatnonzero(ok.any(1)), np.flatnonzero(ok.any(0))
    matches: list[tuple[int, int]] = []
    if len(rs) and len(cs):
        big = 1e6
        c = np.where(ok[np.ix_(rs, cs)], cost[np.ix_(rs, cs)], big)
        a, b = len(rs), len(cs)
        aug = np.full((a + b, a + b), big)
        aug[:a, :b] = c
        aug[np.arange(a), b + np.arange(a)] = max_cost / 2
        aug[a + np.arange(b), np.arange(b)] = max_cost / 2
        aug[a:, b:] = 0.0
        rows, cols = linear_sum_assignment(aug)
        matches = [
            (int(rs[r]), int(cs[k]))
            for r, k in zip(rows, cols, strict=True)
            if r < a and k < b and c[r, k] < big
        ]
    mr = {r for r, _ in matches}
    mc = {k for _, k in matches}
    return matches, [r for r in range(n) if r not in mr], [k for k in range(m) if k not in mc]


class MotionTracker:
    """Track boxes across frames. `update(dets, img)` with `dets` having `xywh` (centre
    format), `conf` and `cls` arrays returns rows (x1, y1, x2, y2, id, score, cls, det index)
    like the library trackers it replaces; `coast(img)` advances without detections."""

    def __init__(self, settings: TrackerSettings | None = None) -> None:
        self.s = settings or TrackerSettings()
        self.kf = BoxKalman(self.s.std_pos, self.s.std_vel)
        self.motion = (
            CameraMotion(
                self.s.cmc_downscale,
                max_corners=self.s.cmc_corners,
                min_distance_px=self.s.cmc_min_distance_px,
                window_px=self.s.cmc_window_px,
                levels=self.s.cmc_levels,
                check_back=self.s.cmc_check_back,
            )
            if self.s.cmc
            else None
        )
        self.frame = 0
        self.next_id = 1
        self.tracks: list[Track] = []  # seen last frame: confirmed and tentative
        self.lost: list[Track] = []
        self._group = (
            {c: i for i, g in enumerate(self.s.class_groups) for c in g}
            if self.s.class_groups
            else None
        )

    # --- motion -----------------------------------------------------------------------
    def _advance(self, img: NDArray, avoid_xyxy: NDArray) -> None:
        """Predict every live track one frame and apply the camera motion."""
        live = self.tracks + self.lost
        if self.motion is not None:
            a = self.motion.estimate(img, avoid_xyxy)
        if not live:
            return
        mean = np.stack([t.mean for t in live])
        cov = np.stack([t.cov for t in live])
        freeze = np.array([t.state == LOST for t in live])
        mean, cov = self.kf.predict(mean, cov, freeze)
        if self.motion is not None:
            mean, cov = self.kf.warp(mean, cov, a)
        for t, m_, c_ in zip(live, mean, cov, strict=True):
            t.mean, t.cov = m_, c_

    # --- costs ------------------------------------------------------------------------
    def _cost(
        self, trk: Sequence[Track], xyxy: NDArray, conf: NDArray, cls: NDArray, fuse: bool
    ) -> NDArray:
        if not len(trk) or not len(xyxy):
            return np.zeros((len(trk), len(xyxy)))
        tb = np.stack([t.xyxy for t in trk])
        sim = iou_xyxy(tb, xyxy, self.s.iou_buffer)
        if fuse:
            sim = sim * conf[None, :]
        cost = 1.0 - sim
        if self._group is not None:
            tg = np.array([self._group.get(t.cls, -1) for t in trk])
            dg = np.array([self._group.get(int(c), -1) for c in cls])
            cost[tg[:, None] != dg[None, :]] = np.inf
        return cost

    def _correct(self, t: Track, z: NDArray, score: float, cls: int) -> None:
        """Mark a match; the filter correction itself is batched in `_apply_corrections`
        (the stages match disjoint tracks, so deferring it changes nothing)."""
        t.score, t.cls, t.last_frame, t.state = float(score), int(cls), self.frame, TRACKED
        t.hits += 1
        self._pending.append((t, z))

    def _apply_corrections(self) -> None:
        if not self._pending:
            return
        trk = [t for t, _ in self._pending]
        m, c = self.kf.update(
            np.stack([t.mean for t in trk]),
            np.stack([t.cov for t in trk]),
            np.stack([z for _, z in self._pending]),
        )
        for t, m_, c_ in zip(trk, m, c, strict=True):
            t.mean, t.cov = m_, c_
        self._pending = []

    # --- public -----------------------------------------------------------------------
    def update(self, dets: Any, img: NDArray) -> NDArray:
        self.frame += 1
        xywh = np.asarray(dets.xywh, float).reshape(-1, 4)
        conf = np.asarray(dets.conf, float).reshape(-1)
        cls = np.asarray(dets.cls, float).reshape(-1).astype(int)
        xyxy = np.column_stack([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2])
        self._advance(img, xyxy)

        t_high, t_new = self.s.thresholds(cls)
        hi = np.flatnonzero(conf >= t_high)
        lo = np.flatnonzero((conf > self.s.low) & (conf < t_high))
        confirmed = [t for t in self.tracks if t.confirmed]
        tentative = [t for t in self.tracks if not t.confirmed]
        pool = confirmed + self.lost
        matched: list[tuple[Track, int]] = []
        self._pending: list[tuple[Track, NDArray]] = []

        # stage one: confirmed and lost tracks against confident detections
        c1 = self._cost(pool, xyxy[hi], conf[hi], cls[hi], self.s.fuse_score)
        m1, u_trk, u_hi = assign(c1, self.s.match_cost)
        for r, k in m1:
            self._correct(pool[r], xywh[hi[k]], conf[hi[k]], cls[hi[k]])
            matched.append((pool[r], int(hi[k])))

        # stage two: tracks seen last frame against weak detections
        rest = [pool[r] for r in u_trk if pool[r].state == TRACKED]
        c2 = self._cost(rest, xyxy[lo], conf[lo], cls[lo], False)
        m2, u_rest, _ = assign(c2, self.s.second_cost)
        for r, k in m2:
            self._correct(rest[r], xywh[lo[k]], conf[lo[k]], cls[lo[k]])
            matched.append((rest[r], int(lo[k])))
        newly_lost = [rest[r] for r in u_rest]
        for t in newly_lost:
            t.state = LOST

        # tentative tracks against the confident detections left
        left = hi[u_hi]
        c3 = self._cost(tentative, xyxy[left], conf[left], cls[left], self.s.fuse_score)
        m3, _, u_left = assign(c3, self.s.tentative_cost)
        kept_tentative = []
        for r, k in m3:
            t = tentative[r]
            self._correct(t, xywh[left[k]], conf[left[k]], cls[left[k]])
            t.confirmed = True
            matched.append((t, int(left[k])))
            kept_tentative.append(t)

        self._apply_corrections()

        # new tracks
        born = []
        for k in left[u_left]:
            if conf[k] < t_new[k]:
                continue
            mean, cov = self.kf.initiate(xywh[k][None])
            born.append(
                Track(
                    self.next_id,
                    int(cls[k]),
                    float(conf[k]),
                    mean[0],
                    cov[0],
                    self.frame,
                    self.frame,
                    confirmed=self.frame == 1,
                )
            )
            self.next_id += 1
            if born[-1].confirmed:  # first frame: output at once
                matched.append((born[-1], int(k)))

        # bookkeeping
        seen = [t for t in pool if t.state == TRACKED and t.last_frame == self.frame]
        self.tracks = seen + kept_tentative + born
        still_lost = [t for t in self.lost if t.state == LOST] + newly_lost
        self.lost = [t for t in still_lost if self.frame - t.last_frame <= self.s.max_lost]
        self._drop_duplicates()

        live = {id(t) for t in self.tracks}
        out = [
            (*t.xyxy, t.track_id, t.score, t.cls, k)
            for t, k in matched
            if t.confirmed and id(t) in live
        ]
        return np.array(out, float).reshape(-1, 8)

    def coast(self, img: NDArray) -> list[Track]:
        """Advance one frame without detections (detection stride): predict and apply camera
        motion; no track is confirmed, lost or dropped. Returns the confirmed tracks seen at
        the last detection frame, at their predicted boxes."""
        self.frame += 1
        live = [t for t in self.tracks if t.confirmed]
        avoid = (
            np.stack([t.xyxy for t in self.tracks + self.lost]) if self.tracks + self.lost else None
        )
        self._advance(img, avoid)
        return live

    def _drop_duplicates(self) -> None:
        if not self.tracks or not self.lost:
            return
        iou = iou_xyxy(
            np.stack([t.xyxy for t in self.tracks]), np.stack([t.xyxy for t in self.lost])
        )
        drop_t, drop_l = set(), set()
        for i, j in zip(*np.nonzero(iou > self.s.duplicate_iou), strict=True):
            a, b = self.tracks[i], self.lost[j]
            if self.frame - a.start_frame > self.frame - b.start_frame:
                drop_l.add(j)
            else:
                drop_t.add(i)
        self.tracks = [t for i, t in enumerate(self.tracks) if i not in drop_t]
        self.lost = [t for j, t in enumerate(self.lost) if j not in drop_l]
