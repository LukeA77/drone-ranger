"""Feature tracks across frames for the filter's camera updates (D-741).

Shi-Tomasi corners and pyramidal Lucas-Kanade tracking checked forward and backward, as in
version 1's terrain from motion (`evaluate.terrain_motion`, same quality and consistency
settings), with persistent track ids, a minimum spacing between features, and outlier rejection
by a fundamental-matrix RANSAC between consecutive frames (OpenCV defaults). Feature count and
spacing follow OpenVINS's EuRoC configuration (num_pts 200, min_px_dist 10).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrackerSettings:
    max_features: int = 200  # OpenVINS num_pts
    min_distance_px: int = 10  # OpenVINS min_px_dist
    quality: float = 0.01  # version 1 terrain_motion (configs/geometry.yaml)
    fb_max_px: float = 1.0  # version 1 terrain_motion (configs/geometry.yaml)
    ransac_px: float = 3.0  # OpenCV findFundamentalMat default threshold
    ransac_conf: float = 0.99  # OpenCV findFundamentalMat default confidence


class FeatureTracker:
    def __init__(self, s: TrackerSettings) -> None:
        self.s = s
        self.prev: NDArray | None = None
        self.ids = np.zeros(0, int)
        self.pts = np.zeros((0, 2), np.float32)
        self.next_id = 0

    def step(self, gray: NDArray) -> tuple[NDArray, NDArray]:
        """Track into a new grayscale frame; returns (ids, pixel positions) in this frame."""
        if self.prev is not None and len(self.pts):
            p0 = self.pts.reshape(-1, 1, 2)
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None)
            fb = np.linalg.norm(p0 - pb, axis=2).ravel()
            h, w = gray.shape[:2]
            q = p1[:, 0]
            inside = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
            ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.s.fb_max_px) & inside
            if ok.sum() >= 8:
                _, mask = cv2.findFundamentalMat(
                    p0[ok, 0], q[ok], cv2.FM_RANSAC, self.s.ransac_px, self.s.ransac_conf
                )
                if mask is not None:
                    idx = np.flatnonzero(ok)
                    ok[idx[mask.ravel() == 0]] = False
            self.ids = self.ids[ok]
            self.pts = q[ok].astype(np.float32)
        else:
            self.ids = np.zeros(0, int)
            self.pts = np.zeros((0, 2), np.float32)
        need = self.s.max_features - len(self.pts)
        if need > 0:
            mask = np.full(gray.shape[:2], 255, np.uint8)
            for u, v in self.pts:
                cv2.circle(mask, (int(u), int(v)), self.s.min_distance_px, 0, -1)
            new = cv2.goodFeaturesToTrack(
                gray, need, self.s.quality, self.s.min_distance_px, mask=mask
            )
            if new is not None:
                new = new[:, 0].astype(np.float32)
                self.pts = np.vstack([self.pts, new])
                self.ids = np.r_[self.ids, np.arange(self.next_id, self.next_id + len(new))]
                self.next_id += len(new)
        self.prev = gray
        return self.ids.copy(), self.pts.astype(float)
