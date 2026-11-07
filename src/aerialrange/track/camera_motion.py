"""Frame-to-frame camera motion in the image, for the tracker's motion compensation.

A drone camera moves, so a stationary car shifts in the image from frame to frame. The
tracker removes that shift before matching by estimating a similarity transform (rotation,
uniform scale, translation) between consecutive frames. Method, as in version 1's
footage analysis (`evaluate/auair_scale.py`): corners in the current frame, away from
detected objects (which move on their own), are followed to the previous frame by
pyramidal Lucas-Kanade optical flow; only points that flow back to where they started
(forward-backward check) are kept, and the transform is fitted by RANSAC. Work is done on a
downscaled grey image (`downscale`), the main speed setting.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class CameraMotion:
    """Stateful estimator: call `estimate` once per frame, in order."""

    def __init__(
        self,
        downscale: int = 2,
        max_corners: int = 1000,
        quality: float = 0.01,
        min_distance_px: float = 3.0,
        fb_max_px: float = 1.0,
        ransac_px: float = 1.0,
        min_inliers: int = 10,
        window_px: int = 21,
        levels: int = 3,
        check_back: bool = True,
    ) -> None:
        self.d = max(1, int(downscale))
        self.lk = {"winSize": (int(window_px), int(window_px)), "maxLevel": int(levels)}
        self.check_back = check_back
        self.max_corners, self.quality, self.min_distance = max_corners, quality, min_distance_px
        self.fb_max, self.ransac, self.min_inliers = fb_max_px, ransac_px, min_inliers
        self.prev: NDArray | None = None
        self.n_inliers = 0  # of the last estimate, for diagnostics

    def reset(self) -> None:
        self.prev = None

    def _grey(self, img: NDArray) -> NDArray:
        """Downscale first, then convert to grey: about 20 times faster than the reverse on a
        4-megapixel frame, and the features only need the coarse image."""
        if img.ndim == 3:
            if self.d > 1:
                img = cv2.resize(
                    img,
                    (img.shape[1] // self.d, img.shape[0] // self.d),
                    interpolation=cv2.INTER_LINEAR,
                )
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.d > 1:
            img = cv2.resize(
                img,
                (img.shape[1] // self.d, img.shape[0] // self.d),
                interpolation=cv2.INTER_LINEAR,
            )
        return img

    def estimate(self, img: NDArray, avoid_xyxy: NDArray | None = None) -> NDArray:
        """2x3 transform taking previous-frame pixels to this frame's (full resolution).
        `avoid_xyxy` (k, 4): boxes in this frame to keep corners out of. Identity on the first
        frame or when too few points agree."""
        cur = self._grey(img)
        prev, self.prev = self.prev, cur
        self.n_inliers = 0
        if prev is None or prev.shape != cur.shape:
            return IDENTITY.copy()
        mask = box_mask(cur.shape, avoid_xyxy, self.d)
        p = cv2.goodFeaturesToTrack(
            cur, self.max_corners, self.quality, self.min_distance, mask=mask
        )
        if p is None or len(p) < self.min_inliers:
            return IDENTITY.copy()
        q, st, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p, None, **self.lk)
        ok = st[:, 0] == 1
        if self.check_back:  # keep points that flow back to where they started
            r, st2, _ = cv2.calcOpticalFlowPyrLK(prev, cur, q, None, **self.lk)
            ok &= (st2[:, 0] == 1) & (np.linalg.norm(r - p, axis=2)[:, 0] < self.fb_max)
        if ok.sum() < self.min_inliers:
            return IDENTITY.copy()
        a, inl = cv2.estimateAffinePartial2D(
            q[ok], p[ok], method=cv2.RANSAC, ransacReprojThreshold=self.ransac
        )
        if a is None or inl is None or int(inl.sum()) < self.min_inliers:
            return IDENTITY.copy()
        self.n_inliers = int(inl.sum())
        a = a.astype(float)
        a[:, 2] *= self.d
        return a


def box_mask(shape: tuple[int, ...], xyxy: NDArray | None, d: int) -> NDArray:
    """uint8 mask of `shape`, 0 inside any box (full-resolution corners, scaled by 1/`d`,
    rounded outwards), 255 elsewhere. Vectorised: box corners are summed into a difference
    image whose 2D running sum counts the boxes covering each pixel."""
    h, w = shape[:2]
    if xyxy is None or not len(xyxy):
        return np.full((h, w), 255, np.uint8)
    b = np.asarray(xyxy, float).reshape(-1, 4) / d
    x0 = np.clip(np.floor(b[:, 0]), 0, w).astype(int)
    y0 = np.clip(np.floor(b[:, 1]), 0, h).astype(int)
    x1 = np.clip(np.ceil(b[:, 2]) + 1, 0, w).astype(int)
    y1 = np.clip(np.ceil(b[:, 3]) + 1, 0, h).astype(int)
    diff = np.zeros((h + 1, w + 1), np.int32)
    np.add.at(diff, (y0, x0), 1)
    np.add.at(diff, (y0, x1), -1)
    np.add.at(diff, (y1, x0), -1)
    np.add.at(diff, (y1, x1), 1)
    cover = diff.cumsum(0).cumsum(1)[:h, :w]
    return np.where(cover > 0, 0, 255).astype(np.uint8)
