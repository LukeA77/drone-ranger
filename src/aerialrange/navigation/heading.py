"""Direction of travel seen by the camera, for the in-flight alignment (D-741).

Between two frames the essential matrix gives the camera's translation direction up to scale
(Nister's five-point method with RANSAC, OpenCV `findEssentialMat` and `recoverPose`).
Compared with the GPS velocity direction in the world frame it fixes yaw even in straight,
steady flight, where the IMU and GPS alone cannot (D-741). Its uncertainty is measured per
frame pair by bootstrapping the inlier matches (resampled with replacement, the direction
re-estimated each time), so no noise level is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from aerialrange.navigation.features import TrackerSettings


@dataclass(frozen=True)
class TravelDirection:
    d_c: NDArray[np.float64]  # unit direction of camera b's centre in camera a's frame
    sigma_rad: float  # bootstrap angular standard deviation
    inliers: int


def travel_direction(
    img_a: NDArray,
    img_b: NDArray,
    K: NDArray,
    ts: TrackerSettings,
    rng: np.random.Generator,
    n_boot: int = 30,
    ransac_px: float = 1.0,
    min_inliers: int = 30,
) -> TravelDirection | None:
    """Camera translation direction from frame a to frame b (None if too few matches).

    `ransac_px`: RANSAC threshold in pixels (OpenCV's default for findEssentialMat is 1.0).
    """
    p0 = cv2.goodFeaturesToTrack(img_a, 2 * ts.max_features, ts.quality, ts.min_distance_px)
    if p0 is None or len(p0) < min_inliers:
        return None
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(img_a, img_b, p0, None)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(img_b, img_a, p1, None)
    fb = np.linalg.norm(p0 - pb, axis=2).ravel()
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < ts.fb_max_px)
    if ok.sum() < min_inliers:
        return None
    Ki = np.linalg.inv(K)
    xa = (np.c_[p0[ok, 0], np.ones(ok.sum())] @ Ki.T)[:, :2]
    xb = (np.c_[p1[ok, 0], np.ones(ok.sum())] @ Ki.T)[:, :2]
    thr = ransac_px / K[0, 0]

    def solve(a: NDArray, b: NDArray) -> tuple[NDArray | None, NDArray | None]:
        E, mask = cv2.findEssentialMat(a, b, np.eye(3), cv2.RANSAC, 0.999, thr)
        if E is None or E.shape != (3, 3):
            return None, None
        _, R, t, mask2 = cv2.recoverPose(E, a, b, np.eye(3), mask=mask)
        c = -R.T @ t.ravel()  # camera b's centre in camera a's frame
        return c / np.linalg.norm(c), mask2.ravel() > 0

    d, inl = solve(xa, xb)
    if d is None or inl.sum() < min_inliers:
        return None
    ia, ib = xa[inl], xb[inl]
    angles = []
    for _ in range(n_boot):
        j = rng.integers(0, len(ia), len(ia))
        db, _ = solve(ia[j], ib[j])
        if db is not None:
            angles.append(np.arccos(np.clip(db @ d, -1.0, 1.0)))
    if len(angles) < n_boot // 2:
        return None
    return TravelDirection(d, float(np.sqrt(np.mean(np.square(angles)))), int(inl.sum()))
