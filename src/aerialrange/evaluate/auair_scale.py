"""AU-AIR scale self-calibration on straight-down frames (D-508).

True height above the ground is h = altitude + h0, where h0 is the altimeter's offset (it
drifts during a flight). Between two straight-down frames:
- the image scale changes by s = h_i / h_j, so h0 = (alt_i - s alt_j) / (s - 1);
- ground displacement d moves the image centre by shift = f d / h, so f = shift h / d.
Image motion comes from a similarity transform (shift, rotation, scale) fitted by RANSAC to
tracked corners between the two frames, measured directly between them (not chained through
intermediate frames, which builds up error: PROGRESS 2026-10-21).
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray


def h0_from_scale(alt_i: float, alt_j: float, s: float) -> float:
    """Altimeter offset from the image scale factor s (frame i to j) and the two readings."""
    if abs(s - 1.0) < 1e-9:
        return float("nan")
    return (alt_i - s * alt_j) / (s - 1.0)


def focal_from_motion(shift_px: float, d_m: float, h_m: float) -> float:
    """Focal length (px) from the image-centre shift, ground displacement and true height."""
    return shift_px * h_m / d_m


def fit_linear_drift(t: ArrayLike, h0: ArrayLike) -> tuple[float, float]:
    """Robust straight line h0 = a + b t (Theil-Sen: median of pairwise slopes, then median
    intercept), so a few bad windows cannot steer the drift."""
    t, y = np.asarray(t, float), np.asarray(h0, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    i, j = np.triu_indices(len(t), 1)
    keep = t[j] != t[i]
    b = float(np.median((y[j] - y[i])[keep] / (t[j] - t[i])[keep])) if keep.any() else 0.0
    a = float(np.median(y - b * t))
    return a, b


def similarity_motion(
    a: NDArray, b: NDArray, mask: NDArray | None = None, min_inliers: int = 40
) -> tuple[float, float, int] | None:
    """Image-centre shift (px), scale factor and inlier count of the similarity transform from
    grey image `a` to `b`, from Shi-Tomasi corners tracked by pyramidal Lucas-Kanade with a
    forward-backward check (< 1 px). None if too few consistent matches."""
    p = cv2.goodFeaturesToTrack(a, 1500, 0.01, 6, mask=mask)
    if p is None or len(p) < min_inliers:
        return None
    lk = {"winSize": (31, 31), "maxLevel": 5}
    q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, **lk)
    r, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, **lk)
    ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((r - p)[:, 0], axis=1) < 1.0)
    if ok.sum() < min_inliers:
        return None
    M, inl = cv2.estimateAffinePartial2D(p[ok], q[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or inl.sum() < min_inliers:
        return None
    c = np.array([a.shape[1] / 2, a.shape[0] / 2])
    shift = float(np.linalg.norm(M[:, :2] @ c + M[:, 2] - c))
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    return shift, scale, int(inl.sum())
