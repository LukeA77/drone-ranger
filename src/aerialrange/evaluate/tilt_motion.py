"""Camera tilt from the ground-plane homography between two frames (D-505, D-513).

For a camera moving over flat ground, image motion of the ground between two frames is the
plane homography H = K (R + t n^T / d) K^-1, where n is the ground normal in the first camera's
frame. With camera axes x right, y down, z along the optical axis, the ground normal of a
camera tilted theta below the horizon is n = (0, cos theta, sin theta), so
theta = atan2(n_z, n_y): 90 degrees straight down, 0 at the horizon.

Decomposing H gives up to four (R, t, n) solutions: two physical ones and their sign flips.
Of the two physical ones, the true one has the rotation the camera actually made between the
frames. The AU-AIR camera is digitally stabilised (D-511), so that rotation is the drone's yaw
change, which is recorded; the candidate whose rotation angle is closest to it is kept.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def camera_rotation(tilt_deg: float) -> NDArray[np.float64]:
    """World (north, east, down) to camera rotation for a camera heading north, tilted
    `tilt_deg` below the horizon, with image x to the east (rows are camera axes in world)."""
    t = np.radians(tilt_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[0.0, 1.0, 0.0], [-s, 0.0, c], [c, 0.0, s]])


def plane_homography(
    K: NDArray, R1: NDArray, C1: NDArray, R2: NDArray, C2: NDArray
) -> NDArray[np.float64]:
    """Homography of the ground plane (down = 0) from view 1 to view 2 (camera centres C in
    north-east-down, height = -C_down > 0)."""
    nw = np.array([0.0, 0.0, 1.0])
    n1, d1 = R1 @ nw, -nw @ C1
    H = K @ (R2 @ R1.T + np.outer(R2 @ (C1 - C2), n1) / d1) @ np.linalg.inv(K)
    return H / H[2, 2]


def tilt_candidates(H: NDArray, K: NDArray) -> list[tuple[float, float, float]]:
    """(tilt_deg, rotation_deg, |t|/d) for each finite decomposition, normal oriented towards
    the camera's view (n_z > 0)."""
    _, Rs, ts, ns = cv2.decomposeHomographyMat(np.asarray(H, float), np.asarray(K, float))
    out = []
    for R, t, n in zip(Rs, ts, ns, strict=True):
        n, t = n[:, 0], t[:, 0]
        if not (np.all(np.isfinite(n)) and np.all(np.isfinite(R))):
            continue
        if n[2] < 0:
            n, t = -n, -t
        rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))
        out.append((float(np.degrees(np.arctan2(n[2], n[1]))), rot, float(np.linalg.norm(t))))
    return out


def tilt_from_homography(
    H: NDArray, K: NDArray, expected_rot_deg: float = 0.0, min_motion: float = 1e-3
) -> float | None:
    """Tilt below the horizon (degrees) of the first camera, or None without enough motion
    (relative translation |t|/d below `min_motion`) or without a candidate below the horizon."""
    cands = [c for c in tilt_candidates(H, K) if 0.0 <= c[0] <= 95.0 and c[2] >= min_motion]
    if not cands:
        return None
    return min(cands, key=lambda c: abs(c[1] - expected_rot_deg))[0]
