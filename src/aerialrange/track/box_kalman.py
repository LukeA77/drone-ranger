"""Constant-velocity Kalman filter on image boxes, used by the tracker (`track/tracker.py`).

State per track: [cx, cy, w, h, vcx, vcy, vw, vh] (pixels, and pixels per frame);
measurement [cx, cy, w, h]. Noise scales with box size, as in the SORT family of trackers
(Wojke, Bewley and Paulus 2017, DeepSORT; Aharon, Orfaig and Bobrovsky 2022, BoT-SORT, which
uses width and height rather than aspect ratio): the standard deviation of position is
`std_pos` times the box size and of velocity `std_vel` times the box size, so a large box may
move proportionally further between frames. All operations are vectorised over tracks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F = np.eye(8)
F[:4, 4:] = np.eye(4)
H = np.eye(4, 8)
MIN_SIZE_PX = 1.0  # widths and heights are kept at least this, so IoU stays defined


class BoxKalman:
    """Box filter with size-proportional noise. Defaults are the published DeepSORT values."""

    def __init__(self, std_pos: float = 1 / 20, std_vel: float = 1 / 160) -> None:
        if std_pos <= 0 or std_vel <= 0:
            raise ValueError("noise weights must be positive")
        self.std_pos, self.std_vel = std_pos, std_vel

    @staticmethod
    def _size(x: NDArray) -> NDArray:
        """(n, 4) [w, h, w, h] from states or measurements."""
        return x[:, [2, 3, 2, 3]]

    def initiate(self, z: NDArray) -> tuple[NDArray, NDArray]:
        """States from first measurements `z` (n, 4) cxcywh: zero velocity, wide velocity
        uncertainty."""
        z = np.asarray(z, float).reshape(-1, 4)
        mean = np.hstack([z, np.zeros_like(z)])
        s = self._size(z)
        std = np.hstack([2 * self.std_pos * s, 10 * self.std_vel * s])
        return mean, _diag(std**2)

    def predict(
        self, mean: NDArray, cov: NDArray, freeze_size: NDArray | None = None
    ) -> tuple[NDArray, NDArray]:
        """One frame ahead. `freeze_size` (n,) bool zeroes the size velocity first (for lost
        tracks, so an unseen box does not keep growing or shrinking)."""
        mean = mean.copy()
        if freeze_size is not None and freeze_size.any():
            mean[freeze_size, 6:8] = 0.0
        s = self._size(mean)
        q = _diag(np.hstack([self.std_pos * s, self.std_vel * s]) ** 2)
        mean = mean @ F.T
        mean[:, 2:4] = np.maximum(mean[:, 2:4], MIN_SIZE_PX)
        return mean, F @ cov @ F.T + q

    def update(self, mean: NDArray, cov: NDArray, z: NDArray) -> tuple[NDArray, NDArray]:
        """Correct states with measurements `z` (n, 4) cxcywh, one per state."""
        z = np.asarray(z, float).reshape(-1, 4)
        r = _diag((self.std_pos * self._size(mean)) ** 2)
        s = H @ cov @ H.T + r
        pht = cov @ H.T  # (n, 8, 4)
        k = np.linalg.solve(s, pht.transpose(0, 2, 1)).transpose(0, 2, 1)
        innov = z - mean @ H.T
        mean = mean + np.einsum("nij,nj->ni", k, innov)
        mean[:, 2:4] = np.maximum(mean[:, 2:4], MIN_SIZE_PX)
        cov = cov - k @ H @ cov
        return mean, 0.5 * (cov + cov.transpose(0, 2, 1))

    @staticmethod
    def warp(mean: NDArray, cov: NDArray, affine: NDArray) -> tuple[NDArray, NDArray]:
        """Move states into the next frame's image coordinates under the camera's motion,
        a 2x3 similarity `affine` (rotation and scale R, translation t): centres map by
        R x + t, velocities by R, and sizes and their velocities scale by |R|^(1/2)."""
        a = np.asarray(affine, float).reshape(2, 3)
        r, t = a[:, :2], a[:, 2]
        sc = float(np.sqrt(abs(np.linalg.det(r))))
        blk = np.zeros((8, 8))
        blk[0:2, 0:2] = r
        blk[2:4, 2:4] = sc * np.eye(2)
        blk[4:6, 4:6] = r
        blk[6:8, 6:8] = sc * np.eye(2)
        mean = mean @ blk.T
        mean[:, 0:2] += t
        return mean, blk @ cov @ blk.T


def _diag(v: NDArray) -> NDArray:
    """(n, k) -> (n, k, k) diagonal matrices."""
    n, k = v.shape
    out = np.zeros((n, k, k))
    idx = np.arange(k)
    out[:, idx, idx] = v
    return out
