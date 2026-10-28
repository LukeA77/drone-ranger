"""Stationary-target discovery from image motion only.

Camera motion between consecutive frames is a homography fitted to background features
(objects masked out). A tracked point's residual step is where it actually is in frame k+1
minus where the background motion would have carried it from frame k. A parked object's
residual steps are detector and homography noise, which cancel when added as vectors; a
moving or creeping object's steps add up. A stationary segment is a run of consecutive frames
whose accumulated residual stays within `thr_px` of its start for at least `min_s` seconds.
No altitude or GPS is used, so the repeatability test built on it is not circular.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def residual_steps(track: NDArray, H: Sequence[NDArray], max_gap: int = 1) -> NDArray[np.float64]:
    """Residual step vectors (n-1, 2) of a track with rows (frame, x, y), frames 1-based and
    increasing. `H[k-1]` maps frame k to frame k+1. A step may span up to `max_gap` frames
    (a short detection miss), carried through the chained camera motion; longer gaps are NaN."""
    track = np.asarray(track, float)
    out = np.full((len(track) - 1, 2), np.nan)
    for i in range(len(track) - 1):
        f0, f1 = int(track[i, 0]), int(track[i + 1, 0])
        if not 1 <= f1 - f0 <= max_gap:
            continue
        M = np.eye(3)
        for k in range(f0, f1):
            M = H[k - 1] @ M
        p = M @ np.array([track[i, 1], track[i, 2], 1.0])
        out[i] = track[i + 1, 1:3] - p[:2] / p[2]
    return out


def stationary_segments(
    steps: NDArray, t: NDArray, thr_px: float, min_s: float
) -> list[tuple[int, int]]:
    """Inclusive (first, last) frame indices of stationary segments. `steps[i]` is the residual
    vector from frame i to i+1 and `t` the frame times in seconds. A segment grows while the
    accumulated residual stays within `thr_px` of its start; a step that breaks this, or a NaN
    step (gap), ends it, and the next segment starts at the frame after that step."""
    steps, t = np.asarray(steps, float), np.asarray(t, float)
    segs, start, cum = [], 0, np.zeros(2)

    def close(last: int) -> None:
        if t[last] - t[start] >= min_s:
            segs.append((start, last))

    i = 0
    while i < len(steps):
        s = steps[i]
        nxt = cum + s
        if np.isnan(s).any() or np.linalg.norm(nxt) > thr_px:
            close(i)
            start, cum = i + 1, np.zeros(2)
        else:
            cum = nxt
        i += 1
    close(len(t) - 1)
    return segs
