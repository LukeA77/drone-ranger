"""Speed and heading truth from SynDrone box positions.

SynDrone saves no velocities, so truth velocity is the derivative of each actor's footprint
centre, sampled at 25 Hz:

- Tracks are cut where consecutive frames are missing or where the implied speed exceeds a
  physical limit `vmax_ms` (respawns, teleports); no derivative window crosses a cut.
- Velocity is a centred Savitzky-Golay derivative (local quadratic fit over `window`
  frames). It is exact for constant acceleration, so its bias comes only from jerk.
- Position jitter is measured, not assumed (`jitter_sigma`), and the velocity's own noise
  follows from it (`velocity_sigma`); both are reported with the truth.

Heading is NED, degrees clockwise from North (Unreal +x), so +y (East) is 90.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _sg_weights(window: int) -> NDArray[np.float64]:
    """Centred first-derivative weights of a local quadratic fit (per sample spacing).

    For a quadratic the derivative at the centre equals the linear least-squares slope, so
    w_i = i / sum(i^2) for i = -m..m.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be odd and at least 3")
    m = window // 2
    i = np.arange(-m, m + 1, dtype=float)
    return i / np.sum(i * i)


def savgol_velocity(pos: NDArray, window: int, dt: float) -> NDArray[np.float64]:
    """(N, D) velocities; NaN where the window does not fit inside the track."""
    pos = np.asarray(pos, float)
    w = _sg_weights(window)
    m = window // 2
    out = np.full(pos.shape, np.nan)
    if len(pos) < window:
        return out
    for d in range(pos.shape[1]):
        # correlate: sum_i w_i * x[k + i]
        out[m : len(pos) - m, d] = np.correlate(pos[:, d], w, mode="valid") / dt
    return out


def velocity_sigma(sigma_pos: float, window: int, dt: float) -> float:
    """Standard deviation of one velocity component from white position noise."""
    w = _sg_weights(window)
    return float(sigma_pos * np.sqrt(np.sum(w * w)) / dt)


def jitter_sigma(pos: NDArray) -> float:
    """Position noise from third differences, robust to smooth motion.

    For white noise of variance s^2, the third difference has variance 20 s^2; smooth
    motion contributes only its third derivative times dt^3. Uses the median absolute
    deviation (scale 1.4826) per component, then averages the components.
    """
    pos = np.asarray(pos, float)
    if len(pos) < 4:
        return float("nan")
    d3 = np.diff(pos, n=3, axis=0)
    mad = np.median(np.abs(d3 - np.median(d3, axis=0)), axis=0) * 1.4826
    return float(np.mean(mad) / np.sqrt(20.0))


def split_at_jumps(frames: NDArray, pos: NDArray, dt: float, vmax_ms: float) -> list[slice]:
    """Index slices of contiguous, physically continuous pieces of one track."""
    frames = np.asarray(frames)
    pos = np.asarray(pos, float)
    if len(frames) == 0:
        return []
    gap = np.diff(frames) != 1
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    cut = np.nonzero(gap | (step > vmax_ms))[0] + 1
    edges = [0, *cut.tolist(), len(frames)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]


def heading_deg_ned(v: NDArray) -> NDArray[np.float64]:
    v = np.asarray(v, float)
    return np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0


def speed_truth(
    frames: NDArray, pos: NDArray, dt: float, window: int, vmax_ms: float, sigma_pos: float
) -> dict[str, NDArray]:
    """Per-sample horizontal speed (m/s), heading (deg), speed sigma and segment index."""
    frames = np.asarray(frames)
    pos = np.asarray(pos, float)
    n = len(frames)
    vel = np.full((n, 3), np.nan)
    seg = np.full(n, -1)
    for k, s in enumerate(split_at_jumps(frames, pos, dt, vmax_ms)):
        vel[s] = savgol_velocity(pos[s], window, dt)
        seg[s] = k
    speed = np.hypot(vel[:, 0], vel[:, 1])
    head = np.full(n, np.nan)
    ok = ~np.isnan(speed)
    head[ok] = heading_deg_ned(vel[ok])
    sig = np.where(ok, velocity_sigma(sigma_pos, window, dt), np.nan)
    return {"speed": speed, "heading_deg": head, "speed_sigma": sig, "segment": seg, "vel": vel}
