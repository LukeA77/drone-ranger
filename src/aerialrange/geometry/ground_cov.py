"""Covariance of a ground point seen by an oblique camera, propagated from its inputs.

The ground point of pixel (u, v) for a camera at height h, tilted `tilt` below the horizon
and heading `yaw` is `ground_offset_oblique` (flat ground, the AU-AIR mount, D-511). Its error
comes from the pixel (the box's contact point), the tilt, the height and the heading. To first
order the 2 x 2 covariance of (north, east) is J diag(sigma^2) J^T, with J the Jacobian of the
ground point with respect to (u, v, tilt, h, yaw), written out here in closed form:

  x = (u - cx) / f, y = (v - cy) / f, c = cos(tilt), s = sin(tilt)
  down = s + c y, along = c - s y, k = h / down
  forward = k along, right = k x                      (camera heading north)
  d forward / d y = -h / down^2          d right / d y = -h c x / down^2
  d forward / d x = 0                    d right / d x = k
  d forward / d tilt = -h (down^2 + along^2) / down^2
  d right / d tilt   = -h x along / down^2
  d (forward, right) / d h = (forward, right) / h
then rotated by the heading; the heading itself turns the offset: d(n, e)/d yaw = (-e, n).

Along the line of sight the error grows as 1 / down^2 (1 / sin^2 of the depression at the
image centre), which is why shallow views give large errors from small pixel or tilt errors
(D-734). The model is first order: near the horizon, where `down` is small, the true error is
larger and skewed, so the covariance is a lower bound there.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def ground_jacobian(
    u: ArrayLike,
    v: ArrayLike,
    h: ArrayLike,
    f: float,
    yaw: ArrayLike,
    tilt_deg: ArrayLike,
    cx: float,
    cy: float,
) -> tuple[NDArray, NDArray]:
    """Ground offsets (n, 2) as (north, east) and their Jacobians (n, 2, 5) with respect to
    (u px, v px, tilt rad, h m, yaw rad). NaN where the pixel is at or above the horizon."""
    u, v, h, yaw, t = np.broadcast_arrays(
        np.asarray(u, float),
        np.asarray(v, float),
        np.asarray(h, float),
        np.asarray(yaw, float),
        np.radians(np.asarray(tilt_deg, float)),
    )
    x, y = (u - cx) / f, (v - cy) / f
    c, s = np.cos(t), np.sin(t)
    down = s + c * y
    along = c - s * y
    with np.errstate(divide="ignore", invalid="ignore"):
        bad = down <= 1e-9
        d2 = np.where(bad, np.nan, down * down)
        k = np.where(bad, np.nan, h / down)
        fwd, right = k * along, k * x
        # columns: u, v, tilt, h, yaw (camera frame: forward, right)
        jf = np.stack(
            [np.zeros_like(x), -h / d2 / f, -h * (d2 + along**2) / d2, fwd / h, np.zeros_like(x)],
            axis=-1,
        )
        jr = np.stack(
            [k / f, -h * c * x / d2 / f, -h * x * along / d2, right / h, np.zeros_like(x)],
            axis=-1,
        )
    cyw, syw = np.cos(yaw), np.sin(yaw)
    n = cyw * fwd - syw * right
    e = syw * fwd + cyw * right
    jn = cyw[..., None] * jf - syw[..., None] * jr
    je = syw[..., None] * jf + cyw[..., None] * jr
    jn[..., 4], je[..., 4] = -e, n
    return np.stack([n, e], axis=-1), np.stack([jn, je], axis=-2)


def ground_covariance(
    u: ArrayLike,
    v: ArrayLike,
    h: ArrayLike,
    f: float,
    yaw: ArrayLike,
    tilt_deg: ArrayLike,
    cx: float,
    cy: float,
    sigma_u_px: ArrayLike,
    sigma_v_px: ArrayLike,
    sigma_tilt_deg: ArrayLike,
    sigma_h_m: ArrayLike,
    sigma_yaw_deg: ArrayLike,
) -> tuple[NDArray, NDArray]:
    """Ground offsets (n, 2) and their covariance (n, 2, 2) in m^2 from independent input
    errors with the given standard deviations (broadcast against the pixels)."""
    off, jac = ground_jacobian(u, v, h, f, yaw, tilt_deg, cx, cy)
    sig = np.stack(
        np.broadcast_arrays(
            np.asarray(sigma_u_px, float),
            np.asarray(sigma_v_px, float),
            np.radians(np.asarray(sigma_tilt_deg, float)),
            np.asarray(sigma_h_m, float),
            np.radians(np.asarray(sigma_yaw_deg, float)),
        ),
        axis=-1,
    )
    sig = np.broadcast_to(sig, jac.shape[:-2] + (5,))
    js = jac * sig[..., None, :]
    return off, js @ np.swapaxes(js, -1, -2)
