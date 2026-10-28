"""Error from turning a detected box into a ground point (D-622).

The Mid-Air evaluation ranges ground pixels, so it measures the ranging geometry alone. The
pipeline instead ranges a box's bottom-centre, which for an object seen at an angle is the near
edge of its footprint, not its centre, and the detector's box is itself noisy. This module
simulates that step with known truth: a 3D box (an object) on flat ground is projected into a
pinhole camera, its axis-aligned image box is taken, the box is perturbed with the measured
detector noise, and the contact pixel is ranged on the same flat ground with the exact pose.
The error is against the range to the centre of the object's footprint.

World frame NED (metres); camera at height h above the ground (Down = 0), heading North,
tilted `tilt_deg` below the horizon, no roll.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PinholeView:
    f_px: float
    width: int
    height: int
    h_m: float
    tilt_deg: float

    @property
    def R_cw(self) -> NDArray:
        """World-to-camera rotation (camera x right, y down, z forward)."""
        t = np.radians(self.tilt_deg)
        fwd = np.array([np.cos(t), 0.0, np.sin(t)])
        right = np.array([0.0, 1.0, 0.0])
        down = np.cross(fwd, right)
        return np.stack([right, down, fwd])

    @property
    def C(self) -> NDArray:
        return np.array([0.0, 0.0, -self.h_m])

    def project(self, X: NDArray) -> tuple[NDArray, NDArray]:
        """Pixels (N, 2) of world points (N, 3) and an in-front mask."""
        Xc = (X - self.C) @ self.R_cw.T
        z = Xc[:, 2]
        uv = np.stack(
            [
                self.width / 2 + self.f_px * Xc[:, 0] / z,
                self.height / 2 + self.f_px * Xc[:, 1] / z,
            ],
            axis=1,
        )
        return uv, z > 0

    def ground_point(self, uv: NDArray) -> NDArray:
        """Intersection of the pixel rays (N, 2) with the ground (Down = 0); NaN above the
        horizon."""
        d_c = np.stack(
            [
                (uv[:, 0] - self.width / 2) / self.f_px,
                (uv[:, 1] - self.height / 2) / self.f_px,
                np.ones(len(uv)),
            ],
            axis=1,
        )
        d_w = d_c @ self.R_cw
        lam = np.where(d_w[:, 2] > 0, self.h_m / d_w[:, 2], np.nan)
        return self.C + lam[:, None] * d_w


def box_corners(centre_ne: NDArray, yaw: NDArray, dims: NDArray) -> NDArray:
    """Eight corners (N, 8, 3) of boxes standing on the ground: footprint centre (N, 2),
    yaw (N,) from North, dims (N, 3) = length, width, height (metres)."""
    sx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * 0.5
    sy = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * 0.5
    sz = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    L, Wd, Hh = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    lx, ly = sx * L, sy * Wd
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    n = centre_ne[:, 0:1] + c * lx - s * ly
    e = centre_ne[:, 1:2] + s * lx + c * ly
    d = -sz * Hh
    return np.stack([n, e, d], axis=2)


def simulate(
    view: PinholeView,
    n: int,
    dims_lo: tuple[float, float, float],
    dims_hi: tuple[float, float, float],
    kappa_u: float,
    kappa_v: float,
    rng: np.random.Generator,
    mode: str = "bottom_centre",
) -> dict[str, NDArray]:
    """Relative range error of the contact pixel for `n` random objects in view.

    Objects: dims uniform in [dims_lo, dims_hi], yaw uniform; footprint centres at the ground
    point of a random pixel in the central 80 % of the image width and the central 60 % of its
    height. Box noise: Gaussian on the contact pixel with sigma `kappa_u` x box width and
    `kappa_v` x box height (the detector's measured noise, relative to box size).
    `mode`: "bottom_centre" (the pipeline's rule) or "centre" (box centre, for comparison).
    Objects not wholly in front of the camera and inside the image are dropped.
    """
    uv0 = np.stack(
        [
            rng.uniform(0.1, 0.9, n) * view.width,
            rng.uniform(0.2, 0.8, n) * view.height,
        ],
        axis=1,
    )
    centre = view.ground_point(uv0)
    ok = np.isfinite(centre[:, 0])
    dims = rng.uniform(dims_lo, dims_hi, (n, 3))
    yaw = rng.uniform(0, 2 * np.pi, n)
    corners = box_corners(centre[:, :2], yaw, dims)
    uv, front = view.project(corners.reshape(-1, 3))
    uv = uv.reshape(n, 8, 2)
    front = front.reshape(n, 8).all(axis=1)
    lo, hi = uv.min(axis=1), uv.max(axis=1)
    inside = (lo[:, 0] >= 0) & (lo[:, 1] >= 0) & (hi[:, 0] < view.width) & (hi[:, 1] < view.height)
    ok &= front & inside
    w, h = hi[:, 0] - lo[:, 0], hi[:, 1] - lo[:, 1]
    if mode == "bottom_centre":
        cp = np.stack([(lo[:, 0] + hi[:, 0]) / 2, hi[:, 1]], axis=1)
    elif mode == "centre":
        cp = (lo + hi) / 2
    else:
        raise ValueError(f"unknown mode {mode!r}")
    noisy = cp + np.stack([rng.normal(0, 1, n) * kappa_u * w, rng.normal(0, 1, n) * kappa_v * h], 1)
    est = view.ground_point(noisy)
    true_r = np.linalg.norm(centre - view.C, axis=1)
    est_r = np.linalg.norm(est - view.C, axis=1)
    ok &= np.isfinite(est_r)
    dep = np.degrees(np.arcsin(view.h_m / true_r))
    return {"rel": ((est_r - true_r) / true_r)[ok], "depression_deg": dep[ok], "box_h_px": h[ok]}
