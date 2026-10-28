"""Pinhole camera, optional Brown-Conrady distortion, and pixel-to-world rays.

Pixels are in pixels, positions in metres (NED world), rotations as 3x3 matrices.
When `Camera.dist` is None the ideal pinhole model is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerialrange.geometry.frames import R_BC_FORWARD, R_wb_from_ypr

# Fixed-point undistortion: iterations and convergence tolerance (in normalised image units).
# These are numerical-method settings, not physical parameters; tests check the round trip.
_UNDISTORT_MAX_ITER = 50
_UNDISTORT_TOL = 1e-12


def _check_rotation(R: NDArray, name: str) -> None:
    if R.shape != (3, 3) or not np.allclose(R @ R.T, np.eye(3), atol=1e-9):
        raise ValueError(f"{name} must be a 3x3 rotation matrix")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-9):
        raise ValueError(f"{name} must have determinant +1")


@dataclass(frozen=True)
class Camera:
    """Intrinsics `K` (pixels), optional distortion `dist` = (k1, k2, p1, p2, k3) in OpenCV order,
    mount rotation `R_bc` (camera to body) and lever arm `t_bc` (metres, body frame)."""

    K: NDArray[np.float64]
    R_bc: NDArray[np.float64] = field(default_factory=lambda: R_BC_FORWARD.copy())
    t_bc: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    dist: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        K = np.asarray(self.K, float)
        if K.shape != (3, 3) or K[2, 2] != 1.0 or np.any(K[[1, 2, 2], [0, 0, 1]] != 0):
            raise ValueError("K must be 3x3 upper triangular with K[2, 2] = 1")
        object.__setattr__(self, "K", K)
        R = np.asarray(self.R_bc, float)
        _check_rotation(R, "R_bc")
        object.__setattr__(self, "R_bc", R)
        object.__setattr__(self, "t_bc", np.asarray(self.t_bc, float).reshape(3))
        if self.dist is not None:
            d = np.asarray(self.dist, float)
            if d.shape != (5,):
                raise ValueError("dist must be (k1, k2, p1, p2, k3)")
            object.__setattr__(self, "dist", d)

    @classmethod
    def pinhole(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        R_bc: ArrayLike = R_BC_FORWARD,
        t_bc: ArrayLike = (0.0, 0.0, 0.0),
        dist: ArrayLike | None = None,
    ) -> Camera:
        """Build from focal lengths and principal point (pixels), zero skew."""
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return cls(
            K,
            np.asarray(R_bc, float),
            np.asarray(t_bc, float),
            None if dist is None else np.asarray(dist, float),
        )

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])


@dataclass(frozen=True)
class Pose:
    """Drone pose: body origin `p_w` (NED, metres) and body-to-world rotation `R_wb`."""

    p_w: NDArray[np.float64]
    R_wb: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "p_w", np.asarray(self.p_w, float).reshape(3))
        R = np.asarray(self.R_wb, float)
        _check_rotation(R, "R_wb")
        object.__setattr__(self, "R_wb", R)

    @classmethod
    def from_ypr(cls, p_w: ArrayLike, yaw: float, pitch: float, roll: float) -> Pose:
        """Build from position (metres) and aerospace yaw, pitch, roll (radians)."""
        return cls(np.asarray(p_w, float), R_wb_from_ypr(yaw, pitch, roll))


def _distort_normalised(cam: Camera, xy: NDArray) -> NDArray:
    k1, k2, p1, p2, k3 = cam.dist  # type: ignore[misc]
    x, y = xy[..., 0], xy[..., 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd, yd], axis=-1)


def _to_normalised(cam: Camera, uv: NDArray) -> NDArray:
    return np.stack([(uv[..., 0] - cam.cx) / cam.fx, (uv[..., 1] - cam.cy) / cam.fy], axis=-1)


def _to_pixels(cam: Camera, xy: NDArray) -> NDArray:
    return np.stack([xy[..., 0] * cam.fx + cam.cx, xy[..., 1] * cam.fy + cam.cy], axis=-1)


def distort_pixels(cam: Camera, uv_ideal: ArrayLike) -> NDArray[np.float64]:
    """Map ideal pinhole pixels to distorted pixels. Identity if `cam.dist` is None."""
    uv = np.asarray(uv_ideal, float)
    if cam.dist is None:
        return uv
    return _to_pixels(cam, _distort_normalised(cam, _to_normalised(cam, uv)))


def undistort_pixels(cam: Camera, uv: ArrayLike) -> NDArray[np.float64]:
    """Map observed (distorted) pixels to ideal pinhole pixels by fixed-point iteration.

    Identity if `cam.dist` is None (pinhole model assumed).
    """
    uv = np.asarray(uv, float)
    if cam.dist is None:
        return uv
    target = _to_normalised(cam, uv)
    xy = target.copy()
    for _ in range(_UNDISTORT_MAX_ITER):
        step = target - _distort_normalised(cam, xy)
        xy = xy + step
        if np.max(np.abs(step)) < _UNDISTORT_TOL:
            break
    return _to_pixels(cam, xy)


def pixel_to_world_ray(
    cam: Camera, pose: Pose, uv: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Camera centre `C_w` (metres, NED) and world ray direction `d_w` for pixel(s) `uv`.

    `d_w = R_wb R_bc K^-1 [u, v, 1]^T` (not normalised: its camera-frame z component is 1).
    `uv` has shape (2,) or (N, 2); `d_w` has shape (3,) or (N, 3).
    """
    uv_i = undistort_pixels(cam, uv)
    ones = np.ones(uv_i.shape[:-1] + (1,))
    d_c = np.concatenate([_to_normalised(cam, uv_i), ones], axis=-1)
    R_wc = pose.R_wb @ cam.R_bc
    d_w = d_c @ R_wc.T
    C_w = pose.p_w + pose.R_wb @ cam.t_bc
    return C_w, d_w


def project_world_point(
    cam: Camera, pose: Pose, X_w: ArrayLike
) -> tuple[NDArray[np.float64], bool]:
    """Pixel of world point `X_w` (metres, NED), including distortion if set.

    Returns (uv, in_front) where `in_front` is False if the point is behind the camera.
    """
    C_w = pose.p_w + pose.R_wb @ cam.t_bc
    X_c = (pose.R_wb @ cam.R_bc).T @ (np.asarray(X_w, float) - C_w)
    in_front = bool(X_c[2] > 0)
    xy = X_c[:2] / X_c[2]
    uv = _to_pixels(cam, xy)
    return distort_pixels(cam, uv), in_front
