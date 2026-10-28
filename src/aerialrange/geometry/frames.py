"""Reference frames and rotations.

Conventions (each one is pinned by a test in `tests/test_frames.py`):

- World W: North, East, Down (NED). Body B: forward, right, down (FRD).
  Camera C: OpenCV, x right, y down, z forward along the optical axis.
- `R_ab` rotates a vector from frame b into frame a: `v_a = R_ab @ v_b`.
- Attitude: aerospace yaw-pitch-roll, `R_wb = Rz(yaw) Ry(pitch) Rx(roll)`. Positive pitch is
  nose up, positive roll is right wing down, yaw is heading clockwise from North.
- Camera mount: camera depression `tilt` below the body's horizontal plane, about body right.
- Quaternions: Hamilton convention, active rotation, unit norm.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Default level, forward-looking mount: camera z -> body forward, x -> right, y -> down.
R_BC_FORWARD: NDArray[np.float64] = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# WGS 84 ellipsoid (NIMA TR8350.2, 3rd edition, 2000, Table 3.1; also EPSG:7030).
WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def rot_x(a: float) -> NDArray[np.float64]:
    """Active right-handed rotation about x by angle `a` (radians)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> NDArray[np.float64]:
    """Active right-handed rotation about y by angle `a` (radians)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> NDArray[np.float64]:
    """Active right-handed rotation about z by angle `a` (radians)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def R_wb_from_ypr(yaw: float, pitch: float, roll: float) -> NDArray[np.float64]:
    """Body-to-world rotation from aerospace yaw, pitch, roll (radians)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def ypr_from_R_wb(R_wb: ArrayLike) -> tuple[float, float, float]:
    """Inverse of `R_wb_from_ypr`. Returns (yaw, pitch, roll) in radians.

    Pitch is in [-pi/2, pi/2]; the decomposition is singular at pitch = +/- pi/2.
    """
    R = np.asarray(R_wb, float)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    return yaw, pitch, roll


def heading_rad(R_wb: ArrayLike) -> float:
    """Heading of the body forward axis projected on the horizontal plane (radians, from North,
    clockwise)."""
    f_w = np.asarray(R_wb, float)[:, 0]
    return float(np.arctan2(f_w[1], f_w[0]))


def R_bc_from_tilt(tilt: float, base: ArrayLike = R_BC_FORWARD) -> NDArray[np.float64]:
    """Camera-to-body rotation for a camera depressed by `tilt` (radians) below body forward.

    `tilt = 0` looks along body forward, `tilt = pi/2` looks straight down (body down).
    The tilt is a rotation about the body right axis by `-tilt` (positive pitch is nose up).
    """
    return rot_y(-tilt) @ np.asarray(base, float)


def _check_order(order: str) -> None:
    if order not in ("wxyz", "xyzw"):
        raise ValueError(f"quaternion order must be 'wxyz' or 'xyzw', got {order!r}")


def quat_to_R(q: ArrayLike, order: str = "wxyz") -> NDArray[np.float64]:
    """Rotation matrix of a Hamilton unit quaternion (normalised here). Active rotation."""
    _check_order(order)
    q = np.asarray(q, float)
    if order == "xyzw":
        q = np.roll(q, 1)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("zero quaternion")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def R_to_quat(R: ArrayLike, order: str = "wxyz") -> NDArray[np.float64]:
    """Hamilton unit quaternion of a rotation matrix (Shepperd's method), with w >= 0."""
    _check_order(order)
    m = np.asarray(R, float)
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * np.sqrt(1.0 + tr)
        q = np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = np.empty(4)
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    if q[0] < 0:
        q = -q
    q /= np.linalg.norm(q)
    return np.roll(q, -1) if order == "xyzw" else q


def _geodetic_to_ecef(lat: NDArray, lon: NDArray, alt: NDArray) -> NDArray[np.float64]:
    n = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def geodetic_to_ned(
    lat_deg: ArrayLike,
    lon_deg: ArrayLike,
    alt_m: ArrayLike,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
) -> NDArray[np.float64]:
    """Local NED position (metres) of WGS 84 points relative to an origin fix.

    Exact ellipsoidal conversion through Earth-centred, Earth-fixed coordinates, so no
    flat-Earth approximation is needed. Inputs in degrees and metres (ellipsoidal height; a
    constant geoid offset cancels in the difference). Output shape (..., 3).
    """
    lat, lon = np.radians(np.asarray(lat_deg, float)), np.radians(np.asarray(lon_deg, float))
    lat0, lon0 = np.radians(lat0_deg), np.radians(lon0_deg)
    d = _geodetic_to_ecef(lat, lon, np.asarray(alt_m, float)) - _geodetic_to_ecef(
        np.asarray(lat0), np.asarray(lon0), np.asarray(alt0_m, float)
    )
    sl, cl, so, co = np.sin(lat0), np.cos(lat0), np.sin(lon0), np.cos(lon0)
    R_ned_ecef = np.array([[-sl * co, -sl * so, cl], [-so, co, 0.0], [-cl * co, -cl * so, -sl]])
    return d @ R_ned_ecef.T
