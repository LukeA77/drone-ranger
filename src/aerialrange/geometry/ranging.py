"""Ray and ground-plane intersection, and the contact-point rules.

Units: metres for positions and distances, degrees for reported angles. World frame is NED.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerialrange.geometry.camera import Camera, Pose, pixel_to_world_ray
from aerialrange.geometry.frames import heading_rad

REASON_OK = "ok"
REASON_ABOVE_HORIZON = "ray_at_or_above_horizon"
REASON_NOT_ABOVE_PLANE = "camera_not_above_plane"
REASON_PARALLEL = "ray_parallel_to_plane"
REASON_BEHIND = "plane_behind_camera"

MODES = ("bottom_centre", "centre_with_height")

_NAN2 = np.full((2, 2), np.nan)


@dataclass(frozen=True)
class MonteCarloSummary:
    """Monte Carlo cross-check of one estimate."""

    n: int
    n_valid: int
    range_percentiles: dict[float, float]
    range_sigma_m: float


@dataclass(frozen=True)
class RangeEstimate:
    """One ground-plane estimate.

    point_w: ground point (metres, NED). range_m: slant range from the camera centre.
    ground_dist_m: horizontal distance from the point below the camera.
    azimuth_deg: world bearing, clockwise from North, in [0, 360).
    rel_azimuth_deg: bearing relative to the drone heading, in (-180, 180].
    depression_deg: angle of the ray below the horizontal.
    cov_ne: 2x2 North/East covariance (m^2). Sigma fields are NaN until propagated.
    """

    point_w: NDArray[np.float64]
    range_m: float
    ground_dist_m: float
    azimuth_deg: float
    rel_azimuth_deg: float
    depression_deg: float
    valid: bool
    reason: str
    cov_ne: NDArray[np.float64] = field(default_factory=lambda: _NAN2.copy())
    range_sigma_m: float = float("nan")
    ground_dist_sigma_m: float = float("nan")
    azimuth_sigma_deg: float = float("nan")
    flags: tuple[str, ...] = ()
    mc: MonteCarloSummary | None = None


def _invalid(reason: str, depression_deg: float = float("nan")) -> RangeEstimate:
    nan = float("nan")
    return RangeEstimate(np.full(3, np.nan), nan, nan, nan, nan, depression_deg, False, reason)


def _wrap180(a: float) -> float:
    """Wrap degrees to (-180, 180]."""
    w = (a + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


def _depression_deg(d: NDArray) -> float:
    return float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1]))))


def _finish(C: NDArray, d: NDArray, lam: float, heading: float) -> RangeEstimate:
    P = C + lam * d
    dn, de = P[0] - C[0], P[1] - C[1]
    az = float(np.degrees(np.arctan2(de, dn))) % 360.0
    return RangeEstimate(
        point_w=P,
        range_m=float(lam * np.linalg.norm(d)),
        ground_dist_m=float(np.hypot(dn, de)),
        azimuth_deg=az,
        rel_azimuth_deg=_wrap180(az - np.degrees(heading)),
        depression_deg=_depression_deg(d),
        valid=True,
        reason=REASON_OK,
    )


def intersect_horizontal_plane(
    origin: ArrayLike, direction: ArrayLike, z_plane: float, heading_rad: float = 0.0
) -> RangeEstimate:
    """Intersect the ray `origin + lam * direction` (NED, metres) with the plane Down = z_plane.

    Invalid if the ray is at or above the horizon (Down component <= 0) or the origin is not
    above the plane. `heading_rad` (from North, clockwise) sets `rel_azimuth_deg`.
    """
    C, d = np.asarray(origin, float), np.asarray(direction, float)
    h = z_plane - C[2]
    if h <= 0:
        return _invalid(REASON_NOT_ABOVE_PLANE, _depression_deg(d))
    if d[2] <= 0:
        return _invalid(REASON_ABOVE_HORIZON, _depression_deg(d))
    return _finish(C, d, h / d[2], heading_rad)


def intersect_plane(
    origin: ArrayLike,
    direction: ArrayLike,
    normal: ArrayLike,
    offset: float,
    heading_rad: float = 0.0,
) -> RangeEstimate:
    """Intersect a ray with the general plane `normal . X = offset` (NED, metres).

    For slope experiments only; the pipeline uses the horizontal plane.
    """
    C, d, n = (np.asarray(a, float) for a in (origin, direction, normal))
    denom = float(n @ d)
    if abs(denom) < np.finfo(float).eps * np.linalg.norm(n) * np.linalg.norm(d):
        return _invalid(REASON_PARALLEL, _depression_deg(d))
    lam = (offset - float(n @ C)) / denom
    if lam <= 0:
        return _invalid(REASON_BEHIND, _depression_deg(d))
    return _finish(C, d, lam, heading_rad)


def contact_pixel(box_ltwh: ArrayLike, mode: str) -> tuple[float, float]:
    """Pixel representing the object for a box (left, top, width, height) in pixels.

    `bottom_centre`: bottom-centre of the box (oblique views).
    `centre_with_height`: box centre, used with a plane raised by the object height.
    """
    left, top, w, h = (float(x) for x in box_ltwh)
    if mode == "bottom_centre":
        return left + w / 2, top + h
    if mode == "centre_with_height":
        return left + w / 2, top + h / 2
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def select_contact_mode(depression_deg: float, nadir_threshold_deg: float) -> str:
    """Use `centre_with_height` when the view is steeper than `nadir_threshold_deg` (degrees)."""
    return "centre_with_height" if depression_deg > nadir_threshold_deg else "bottom_centre"


def estimate_range(
    cam: Camera,
    pose: Pose,
    uv: ArrayLike,
    h_above_plane: float,
    mode: str = "bottom_centre",
    obj_height_m: float | None = None,
) -> RangeEstimate:
    """Range and bearing of the ground point seen at pixel `uv`.

    `h_above_plane`: height of the camera centre above the horizontal ground plane (metres).
    In `centre_with_height` mode the ray is intersected with a plane raised by
    `obj_height_m`, and the returned point is the object's ground contact directly below the
    intersection; `range_m` is measured to that ground contact.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    C_w, d_w = pixel_to_world_ray(cam, pose, uv)
    z_ground = C_w[2] + h_above_plane
    heading = heading_rad(pose.R_wb)
    if mode == "bottom_centre":
        return intersect_horizontal_plane(C_w, d_w, z_ground, heading)
    if obj_height_m is None:
        raise ValueError("centre_with_height mode needs obj_height_m")
    top = intersect_horizontal_plane(C_w, d_w, z_ground - obj_height_m, heading)
    if not top.valid:
        return top
    P = top.point_w + np.array([0.0, 0.0, obj_height_m])
    return RangeEstimate(
        point_w=P,
        range_m=float(np.linalg.norm(P - C_w)),
        ground_dist_m=top.ground_dist_m,
        azimuth_deg=top.azimuth_deg,
        rel_azimuth_deg=top.rel_azimuth_deg,
        depression_deg=top.depression_deg,
        valid=True,
        reason=REASON_OK,
    )


def estimate_range_batch(
    cam: Camera, pose: Pose, uv: ArrayLike, h_above_plane: ArrayLike
) -> dict[str, NDArray]:
    """Vectorised `estimate_range` for `bottom_centre` mode on a horizontal plane.

    `uv` is (N, 2) pixels; `h_above_plane` is a scalar or (N,) array (metres). Returns arrays:
    point_w (N, 3), range_m, ground_dist_m, azimuth_deg, depression_deg, valid (bool), and
    reason (str). Invalid entries hold NaN. Matches `estimate_range` element by element
    (tested).
    """
    C, d = pixel_to_world_ray(cam, pose, np.atleast_2d(np.asarray(uv, float)))
    n = len(d)
    h = np.broadcast_to(np.asarray(h_above_plane, float), (n,))
    horiz = np.hypot(d[:, 0], d[:, 1])
    depression = np.degrees(np.arctan2(d[:, 2], horiz))
    above = h > 0
    below_horizon = d[:, 2] > 0
    valid = above & below_horizon
    reason = np.where(
        ~above, REASON_NOT_ABOVE_PLANE, np.where(~below_horizon, REASON_ABOVE_HORIZON, REASON_OK)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(valid, h / d[:, 2], np.nan)
    P = C + lam[:, None] * d
    dn, de = P[:, 0] - C[0], P[:, 1] - C[1]
    return {
        "point_w": P,
        "range_m": lam * np.linalg.norm(d, axis=1),
        "ground_dist_m": np.hypot(dn, de),
        "azimuth_deg": np.degrees(np.arctan2(de, dn)) % 360.0,
        "depression_deg": depression,
        "valid": valid,
        "reason": reason,
    }
