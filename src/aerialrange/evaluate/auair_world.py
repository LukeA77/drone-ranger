"""Ground position of an image point in a straight-down AU-AIR frame.

Image axes: x right, y down, principal point assumed at the image centre. At true height h and
focal length f, a pixel's ground offset in image axes is ((u - cx) h / f, (v - cy) h / f).
How the image axes sit on the drone is unknown, so it is a hypothesis: the base mapping takes
image up as the drone's forward and image right as its right; `rot` (0 to 3) turns that by
multiples of 90 degrees and `mirror` flips left and right (8 hypotheses, chosen on the
calibration clips by the smallest repeatability spread, D-511). The body offset is rotated to
north and east by the yaw (`angle_psi`, radians, assumed clockwise from north as in NED).
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

HYPOTHESES = tuple(itertools.product(range(4), (False, True)))
CX, CY = 960.0, 540.0


def ground_offset_nadir(
    u: float | NDArray,
    v: float | NDArray,
    h: float | NDArray,
    f: float,
    yaw: float | NDArray,
    rot: int,
    mirror: bool,
    cx: float = CX,
    cy: float = CY,
) -> tuple[NDArray, NDArray]:
    """North and east offset (m) of the ground point seen at pixel (u, v) from the point below
    the camera, for a straight-down view."""
    x = (np.asarray(u, float) - cx) * np.asarray(h, float) / f  # image right
    y = (np.asarray(v, float) - cy) * np.asarray(h, float) / f  # image down
    fwd, right = -y, (-x if mirror else x)
    for _ in range(rot % 4):  # turn the image frame 90 degrees clockwise on the drone
        fwd, right = -right, fwd
    c, s = np.cos(yaw), np.sin(yaw)
    return c * fwd - s * right, s * fwd + c * right


def spread_rms(points: NDArray) -> float:
    """Root-mean-square distance of 2-D points about their mean (repeatability)."""
    p = np.asarray(points, float)
    return float(np.sqrt(np.mean(np.sum((p - p.mean(axis=0)) ** 2, axis=1))))


def ground_offset_oblique(
    u: float | NDArray,
    v: float | NDArray,
    h: float | NDArray,
    f: float,
    yaw: float | NDArray,
    tilt_deg: float | NDArray,
    cx: float = CX,
    cy: float = CY,
) -> tuple[NDArray, NDArray]:
    """North and east offset (m) of the ground point seen at pixel (u, v), for a camera at
    height h tilted `tilt_deg` below the horizon (90 = straight down) and heading `yaw`, with
    the mount found on the calibration clips (image up = forward, no mirror; D-511). The pixel's
    ray is intersected with flat ground; pixels at or above the horizon give NaN."""
    x = (np.asarray(u, float) - cx) / f  # ray in camera axes: x right, y down, z forward
    y = (np.asarray(v, float) - cy) / f
    t = np.radians(np.asarray(tilt_deg, float))
    c, s = np.cos(t), np.sin(t)
    fwd = c - s * y  # world components of the ray for a camera heading north
    down = s + c * y
    right = x
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(down > 1e-9, np.asarray(h, float) / down, np.nan)
    fwd, right = k * fwd, k * right
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    return cy_ * fwd - sy_ * right, sy_ * fwd + cy_ * right


def project_ground(
    dn: float | NDArray,
    de: float | NDArray,
    h: float | NDArray,
    f: float,
    yaw: float | NDArray,
    tilt_deg: float | NDArray,
    cx: float = CX,
    cy: float = CY,
) -> tuple[NDArray, NDArray]:
    """Pixel of the ground point at north/east offset (dn, de) from the point below a camera at
    height h: the inverse of `ground_offset_oblique` (same mount and conventions). Points at or
    behind the image plane give NaN."""
    c, s = np.cos(yaw), np.sin(yaw)
    dn, de = np.asarray(dn, float), np.asarray(de, float)
    fwd, right = c * dn + s * de, -s * dn + c * de  # into the heading frame
    down = np.broadcast_to(np.asarray(h, float), np.shape(fwd))
    t = np.radians(np.asarray(tilt_deg, float))
    ct, st = np.cos(t), np.sin(t)
    z = ct * fwd + st * down  # along the optical axis
    y = -st * fwd + ct * down  # image down
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = z > 1e-9
        u = np.where(ok, cx + f * right / z, np.nan)
        v = np.where(ok, cy + f * y / z, np.nan)
    return u, v
