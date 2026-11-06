"""CARLA (Unreal Engine) coordinates to the project's frames.

Unreal world: left-handed, x forward, y right, z up, metres; rotations in degrees
(CARLA 0.9.12 Python API, `carla.Rotation`). The project world is NED and the camera frame
OpenCV (`geometry/frames.py`). With Unreal x as North and y as East, the map is a flip of
z, `NED = (x, y, -z)`, which turns the left-handed basis right-handed. Unreal yaw (x towards
y) is then heading clockwise from North, and Unreal pitch (nose up) is aerospace pitch, so a
CARLA sensor at (pitch, yaw) has `R_wb = Rz(yaw) Ry(pitch)` with an FRD body whose forward
axis is the camera's optical axis. Signs are pinned in `tests/test_carla_frames.py` and
checked on SynDrone data by reprojection (`scripts/syndrone_reproject.py`).

Roll is not supported: SynDrone cameras always have roll 0, and its sign convention was not
checked against data, so a non-zero roll is refused rather than guessed.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerialrange.geometry.camera import Camera, Pose
from aerialrange.geometry.frames import R_BC_FORWARD, R_wb_from_ypr

_FLIP = np.array([1.0, 1.0, -1.0])


def ned_from_ue(p: ArrayLike) -> NDArray[np.float64]:
    """Unreal world point(s) (..., 3) to NED (metres)."""
    return np.asarray(p, float) * _FLIP


def ue_from_ned(p: ArrayLike) -> NDArray[np.float64]:
    """NED point(s) (..., 3) to Unreal world (metres)."""
    return np.asarray(p, float) * _FLIP


def carla_camera(width: int, height: int, fov_deg: float) -> Camera:
    """Ideal pinhole CARLA camera; `fov_deg` is the horizontal field of view (CARLA 0.9.12
    sensor reference: "Horizontal field of view in degrees"), square pixels, principal point
    at the image centre, as in CARLA's own projection example."""
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return Camera.pinhole(f, f, width / 2.0, height / 2.0, R_bc=R_BC_FORWARD)


def carla_pose(
    x: float, y: float, z: float, pitch: float, yaw: float, roll: float = 0.0, z0: float = 0.0
) -> Pose:
    """Pose of a CARLA sensor at Unreal location (x, y, z + z0) and rotation in degrees.

    `z0` is the base height SynDrone adds to the stored camera height `z`.
    """
    if roll != 0.0:
        raise ValueError("non-zero roll is not supported (sign convention not verified)")
    p = ned_from_ue([x, y, z + z0])
    return Pose(p, R_wb_from_ypr(math.radians(yaw), math.radians(pitch), 0.0))
