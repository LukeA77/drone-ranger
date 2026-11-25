"""Conversions between ROS 2 conventions and the pipeline's, without any ROS import (Phase 12).

ROS (REP 103, REP 105): the world frame is East, North, Up (ENU); a body frame is forward,
left, up (FLU); orientations are Hamilton quaternions (x, y, z, w). The pipeline: world North,
East, Down (NED), body forward, right, down (FRD), and a camera pose given as local north and
east, height above the ground, heading of the optical axis clockwise from north, and its
depression below the horizon (`pipeline.ranging.CameraPose`).

The ROS node (`ros2/drone_ranger_ros`) uses these functions; the tests in
`tests/test_ros_bridge.py` pin them, so the node needs no geometry of its own.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerialrange.geometry.frames import R_to_quat, quat_to_R
from aerialrange.pipeline.ranging import CameraPose

# ENU <-> NED is a fixed permutation and sign (its own inverse); FLU <-> FRD flips y and z.
T_ENU_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
T_FLU_FRD = np.diag([1.0, -1.0, -1.0])


def R_enu_flu_from_ned_frd(R_ned_frd: ArrayLike) -> NDArray[np.float64]:
    """Body attitude in ROS conventions from the same attitude in the pipeline's."""
    return T_ENU_NED @ np.asarray(R_ned_frd, float) @ T_FLU_FRD


def R_ned_frd_from_enu_flu(R_enu_flu: ArrayLike) -> NDArray[np.float64]:
    return T_ENU_NED @ np.asarray(R_enu_flu, float) @ T_FLU_FRD


def camera_axis_flu(mount_tilt_deg: float) -> NDArray[np.float64]:
    """Optical axis in the body FLU frame of a camera looking forward and `mount_tilt_deg`
    below the body's horizontal plane (pitched about the body's right axis)."""
    t = np.radians(mount_tilt_deg)
    return np.array([np.cos(t), 0.0, -np.sin(t)])


# The tf2 frame tree (REP 103, REP 105): map -> base_link (the odometry) -> camera_link (the
# fixed mount: forward, left, up, pitched down by the mount tilt) -> camera_optical_frame
# (z along the optical axis, x right, y down in the image; images are stamped in this frame).
# camera_link -> camera_optical_frame is the standard fixed rotation, roll -90 deg and
# yaw -90 deg, as static_transform_publisher and tf2's setRPY compose it (Rz Ry Rx); the
# columns of R_LINK_OPTICAL are the optical frame's axes in camera_link.
OPTICAL_RPY = (-np.pi / 2, 0.0, -np.pi / 2)
R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def R_from_rpy(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """Rotation of fixed-axis roll, then pitch, then yaw (Rz Ry Rx), as tf2's setRPY."""
    cr, sr, cp, sp, cy, sy = (np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch),
                              np.cos(yaw), np.sin(yaw))  # fmt: skip
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def mount_rotation(mount_tilt_deg: float) -> NDArray[np.float64]:
    """base_link -> camera_optical_frame rotation of a camera looking forward and
    `mount_tilt_deg` below the body's horizontal plane: the base_link -> camera_link pitch
    (positive pitch about the body's left axis tips forward down) times the optical rotation."""
    return R_from_rpy(0.0, np.radians(mount_tilt_deg), 0.0) @ R_LINK_OPTICAL


def camera_pose_from_frames(
    position_enu: ArrayLike,
    quat_body_xyzw: ArrayLike,
    mount_translation: ArrayLike,
    mount_quat_xyzw: ArrayLike,
    ground_up_m: float = 0.0,
) -> CameraPose:
    """The pipeline's camera pose from the body's pose in the world (a `nav_msgs/Odometry`
    pose: body FLU in world ENU) and the camera's pose in the body (the base_link ->
    camera_optical_frame transform that tf2 gives, whatever frames lie between): heading and
    depression of the optical axis (the optical frame's z), and the camera's height above a
    flat ground at `ground_up_m` (world up). The pipeline's camera model has no roll about the
    optical axis, so roll is not represented (as in `aerialrange run`)."""
    p = np.asarray(position_enu, float)
    R_body = quat_to_R(np.asarray(quat_body_xyzw, float), order="xyzw")
    R_mount = quat_to_R(np.asarray(mount_quat_xyzw, float), order="xyzw")
    cam = p + R_body @ np.asarray(mount_translation, float)  # ENU
    east, north, up = R_body @ R_mount[:, 2]
    yaw = float(np.arctan2(east, north))
    tilt = float(np.degrees(np.arctan2(-up, np.hypot(east, north))))
    return CameraPose(float(cam[1]), float(cam[0]), float(cam[2] - ground_up_m), yaw, tilt)


def camera_pose_from_odometry(
    position_enu: ArrayLike,
    quat_xyzw: ArrayLike,
    mount_tilt_deg: float,
    ground_up_m: float = 0.0,
) -> CameraPose:
    """`camera_pose_from_frames` for a camera at the body's origin on a fixed mount tilted
    `mount_tilt_deg` below the body's horizontal plane."""
    q_mount = R_to_quat(mount_rotation(mount_tilt_deg), order="xyzw")
    return camera_pose_from_frames(position_enu, quat_xyzw, np.zeros(3), q_mount, ground_up_m)


def odometry_from_camera_pose(
    n_m: float, e_m: float, h_m: float, yaw_rad: float, tilt_deg: float, mount_tilt_deg: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(position ENU, quaternion xyzw) of a level-rolled body whose camera, mounted
    `mount_tilt_deg` down, looks along heading `yaw_rad` and `tilt_deg` below the horizon, at
    height `h_m` over ground at zero: the inverse of `camera_pose_from_odometry`, used to turn a
    recorded flight's camera poses into odometry for a test recording."""
    pitch = np.radians(tilt_deg - mount_tilt_deg)  # body nose-down pitch, as a positive angle
    cy, sy, cp, sp = np.cos(yaw_rad), np.sin(yaw_rad), np.cos(pitch), np.sin(pitch)
    # NED/FRD: R = Rz(yaw) Ry(-pitch) (nose down is negative aerospace pitch)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
    R_enu_flu = R_enu_flu_from_ned_frd(Rz @ Ry)
    return np.array([e_m, n_m, h_m], float), R_to_quat(R_enu_flu, order="xyzw")


def velocity_enu(
    speed_ms: float, speed_sigma_ms: float, heading_deg: float, heading_sigma_deg: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Ground velocity (east, north) and its 2 x 2 covariance from speed and heading (clockwise
    from north) with their sigmas, to first order (independent errors)."""
    h = np.radians(heading_deg)
    sh = np.radians(heading_sigma_deg)
    v = np.array([speed_ms * np.sin(h), speed_ms * np.cos(h)])
    J = np.array([[np.sin(h), speed_ms * np.cos(h)], [np.cos(h), -speed_ms * np.sin(h)]])
    return v, J @ np.diag([speed_sigma_ms**2, sh**2]) @ J.T


def cov_en_from_ne(cov_ne: ArrayLike) -> NDArray[np.float64]:
    """A 2 x 2 (north, east) covariance as (east, north)."""
    c = np.asarray(cov_ne, float)
    return c[::-1, ::-1].copy()
