"""ROS convention conversions (Phase 12, REP 103 and REP 105)."""

from __future__ import annotations

import numpy as np
import pytest

from aerialrange import ros_bridge as rb
from aerialrange.geometry.frames import ned_to_enu, quat_to_R


def test_level_body_heading_north_with_a_tilted_mount() -> None:
    pose = rb.camera_pose_from_odometry([3.0, 4.0, 25.0], [0, 0, 0, 1], 30.0)
    # identity quaternion: body forward is ENU x, i.e. east (REP 103)
    assert pose.yaw_rad == pytest.approx(np.pi / 2)
    assert pose.tilt_deg == pytest.approx(30.0)
    assert (pose.n_m, pose.e_m, pose.h_m) == (4.0, 3.0, 25.0)


def test_yaw_about_up_turns_the_heading_anticlockwise() -> None:
    q = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]  # +90 deg about ENU up: forward north
    pose = rb.camera_pose_from_odometry([0, 0, 10.0], q, 45.0)
    assert pose.yaw_rad == pytest.approx(0.0, abs=1e-12)
    assert pose.tilt_deg == pytest.approx(45.0)


def test_ground_height_is_subtracted() -> None:
    assert rb.camera_pose_from_odometry([0, 0, 40.0], [0, 0, 0, 1], 0.0, 15.0).h_m == 25.0


@pytest.mark.parametrize("mount", [0.0, 20.0, 90.0])
def test_odometry_round_trip(mount: float) -> None:
    rng = np.random.default_rng(3)
    for _ in range(200):
        n, e, h = rng.uniform(-500, 500), rng.uniform(-500, 500), rng.uniform(5, 120)
        yaw = rng.uniform(-np.pi, np.pi)
        tilt = rng.uniform(5, 89) if mount < 90 else 90.0
        p, q = rb.odometry_from_camera_pose(n, e, h, yaw, tilt, mount)
        pose = rb.camera_pose_from_odometry(p, q, mount)
        assert (pose.n_m, pose.e_m, pose.h_m) == pytest.approx((n, e, h))
        assert pose.tilt_deg == pytest.approx(tilt, abs=1e-9)
        if tilt < 89.9:
            assert np.angle(np.exp(1j * (pose.yaw_rad - yaw))) == pytest.approx(0.0, abs=1e-9)


def test_optical_frame_rotation_is_the_standard_static_transform() -> None:
    # static_transform_publisher --roll -pi/2 --yaw -pi/2 gives exactly R_LINK_OPTICAL: the
    # optical z is camera_link forward, x is right (-left) and y is down (-up)
    np.testing.assert_allclose(rb.R_from_rpy(*rb.OPTICAL_RPY), rb.R_LINK_OPTICAL, atol=1e-15)
    np.testing.assert_allclose(rb.R_LINK_OPTICAL[:, 2], [1, 0, 0])
    np.testing.assert_allclose(rb.R_LINK_OPTICAL[:, 0], [0, -1, 0])
    np.testing.assert_allclose(rb.R_LINK_OPTICAL[:, 1], [0, 0, -1])


@pytest.mark.parametrize("mount", [0.0, 30.0, 50.0, 90.0])
def test_mount_rotation_points_the_optical_axis_down_the_mount(mount: float) -> None:
    np.testing.assert_allclose(
        rb.mount_rotation(mount)[:, 2], rb.camera_axis_flu(mount), atol=1e-15
    )


def test_frame_tree_pose_equals_the_mount_angle_pose() -> None:
    # the tf2 route (body pose and base_link -> optical transform) gives the same pose as the
    # fixed mount angle, for random attitudes: the node's two ways in agree
    rng = np.random.default_rng(11)
    for _ in range(200):
        p, q = rng.uniform(-300, 300, 3), rng.normal(size=4)
        mount = rng.uniform(0, 90)
        q_mount = rb.R_to_quat(rb.mount_rotation(mount), order="xyzw")
        a = rb.camera_pose_from_frames(p, q, np.zeros(3), q_mount, 2.0)
        b = rb.camera_pose_from_odometry(p, q, mount, 2.0)
        assert (a.n_m, a.e_m, a.h_m, a.tilt_deg) == pytest.approx(
            (b.n_m, b.e_m, b.h_m, b.tilt_deg), abs=1e-9
        )
        if abs(a.tilt_deg) < 89.9:
            assert np.angle(np.exp(1j * (a.yaw_rad - b.yaw_rad))) == pytest.approx(0, abs=1e-9)


def test_camera_offset_on_the_body_moves_the_camera() -> None:
    # body yawed to face north (+90 deg about up), camera 0.2 m forward and 0.1 m below:
    # the camera sits 0.2 m north of the body and 0.1 m lower
    q = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    q_mount = rb.R_to_quat(rb.mount_rotation(30.0), order="xyzw")
    pose = rb.camera_pose_from_frames([5.0, 7.0, 40.0], q, [0.2, 0.0, -0.1], q_mount)
    assert (pose.n_m, pose.e_m, pose.h_m) == pytest.approx((7.2, 5.0, 39.9))
    assert pose.yaw_rad == pytest.approx(0.0, abs=1e-12)
    assert pose.tilt_deg == pytest.approx(30.0)


def test_attitude_conversions_are_inverse_and_proper() -> None:
    rng = np.random.default_rng(5)
    q = rng.normal(size=4)
    R = quat_to_R(q)
    R2 = rb.R_ned_frd_from_enu_flu(rb.R_enu_flu_from_ned_frd(R))
    np.testing.assert_allclose(R2, R, atol=1e-12)
    assert np.linalg.det(rb.R_enu_flu_from_ned_frd(R)) == pytest.approx(1.0)
    # a world vector maps the same way as the position conversion
    v = rng.normal(size=3)
    np.testing.assert_allclose(rb.T_ENU_NED @ v, ned_to_enu(v))


def test_velocity_from_speed_and_heading() -> None:
    v, cov = rb.velocity_enu(2.0, 0.1, 90.0, 5.0)
    np.testing.assert_allclose(v, [2.0, 0.0], atol=1e-12)
    # heading error is across the motion (north), speed error along it (east)
    assert cov[0, 0] == pytest.approx(0.01)
    assert cov[1, 1] == pytest.approx((2.0 * np.radians(5.0)) ** 2)
    rng = np.random.default_rng(1)
    s = 3.0 + rng.normal(size=200_000) * 0.05
    hd = np.radians(30.0 + rng.normal(size=200_000) * 1.0)
    mc = np.cov(np.c_[s * np.sin(hd), s * np.cos(hd)].T)
    _, cov = rb.velocity_enu(3.0, 0.05, 30.0, 1.0)
    np.testing.assert_allclose(cov, mc, rtol=0.03, atol=1e-6)


def test_covariance_reorder() -> None:
    c = np.array([[4.0, 1.0], [1.0, 2.0]])  # (north, east)
    np.testing.assert_allclose(rb.cov_en_from_ne(c), [[2.0, 1.0], [1.0, 4.0]])
