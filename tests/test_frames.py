"""Frame conventions, pinned by tests.

World W = North, East, Down (NED). Body B = forward, right, down (FRD).
Camera C = OpenCV (x right, y down, z forward). R_ab maps vectors from frame b to frame a.
"""

import numpy as np
import pytest

from aerialrange.geometry import frames as fr

DEG = np.pi / 180
FWD, RIGHT, DOWN = np.eye(3)


def _proper(R):
    return np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1.0)


@pytest.mark.parametrize("fn", [fr.rot_x, fr.rot_y, fr.rot_z])
@pytest.mark.parametrize("a", [-2.0, -0.3, 0.0, 0.7, 3.1])
def test_elementary_rotations_are_proper(fn, a):
    assert _proper(fn(a))


def test_rot_z_is_right_handed():
    np.testing.assert_allclose(fr.rot_z(90 * DEG) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


# --- attitude: aerospace yaw-pitch-roll (Z-Y-X) ---------------------------------------------


def test_level_zero_attitude_is_identity():
    np.testing.assert_allclose(fr.R_wb_from_ypr(0, 0, 0), np.eye(3), atol=1e-15)


def test_yaw_90_points_body_forward_east():
    np.testing.assert_allclose(fr.R_wb_from_ypr(90 * DEG, 0, 0) @ FWD, [0, 1, 0], atol=1e-12)


def test_positive_pitch_is_nose_up():
    f_w = fr.R_wb_from_ypr(0, 10 * DEG, 0) @ FWD
    assert f_w[2] < 0  # negative Down = up


def test_positive_roll_is_right_wing_down():
    r_w = fr.R_wb_from_ypr(0, 0, 10 * DEG) @ RIGHT
    assert r_w[2] > 0


def test_ypr_sequence_is_yaw_then_pitch_then_roll():
    y, p, r = 0.4, -0.2, 0.15
    expected = fr.rot_z(y) @ fr.rot_y(p) @ fr.rot_x(r)
    np.testing.assert_allclose(fr.R_wb_from_ypr(y, p, r), expected, atol=1e-15)


@pytest.mark.parametrize("ypr", [(0.4, -0.2, 0.15), (-2.9, 0.5, -0.6), (1.26, 0.0, 0.011)])
def test_ypr_round_trip(ypr):
    np.testing.assert_allclose(fr.ypr_from_R_wb(fr.R_wb_from_ypr(*ypr)), ypr, atol=1e-12)


def test_heading_follows_yaw_regardless_of_pitch_and_roll():
    # AU-AIR paper Figure 1 sample attitude: roll 0.011 rad, pitch 0, yaw 1.26 rad.
    assert fr.heading_rad(fr.R_wb_from_ypr(1.26, 0.0, 0.011)) == pytest.approx(1.26, abs=1e-12)
    assert fr.heading_rad(fr.R_wb_from_ypr(-0.5, 0.3, -0.2)) == pytest.approx(-0.5, abs=1e-12)


# --- camera mount ----------------------------------------------------------------------------


def test_default_mount_maps_camera_axes_to_body():
    R = fr.R_BC_FORWARD
    np.testing.assert_allclose(R @ [0, 0, 1], FWD)  # optical axis -> forward
    np.testing.assert_allclose(R @ [1, 0, 0], RIGHT)  # image right -> right
    np.testing.assert_allclose(R @ [0, 1, 0], DOWN)  # image down -> down
    assert _proper(R)


def test_tilt_0_puts_optical_axis_along_body_forward():
    np.testing.assert_allclose(fr.R_bc_from_tilt(0.0) @ [0, 0, 1], FWD, atol=1e-15)


def test_tilt_90_puts_optical_axis_along_body_down():
    np.testing.assert_allclose(fr.R_bc_from_tilt(90 * DEG) @ [0, 0, 1], DOWN, atol=1e-15)


def test_positive_tilt_is_depression_below_horizon():
    z_b = fr.R_bc_from_tilt(30 * DEG) @ [0, 0, 1]
    np.testing.assert_allclose(z_b, [np.cos(30 * DEG), 0, np.sin(30 * DEG)], atol=1e-15)


def test_nadir_mount_keeps_image_right_as_body_right_and_image_top_forward():
    R = fr.R_bc_from_tilt(90 * DEG)
    np.testing.assert_allclose(R @ [1, 0, 0], RIGHT, atol=1e-15)
    np.testing.assert_allclose(R @ [0, -1, 0], FWD, atol=1e-15)


# --- quaternions (Hamilton, active) ----------------------------------------------------------


def test_identity_quaternion():
    np.testing.assert_allclose(fr.quat_to_R([1, 0, 0, 0]), np.eye(3), atol=1e-15)


def test_quaternion_90_about_z_matches_rot_z():
    q = [np.cos(45 * DEG), 0, 0, np.sin(45 * DEG)]
    np.testing.assert_allclose(fr.quat_to_R(q), fr.rot_z(90 * DEG), atol=1e-12)


def test_quaternion_xyzw_order():
    q_wxyz = np.array([0.9, 0.1, -0.3, 0.2])
    q_xyzw = np.roll(q_wxyz, -1)
    np.testing.assert_allclose(fr.quat_to_R(q_xyzw, order="xyzw"), fr.quat_to_R(q_wxyz))


def test_quaternion_round_trip_and_normalisation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        R = fr.quat_to_R(q)
        assert _proper(R)
        np.testing.assert_allclose(fr.quat_to_R(fr.R_to_quat(R)), R, atol=1e-12)
        q_back = fr.R_to_quat(R, order="xyzw")
        np.testing.assert_allclose(fr.quat_to_R(q_back, order="xyzw"), R, atol=1e-12)


def test_quaternion_rejects_zero_and_bad_order():
    with pytest.raises(ValueError):
        fr.quat_to_R([0, 0, 0, 0])
    with pytest.raises(ValueError):
        fr.quat_to_R([1, 0, 0, 0], order="zyxw")
    with pytest.raises(ValueError):
        fr.R_to_quat(np.eye(3), order="zyxw")


# --- geodetic to local NED -------------------------------------------------------------------

A, F_WGS = 6378137.0, 1 / 298.257223563
E2 = F_WGS * (2 - F_WGS)


def test_ned_origin_is_zero():
    np.testing.assert_allclose(fr.geodetic_to_ned(56.2, 10.2, 30.0, 56.2, 10.2, 30.0), 0, atol=1e-9)


def test_ned_north_offset_matches_meridian_radius():
    lat0 = 56.2
    dlat = 1e-4  # degrees, about 11 m
    phi = np.radians(lat0)
    M = A * (1 - E2) / (1 - E2 * np.sin(phi) ** 2) ** 1.5
    ned = fr.geodetic_to_ned(lat0 + dlat, 10.2, 0.0, lat0, 10.2, 0.0)
    assert ned[0] == pytest.approx(M * np.radians(dlat), rel=1e-4)
    assert abs(ned[1]) < 1e-6


def test_ned_east_offset_matches_prime_vertical_radius():
    lat0 = 56.2
    dlon = 1e-4
    phi = np.radians(lat0)
    N = A / np.sqrt(1 - E2 * np.sin(phi) ** 2)
    ned = fr.geodetic_to_ned(lat0, 10.2 + dlon, 0.0, lat0, 10.2, 0.0)
    east = N * np.cos(phi) * np.radians(dlon)
    assert ned[1] == pytest.approx(east, rel=1e-4)
    # A parallel curves poleward of the tangent plane: north offset ~ E^2 tan(phi) / (2N).
    assert ned[0] == pytest.approx(east**2 * np.tan(phi) / (2 * N), rel=1e-2)


def test_ned_higher_altitude_is_negative_down():
    ned = fr.geodetic_to_ned(56.2, 10.2, 25.0, 56.2, 10.2, 5.0)
    assert ned[2] == pytest.approx(-20.0, abs=1e-6)


def test_ned_vectorised():
    lat = np.array([56.2, 56.2001])
    out = fr.geodetic_to_ned(lat, np.full(2, 10.2), np.zeros(2), 56.2, 10.2, 0.0)
    assert out.shape == (2, 3)
