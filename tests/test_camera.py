"""Camera model and pixel-to-ray mapping."""

import numpy as np
import pytest

from aerialrange.geometry import frames as fr
from aerialrange.geometry.camera import (
    Camera,
    Pose,
    distort_pixels,
    pixel_to_world_ray,
    project_world_point,
    undistort_pixels,
)

DEG = np.pi / 180
# Synthetic intrinsics for tests only (not from any dataset).
FX, FY, CX, CY = 800.0, 820.0, 640.0, 360.0


def nadir_cam(**kw):
    return Camera.pinhole(FX, FY, CX, CY, R_bc=fr.R_bc_from_tilt(90 * DEG), **kw)


def level_pose(p=(0.0, 0.0, -20.0), yaw=0.0):
    return Pose.from_ypr(p, yaw, 0.0, 0.0)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_intrinsic_accessors():
    cam = nadir_cam()
    assert (cam.fx, cam.fy, cam.cx, cam.cy) == (FX, FY, CX, CY)


def test_camera_rejects_malformed_K():
    with pytest.raises(ValueError):
        Camera(K=np.eye(2))
    with pytest.raises(ValueError):
        Camera(K=np.array([[1.0, 0, 0], [0, 1, 0], [1, 0, 1]]))


def test_pose_rejects_non_rotation():
    with pytest.raises(ValueError):
        Pose(p_w=np.zeros(3), R_wb=2 * np.eye(3))


def test_principal_point_of_nadir_camera_on_level_drone_gives_world_down():
    _, d_w = pixel_to_world_ray(nadir_cam(), level_pose(), (CX, CY))
    np.testing.assert_allclose(unit(d_w), [0, 0, 1], atol=1e-15)


def test_principal_point_of_forward_camera_gives_heading():
    cam = Camera.pinhole(FX, FY, CX, CY)  # default forward mount
    _, d_w = pixel_to_world_ray(cam, level_pose(yaw=90 * DEG), (CX, CY))
    np.testing.assert_allclose(unit(d_w), [0, 1, 0], atol=1e-12)


def test_nadir_pixel_right_of_centre_points_east_and_below_centre_points_south():
    cam, pose = nadir_cam(), level_pose()
    _, d_right = pixel_to_world_ray(cam, pose, (CX + 100, CY))
    _, d_below = pixel_to_world_ray(cam, pose, (CX, CY + 100))
    assert d_right[1] > 0 and abs(d_right[0]) < 1e-12
    assert d_below[0] < 0 and abs(d_below[1]) < 1e-12


def test_ray_is_unnormalised_K_inverse_with_unit_optical_component():
    cam, pose = nadir_cam(), level_pose()
    _, d_w = pixel_to_world_ray(cam, pose, (CX + FX, CY))
    np.testing.assert_allclose(d_w, [0, 1, 1], atol=1e-12)


def test_lever_arm_moves_camera_centre():
    t_bc = np.array([0.1, 0.0, 0.05])
    pose = level_pose(yaw=90 * DEG)
    C_w, _ = pixel_to_world_ray(nadir_cam(t_bc=t_bc), pose, (CX, CY))
    np.testing.assert_allclose(C_w, pose.p_w + pose.R_wb @ t_bc, atol=1e-12)


def test_batched_pixels():
    uv = np.array([[CX, CY], [CX + 10, CY], [CX, CY - 5]])
    _, d_w = pixel_to_world_ray(nadir_cam(), level_pose(), uv)
    assert d_w.shape == (3, 3)


def test_project_world_point_inverts_ray():
    rng = np.random.default_rng(1)
    cam = Camera.pinhole(FX, FY, CX, CY, R_bc=fr.R_bc_from_tilt(40 * DEG))
    pose = Pose.from_ypr([5.0, -3.0, -25.0], 0.7, 0.05, -0.03)
    uv = np.array([CX, CY]) + rng.uniform(-300, 300, size=2)
    C_w, d_w = pixel_to_world_ray(cam, pose, uv)
    X_w = C_w + 37.0 * d_w
    uv_back, in_front = project_world_point(cam, pose, X_w)
    assert in_front
    np.testing.assert_allclose(uv_back, uv, atol=1e-9)


def test_project_point_behind_camera_is_flagged():
    cam, pose = nadir_cam(), level_pose()
    _, in_front = project_world_point(cam, pose, [0.0, 0.0, -100.0])  # above the drone
    assert not in_front


# --- lens distortion (Brown-Conrady k1, k2, p1, p2, k3; OpenCV order) ------------------------


def test_no_distortion_is_identity():
    uv = np.array([[100.0, 50.0], [CX, CY]])
    np.testing.assert_array_equal(undistort_pixels(nadir_cam(), uv), uv)


def test_undistort_inverts_distort():
    # Synthetic coefficients for the round-trip test only.
    cam = nadir_cam(dist=np.array([-0.2, 0.05, 0.001, -0.0005, 0.0]))
    uv_ideal = np.array([[CX, CY], [CX + 300, CY - 200], [100.0, 650.0]])
    uv_dist = distort_pixels(cam, uv_ideal)
    assert not np.allclose(uv_dist[1:], uv_ideal[1:])
    np.testing.assert_allclose(undistort_pixels(cam, uv_dist), uv_ideal, atol=1e-6)


def test_distortion_coefficients_validated():
    with pytest.raises(ValueError):
        nadir_cam(dist=np.zeros(3))


def test_ray_uses_undistorted_pixel():
    cam_d = nadir_cam(dist=np.array([-0.2, 0.05, 0.0, 0.0, 0.0]))
    uv_ideal = np.array([CX + 300.0, CY + 100.0])
    uv_dist = distort_pixels(cam_d, uv_ideal)
    _, d_dist = pixel_to_world_ray(cam_d, level_pose(), uv_dist)
    _, d_ideal = pixel_to_world_ray(nadir_cam(), level_pose(), uv_ideal)
    np.testing.assert_allclose(d_dist, d_ideal, atol=1e-6)


def test_rotation_with_negative_determinant_rejected():
    with pytest.raises(ValueError, match="determinant"):
        Pose(p_w=np.zeros(3), R_wb=np.diag([1.0, 1.0, -1.0]))
