"""CARLA / Unreal Engine conventions, pinned by tests.

Unreal: left-handed, x forward, y right, z up, metres, angles in degrees; positive yaw turns
x towards y (clockwise seen from above), positive pitch is nose up (CARLA 0.9.12 Python API:
"Z-up left-handed system"; the signs are fixed here and checked against SynDrone data in
scripts/syndrone_reproject.py).
"""

import numpy as np
import pytest

from aerialrange.geometry.camera import project_world_point
from aerialrange.geometry.carla import (
    carla_camera,
    carla_pose,
    ned_from_ue,
    ue_from_ned,
)


def test_points_flip_z_only():
    p = ned_from_ue([1.0, 2.0, 3.0])
    assert np.allclose(p, [1.0, 2.0, -3.0])
    assert np.allclose(ue_from_ned(p), [1.0, 2.0, 3.0])
    pts = np.array([[1, 2, 3], [4, 5, 6]], float)
    assert np.allclose(ue_from_ned(ned_from_ue(pts)), pts)


def test_left_handed_to_right_handed():
    # Unreal's x, y, z basis is left-handed; mapped into NED it must be right-handed.
    ex, ey, ez = ned_from_ue(np.eye(3))
    assert np.isclose(np.dot(np.cross(ex, ey), -ez), 1.0)  # x cross y = -z (left-handed)
    assert np.isclose(np.linalg.det(np.stack([ex, ey, -ez])), 1.0)


def test_intrinsics_from_horizontal_fov():
    cam = carla_camera(1920, 1080, 90.0)
    assert cam.fx == pytest.approx(960.0) and cam.fy == pytest.approx(960.0)
    assert (cam.cx, cam.cy) == (960.0, 540.0)
    assert cam.dist is None


@pytest.mark.parametrize("yaw", [0.0, 37.0, -120.0, 179.0])
def test_nadir_camera_looks_down(yaw):
    cam = carla_camera(1920, 1080, 90.0)
    pose = carla_pose(x=10.0, y=-5.0, z=80.0, pitch=-90.0, yaw=yaw)
    axis_w = pose.R_wb @ cam.R_bc @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(axis_w, [0.0, 0.0, 1.0], atol=1e-12)  # NED down
    below = ned_from_ue([10.0, -5.0, 0.0])
    uv, front = project_world_point(cam, pose, below)
    assert front and np.allclose(uv, [960.0, 540.0])


def test_oblique_camera_depression_and_heading():
    cam = carla_camera(1920, 1080, 90.0)
    pose = carla_pose(x=0.0, y=0.0, z=20.0, pitch=-30.0, yaw=90.0)  # facing Unreal +y
    axis_w = pose.R_wb @ cam.R_bc @ np.array([0.0, 0.0, 1.0])
    # Heading +90 deg from Unreal x (north) is +y (east); 30 deg below the horizon.
    assert np.allclose(axis_w, [0.0, np.cos(np.radians(30)), np.sin(np.radians(30))])


def test_image_axes_forward_camera():
    cam = carla_camera(1920, 1080, 90.0)
    pose = carla_pose(x=0.0, y=0.0, z=10.0, pitch=0.0, yaw=0.0)
    ahead = ned_from_ue([20.0, 0.0, 10.0])
    right = ned_from_ue([20.0, 5.0, 10.0])  # Unreal +y is to the right
    up = ned_from_ue([20.0, 0.0, 15.0])  # Unreal +z is up
    uv0, _ = project_world_point(cam, pose, ahead)
    uvr, _ = project_world_point(cam, pose, right)
    uvu, _ = project_world_point(cam, pose, up)
    assert np.allclose(uv0, [960.0, 540.0])
    assert uvr[0] > 960.0 and np.isclose(uvr[1], 540.0)
    assert uvu[1] < 540.0 and np.isclose(uvu[0], 960.0)
    # 45 deg off-axis lands exactly on the image edge for a 90 deg horizontal field of view.
    edge = ned_from_ue([20.0, 20.0, 10.0])
    uve, _ = project_world_point(cam, pose, edge)
    assert np.isclose(uve[0], 1920.0)


def test_nadir_image_up_is_forward():
    cam = carla_camera(1920, 1080, 90.0)
    pose = carla_pose(x=0.0, y=0.0, z=50.0, pitch=-90.0, yaw=0.0)
    fwd = ned_from_ue([5.0, 0.0, 0.0])
    uv, _ = project_world_point(cam, pose, fwd)
    assert uv[1] < 540.0 and np.isclose(uv[0], 960.0)


def test_base_height_adds_to_camera_height():
    pose = carla_pose(x=0.0, y=0.0, z=20.0, pitch=-30.0, yaw=0.0, z0=7.5)
    assert np.allclose(pose.p_w, [0.0, 0.0, -27.5])


def test_roll_not_supported():
    with pytest.raises(ValueError):
        carla_pose(x=0.0, y=0.0, z=20.0, pitch=-30.0, yaw=0.0, roll=3.0)


@pytest.mark.parametrize("h,pitch,yaw", [(20, -30.0, 3.4), (50, -60.0, -179.8), (80, -90.0, 75.0)])
def test_pipeline_flat_ground_model_matches_carla_pose(h, pitch, yaw):
    """The version 1 pipeline's ground model (auair_world.project_ground) and a CARLA pose
    must put the same ground point on the same pixel (SynDrone evaluation)."""
    from aerialrange.evaluate.auair_world import project_ground

    cam = carla_camera(1920, 1080, 90.0)
    pose = carla_pose(x=100.0, y=50.0, z=float(h), pitch=pitch, yaw=yaw)
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    n_checked = 0
    centre = h / np.tan(np.radians(-pitch))  # where the optical axis meets the ground
    for fwd, right in [(centre, 0.0), (centre + 0.2 * h, 0.25 * h), (centre - 0.2 * h, -0.2 * h)]:
        dn, de = c * fwd - s * right, s * fwd + c * right  # ahead of the camera, in NED
        uv, front = project_world_point(cam, pose, ned_from_ue([100.0 + dn, 50.0 + de, 0.0]))
        u, v = project_ground(dn, de, float(h), cam.fx, np.radians(yaw), -pitch, cam.cx, cam.cy)
        if front and 0 <= uv[0] <= 1920 and 0 <= uv[1] <= 1080:
            assert np.allclose(uv, [u, v], atol=1e-6)
            n_checked += 1
    assert n_checked == 3
