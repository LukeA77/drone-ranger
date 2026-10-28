"""Two-view triangulation with known camera poses (terrain from motion, plan D).

Analytic oracle: two parallel cameras with baseline B (metres) looking at a point at depth Z
on the centre line: depth sigma = Z^2 / (f B) * sigma_disparity, with
sigma_disparity = sqrt(2) * sigma_px for independent noise in both images.
"""

import numpy as np
import pytest

from aerialrange.geometry import frames as fr
from aerialrange.geometry.camera import Camera, Pose, project_world_point
from aerialrange.geometry.triangulation import triangulate, triangulate_with_covariance
from aerialrange.utils.seeds import make_rng

DEG = np.pi / 180
F = 500.0  # synthetic focal length (px)
CAM = Camera.pinhole(F, F, 320.0, 240.0)  # forward-looking default mount


def pose_at(north, east, down, yaw=0.0, pitch=0.0, roll=0.0):
    return Pose.from_ypr([north, east, down], yaw, pitch, roll)


def test_exact_triangulation_of_a_known_point():
    rng = np.random.default_rng(0)
    p1 = pose_at(0, 0, -20, yaw=0.1, pitch=-0.4)
    p2 = pose_at(3, 1, -21, yaw=0.12, pitch=-0.38, roll=0.02)
    for _ in range(20):
        X = np.array([rng.uniform(20, 60), rng.uniform(-10, 10), rng.uniform(-2, 2)])
        uv1, ok1 = project_world_point(CAM, p1, X)
        uv2, ok2 = project_world_point(CAM, p2, X)
        assert ok1 and ok2
        r = triangulate(CAM, p1, uv1, CAM, p2, uv2)
        assert r["valid"]
        np.testing.assert_allclose(r["point_w"], X, atol=1e-8)
        assert r["ray_gap_m"] < 1e-8


def test_parallax_angle():
    p1, p2 = pose_at(0, 0, 0), pose_at(0, 10, 0)  # 10 m baseline, east
    X = np.array([10.0, 5.0, 0.0])  # 10 m north of the baseline midpoint
    r = triangulate(
        CAM, p1, project_world_point(CAM, p1, X)[0], CAM, p2, project_world_point(CAM, p2, X)[0]
    )
    assert r["parallax_deg"] == pytest.approx(2 * np.degrees(np.arctan(5 / 10)))


def test_parallel_rays_are_invalid():
    p1, p2 = pose_at(0, 0, 0), pose_at(0, 1, 0)
    r = triangulate(CAM, p1, (320.0, 240.0), CAM, p2, (320.0, 240.0))  # both straight ahead
    assert not r["valid"] and r["reason"] == "parallel_rays"


def test_point_behind_a_camera_is_invalid():
    p1, p2 = pose_at(0, 0, 0), pose_at(0, 2, 0)
    # Rays diverge forwards and meet behind the cameras.
    r = triangulate(CAM, p1, (300.0, 240.0), CAM, p2, (340.0, 240.0))
    assert not r["valid"] and r["reason"] == "behind_camera"


@pytest.mark.parametrize("Z, B", [(20.0, 1.0), (50.0, 2.0), (80.0, 5.0)])
def test_depth_sigma_matches_stereo_oracle(Z, B):
    p1, p2 = pose_at(0, -B / 2, 0), pose_at(0, B / 2, 0)  # forward cameras, baseline east
    X = np.array([Z, 0.0, 0.0])
    uv1, _ = project_world_point(CAM, p1, X)
    uv2, _ = project_world_point(CAM, p2, X)
    sigma_px = 0.5
    r = triangulate_with_covariance(CAM, p1, uv1, CAM, p2, uv2, sigma_px=sigma_px)
    expected = Z**2 / (F * B) * np.sqrt(2) * sigma_px
    assert np.sqrt(r["cov_w"][0, 0]) == pytest.approx(expected, rel=0.02)  # depth = north


def test_first_order_covariance_agrees_with_monte_carlo():
    p1 = pose_at(0, 0, -20, pitch=-0.5)
    p2 = pose_at(4, 0.5, -20.5, pitch=-0.5)
    X = np.array([35.0, 3.0, 0.0])
    uv1, _ = project_world_point(CAM, p1, X)
    uv2, _ = project_world_point(CAM, p2, X)
    s = 0.7
    fo = triangulate_with_covariance(CAM, p1, uv1, CAM, p2, uv2, sigma_px=s)
    rng = make_rng(0)
    pts = []
    for _ in range(4000):
        r = triangulate(CAM, p1, uv1 + rng.normal(0, s, 2), CAM, p2, uv2 + rng.normal(0, s, 2))
        if r["valid"]:
            pts.append(r["point_w"])
    mc = np.cov(np.array(pts).T)
    np.testing.assert_allclose(np.sqrt(np.diag(mc)), np.sqrt(np.diag(fo["cov_w"])), rtol=0.1)


def test_pose_noise_adds_to_covariance():
    p1 = pose_at(0, 0, -20, pitch=-0.5)
    p2 = pose_at(4, 0.5, -20.5, pitch=-0.5)
    X = np.array([35.0, 3.0, 0.0])
    uv1, _ = project_world_point(CAM, p1, X)
    uv2, _ = project_world_point(CAM, p2, X)
    a = triangulate_with_covariance(CAM, p1, uv1, CAM, p2, uv2, sigma_px=0.5)
    b = triangulate_with_covariance(
        CAM, p1, uv1, CAM, p2, uv2, sigma_px=0.5, sigma_pos_m=0.1, sigma_att_rad=0.2 * DEG
    )
    assert np.trace(b["cov_w"]) > np.trace(a["cov_w"])


def test_works_with_tilted_mount():
    cam = Camera.pinhole(F, F, 320.0, 240.0, R_bc=fr.R_bc_from_tilt(45 * DEG))
    p1, p2 = pose_at(0, 0, -30), pose_at(5, 0, -30)
    X = np.array([30.0, -4.0, 0.0])
    r = triangulate(
        cam, p1, project_world_point(cam, p1, X)[0], cam, p2, project_world_point(cam, p2, X)[0]
    )
    np.testing.assert_allclose(r["point_w"], X, atol=1e-8)


def test_batch_matches_single_point_triangulation_and_covariance():
    from aerialrange.geometry.triangulation import triangulate_batch

    rng = np.random.default_rng(3)
    p1 = pose_at(0, 0, -20, yaw=0.1, pitch=-0.4)
    p2 = pose_at(3, 1, -21, yaw=0.12, pitch=-0.38, roll=0.02)
    X = np.column_stack([rng.uniform(20, 60, 40), rng.uniform(-10, 10, 40), rng.uniform(-2, 2, 40)])
    uv1 = np.array([project_world_point(CAM, p1, x)[0] for x in X])
    uv2 = np.array([project_world_point(CAM, p2, x)[0] for x in X])
    uv2[0] = uv1[0]  # make one pair inconsistent (rays may not meet in front)
    b = triangulate_batch(
        CAM, p1, uv1, CAM, p2, uv2, sigma_px=0.5, sigma_pos_m=0.05, sigma_att_rad=0.1 * DEG
    )
    for i in range(len(X)):
        s = triangulate_with_covariance(
            CAM,
            p1,
            uv1[i],
            CAM,
            p2,
            uv2[i],
            sigma_px=0.5,
            sigma_pos_m=0.05,
            sigma_att_rad=0.1 * DEG,
        )
        assert b["valid"][i] == s["valid"]
        if s["valid"]:
            np.testing.assert_allclose(b["point_w"][i], s["point_w"], atol=1e-9)
            np.testing.assert_allclose(b["cov_w"][i], s["cov_w"], rtol=1e-4, atol=1e-10)
            assert b["parallax_deg"][i] == pytest.approx(s["parallax_deg"])
