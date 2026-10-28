"""Ray and plane intersection, outputs, validity and contact-point rules."""

import numpy as np
import pytest

from aerialrange.geometry import frames as fr
from aerialrange.geometry.camera import Camera, Pose, pixel_to_world_ray, project_world_point
from aerialrange.geometry.ranging import (
    REASON_ABOVE_HORIZON,
    REASON_NOT_ABOVE_PLANE,
    REASON_OK,
    REASON_PARALLEL,
    contact_pixel,
    estimate_range,
    intersect_horizontal_plane,
    intersect_plane,
    select_contact_mode,
)

DEG = np.pi / 180
FX, FY, CX, CY = 800.0, 800.0, 640.0, 360.0  # synthetic


def cam_tilt(deg):
    return Camera.pinhole(FX, FY, CX, CY, R_bc=fr.R_bc_from_tilt(deg * DEG))


def pose(yaw=0.0, pitch=0.0, roll=0.0, p=(0.0, 0.0, -20.0)):
    return Pose.from_ypr(p, yaw, pitch, roll)


def test_nadir_principal_point_hits_point_below():
    est = estimate_range(cam_tilt(90), pose(), (CX, CY), h_above_plane=20.0)
    assert est.valid and est.reason == REASON_OK
    np.testing.assert_allclose(est.point_w, [0, 0, 0], atol=1e-12)
    assert est.range_m == pytest.approx(20.0)
    assert est.ground_dist_m == pytest.approx(0.0, abs=1e-12)
    assert est.depression_deg == pytest.approx(90.0)


@pytest.mark.parametrize("tilt", [15.0, 30.0, 60.0])
def test_oblique_principal_point_closed_form(tilt):
    h = 20.0
    est = estimate_range(cam_tilt(tilt), pose(), (CX, CY), h_above_plane=h)
    d = tilt * DEG
    assert est.range_m == pytest.approx(h / np.sin(d))
    assert est.ground_dist_m == pytest.approx(h / np.tan(d))
    assert est.depression_deg == pytest.approx(tilt)
    assert est.azimuth_deg == pytest.approx(0.0, abs=1e-9)
    assert est.rel_azimuth_deg == pytest.approx(0.0, abs=1e-9)


def test_drone_pitch_adds_to_camera_tilt():
    est = estimate_range(cam_tilt(30), pose(pitch=-10 * DEG), (CX, CY), h_above_plane=20.0)
    assert est.depression_deg == pytest.approx(40.0)


def test_azimuth_is_clockwise_from_north_and_relative_azimuth_wraps():
    est = estimate_range(cam_tilt(30), pose(yaw=90 * DEG), (CX, CY), h_above_plane=20.0)
    assert est.azimuth_deg == pytest.approx(90.0)
    assert est.rel_azimuth_deg == pytest.approx(0.0, abs=1e-9)
    est = estimate_range(cam_tilt(30), pose(yaw=-90 * DEG), (CX, CY), h_above_plane=20.0)
    assert est.azimuth_deg == pytest.approx(270.0)
    # Pixel left of centre with a forward camera on a north-facing drone -> west of north.
    est = estimate_range(cam_tilt(30), pose(), (CX - 400, CY), h_above_plane=20.0)
    assert 270 < est.azimuth_deg < 360
    assert -90 < est.rel_azimuth_deg < 0


def test_horizon_ray_is_invalid():
    est = estimate_range(cam_tilt(0), pose(), (CX, CY), h_above_plane=20.0)
    assert not est.valid and est.reason == REASON_ABOVE_HORIZON


def test_ray_above_horizon_is_invalid():
    est = estimate_range(cam_tilt(0), pose(), (CX, CY - 100), h_above_plane=20.0)
    assert not est.valid and est.reason == REASON_ABOVE_HORIZON
    assert np.isnan(est.range_m)


@pytest.mark.parametrize("h", [0.0, -5.0])
def test_camera_not_above_plane_is_invalid(h):
    est = estimate_range(cam_tilt(90), pose(), (CX, CY), h_above_plane=h)
    assert not est.valid and est.reason == REASON_NOT_ABOVE_PLANE


def test_round_trip_random_poses_recovers_ground_point():
    rng = np.random.default_rng(2)
    for _ in range(50):
        tilt = rng.uniform(20, 90)
        cam = cam_tilt(tilt)
        P = pose(
            rng.uniform(-np.pi, np.pi),
            rng.uniform(-0.2, 0.2),
            rng.uniform(-0.2, 0.2),
            p=(rng.uniform(-50, 50), rng.uniform(-50, 50), -rng.uniform(5, 60)),
        )
        uv = np.array([CX, CY]) + rng.uniform(-200, 200, size=2)
        C_w, d_w = pixel_to_world_ray(cam, P, uv)
        if d_w[2] <= 0:
            continue
        z_g = 3.0  # ground at down = +3 m
        h = z_g - C_w[2]
        est = estimate_range(cam, P, uv, h_above_plane=h)
        assert est.valid
        assert est.point_w[2] == pytest.approx(z_g)
        uv_back, _ = project_world_point(cam, P, est.point_w)
        np.testing.assert_allclose(uv_back, uv, atol=1e-8)
        assert est.range_m == pytest.approx(np.linalg.norm(est.point_w - C_w))


def test_intersect_horizontal_plane_direct():
    est = intersect_horizontal_plane(np.array([1.0, 2.0, -10.0]), np.array([1.0, 0.0, 1.0]), 0.0)
    np.testing.assert_allclose(est.point_w, [11.0, 2.0, 0.0])
    assert est.range_m == pytest.approx(10 * np.sqrt(2))
    assert est.ground_dist_m == pytest.approx(10.0)
    assert est.depression_deg == pytest.approx(45.0)


def test_general_plane_matches_horizontal_plane():
    C, d = np.array([0.0, 0.0, -10.0]), np.array([0.3, -0.2, 1.0])
    a = intersect_horizontal_plane(C, d, 2.0)
    b = intersect_plane(C, d, normal=np.array([0, 0, 1.0]), offset=2.0)
    np.testing.assert_allclose(a.point_w, b.point_w)
    assert a.range_m == pytest.approx(b.range_m)


def test_general_sloped_plane():
    # Plane through origin rising to the north by 10 degrees: n . X = 0.
    s = 10 * DEG
    n = np.array([np.sin(s), 0.0, np.cos(s)])
    C, d = np.array([0.0, 0.0, -20.0]), np.array([1.0, 0.0, 1.0])
    est = intersect_plane(C, d, n, 0.0)
    assert est.valid
    assert n @ est.point_w == pytest.approx(0.0, abs=1e-12)


def test_general_plane_parallel_or_behind_is_invalid():
    n = np.array([0.0, 0.0, 1.0])
    est = intersect_plane(np.array([0, 0, -10.0]), np.array([1.0, 0, 0]), n, 0.0)
    assert not est.valid and est.reason == REASON_PARALLEL
    est = intersect_plane(np.array([0, 0, -10.0]), np.array([0, 0, -1.0]), n, 0.0)
    assert not est.valid


# --- contact-point rules ---------------------------------------------------------------------


def test_contact_pixel_modes():
    box = (100.0, 200.0, 40.0, 60.0)  # left, top, width, height
    assert contact_pixel(box, "bottom_centre") == pytest.approx((120.0, 260.0))
    assert contact_pixel(box, "centre_with_height") == pytest.approx((120.0, 230.0))
    with pytest.raises(ValueError):
        contact_pixel(box, "top")


def test_select_contact_mode_uses_threshold():
    assert select_contact_mode(85.0, nadir_threshold_deg=70.0) == "centre_with_height"
    assert select_contact_mode(40.0, nadir_threshold_deg=70.0) == "bottom_centre"


def test_centre_with_height_recovers_object_ground_position_at_nadir():
    """A 1.5 m object seen straight down: its box centre images the top, not the ground."""
    cam, P = cam_tilt(90), pose()
    obj_h = 1.5  # synthetic test height
    ground = np.array([6.0, -4.0, 0.0])
    top = ground - [0.0, 0.0, obj_h]
    uv_top, _ = project_world_point(cam, P, top)
    good = estimate_range(cam, P, uv_top, 20.0, mode="centre_with_height", obj_height_m=obj_h)
    np.testing.assert_allclose(good.point_w, ground, atol=1e-9)
    assert good.range_m == pytest.approx(np.linalg.norm(ground - P.p_w))
    naive = estimate_range(cam, P, uv_top, 20.0, mode="bottom_centre")
    assert np.linalg.norm(naive.point_w[:2] - ground[:2]) > 0.1  # biased outwards


def test_estimate_range_argument_checks():
    with pytest.raises(ValueError):
        estimate_range(cam_tilt(90), pose(), (CX, CY), 20.0, mode="centre_with_height")
    with pytest.raises(ValueError):
        estimate_range(cam_tilt(90), pose(), (CX, CY), 20.0, mode="nope")


def test_object_taller_than_camera_height_is_invalid():
    est = estimate_range(
        cam_tilt(90), pose(), (CX, CY), 2.0, mode="centre_with_height", obj_height_m=3.0
    )
    assert not est.valid and est.reason == REASON_NOT_ABOVE_PLANE
