"""Pixel to ground position for straight-down AU-AIR frames."""

import numpy as np
import pytest

from aerialrange.evaluate.auair_world import HYPOTHESES, ground_offset_nadir, spread_rms


def test_image_up_is_forward_under_the_base_hypothesis():
    # 100 px above the centre at 20 m with f = 1000: 2 m ahead; heading north -> 2 m north
    dn, de = ground_offset_nadir(960.0, 440.0, h=20.0, f=1000.0, yaw=0.0, rot=0, mirror=False)
    assert (dn, de) == pytest.approx((2.0, 0.0))


def test_yaw_rotates_the_offset_into_north_east():
    dn, de = ground_offset_nadir(960.0, 440.0, h=20.0, f=1000.0, yaw=np.pi / 2, rot=0, mirror=False)
    assert (dn, de) == pytest.approx((0.0, 2.0), abs=1e-9)  # heading east: ahead is east
    dn, de = ground_offset_nadir(1060.0, 540.0, h=20.0, f=1000.0, yaw=0.0, rot=0, mirror=False)
    assert (dn, de) == pytest.approx((0.0, 2.0), abs=1e-9)  # image right is the drone's right


def test_mount_rotation_and_mirror():
    # rot=1: image up is the drone's right (rotate 90 deg); mirror flips left and right
    dn, de = ground_offset_nadir(960.0, 440.0, h=20.0, f=1000.0, yaw=0.0, rot=1, mirror=False)
    assert (dn, de) == pytest.approx((0.0, 2.0), abs=1e-9)
    dn, de = ground_offset_nadir(1060.0, 540.0, h=20.0, f=1000.0, yaw=0.0, rot=0, mirror=True)
    assert (dn, de) == pytest.approx((0.0, -2.0), abs=1e-9)


def test_eight_hypotheses():
    assert len(HYPOTHESES) == 8 and len(set(HYPOTHESES)) == 8


def test_spread_rms_about_the_mean():
    p = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    assert spread_rms(p) == pytest.approx(np.sqrt(2.0))


def test_oblique_matches_nadir_at_90_degrees():
    from aerialrange.evaluate.auair_world import ground_offset_oblique

    for u, v, yaw in ((960.0, 440.0, 0.0), (1200.0, 700.0, 0.7), (300.0, 100.0, -2.0)):
        a = ground_offset_nadir(u, v, h=20.0, f=1259.0, yaw=yaw, rot=0, mirror=False)
        b = ground_offset_oblique(u, v, h=20.0, f=1259.0, yaw=yaw, tilt_deg=90.0)
        np.testing.assert_allclose(b, a, atol=1e-9)


def test_oblique_centre_pixel_lands_at_h_over_tan_tilt_ahead():
    from aerialrange.evaluate.auair_world import ground_offset_oblique

    dn, de = ground_offset_oblique(960.0, 540.0, h=20.0, f=1259.0, yaw=0.0, tilt_deg=45.0)
    assert (dn, de) == pytest.approx((20.0, 0.0))
    dn, de = ground_offset_oblique(960.0, 540.0, h=20.0, f=1259.0, yaw=np.pi / 2, tilt_deg=30.0)
    assert (dn, de) == pytest.approx((0.0, 20.0 / np.tan(np.radians(30.0))), abs=1e-9)


def test_oblique_pixel_above_the_horizon_has_no_ground_point():
    from aerialrange.evaluate.auair_world import ground_offset_oblique

    dn, de = ground_offset_oblique(960.0, 0.0, h=20.0, f=1259.0, yaw=0.0, tilt_deg=10.0)
    assert np.isnan(dn) and np.isnan(de)


@pytest.mark.parametrize("tilt", [90.0, 60.0, 35.0, 15.0])
def test_project_ground_inverts_ground_offset_oblique(tilt):
    from aerialrange.evaluate.auair_world import ground_offset_oblique, project_ground

    rng = np.random.default_rng(int(tilt))
    u = rng.uniform(0, 1920, 50)
    v = rng.uniform(560 if tilt < 25 else 0, 1080, 50)  # below the horizon for shallow tilts
    dn, de = ground_offset_oblique(u, v, h=18.0, f=1259.0, yaw=0.7, tilt_deg=tilt)
    ok = np.isfinite(dn)
    u2, v2 = project_ground(dn[ok], de[ok], h=18.0, f=1259.0, yaw=0.7, tilt_deg=tilt)
    np.testing.assert_allclose(u2, u[ok], atol=1e-6)
    np.testing.assert_allclose(v2, v[ok], atol=1e-6)


def test_project_ground_behind_the_camera_is_nan():
    from aerialrange.evaluate.auair_world import project_ground

    u, v = project_ground(-50.0, 0.0, h=10.0, f=1259.0, yaw=0.0, tilt_deg=20.0)
    assert np.isnan(u) and np.isnan(v)
