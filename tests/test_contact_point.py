import numpy as np
import pytest

from aerialrange.evaluate.contact_point import PinholeView, box_corners, simulate

VIEW = PinholeView(f_px=1259, width=1920, height=1080, h_m=20.0, tilt_deg=30.0)


def test_project_and_ground_point_are_inverse():
    rng = np.random.default_rng(0)
    uv = np.stack([rng.uniform(0, 1920, 50), rng.uniform(300, 1080, 50)], axis=1)
    P = VIEW.ground_point(uv)
    back, front = VIEW.project(P)
    assert front.all()
    np.testing.assert_allclose(back, uv, atol=1e-6)
    np.testing.assert_allclose(P[:, 2], 0.0, atol=1e-9)


def test_box_corners_footprint_and_height():
    c = box_corners(np.array([[10.0, 5.0]]), np.array([0.0]), np.array([[4.0, 2.0, 1.5]]))[0]
    assert c[:4, 2].max() == 0 and np.allclose(c[4:, 2], -1.5)
    np.testing.assert_allclose(c[:, 0].min(), 8.0)
    np.testing.assert_allclose(c[:, 1].max(), 6.0)


def test_point_object_without_noise_has_no_error():
    tiny = (1e-4, 1e-4, 1e-4)
    r = simulate(VIEW, 500, tiny, tiny, 0.0, 0.0, np.random.default_rng(1))
    assert len(r["rel"]) > 400
    assert np.abs(r["rel"]).max() < 1e-4


def test_car_at_an_angle_reads_short_with_bottom_centre():
    car = (4.5, 1.8, 1.5)
    r = simulate(VIEW, 2000, car, car, 0.0, 0.0, np.random.default_rng(2))
    assert np.median(r["rel"]) < 0  # the box bottom is the footprint's near edge


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        simulate(VIEW, 10, (1, 1, 1), (1, 1, 1), 0, 0, np.random.default_rng(0), mode="top")
