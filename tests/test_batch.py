"""Batch ranging and propagation must match the scalar, already-tested functions."""

import numpy as np
import pytest

from aerialrange.geometry import frames as fr
from aerialrange.geometry.camera import Camera, Pose
from aerialrange.geometry.ranging import estimate_range, estimate_range_batch
from aerialrange.geometry.uncertainty import propagate_batch, propagate_uncertainty

DEG = np.pi / 180
CAM = Camera.pinhole(512.0, 512.0, 512.0, 512.0, R_bc=fr.R_bc_from_tilt(10 * DEG))  # synthetic
POSE = Pose.from_ypr([3.0, -2.0, -40.0], 0.7, -0.25, 0.08)
H = 38.0


def _uv(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1024, size=(n, 2))


def test_batch_matches_scalar_estimates():
    uv = _uv()
    b = estimate_range_batch(CAM, POSE, uv, H)
    for i, p in enumerate(uv):
        s = estimate_range(CAM, POSE, p, H)
        assert b["valid"][i] == s.valid
        if s.valid:
            np.testing.assert_allclose(b["point_w"][i], s.point_w, rtol=1e-12, atol=1e-9)
            assert b["range_m"][i] == pytest.approx(s.range_m)
            assert b["ground_dist_m"][i] == pytest.approx(s.ground_dist_m)
            assert b["depression_deg"][i] == pytest.approx(s.depression_deg)
        else:
            assert np.isnan(b["range_m"][i])
    assert b["valid"].any() and not b["valid"].all()  # both cases exercised


def test_batch_accepts_per_sample_height():
    uv = _uv(5)
    h = np.linspace(20, 40, 5)
    b = estimate_range_batch(CAM, POSE, uv, h)
    for i in range(5):
        s = estimate_range(CAM, POSE, uv[i], h[i])
        if s.valid:
            assert b["range_m"][i] == pytest.approx(s.range_m)


def test_batch_propagation_matches_scalar():
    uv = _uv(25, seed=1)
    sig = {
        "u": 1.5,
        "v": 2.0,
        "h": 0.4,
        "roll": 0.3 * DEG,
        "pitch": 0.3 * DEG,
        "yaw": 0.5 * DEG,
        "tilt": 0.2 * DEG,
        "fx": 3.0,
        "fy": 3.0,
    }
    b = propagate_batch(CAM, POSE, uv, H, sig)
    n_valid = 0
    for i, p in enumerate(uv):
        s = propagate_uncertainty(CAM, POSE, p, H, sig)
        if not s.valid:
            assert not b["valid"][i]
            continue
        n_valid += 1
        np.testing.assert_allclose(b["cov_ne"][i], s.cov_ne, rtol=1e-5, atol=1e-10)
        assert b["range_sigma_m"][i] == pytest.approx(s.range_sigma_m, rel=1e-5)
    assert n_valid > 5


def test_batch_propagation_per_pixel_sigma():
    uv = _uv(4, seed=2)
    su = np.array([1.0, 2.0, 3.0, 4.0])
    b = propagate_batch(CAM, POSE, uv, H, {"u": su})
    for i in range(4):
        s = propagate_uncertainty(CAM, POSE, uv[i], H, {"u": float(su[i])})
        if s.valid:
            assert b["range_sigma_m"][i] == pytest.approx(s.range_sigma_m, rel=1e-5)


def test_batch_sigma_is_zero_not_nan_for_an_input_with_no_range_effect():
    """Yaw does not change range for a level drone; rounding must not give NaN (regression)."""
    level = Pose.from_ypr([0.0, 0.0, -30.0], 0.4, 0.0, 0.0)
    b = propagate_batch(CAM, level, _uv(50, seed=3), 30.0, {"yaw": 1.0 * DEG})
    ok = b["valid"]
    assert ok.any()
    assert np.all(np.isfinite(b["range_sigma_m"][ok]))
    np.testing.assert_allclose(b["range_sigma_m"][ok], 0.0, atol=1e-6)
