"""AU-AIR scale self-calibration (D-508): altimeter offset, focal length, frame motion."""

import cv2
import numpy as np
import pytest

from aerialrange.evaluate.auair_scale import (
    fit_linear_drift,
    focal_from_motion,
    h0_from_scale,
    similarity_motion,
)


def test_h0_from_scale_recovers_a_known_offset():
    # true heights 12 and 18 m with altitude reading 2 m too high: alt = true + 2, h0 = -2
    h0 = -2.0
    alt_i, alt_j = 12.0 - h0, 18.0 - h0
    s = 12.0 / 18.0  # image shrinks from i to j by true-height ratio (true_i / true_j)
    assert h0_from_scale(alt_i, alt_j, s) == pytest.approx(h0)


def test_h0_from_scale_is_nan_without_scale_change():
    assert np.isnan(h0_from_scale(10.0, 10.0, 1.0))


def test_focal_from_motion():
    # 3 m of ground motion seen at 20 m true height moves the image 225 px: f = 1500
    assert focal_from_motion(225.0, 3.0, 20.0) == pytest.approx(1500.0)


def test_fit_linear_drift_and_its_prediction():
    t = np.array([0.0, 100, 200, 300])
    a, b = fit_linear_drift(t, -1.0 - 0.01 * t)
    assert a == pytest.approx(-1.0) and b == pytest.approx(-0.01)


def test_fit_linear_drift_is_robust_to_one_outlier():
    t = np.arange(10.0)
    y = 2.0 + 0.5 * t
    y[3] = 40.0
    a, b = fit_linear_drift(t, y)
    assert a == pytest.approx(2.0, abs=0.2) and b == pytest.approx(0.5, abs=0.05)


def _texture(h=360, w=640, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.uint8), (0, 0), 2)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def test_similarity_motion_recovers_shift_and_scale():
    a = _texture()
    c = (a.shape[1] / 2, a.shape[0] / 2)
    M = cv2.getRotationMatrix2D(c, 0.0, 0.95)  # zoom out about the centre (climb)
    M[:, 2] += (12.0, -7.0)  # plus a translation
    b = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_REFLECT)
    r = similarity_motion(a, b)
    assert r is not None
    shift, scale, n = r
    assert scale == pytest.approx(0.95, abs=0.005)
    assert shift == pytest.approx(np.hypot(12.0, -7.0), abs=0.5)
    assert n >= 40
