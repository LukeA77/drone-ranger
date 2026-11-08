"""Footprint-centre correction (pipeline/footprint.py)."""

import numpy as np
import pytest

from aerialrange.pipeline.footprint import FootprintSize, forward_offset

CAR = FootprintSize(4.6, 1.9, 0.0, 0.0, 0.0)  # a fixed size: only the heading varies


def test_known_heading_gives_the_rectangle_support_half_width():
    m, v = forward_offset(CAR, [0.0, np.pi / 2, np.pi], [1e-9, 1e-9, 1e-9])
    np.testing.assert_allclose(m, [2.3, 0.95, 2.3], atol=1e-6)
    np.testing.assert_allclose(v, 0.0, atol=1e-9)
    m, _ = forward_offset(CAR, [np.pi / 4], [1e-9])
    assert m[0] == pytest.approx((2.3 + 0.95) * np.sqrt(0.5), abs=1e-6)


def test_unknown_heading_averages_to_l_plus_w_over_pi():
    for sig in (np.nan, 10.0):
        m, v = forward_offset(CAR, [np.nan if np.isnan(sig) else 0.3], [sig])
        assert m[0] == pytest.approx((4.6 + 1.9) / np.pi, rel=1e-3)
        assert v[0] > 0
    # a broad but finite heading sigma tends to the uniform answer
    m_broad, _ = forward_offset(CAR, [0.0], [2.5])
    assert abs(m_broad[0] - (4.6 + 1.9) / np.pi) < 0.1


def test_size_spread_adds_variance():
    spread = FootprintSize(4.6, 1.9, 4.0, 0.2, 0.5)
    _, v0 = forward_offset(CAR, [0.0], [1e-9])
    _, v1 = forward_offset(spread, [0.0], [1e-9])
    assert v1[0] == pytest.approx(4.0 / 4, rel=1e-6) and v0[0] == pytest.approx(0.0, abs=1e-12)
