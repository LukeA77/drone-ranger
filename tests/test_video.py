import numpy as np
import pytest

from aerialrange.viz import video as v


def test_target_kbps_fits_budget():
    kbps = v.target_kbps(20, 80)
    assert kbps * 1000 * 80 / 8 <= 20e6  # bits/s x seconds, in bytes
    assert kbps * 1000 * 80 / 8 >= 0.85 * 20e6  # and does not waste much of it
    with pytest.raises(ValueError):
        v.target_kbps(20, 0)


def test_hex_bgr():
    assert v.hex_bgr("#0ca30c") == (0x0C, 0xA3, 0x0C)
    assert v.hex_bgr("fab219") == (0x19, 0xB2, 0xFA)


def test_fit_keeps_aspect():
    img = np.zeros((1024, 2048, 3), np.uint8)
    out, s = v.fit(img, 640, 640)
    assert out.shape == (320, 640, 3) and s == pytest.approx(0.3125)


def test_text_draws_and_wraps():
    img = v.blank()
    v.text(img, [("Hello", (10, 10), 40, v.INK, True)])
    assert (img != v.blank()).any()
    lines = v.wrap("one two three four five six seven eight", 30, 200)
    assert len(lines) > 1
    assert all(v.font(30).getlength(s) <= 200 for s in lines)
