"""Stationary-target discovery without flight metadata."""

import numpy as np
import pytest

from aerialrange.evaluate.stationary import residual_steps, stationary_segments


def _shift(dx, dy):
    return np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])


def test_residual_steps_remove_camera_motion():
    # camera pans: the background moves 10 px/frame left in the image; a parked car moves with
    # it (zero residual); a car holding its image position is moving 10 px/frame against it
    H = [_shift(-10, 0)] * 4  # maps frame k to frame k+1
    frames = np.arange(1, 6)
    parked = np.column_stack([frames, 100 - 10 * (frames - 1), np.full(5, 50.0)])
    moving = np.column_stack([frames, np.full(5, 100.0), np.full(5, 50.0)])
    np.testing.assert_allclose(residual_steps(parked, H), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(residual_steps(moving, H), axis=1), 10.0)


def test_residual_steps_mark_gaps_in_a_track():
    H = [_shift(0, 0)] * 5
    track = np.array([[1, 0.0, 0.0], [2, 1.0, 0.0], [4, 1.0, 0.0]])  # frame 3 missing
    r = residual_steps(track, H)
    assert r[0, 0] == pytest.approx(1.0) and np.isnan(r[1]).all()


def _steps(n, v=(0.0, 0.0)):
    return np.tile(np.asarray(v, float), (n, 1))


def test_stationary_segments_split_at_a_jump():
    t = np.arange(0, 6.0, 0.2)  # 5 fps, 30 frames
    steps = (
        _steps(29, (0.3, 0.0)) * np.array([[1, 0]]) * np.where(np.arange(29) % 2, 1, -1)[:, None]
    )
    steps[14] = (9.0, 0.0)
    assert stationary_segments(steps, t, thr_px=2.0, min_s=2.0) == [(0, 14), (15, 29)]


def test_jitter_cancels_but_creeping_adds_up():
    t = np.arange(0, 6.0, 0.2)
    rng = np.random.default_rng(0)
    jitter = rng.normal(0, 0.3, (29, 2))
    jitter -= jitter.mean(axis=0)  # zero net drift
    assert stationary_segments(jitter, t, thr_px=3.0, min_s=2.0) == [(0, 29)]
    assert stationary_segments(_steps(29, (0.5, 0.0)), t, thr_px=2.0, min_s=2.0) == []


def test_stationary_segments_reject_short_stops_and_gaps():
    t = np.arange(0, 1.0, 0.2)
    assert stationary_segments(_steps(4), t, thr_px=2.0, min_s=2.0) == []
    t = np.arange(0, 6.0, 0.2)
    s = _steps(29)
    s[10] = np.nan  # a gap ends a segment
    assert stationary_segments(s, t, thr_px=2.0, min_s=2.0) == [(0, 10), (11, 29)]


def test_residual_steps_bridge_short_gaps_by_chaining_camera_motion():
    # background moves 10 px/frame left; a parked car is missed in frame 3
    H = [_shift(-10, 0)] * 5
    track = np.array([[1, 100.0, 50.0], [2, 90.0, 50.0], [4, 70.0, 50.0]])
    r = residual_steps(track, H, max_gap=5)
    np.testing.assert_allclose(r, 0.0, atol=1e-9)
    assert np.isnan(residual_steps(track, H, max_gap=1)[1]).all()  # default: no bridging
