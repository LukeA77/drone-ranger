"""Speed truth from box positions, on synthetic tracks (labelled synthetic)."""

import numpy as np
import pytest

from aerialrange.evaluate.syndrone_speed import (
    heading_deg_ned,
    jitter_sigma,
    savgol_velocity,
    speed_truth,
    split_at_jumps,
    velocity_sigma,
)

DT = 0.04


def test_constant_velocity_exact():
    t = np.arange(100) * DT
    pos = np.stack([3.0 + 5.0 * t, -2.0 + 1.0 * t, np.zeros_like(t)], 1)
    v = savgol_velocity(pos, 9, DT)
    ok = ~np.isnan(v[:, 0])
    assert ok.sum() == 100 - 8  # half a window lost at each end
    assert np.allclose(v[ok], [5.0, 1.0, 0.0])


def test_quadratic_motion_exact_at_centre():
    t = np.arange(60) * DT
    pos = np.stack([0.5 * 2.0 * t**2, np.zeros_like(t), np.zeros_like(t)], 1)  # a = 2 m/s^2
    v = savgol_velocity(pos, 7, DT)
    ok = ~np.isnan(v[:, 0])
    assert np.allclose(v[ok, 0], 2.0 * t[ok], atol=1e-9)


def test_noise_level_and_velocity_sigma():
    rng = np.random.default_rng(1)
    t = np.arange(4000) * DT
    clean = np.stack([10 * np.sin(0.2 * t), 4 * t, np.zeros_like(t)], 1)
    sig = 0.02
    pos = clean + rng.normal(0, sig, clean.shape)
    assert jitter_sigma(pos) == pytest.approx(sig, rel=0.1)
    n = 11
    v = savgol_velocity(pos, n, DT)
    vt = np.stack([2 * np.cos(0.2 * t), 4 + 0 * t], 1)
    ok = ~np.isnan(v[:, 0])
    emp = np.std(v[ok, :2] - vt[ok], axis=0).mean()
    assert emp == pytest.approx(velocity_sigma(sig, n, DT), rel=0.1)


def test_split_at_jumps():
    frames = np.arange(20)
    pos = np.zeros((20, 3))
    pos[:, 0] = np.arange(20) * 0.4  # 10 m/s
    pos[10:, 0] += 300.0  # respawn
    segs = split_at_jumps(frames, pos, DT, vmax_ms=70.0)
    assert [(s.start, s.stop) for s in segs] == [(0, 10), (10, 20)]
    frames2 = np.r_[0:5, 8:12]  # a gap in frames also splits
    segs2 = split_at_jumps(frames2, np.zeros((9, 3)), DT, vmax_ms=70.0)
    assert [(s.start, s.stop) for s in segs2] == [(0, 5), (5, 9)]


def test_heading_convention():
    # Unreal +x is North (heading 0), +y is East (heading 90).
    assert heading_deg_ned(np.array([[1.0, 0, 0]]))[0] == pytest.approx(0.0)
    assert heading_deg_ned(np.array([[0, 2.0, 0]]))[0] == pytest.approx(90.0)
    assert heading_deg_ned(np.array([[-1.0, -1e-9, 0]]))[0] == pytest.approx(180.0, abs=1e-6)


def test_speed_truth_end_to_end():
    frames = np.arange(80)
    pos = np.zeros((80, 3))
    pos[:, 1] = frames * DT * 8.0  # 8 m/s east
    pos[40:, 1] += 500.0  # teleport at frame 40
    out = speed_truth(frames, pos, DT, window=9, vmax_ms=70.0, sigma_pos=0.0)
    assert np.isnan(out["speed"][:4]).all() and np.isnan(out["speed"][36:44]).all()
    good = ~np.isnan(out["speed"])
    assert np.allclose(out["speed"][good], 8.0)
    assert np.allclose(out["heading_deg"][good], 90.0)
    assert np.all(out["segment"][:40] == 0) and np.all(out["segment"][40:] == 1)
