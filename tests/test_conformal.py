"""Split conformal calibration of range intervals (plan B), on SYNTHETIC data."""

import numpy as np
import pytest

from aerialrange.evaluate.conformal import apply_conformal, conformal_quantile, fit_conformal

LEVELS = (0.5, 0.68, 0.9, 0.95)


def data(n, scale, seed, bins=None):
    rng = np.random.default_rng(seed)
    sig = rng.uniform(0.5, 5.0, n)
    s = np.ones(n) * scale if bins is None else np.where(bins == 1, scale[1], scale[0])
    return s * sig * rng.normal(size=n), sig


def test_conformal_quantile_finite_sample_rule():
    scores = np.arange(1, 101, dtype=float)  # 1..100
    # ceil((n + 1) * level) / n quantile: level 0.9 -> rank ceil(101 * 0.9) = 91
    assert conformal_quantile(scores, 0.9) == 91.0
    assert conformal_quantile(scores, 0.5) == 51.0
    assert np.isinf(conformal_quantile(np.arange(5.0), 0.99))  # too few points: no finite bound


def test_fixes_coverage_of_a_misscaled_sigma():
    e_cal, s_cal = data(5000, 2.3, seed=0)
    model = fit_conformal(e_cal, s_cal, LEVELS)
    e_new, s_new = data(20000, 2.3, seed=1)
    for lv in LEVELS:
        hw = apply_conformal(s_new, model, lv)
        cov = np.mean(np.abs(e_new) <= hw)
        assert cov == pytest.approx(lv, abs=0.015)
        naive = np.mean(
            np.abs(e_new) <= s_new * {0.5: 0.674, 0.68: 0.994, 0.9: 1.645, 0.95: 1.96}[lv]
        )
        assert naive < lv - 0.1  # the uncalibrated sigma under-covers


def test_mondrian_bins_give_conditional_coverage():
    rng = np.random.default_rng(2)
    bins_cal = rng.integers(0, 2, 6000)
    e_cal, s_cal = data(6000, (1.0, 4.0), seed=3, bins=bins_cal)
    model = fit_conformal(e_cal, s_cal, LEVELS, groups=bins_cal, min_group=200)
    bins_new = rng.integers(0, 2, 20000)
    e_new, s_new = data(20000, (1.0, 4.0), seed=4, bins=bins_new)
    hw = apply_conformal(s_new, model, 0.9, groups=bins_new)
    for b in (0, 1):
        m = bins_new == b
        assert np.mean(np.abs(e_new[m]) <= hw[m]) == pytest.approx(0.9, abs=0.02)
    hw_global = apply_conformal(s_new, fit_conformal(e_cal, s_cal, LEVELS), 0.9)
    assert np.mean(np.abs(e_new[bins_new == 1]) <= hw_global[bins_new == 1]) < 0.85


def test_small_group_falls_back_to_global():
    e, s = data(1000, 1.0, seed=5)
    g = np.zeros(1000, int)
    g[:20] = 7  # tiny group
    model = fit_conformal(e, s, (0.9,), groups=g, min_group=100)
    assert 7 not in model["groups"]
    hw = apply_conformal(np.array([1.0]), model, 0.9, groups=np.array([7]))
    assert hw[0] == pytest.approx(model["global"][0.9])


def test_unknown_level_rejected():
    e, s = data(100, 1.0, seed=6)
    with pytest.raises(KeyError):
        apply_conformal(s, fit_conformal(e, s, (0.9,)), 0.8)
