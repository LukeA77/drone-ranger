"""Speed display: filter outputs and the rule that decides when to show speed."""

import numpy as np
import pytest

from aerialrange.geometry.kalman import GroundKalmanBatch
from aerialrange.pipeline.filterbank import TrackFilterBank
from aerialrange.pipeline.speed_display import (
    SpeedDisplayRule,
    format_speed_label,
    nis_bounds,
)


def test_batch_update_returns_nis():
    b = GroundKalmanBatch.from_two_points(
        np.zeros((1, 2)), np.eye(2)[None], np.zeros((1, 2)), np.eye(2)[None], 1.0, 0.0
    )
    b.predict(1.0)
    P = b.P[0, :2, :2].copy()
    nis = b.update(np.array([[1.0, 2.0]]), np.eye(2)[None] * 0.5)
    S = P + np.eye(2) * 0.5
    assert nis[0] == pytest.approx(np.array([1.0, 2.0]) @ np.linalg.solve(S, [1.0, 2.0]))


def test_batch_heading_convention_and_sigma():
    b = GroundKalmanBatch(
        np.array([[0, 0, 0.0, 5.0], [0, 0, -3.0, 0.0]]), np.tile(np.eye(4) * 0.01, (2, 1, 1)), 1.0
    )
    h, hs = b.heading()
    assert h == pytest.approx([90.0, 180.0])  # east, south
    assert hs[0] == pytest.approx(np.degrees(0.1 / 5.0))


def test_bank_reports_age_heading_and_nis_window():
    bank = TrackFilterBank(q=0.5, max_unseen=30, nis_window=10)
    rng = np.random.default_rng(3)
    dt, sig = 0.08, 0.1
    for k in range(80):
        z = np.array([[2.0 * k * dt, 1.0 * k * dt]]) + rng.normal(0, sig, (1, 2))
        out = bank.step(dt, np.array([4]), z, np.array([sig]))
    assert out["age"][0] == 78  # 80 measurements: two initialise, 78 update
    assert out["heading_deg"][0] == pytest.approx(np.degrees(np.arctan2(1, 2)), abs=3)
    assert out["nis_n"][0] == 10
    lo, hi = nis_bounds(10, 0.95)
    assert lo < out["nis_mean"][0] < hi  # a consistent filter on its own model


def test_nis_bounds_two_sided_chi_square():
    lo, hi = nis_bounds(10, 0.95)
    # mean of 10 two-dof NIS values: chi-square with 20 dof divided by 10
    assert lo == pytest.approx(9.591 / 10, abs=1e-3) and hi == pytest.approx(34.170 / 10, abs=1e-3)


@pytest.mark.parametrize(
    "sigma,age,nis_mean,nis_n,ok",
    [
        (0.5, 30, 2.0, 10, True),
        (2.0, 30, 2.0, 10, False),  # too uncertain
        (0.5, 3, 2.0, 10, False),  # too young
        (0.5, 30, 9.0, 10, False),  # innovations too large: inconsistent
        (0.5, 30, 2.0, 4, False),  # not enough recent updates to judge
        (np.nan, 30, 2.0, 10, False),
    ],
)
def test_display_rule(sigma, age, nis_mean, nis_n, ok):
    rule = SpeedDisplayRule(sigma_max_ms=1.0, min_age=10, nis_window=10, nis_conf=0.95)
    assert rule.ok(sigma, age, nis_mean, nis_n) is ok


def test_label_format():
    rule = SpeedDisplayRule(sigma_max_ms=1.0, min_age=10, nis_window=10, nis_conf=0.95)
    s = format_speed_label(rule, 5.0, 0.8, 30, 2.0, 10)
    assert s == "18 ± 3 km/h"
    assert format_speed_label(rule, 5.0, 3.0, 30, 2.0, 10) == "speed: settling"
